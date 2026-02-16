from __future__ import annotations

import json
import logging
import os
import secrets
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from atlas.backtest.engine import BacktestConfig, run_backtest
from atlas.backtest.plots import write_equity_vs_benchmark_artifacts
from atlas.backtest.window_analysis import rolling_window_summary, write_window_analysis_json
from atlas.backtest.derivatives_engine import run_derivatives_backtest
from atlas.config import (
    AlpacaSettings,
    get_alpaca_settings,
    get_default_max_position_notional_usd,
    get_log_level,
)
from atlas.evaluation.orchestrator import EvaluationConfig, run_evaluate_all
from atlas.data.benchmarks import spy_total_return
from atlas.data.bars import parse_bar_timeframe
from atlas.data.universe import load_universe_bars
from atlas.logging_utils import setup_logging
from atlas.market import Market, coerce_symbols_for_market, default_symbols, parse_market
from atlas.paper.runner import PaperConfig, run_paper_loop
from atlas.strategies.registry import build_strategy
# Textual (TUI) is an optional dependency. Import lazily in the `tui` command.
from atlas.utils.time import now_ny, parse_iso_datetime

from atlas.ml.tune import (
    ObjectiveConfig,
    TuneConfig,
    WalkForwardConfig,
    parse_duration_spec,
    tune_walk_forward,
)
from atlas.ml.validate import walk_forward_evaluate

app = typer.Typer(add_completion=False)
logger = logging.getLogger(__name__)


def _parse_float_csv(raw: Optional[str]) -> tuple[float, ...]:
    if raw is None:
        return tuple()
    parts = [p.strip() for p in str(raw).split(",") if p.strip()]
    return tuple(float(p) for p in parts)


def _infer_bar_minutes(index: pd.DatetimeIndex) -> float:
    if len(index) < 3:
        return 0.0
    diffs = index.to_series().diff().dropna().dt.total_seconds() / 60.0
    median = float(diffs.median())
    return median if median > 0 else 0.0


def _print_backtest_summary(
    *,
    run_dir: Path,
    symbols: list[str],
    data_source: str,
    data_hint: str,
    bar_index: pd.DatetimeIndex,
    strategy_name: str,
    strategy_params_hint: str,
    warmup_bars: int,
    cfg: BacktestConfig,
    elapsed_s: float,
) -> None:
    metrics = json.loads((run_dir / "metrics.json").read_text())
    strategy_total_return = float(metrics.get("total_return", 0.0) or 0.0)

    equity_curve = pd.read_csv(run_dir / "equity_curve.csv", parse_dates=["timestamp"])
    final_equity = float(equity_curve["equity"].iloc[-1])

    trades_path = run_dir / "trades.csv"
    trades = pd.read_csv(trades_path) if trades_path.exists() else pd.DataFrame()
    gross_notional = (
        float(trades["notional"].sum())
        if len(trades) and "notional" in trades.columns
        else 0.0
    )

    start_ts = pd.Timestamp(bar_index[0])
    end_ts = pd.Timestamp(bar_index[-1])
    duration = end_ts - start_ts
    bar_minutes = _infer_bar_minutes(bar_index)
    sessions = int(pd.Series(bar_index.date).nunique())

    benchmark_row: Optional[str] = None
    benchmark_alpha_row: Optional[str] = None
    try:
        spy = spy_total_return(start=start_ts.to_pydatetime(), end=end_ts.to_pydatetime())
        if spy is not None:
            (run_dir / "benchmark.json").write_text(
                json.dumps({"spy": spy.to_dict()}, indent=2)
            )
            benchmark_row = f"SPY {float(spy.total_return):.4%} ({spy.start_observed} → {spy.end_observed})"
            alpha = float(strategy_total_return) - float(spy.total_return)
            benchmark_alpha_row = f"alpha={alpha:.4%}  beat_spy={alpha > 0.0}"
    except Exception as exc:
        logger.warning("Failed to compute SPY benchmark for %s: %s", run_dir, exc)

    try:
        _ = Table
    except Exception:
        typer.echo(f"run_dir: {run_dir}")
        typer.echo(f"symbols: {','.join(symbols)}")
        typer.echo(f"data: {data_source} ({data_hint})")
        typer.echo(f"window: {start_ts.isoformat()} -> {end_ts.isoformat()}")
        typer.echo(
            f"bars: {len(bar_index)} sessions: {sessions} bar: {bar_minutes:.2f}m duration: {duration}"
        )
        typer.echo(f"strategy: {strategy_name} ({strategy_params_hint}) warmup_bars={warmup_bars}")
        typer.echo(
            "config: "
            f"initial_cash={cfg.initial_cash:.2f} "
            f"max_notional={cfg.max_position_notional_usd:.2f} "
            f"slippage_bps={cfg.slippage_bps:.2f} "
            f"taker_fee_bps={cfg.taker_fee_bps:.2f} "
            f"fixed_fee_per_contract_usd={cfg.fixed_fee_per_contract_usd:.4f} "
            f"contract_size_units={cfg.contract_size_units:.6g} "
            f"allow_short={cfg.allow_short}"
        )
        typer.echo(
            "results: "
            f"final_equity={final_equity:.2f} "
            f"total_return={strategy_total_return:.4%} "
            f"max_drawdown={metrics['max_drawdown']:.4%} "
            f"sharpe={metrics['sharpe']:.2f} "
            f"sharpe_daily={metrics.get('sharpe_daily', 0.0):.2f} "
            f"fills={metrics['trades']} "
            f"gross_notional={gross_notional:.2f}"
        )
        if benchmark_row is not None:
            typer.echo(f"benchmark: {benchmark_row}")
        if benchmark_alpha_row is not None:
            typer.echo(f"vs_spy: {benchmark_alpha_row}")
        typer.echo(f"elapsed: {elapsed_s:.2f}s")
        return

    console = Console()
    table = Table(title="Backtest summary", show_header=False)
    table.add_column("k", style="bold")
    table.add_column("v")

    table.add_row("run_dir", str(run_dir))
    table.add_row("symbols", ",".join(symbols))
    table.add_row("data", f"{data_source} ({data_hint})")
    table.add_row(
        "window",
        f"{start_ts.isoformat()} → {end_ts.isoformat()}  |  bars={len(bar_index)}  sessions={sessions}  bar={bar_minutes:.2f}m",
    )
    table.add_row("duration", str(duration))
    table.add_row("strategy", f"{strategy_name} ({strategy_params_hint})")
    table.add_row("warmup_bars", str(warmup_bars))
    table.add_row(
        "config",
        "  ".join(
            [
                f"initial_cash={cfg.initial_cash:.2f}",
                f"max_notional={cfg.max_position_notional_usd:.2f}",
                f"slippage_bps={cfg.slippage_bps:.2f}",
                f"taker_fee_bps={cfg.taker_fee_bps:.2f}",
                f"fixed_fee_per_contract_usd={cfg.fixed_fee_per_contract_usd:.4f}",
                f"contract_size_units={cfg.contract_size_units:.6g}",
                f"allow_short={cfg.allow_short}",
            ]
        ),
    )
    table.add_row(
        "results",
        "  ".join(
            [
                f"final_equity={final_equity:.2f}",
                f"total_return={strategy_total_return:.4%}",
                f"max_drawdown={metrics['max_drawdown']:.4%}",
                f"sharpe={metrics['sharpe']:.2f}",
                f"sharpe_daily={metrics.get('sharpe_daily', 0.0):.2f}",
                f"fills={metrics['trades']}",
                f"gross_notional={gross_notional:.2f}",
            ]
        ),
    )
    if benchmark_row is not None:
        table.add_row("benchmark", benchmark_row)
    if benchmark_alpha_row is not None:
        table.add_row("vs_spy", benchmark_alpha_row)
    table.add_row("elapsed", f"{elapsed_s:.2f}s")

    console.print(table)


def _run_id(prefix: str) -> str:
    # Include microseconds + pid + random suffix so parallel invocations don't collide.
    ts = datetime.now().strftime(f"{prefix}_%Y%m%d_%H%M%S_%f")
    return f"{ts}_{os.getpid()}_{secrets.token_hex(2)}"


def _load_strategy_params_for_name(path: Optional[Path], strategy_name: str) -> dict:
    if path is None:
        return {}
    raw = json.loads(path.read_text())
    if isinstance(raw, dict) and "params" in raw and isinstance(raw["params"], dict):
        raw = raw["params"]
    if isinstance(raw, dict) and "parameters" in raw and isinstance(raw["parameters"], dict):
        raw = raw["parameters"]
    if not isinstance(raw, dict):
        raise ValueError("strategy params json must be an object")

    canonical = strategy_name.replace("-", "_")
    if canonical in raw and isinstance(raw[canonical], dict):
        return dict(raw[canonical])
    if strategy_name in raw and isinstance(raw[strategy_name], dict):
        return dict(raw[strategy_name])
    return dict(raw)


@app.command()
def tui() -> None:
    try:
        from atlas.tui.app import run_tui
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "The TUI requires the optional 'textual' dependency. "
            "Install it (e.g. `pip install textual`) to use `atlas tui`."
        ) from exc
    run_tui()


@app.command()
def download_bars(
    symbol: str = typer.Option(..., help="US equity symbol, e.g. SPY"),
    start: str = typer.Option(..., help="ISO datetime, e.g. 2024-01-02T09:30:00-05:00"),
    end: str = typer.Option(..., help="ISO datetime, e.g. 2024-01-02T16:00:00-05:00"),
    timeframe: str = typer.Option(
        "1Min", help="Bar timeframe, e.g. 1Min, 5Min, 15Min, 30Min, 1H, 4H, 6H, 1D"
    ),
    feed: str = typer.Option(
        "delayed_sip",
        help="Alpaca data feed: iex, sip, delayed_sip (alias: uses sip but clamps end >=15m old).",
    ),
    out: Optional[Path] = typer.Option(None, help="Optional explicit output CSV path"),
) -> None:
    from atlas.data.alpaca_data import download_stock_bars_to_csv

    settings = get_alpaca_settings(require_keys=True)
    run_dir = Path("outputs") / "downloads" / _run_id("download")
    setup_logging(level=get_log_level(), log_file=run_dir / "run.log")

    download_stock_bars_to_csv(
        settings=settings,
        symbol=symbol,
        start=parse_iso_datetime(start),
        end=parse_iso_datetime(end),
        timeframe=timeframe,
        out_path=out,
        feed=feed,
    )


@app.command()
def backtest(
    market: str = typer.Option("equity", help="Market mode: equity|crypto|derivatives"),
    symbol: str = typer.Option("SPY", help="Symbol to backtest"),
    symbols: Optional[str] = typer.Option(
        None, help="Comma-separated symbols, e.g. SPY,QQQ (overrides --symbol)"
    ),
    data_source: str = typer.Option(
        "sample", help="sample|csv|alpaca|coinbase", show_default=True
    ),
    csv_path: Optional[Path] = typer.Option(None, help="CSV path when data-source=csv"),
    csv_dir: Optional[Path] = typer.Option(
        None, help="CSV directory with per-symbol files when data-source=csv and multiple symbols"
    ),
    bar_timeframe: str = typer.Option(
        "1Min", help="Bar timeframe, e.g. 1Min, 5Min, 15Min, 30Min, 1H, 4H, 6H, 1D"
    ),
    start: Optional[str] = typer.Option(
        None, help="ISO datetime (required for alpaca; optional filter otherwise)"
    ),
    end: Optional[str] = typer.Option(
        None, help="ISO datetime (required for alpaca; optional filter otherwise)"
    ),
    prewarm: Optional[str] = typer.Option(
        None,
        help="Load extra history before --start for indicator warmup (e.g. 30d, 12h). "
        "Trades/metrics are scored only on [start,end). Requires --start/--end.",
    ),
    alpaca_feed: str = typer.Option(
        "delayed_sip",
        help="When data-source=alpaca: iex, sip, delayed_sip (alias: uses sip but clamps end >=15m old).",
    ),
    strategy: str = typer.Option("spy_open_close", help="Strategy name"),
    strategy_params: Optional[Path] = typer.Option(
        None, help="JSON file with strategy parameters"
    ),
    fast_window: int = typer.Option(10, help="ma_crossover/ema_crossover fast window"),
    slow_window: int = typer.Option(30, help="ma_crossover/ema_crossover slow window"),
    initial_cash: float = typer.Option(100_000.0, help="Starting cash"),
    max_position_notional_usd: Optional[float] = typer.Option(
        None, help="Max notional per symbol"
    ),
    slippage_bps: Optional[float] = typer.Option(
        None,
        help="Fill cost per side in basis points (slippage/spread proxy). If omitted, nec_x/orb_trend default to 1.25 bps/side and nec_pdt defaults to 3.8 bps/side. Otherwise: equity=0.0, crypto=3.0, derivatives=1.25.",
    ),
    taker_fee_bps: Optional[float] = typer.Option(
        None,
        help="Taker fee in bps (per side). If omitted: equity=0.0, crypto=25.0, derivatives=10.0 on coinbase (otherwise 3.0).",
    ),
    fixed_fee_per_contract_usd: Optional[float] = typer.Option(
        None,
        help=(
            "Derivatives only: fixed fee in USD charged per contract per side. "
            "If omitted with market=derivatives and data-source=coinbase: defaults to 0.15."
        ),
    ),
    contract_size_units: Optional[float] = typer.Option(
        None,
        help=(
            "Derivatives only: contract size in underlying units (e.g. BTC nano perp = 0.01 BTC). "
            "If omitted with market=derivatives and data-source=coinbase: defaults to 0.01."
        ),
    ),
    coinbase_fee_model: bool = typer.Option(
        True,
        "--coinbase-fee-model/--no-coinbase-fee-model",
        help="Apply Coinbase fixed per-contract fee model (only active for market=derivatives + data-source=coinbase).",
    ),
    maintenance_margin_rate: float = typer.Option(
        0.05,
        help="Derivatives only: maintenance margin rate (e.g. 0.05 = 5%)",
    ),
    liquidation_fee_rate: float = typer.Option(
        0.005,
        help="Derivatives only: additional liquidation fee rate (e.g. 0.005 = 0.5%)",
    ),
    debug: bool = typer.Option(
        False,
        help="Write extra debug JSONL (decision snapshots + trade_debug.jsonl) under the run_dir.",
    ),
    allow_short: bool = typer.Option(False, help="Allow negative exposure"),
) -> None:
    run_dir = Path("outputs") / "backtests" / _run_id("backtest")
    setup_logging(level=get_log_level(), log_file=run_dir / "run.log")

    if max_position_notional_usd is None:
        max_position_notional_usd = get_default_max_position_notional_usd(mode="backtest")

    mkt = parse_market(market)

    tf = parse_bar_timeframe(bar_timeframe)
    start_dt = parse_iso_datetime(start) if start is not None else None
    end_dt = parse_iso_datetime(end) if end is not None else None
    score_start_dt: Optional[datetime] = None
    score_end_dt: Optional[datetime] = None
    load_start_dt = start_dt
    if prewarm is not None:
        if start_dt is None or end_dt is None:
            raise typer.BadParameter("--prewarm requires --start and --end")
        try:
            load_start_dt = start_dt - parse_duration_spec(prewarm)
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc
        score_start_dt = start_dt
        score_end_dt = end_dt

    canonical_strategy = str(strategy).strip().lower().replace("-", "_")
    if symbols is not None:
        raw_symbols = [s.strip() for s in (symbols or "").split(",") if s.strip()]
    else:
        if canonical_strategy in {"basis_carry", "hedge"} and mkt in {Market.CRYPTO, Market.DERIVATIVES}:
            raw_symbols = ["BTC/USD", "BTC-PERP"]
        else:
            raw_symbols = (
                default_symbols(mkt, count=2)
                if canonical_strategy in {"nec_x", "nec_pdt", "basis_carry", "hedge"}
                else [symbol.strip()]
            )
    universe_symbols = coerce_symbols_for_market(raw_symbols, mkt)

    alpaca_settings = get_alpaca_settings(require_keys=True) if data_source == "alpaca" else None
    try:
        universe = load_universe_bars(
            symbols=universe_symbols,
            data_source=data_source,
            timeframe=tf,
            start=load_start_dt,
            end=end_dt,
            csv_path=csv_path,
            csv_dir=csv_dir,
            alpaca_settings=alpaca_settings,
            alpaca_feed=alpaca_feed,
            market=mkt.value,
        )
    except FileNotFoundError as exc:
        raise typer.BadParameter(str(exc)) from exc
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    strat = build_strategy(
        name=strategy,
        params_path=strategy_params,
        symbols=universe_symbols,
        fast_window=fast_window,
        slow_window=slow_window,
    )

    default_slippage_bps = 0.0
    if mkt == Market.CRYPTO:
        default_slippage_bps = 3.0
    elif mkt == Market.DERIVATIVES:
        default_slippage_bps = 1.25

    default_taker_fee_bps = 0.0
    if mkt == Market.CRYPTO:
        default_taker_fee_bps = 25.0
    elif mkt == Market.DERIVATIVES:
        default_taker_fee_bps = 10.0 if str(data_source).strip().lower() == "coinbase" else 3.0

    use_coinbase_fee_model = bool(
        coinbase_fee_model and mkt == Market.DERIVATIVES and str(data_source).strip().lower() == "coinbase"
    )
    default_fixed_fee_per_contract_usd = 0.0
    default_contract_size_units = 1.0
    if use_coinbase_fee_model:
        default_fixed_fee_per_contract_usd = 0.15
        default_contract_size_units = 0.01
    effective_fixed_fee_per_contract_usd = (
        float(default_fixed_fee_per_contract_usd)
        if fixed_fee_per_contract_usd is None
        else float(fixed_fee_per_contract_usd)
    )
    effective_contract_size_units = (
        float(default_contract_size_units) if contract_size_units is None else float(contract_size_units)
    )
    if not use_coinbase_fee_model:
        effective_fixed_fee_per_contract_usd = 0.0
        effective_contract_size_units = 1.0
    if effective_contract_size_units <= 0.0:
        effective_contract_size_units = 1.0

    cfg = BacktestConfig(
        symbols=universe_symbols,
        initial_cash=initial_cash,
        max_position_notional_usd=float(max_position_notional_usd),
        slippage_bps=float(
            (
                1.25
                if strategy in {"nec_x", "nec-x", "orb_trend", "orb-trend"}
                else 3.8
                if strategy in {"nec_pdt", "nec-pdt"}
                else default_slippage_bps
            )
            if slippage_bps is None
            else slippage_bps
        ),
        allow_short=allow_short,
        taker_fee_bps=float(default_taker_fee_bps if taker_fee_bps is None else taker_fee_bps),
        fixed_fee_per_contract_usd=float(effective_fixed_fee_per_contract_usd),
        contract_size_units=float(effective_contract_size_units),
        maintenance_margin_rate=float(maintenance_margin_rate),
        liquidation_fee_rate=float(liquidation_fee_rate),
    )

    common_index: Optional[pd.DatetimeIndex] = None
    for sym in universe_symbols:
        idx = universe.bars_by_symbol[sym].index
        common_index = idx if common_index is None else common_index.intersection(idx)
    if common_index is None or len(common_index) < 3:
        raise typer.BadParameter("backtest window has too few aligned bars")
    common_index = common_index.sort_values()

    t0 = time.perf_counter()
    if mkt == Market.DERIVATIVES:
        run_derivatives_backtest(
            bars_by_symbol=universe.bars_by_symbol,
            strategy=strat,
            cfg=cfg,
            run_dir=run_dir,
            debug=debug,
            score_start=score_start_dt,
            score_end=score_end_dt,
            no_trade_before=score_start_dt,
        )
    else:
        run_backtest(
            bars_by_symbol=universe.bars_by_symbol,
            strategy=strat,
            cfg=cfg,
            run_dir=run_dir,
            debug=debug,
            score_start=score_start_dt,
            score_end=score_end_dt,
            no_trade_before=score_start_dt,
        )
    elapsed_s = time.perf_counter() - t0

    if strategy_params is not None:
        strategy_params_hint = f"params={strategy_params}"
    elif strategy in {"ma_crossover", "ema_crossover"}:
        strategy_params_hint = f"fast={fast_window} slow={slow_window}"
    else:
        strategy_params_hint = "defaults"

    score_index = common_index
    if score_start_dt is not None:
        score_index = score_index[score_index >= pd.Timestamp(score_start_dt)]
    if score_end_dt is not None:
        score_index = score_index[score_index < pd.Timestamp(score_end_dt)]
    if len(score_index) < 3:
        raise typer.BadParameter("backtest scoring window has too few aligned bars")

    _print_backtest_summary(
        run_dir=run_dir,
        symbols=universe_symbols,
        data_source=data_source,
        data_hint=universe.hint,
        bar_index=score_index,
        strategy_name=strategy,
        strategy_params_hint=strategy_params_hint,
        warmup_bars=strat.warmup_bars(),
        cfg=cfg,
        elapsed_s=elapsed_s,
    )
    try:
        csv_path, png_path = write_equity_vs_benchmark_artifacts(
            run_dir=run_dir,
            benchmark="spy.us",
        )
        if csv_path is not None:
            typer.echo(f"plot_csv: {csv_path}")
        if png_path is not None:
            typer.echo(f"plot_png: {png_path}")
    except Exception as exc:
        logger.warning("Failed to write benchmark plot for %s: %s", run_dir, exc)


@app.command("plot-run")
def plot_run(
    run_dir: Path = typer.Argument(..., help="Existing backtest run directory under outputs/"),
    benchmark: str = typer.Option("spy.us", help="Stooq benchmark symbol (e.g. spy.us)"),
    out_csv: Optional[Path] = typer.Option(None, help="Optional output CSV path"),
    out_png: Optional[Path] = typer.Option(None, help="Optional output PNG path"),
) -> None:
    """
    Generate a per-run plot comparing strategy equity vs a benchmark (default: SPY).
    """
    csv_path, png_path = write_equity_vs_benchmark_artifacts(
        run_dir=run_dir,
        benchmark=benchmark,
        out_csv=out_csv,
        out_png=out_png,
    )
    table = Table(title="Plot run", show_header=False)
    table.add_column("k", style="bold")
    table.add_column("v")
    table.add_row("run_dir", str(run_dir))
    table.add_row("benchmark", str(benchmark))
    if csv_path is not None:
        table.add_row("csv", str(csv_path))
    if png_path is not None:
        table.add_row("png", str(png_path))
    Console().print(table)


@app.command()
def tune(
    market: str = typer.Option("derivatives", help="Market mode: equity|crypto|derivatives"),
    symbol: str = typer.Option("BTC-PERP", help="Primary symbol to tune"),
    symbols: Optional[str] = typer.Option(
        None, help="Comma-separated symbols, e.g. BTC-PERP,ETH-PERP (overrides --symbol)"
    ),
    data_source: str = typer.Option(
        "coinbase", help="sample|csv|alpaca|coinbase", show_default=True
    ),
    csv_path: Optional[Path] = typer.Option(None, help="CSV path when data-source=csv"),
    csv_dir: Optional[Path] = typer.Option(
        None, help="CSV directory with per-symbol files when data-source=csv and multiple symbols"
    ),
    bar_timeframe: str = typer.Option(
        "5Min", help="Bar timeframe, e.g. 1Min, 5Min, 15Min, 30Min, 1H, 4H, 6H, 1D"
    ),
    start: Optional[str] = typer.Option(None, help="ISO datetime (optional if --timeframe is used)"),
    end: Optional[str] = typer.Option(None, help="ISO datetime (optional if --timeframe is used)"),
    timeframe: Optional[str] = typer.Option(
        None, help="Relative lookback like 60d/6h/1y; sets end=now and start=end-timeframe"
    ),
    alpaca_feed: str = typer.Option(
        "delayed_sip",
        help="When data-source=alpaca: iex, sip, delayed_sip (alias: uses sip but clamps end >=15m old).",
    ),
    strategy: str = typer.Option("perp_flare", help="Strategy name to tune"),
    strategy_params: Optional[Path] = typer.Option(
        None, help="Optional JSON file with base strategy params (incumbent)"
    ),
    initial_cash: float = typer.Option(10_000.0, help="Starting cash"),
    max_position_notional_usd: Optional[float] = typer.Option(
        None, help="Max notional per symbol"
    ),
    slippage_bps: Optional[float] = typer.Option(
        None,
        help="Fill cost per side in bps (slippage/spread proxy). If omitted: equity=0.0, crypto=3.0, derivatives=1.25.",
    ),
    taker_fee_bps: Optional[float] = typer.Option(
        None,
        help="Taker fee in bps (per side). If omitted: equity=0.0, crypto=25.0, derivatives=10.0 on coinbase (otherwise 3.0).",
    ),
    fixed_fee_per_contract_usd: Optional[float] = typer.Option(
        None,
        help=(
            "Derivatives only: fixed fee in USD charged per contract per side. "
            "If omitted with market=derivatives and data-source=coinbase: defaults to 0.15."
        ),
    ),
    contract_size_units: Optional[float] = typer.Option(
        None,
        help=(
            "Derivatives only: contract size in underlying units (e.g. BTC nano perp = 0.01 BTC). "
            "If omitted with market=derivatives and data-source=coinbase: defaults to 0.01."
        ),
    ),
    coinbase_fee_model: bool = typer.Option(
        True,
        "--coinbase-fee-model/--no-coinbase-fee-model",
        help="Apply Coinbase fixed per-contract fee model (only active for market=derivatives + data-source=coinbase).",
    ),
    allow_short: bool = typer.Option(True, help="Allow negative exposure"),
    trials_per_segment: int = typer.Option(60, help="Random trials per walk-forward segment"),
    jobs: int = typer.Option(
        1,
        help="Parallel worker processes for trials (1=disable, 0=auto).",
    ),
    seed: int = typer.Option(7, help="RNG seed"),
    train: str = typer.Option("30d", help="Train window size (e.g. 30d)"),
    validate: str = typer.Option("7d", help="Validation window size (e.g. 7d)"),
    test: str = typer.Option("7d", help="Test window size (e.g. 7d)"),
    step: str = typer.Option("7d", help="Walk-forward step (e.g. 7d)"),
    drift_frac: Optional[float] = typer.Option(
        0.50,
        help="Limit parameter drift vs previous segment by this fraction (set 0 to disable).",
    ),
    improvement_margin: float = typer.Option(
        0.0,
        help="Require selected params to beat the incumbent selection score by this margin; otherwise keep incumbent.",
    ),
    optimize: str = typer.Option(
        "balanced",
        help="Objective preset: balanced|sharpe_daily|return. (Selection uses train/validate; test is out-of-sample.)",
        show_default=True,
    ),
    out: Optional[Path] = typer.Option(
        None, help="Optional path to write best params JSON (strategy-keyed)."
    ),
) -> None:
    run_dir = Path("outputs") / "tuning" / _run_id("tune")
    setup_logging(level=get_log_level(), log_file=run_dir / "run.log")

    mkt = parse_market(market)
    tf = parse_bar_timeframe(bar_timeframe)
    start_dt = parse_iso_datetime(start) if start is not None else None
    end_dt = parse_iso_datetime(end) if end is not None else None
    if timeframe:
        delta = parse_duration_spec(timeframe)
        end_dt = now_ny()
        start_dt = end_dt - delta

    if max_position_notional_usd is None:
        max_position_notional_usd = get_default_max_position_notional_usd(mode="backtest")

    if symbols is not None:
        raw_symbols = [s.strip() for s in (symbols or "").split(",") if s.strip()]
    else:
        raw_symbols = [symbol.strip()]
    universe_symbols = coerce_symbols_for_market(raw_symbols, mkt)

    alpaca_settings = get_alpaca_settings(require_keys=True) if data_source == "alpaca" else None
    try:
        universe = load_universe_bars(
            symbols=universe_symbols,
            data_source=data_source,
            timeframe=tf,
            start=start_dt,
            end=end_dt,
            csv_path=csv_path,
            csv_dir=csv_dir,
            alpaca_settings=alpaca_settings,
            alpaca_feed=alpaca_feed,
            market=mkt.value,
        )
    except FileNotFoundError as exc:
        raise typer.BadParameter(str(exc)) from exc
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    base_params = _load_strategy_params_for_name(strategy_params, strategy) if strategy_params else {}

    default_slippage_bps = 0.0
    if mkt == Market.DERIVATIVES:
        default_slippage_bps = 1.25
    elif mkt == Market.CRYPTO:
        default_slippage_bps = 3.0

    default_taker_fee_bps = 0.0
    if mkt == Market.DERIVATIVES:
        default_taker_fee_bps = 10.0 if str(data_source).strip().lower() == "coinbase" else 3.0
    elif mkt == Market.CRYPTO:
        default_taker_fee_bps = 25.0

    use_coinbase_fee_model = bool(
        coinbase_fee_model and mkt == Market.DERIVATIVES and str(data_source).strip().lower() == "coinbase"
    )
    default_fixed_fee_per_contract_usd = 0.0
    default_contract_size_units = 1.0
    if use_coinbase_fee_model:
        default_fixed_fee_per_contract_usd = 0.15
        default_contract_size_units = 0.01
    effective_fixed_fee_per_contract_usd = (
        float(default_fixed_fee_per_contract_usd)
        if fixed_fee_per_contract_usd is None
        else float(fixed_fee_per_contract_usd)
    )
    effective_contract_size_units = (
        float(default_contract_size_units) if contract_size_units is None else float(contract_size_units)
    )
    if not use_coinbase_fee_model:
        effective_fixed_fee_per_contract_usd = 0.0
        effective_contract_size_units = 1.0
    if effective_contract_size_units <= 0.0:
        effective_contract_size_units = 1.0

    backtest_cfg = BacktestConfig(
        symbols=universe_symbols,
        initial_cash=float(initial_cash),
        max_position_notional_usd=float(max_position_notional_usd),
        slippage_bps=float(default_slippage_bps if slippage_bps is None else slippage_bps),
        allow_short=bool(allow_short),
        taker_fee_bps=float(default_taker_fee_bps if taker_fee_bps is None else taker_fee_bps),
        fixed_fee_per_contract_usd=float(effective_fixed_fee_per_contract_usd),
        contract_size_units=float(effective_contract_size_units),
    )

    optimize = (optimize or "").strip().lower()
    if optimize == "balanced":
        objective = ObjectiveConfig()
    elif optimize in {"sharpe_daily", "sharpe-daily", "daily_sharpe", "daily-sharpe"}:
        objective = ObjectiveConfig(
            # Prioritize daily Sharpe (scale-free) and penalize tail/drawdowns.
            w_total_return=0.25,
            w_sharpe_daily=1.00,
            w_sharpe=0.00,
            w_positive_trading_days=0.00,
            w_drawdown=1.25,
            w_turnover=0.006,
            w_worst_day=0.75,
        )
    elif optimize == "return":
        objective = ObjectiveConfig(
            w_total_return=1.00,
            w_sharpe_daily=0.10,
            w_sharpe=0.00,
            w_positive_trading_days=0.10,
            w_drawdown=1.00,
            w_turnover=0.003,
            w_worst_day=0.50,
        )
    else:
        raise typer.BadParameter("optimize must be one of: balanced, sharpe_daily, return")

    tune_cfg = TuneConfig(
        trials_per_segment=int(trials_per_segment),
        jobs=int(jobs),
        seed=int(seed),
        drift_frac=None if (drift_frac is None or float(drift_frac) == 0.0) else float(drift_frac),
        improvement_margin=float(improvement_margin),
        objective=objective,
        walk_forward=WalkForwardConfig(train=train, validate=validate, test=test, step=step),
        keep_best_test_runs=True,
    )

    t0 = time.perf_counter()
    result = tune_walk_forward(
        bars_by_symbol=universe.bars_by_symbol,
        market=mkt.value,
        symbols=universe_symbols,
        strategy=strategy,
        backtest_cfg=backtest_cfg,
        tune_cfg=tune_cfg,
        run_dir=run_dir,
        base_params=base_params,
    )
    elapsed_s = time.perf_counter() - t0

    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({result.strategy: result.best_params_latest}, indent=2))

    table = Table(title="Tune summary", show_header=False)
    table.add_column("k", style="bold")
    table.add_column("v")
    table.add_row("run_dir", str(result.run_dir))
    table.add_row("strategy", result.strategy)
    table.add_row("market", result.market)
    table.add_row("symbols", ",".join(result.symbols))
    table.add_row("segments", str(len(result.selections)))
    table.add_row("elapsed", f"{elapsed_s:.2f}s")
    table.add_row("best_params_latest", json.dumps(result.best_params_latest, sort_keys=True))
    table.add_row("best_params_stable", json.dumps(result.best_params_stable, sort_keys=True))
    table.add_row("best_params_file", str(run_dir / "best_params.json"))
    table.add_row("best_params_stable_file", str(run_dir / "best_params_stable.json"))
    if out is not None:
        table.add_row("out", str(out))
    Console().print(table)


@app.command()
def validate(
    market: str = typer.Option("crypto", help="Market mode: equity|crypto|derivatives"),
    symbol: str = typer.Option("BTC/USD", help="Primary symbol"),
    symbols: Optional[str] = typer.Option(
        None, help="Comma-separated symbols, e.g. BTC/USD,ETH/USD (overrides --symbol)"
    ),
    data_source: str = typer.Option(
        "coinbase", help="sample|csv|alpaca|coinbase", show_default=True
    ),
    csv_path: Optional[Path] = typer.Option(None, help="CSV path when data-source=csv"),
    csv_dir: Optional[Path] = typer.Option(
        None, help="CSV directory with per-symbol files when data-source=csv and multiple symbols"
    ),
    bar_timeframe: str = typer.Option(
        "15Min", help="Bar timeframe, e.g. 1Min, 5Min, 15Min, 30Min, 1H, 4H, 6H, 1D"
    ),
    start: Optional[str] = typer.Option(None, help="ISO datetime (optional if --timeframe is used)"),
    end: Optional[str] = typer.Option(None, help="ISO datetime (optional if --timeframe is used)"),
    timeframe: Optional[str] = typer.Option(
        None, help="Relative lookback like 60d/6h/2y; sets end=now and start=end-timeframe"
    ),
    alpaca_feed: str = typer.Option(
        "delayed_sip",
        help="When data-source=alpaca: iex, sip, delayed_sip (alias: uses sip but clamps end >=15m old).",
    ),
    strategy: str = typer.Option("crypto_ensemble", help="Strategy name to validate"),
    strategy_params: Optional[Path] = typer.Option(
        None, help="Optional JSON file with strategy parameters"
    ),
    initial_cash: float = typer.Option(100_000.0, help="Starting cash"),
    max_position_notional_usd: Optional[float] = typer.Option(
        None, help="Max notional per symbol"
    ),
    slippage_bps: Optional[float] = typer.Option(
        None,
        help="Fill cost per side in bps (slippage/spread proxy). If omitted: equity=0.0, crypto=3.0, derivatives=1.25.",
    ),
    taker_fee_bps: Optional[float] = typer.Option(
        None,
        help="Taker fee in bps (per side). If omitted: equity=0.0, crypto=25.0, derivatives=3.0.",
    ),
    slippage_grid: Optional[str] = typer.Option(
        None, help="Optional comma-separated slippage scenarios, e.g. 3,5,8 (overrides --slippage-bps)"
    ),
    taker_fee_grid: Optional[str] = typer.Option(
        None, help="Optional comma-separated taker fee scenarios, e.g. 10,20,40 (overrides --taker-fee-bps)"
    ),
    allow_short: bool = typer.Option(False, help="Allow negative exposure"),
    train: str = typer.Option("180d", help="Train window size (e.g. 180d)"),
    validate_window: str = typer.Option("30d", help="Validation window size (e.g. 30d)"),
    test: str = typer.Option("30d", help="Test window size (e.g. 30d)"),
    step: str = typer.Option("30d", help="Walk-forward step (e.g. 30d)"),
    min_trades: int = typer.Option(2, help="Reject segments with < min trades"),
    max_drawdown_limit: float = typer.Option(
        0.40, help="Reject segments with max_drawdown < -limit (e.g. 0.40 = -40%)"
    ),
    worst_day_limit: float = typer.Option(
        0.20, help="Reject segments with worst day < -limit (e.g. 0.20 = -20%)"
    ),
    turnover_cap: float = typer.Option(
        250.0, help="Reject segments with turnover > cap (gross_notional / avg_equity)"
    ),
    keep_test_runs: bool = typer.Option(True, help="Keep per-segment test run outputs on disk"),
    overfit_report: bool = typer.Option(
        False,
        help="Write overfit_report.json (bootstrap CI of daily Sharpe/total_return) under each scenario dir. Requires --keep-test-runs.",
    ),
) -> None:
    run_dir = Path("outputs") / "validation" / _run_id("validate")
    setup_logging(level=get_log_level(), log_file=run_dir / "run.log")

    mkt = parse_market(market)
    tf = parse_bar_timeframe(bar_timeframe)
    start_dt = parse_iso_datetime(start) if start is not None else None
    end_dt = parse_iso_datetime(end) if end is not None else None
    if timeframe:
        delta = parse_duration_spec(timeframe)
        end_dt = now_ny()
        start_dt = end_dt - delta

    if max_position_notional_usd is None:
        max_position_notional_usd = get_default_max_position_notional_usd(mode="backtest")

    if symbols is not None:
        raw_symbols = [s.strip() for s in (symbols or "").split(",") if s.strip()]
    else:
        raw_symbols = [symbol.strip()]
    universe_symbols = coerce_symbols_for_market(raw_symbols, mkt)

    alpaca_settings = get_alpaca_settings(require_keys=True) if data_source == "alpaca" else None
    try:
        universe = load_universe_bars(
            symbols=universe_symbols,
            data_source=data_source,
            timeframe=tf,
            start=start_dt,
            end=end_dt,
            csv_path=csv_path,
            csv_dir=csv_dir,
            alpaca_settings=alpaca_settings,
            alpaca_feed=alpaca_feed,
            market=mkt.value,
        )
    except FileNotFoundError as exc:
        raise typer.BadParameter(str(exc)) from exc
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    default_slippage = 0.0
    if mkt == Market.DERIVATIVES:
        default_slippage = 1.25
    elif mkt == Market.CRYPTO:
        default_slippage = 3.0

    default_fee = 0.0
    if mkt == Market.DERIVATIVES:
        default_fee = 3.0
    elif mkt == Market.CRYPTO:
        default_fee = 25.0

    params = _load_strategy_params_for_name(strategy_params, strategy) if strategy_params else {}
    wf = WalkForwardConfig(train=train, validate=validate_window, test=test, step=step)
    objective = ObjectiveConfig(
        min_trades=int(min_trades),
        max_drawdown_limit=float(max_drawdown_limit),
        worst_day_limit=float(worst_day_limit),
        turnover_cap=float(turnover_cap),
    )

    def _parse_grid(raw: Optional[str]) -> list[float]:
        if raw is None:
            return []
        parts = [p.strip() for p in str(raw).split(",") if p.strip()]
        out: list[float] = []
        for p in parts:
            out.append(float(p))
        return out

    slip_values = _parse_grid(slippage_grid) or [
        float(default_slippage if slippage_bps is None else slippage_bps)
    ]
    fee_values = _parse_grid(taker_fee_grid) or [
        float(default_fee if taker_fee_bps is None else taker_fee_bps)
    ]

    scenario_rows: list[dict[str, object]] = []
    t0 = time.perf_counter()
    for slip in slip_values:
        for fee in fee_values:
            cfg = BacktestConfig(
                symbols=universe_symbols,
                initial_cash=float(initial_cash),
                max_position_notional_usd=float(max_position_notional_usd),
                slippage_bps=float(slip),
                allow_short=bool(allow_short),
                taker_fee_bps=float(fee),
            )
            scenario_dir = run_dir / f"slip{slip:g}_fee{fee:g}"
            result = walk_forward_evaluate(
                bars_by_symbol=universe.bars_by_symbol,
                market=mkt.value,
                symbols=universe_symbols,
                strategy=strategy,
                params=params,
                backtest_cfg=cfg,
                walk_forward=wf,
                objective=objective,
                run_dir=scenario_dir,
                keep_test_runs=bool(keep_test_runs),
            )
            if overfit_report:
                if not keep_test_runs:
                    logger.warning(
                        "overfit_report requested but keep_test_runs=False; skipping (needs per-segment equity_curve.csv)"
                    )
                else:
                    try:
                        from atlas.ml.overfit import overfit_report_from_walk_forward

                        overfit_report_from_walk_forward(Path(str(scenario_dir)))
                    except Exception as exc:
                        logger.warning("failed to write overfit_report.json for %s: %s", scenario_dir, exc)
            scenario_rows.append(
                {
                    "slippage_bps": float(slip),
                    "taker_fee_bps": float(fee),
                    "segments": int(result.summary.get("segments", 0)),
                    "accepted": int(result.summary.get("accepted", 0)),
                    "positive_segment_frac": float(result.summary.get("positive_segment_frac", 0.0)),
                    "median_return": float(result.summary.get("median_return", 0.0)),
                    "mean_return": float(result.summary.get("mean_return", 0.0)),
                    "run_dir": str(scenario_dir),
                }
            )
    elapsed_s = time.perf_counter() - t0

    (run_dir / "scenario_summary.json").write_text(json.dumps(scenario_rows, indent=2))

    if len(scenario_rows) == 1:
        row = scenario_rows[0]
        table = Table(title="Walk-forward validation summary", show_header=False)
        table.add_column("k", style="bold")
        table.add_column("v")
        table.add_row("run_dir", str(run_dir))
        table.add_row("market", mkt.value)
        table.add_row("symbols", ",".join(universe_symbols))
        table.add_row("strategy", str(strategy))
        table.add_row("bar_timeframe", tf.name)
        table.add_row(
            "costs",
            f"slippage_bps={float(row['slippage_bps']):.2f} taker_fee_bps={float(row['taker_fee_bps']):.2f}",
        )
        table.add_row("segments", str(int(row["segments"])))
        table.add_row("accepted", str(int(row["accepted"])))
        table.add_row("pos_seg_frac", f"{float(row['positive_segment_frac']):.2%}")
        table.add_row("median_seg_return", f"{float(row['median_return']):.2%}")
        table.add_row("mean_seg_return", f"{float(row['mean_return']):.2%}")
        table.add_row("elapsed", f"{elapsed_s:.2f}s")
        Console().print(table)
    else:
        table = Table(title="Walk-forward validation (cost grid)", show_header=True)
        table.add_column("slip_bps", justify="right")
        table.add_column("fee_bps", justify="right")
        table.add_column("segments", justify="right")
        table.add_column("accepted", justify="right")
        table.add_column("pos_seg_frac", justify="right")
        table.add_column("median_ret", justify="right")
        table.add_column("mean_ret", justify="right")
        table.add_column("run_dir")
        for row in scenario_rows:
            table.add_row(
                f"{float(row['slippage_bps']):.2f}",
                f"{float(row['taker_fee_bps']):.2f}",
                str(int(row["segments"])),
                str(int(row["accepted"])),
                f"{float(row['positive_segment_frac']):.2%}",
                f"{float(row['median_return']):.2%}",
                f"{float(row['mean_return']):.2%}",
                str(row["run_dir"]),
            )
        Console().print(table)
        typer.echo(f"elapsed: {elapsed_s:.2f}s  outputs: {run_dir}")


@app.command()
def analyze_run(
    run_dir: Path = typer.Argument(..., help="Existing backtest run directory under outputs/"),
    window: str = typer.Option("7d", help="Rolling window size (e.g. 7d, 30d)"),
    step: str = typer.Option("7d", help="Step size between windows (e.g. 7d, 1d)"),
    benchmark: Optional[str] = typer.Option(
        "spy.us",
        help="Optional Stooq benchmark symbol (e.g. spy.us). Use '' to disable.",
    ),
) -> None:
    """
    Analyze a completed run directory: rolling window returns + trade frequency.

    Writes `window_analysis.json` under `run_dir`.
    """
    run_dir = Path(str(run_dir))
    win_td = parse_duration_spec(window)
    step_td = parse_duration_spec(step)

    bench = (benchmark or "").strip() or None
    out_path = write_window_analysis_json(
        run_dir=run_dir, window=win_td, step=step_td, benchmark=bench
    )
    summary, _rows = rolling_window_summary(
        run_dir=run_dir, window=win_td, step=step_td, benchmark=bench
    )

    table = Table(title="Run window analysis", show_header=False)
    table.add_column("k", style="bold")
    table.add_column("v")
    table.add_row("run_dir", str(run_dir))
    table.add_row("window", str(summary.window))
    table.add_row("step", str(summary.step))
    table.add_row("windows", str(int(summary.windows)))
    table.add_row("windows_with_trades", str(int(summary.windows_with_trades)))
    table.add_row("trade_window_frac", f"{float(summary.trade_window_frac):.2%}")
    table.add_row("mean_return", f"{float(summary.mean_return):.4%}")
    table.add_row("median_return", f"{float(summary.median_return):.4%}")
    table.add_row("p05_return", f"{float(summary.p05_return):.4%}")
    table.add_row("p95_return", f"{float(summary.p95_return):.4%}")
    table.add_row("best_return", f"{float(summary.best_return):.4%}")
    table.add_row("worst_return", f"{float(summary.worst_return):.4%}")
    if summary.benchmark is not None:
        table.add_row("benchmark", str(summary.benchmark))
        if summary.beat_benchmark_frac is not None:
            table.add_row("beat_benchmark_frac", f"{float(summary.beat_benchmark_frac):.2%}")
        if summary.mean_benchmark_return is not None:
            table.add_row("mean_benchmark_return", f"{float(summary.mean_benchmark_return):.4%}")
        if summary.mean_alpha is not None:
            table.add_row("mean_alpha", f"{float(summary.mean_alpha):.4%}")
        if summary.p05_alpha is not None:
            table.add_row("p05_alpha", f"{float(summary.p05_alpha):.4%}")
        if summary.worst_alpha is not None:
            table.add_row("worst_alpha", f"{float(summary.worst_alpha):.4%}")
    table.add_row("out", str(out_path))
    Console().print(table)


@app.command("evaluate-all")
def evaluate_all(
    strategies: Optional[str] = typer.Option(
        None,
        help="Optional comma-separated strategy list. Default: all registered strategies.",
    ),
    strategy_params_prefixes: Optional[str] = typer.Option(
        None,
        help="Optional comma-separated prefixes for strategy_params/*.json expansion (e.g. crypto_rotation,crypto_ensemble).",
    ),
    initial_cash: float = typer.Option(500.0, help="Starting cash per candidate run"),
    max_position_notional_usd: float = typer.Option(
        500.0, help="Max notional per symbol in candidate runs"
    ),
    prewarm: str = typer.Option(
        "90d", help="Prewarm lookback loaded before score window (e.g. 30d, 12h)"
    ),
    baseline_slippage_bps: float = typer.Option(
        3.0, help="Baseline slippage bps per side for initial ranking"
    ),
    baseline_taker_fee_bps: float = typer.Option(
        25.0, help="Baseline taker fee bps per side for initial ranking"
    ),
    coinbase_fee_model: bool = typer.Option(
        True,
        "--coinbase-fee-model/--no-coinbase-fee-model",
        help="Enable Coinbase fixed per-contract fee model for derivatives candidates that use coinbase data.",
    ),
    coinbase_fixed_fee_per_contract_usd: float = typer.Option(
        0.15,
        help="Coinbase fixed fee in USD per contract per side (used when coinbase fee model is enabled).",
    ),
    coinbase_contract_size_units: float = typer.Option(
        0.01,
        help="Coinbase contract size in underlying units (e.g. nano BTC perp = 0.01 BTC).",
    ),
    stress_slippage_grid: str = typer.Option(
        "3,5,8", help="Stress grid slippage bps values, comma-separated"
    ),
    stress_taker_fee_grid: str = typer.Option(
        "25,40,60", help="Stress grid taker fee bps values, comma-separated"
    ),
    stress_min_mean_return: float = typer.Option(
        -0.0025,
        help="Stress scenario pass floor for mean walk-forward return (e.g. -0.0025 = -0.25%).",
    ),
    stress_min_positive_segment_frac: float = typer.Option(
        0.45,
        help="Stress scenario pass floor for positive walk-forward segment fraction.",
    ),
    stress_min_accepted_segment_frac: float = typer.Option(
        0.40,
        help="Stress scenario pass floor for accepted/total walk-forward segments.",
    ),
    top_n_validate: int = typer.Option(
        8, help="Number of top baseline candidates to stress-validate"
    ),
    validate_train: str = typer.Option("180d", help="Walk-forward train window"),
    validate_validate: str = typer.Option(
        "30d", help="Walk-forward validation window"
    ),
    validate_test: str = typer.Option("30d", help="Walk-forward test window"),
    validate_step: str = typer.Option("30d", help="Walk-forward step"),
    validate_min_trades: int = typer.Option(
        1, help="Validation reject threshold: min trades"
    ),
    validate_max_drawdown_limit: float = typer.Option(
        0.20,
        help="Validation reject threshold: max drawdown limit (0.20 => reject below -20%)",
    ),
    validate_worst_day_limit: float = typer.Option(
        0.20,
        help="Validation reject threshold: worst day limit (0.20 => reject below -20%)",
    ),
    validate_turnover_cap: float = typer.Option(
        250.0, help="Validation reject threshold: turnover cap"
    ),
    gate_max_drawdown: float = typer.Option(
        -0.20, help="Final gate: max drawdown must be >= this value"
    ),
    gate_min_positive_week_frac: float = typer.Option(
        0.70, help="Final gate: fraction of positive 7d windows must be >= this value"
    ),
    gate_min_stress_pass_frac: float = typer.Option(
        0.66, help="Final gate: stress scenario pass fraction must be >= this value"
    ),
    require_beat_spy: bool = typer.Option(
        True, help="Final gate: require total return to beat SPY over same dates"
    ),
    equity_fallback_sample: bool = typer.Option(
        True,
        help="If Alpaca equity data unavailable, fall back to bundled sample data.",
    ),
) -> None:
    run_dir = Path("outputs") / "evaluations" / _run_id("evaluate_all")
    setup_logging(level=get_log_level(), log_file=run_dir / "run.log")

    strategy_list = None
    if strategies is not None:
        strategy_list = [
            s.strip().lower().replace("-", "_")
            for s in str(strategies).split(",")
            if s.strip()
        ]
        if not strategy_list:
            strategy_list = None

    params_prefix_list = None
    if strategy_params_prefixes is not None:
        params_prefix_list = [
            s.strip().lower().replace("-", "_")
            for s in str(strategy_params_prefixes).split(",")
            if s.strip()
        ]
        if not params_prefix_list:
            params_prefix_list = None

    slip_grid = _parse_float_csv(stress_slippage_grid)
    fee_grid = _parse_float_csv(stress_taker_fee_grid)
    if not slip_grid:
        raise typer.BadParameter("stress_slippage_grid must contain at least one value")
    if not fee_grid:
        raise typer.BadParameter("stress_taker_fee_grid must contain at least one value")

    cfg = EvaluationConfig(
        initial_cash=float(initial_cash),
        max_position_notional_usd=float(max_position_notional_usd),
        prewarm=str(prewarm),
        baseline_slippage_bps=float(baseline_slippage_bps),
        baseline_taker_fee_bps=float(baseline_taker_fee_bps),
        use_coinbase_fee_model=bool(coinbase_fee_model),
        coinbase_fixed_fee_per_contract_usd=float(coinbase_fixed_fee_per_contract_usd),
        coinbase_contract_size_units=float(coinbase_contract_size_units),
        stress_slippage_grid=tuple(float(v) for v in slip_grid),
        stress_taker_fee_grid=tuple(float(v) for v in fee_grid),
        stress_min_mean_return=float(stress_min_mean_return),
        stress_min_positive_segment_frac=float(stress_min_positive_segment_frac),
        stress_min_accepted_segment_frac=float(stress_min_accepted_segment_frac),
        top_n_validate=int(top_n_validate),
        validate_train=str(validate_train),
        validate_validate=str(validate_validate),
        validate_test=str(validate_test),
        validate_step=str(validate_step),
        validate_min_trades=int(validate_min_trades),
        validate_max_drawdown_limit=float(validate_max_drawdown_limit),
        validate_worst_day_limit=float(validate_worst_day_limit),
        validate_turnover_cap=float(validate_turnover_cap),
        gate_max_drawdown=float(gate_max_drawdown),
        gate_min_positive_week_frac=float(gate_min_positive_week_frac),
        gate_min_stress_pass_frac=float(gate_min_stress_pass_frac),
        require_beat_spy=bool(require_beat_spy),
        equity_fallback_sample=bool(equity_fallback_sample),
    )

    started = time.perf_counter()
    result = run_evaluate_all(
        run_dir=run_dir,
        cfg=cfg,
        strategies=strategy_list,
        strategy_params_prefixes=params_prefix_list,
    )
    elapsed = time.perf_counter() - started

    table = Table(title="Evaluate-all summary", show_header=False)
    table.add_column("k", style="bold")
    table.add_column("v")
    table.add_row("run_dir", str(result.run_dir))
    table.add_row("generated_at", str(result.generated_at))
    table.add_row("total_candidates", str(result.total_candidates))
    table.add_row("validated_candidates", str(result.validated_candidates))
    table.add_row("passed_candidates", str(result.passed_candidates))
    table.add_row("algorithm_a", str(result.algorithm_a_candidate_id))
    table.add_row("algorithm_b", str(result.algorithm_b_candidate_id))
    table.add_row("winner", str(result.winner_candidate_id))
    table.add_row("leaderboard_csv", str(result.leaderboard_csv))
    table.add_row("leaderboard_json", str(result.leaderboard_json))
    table.add_row("full_json", str(result.full_json))
    table.add_row("state_md", "docs/EXECUTION_STATE_STRATEGY_SEARCH.md")
    table.add_row("state_json", "outputs/evaluations/latest_state.json")
    table.add_row("elapsed", f"{elapsed:.2f}s")
    Console().print(table)


@app.command()
def paper(
    market: str = typer.Option("equity", help="Market mode: equity|crypto|derivatives"),
    symbols: list[str] = typer.Option(["SPY"], help="Symbols to trade, repeatable"),
    bar_timeframe: str = typer.Option(
        "1Min", help="Bar timeframe, e.g. 1Min, 5Min, 15Min, 30Min, 1H, 4H, 6H, 1D"
    ),
    data_source: str = typer.Option(
        "alpaca",
        help="Bars source for paper loop: alpaca|coinbase",
        show_default=True,
    ),
    execution_venue: str = typer.Option(
        "alpaca",
        help="Order execution venue: alpaca|coinbase",
        show_default=True,
    ),
    alpaca_feed: str = typer.Option(
        "iex",
        help="Alpaca data feed for bars: iex, sip, delayed_sip (alias: uses sip but clamps end >=15m old).",
    ),
    strategy: str = typer.Option("spy_open_close", help="Strategy name"),
    strategy_params: Optional[Path] = typer.Option(
        None, help="JSON file with strategy parameters"
    ),
    fast_window: int = typer.Option(10, help="ma_crossover/ema_crossover fast window"),
    slow_window: int = typer.Option(30, help="ma_crossover/ema_crossover slow window"),
    lookback_bars: int = typer.Option(200, help="Bars fetched each loop"),
    poll_seconds: int = typer.Option(60, help="Minimum seconds between loops"),
    initial_cash: float = typer.Option(
        500.0,
        help="Synthetic starting cash for paper risk accounting (used for coinbase execution state).",
    ),
    max_position_notional_usd: Optional[float] = typer.Option(
        None, help="Max notional per symbol"
    ),
    slippage_bps: Optional[float] = typer.Option(
        None,
        help="Fill cost proxy per side in bps (used for strategy cost gating only). If omitted: equity=0.0, crypto=3.0, derivatives=1.25.",
    ),
    taker_fee_bps: Optional[float] = typer.Option(
        None,
        help="Taker fee proxy per side in bps (used for strategy cost gating only). If omitted: equity=0.0, crypto=25.0, derivatives=10.0 on coinbase (otherwise 3.0).",
    ),
    fixed_fee_per_contract_usd: Optional[float] = typer.Option(
        None,
        help=(
            "Derivatives/coinbase: fixed fee in USD per contract per side for synthetic PnL accounting. "
            "If omitted with market=derivatives and coinbase venue/source: defaults to 0.15."
        ),
    ),
    contract_size_units: Optional[float] = typer.Option(
        None,
        help=(
            "Derivatives/coinbase: contract size in underlying units (e.g. BTC nano perp = 0.01 BTC). "
            "If omitted with market=derivatives and coinbase venue/source: defaults to 0.01."
        ),
    ),
    coinbase_fee_model: bool = typer.Option(
        True,
        "--coinbase-fee-model/--no-coinbase-fee-model",
        help="Apply Coinbase fixed per-contract fee model (only active for derivatives + coinbase venue/source).",
    ),
    allow_short: bool = typer.Option(False, help="Allow negative target exposure (shorting)"),
    regular_hours_only: bool = typer.Option(
        True, help="Filter bars to regular market hours (09:30–16:00 ET)"
    ),
    allow_trading_when_closed: bool = typer.Option(
        False, help="Allow trading when market is closed (uses limit orders w/ extended hours)"
    ),
    limit_offset_bps: float = typer.Option(
        5.0,
        help="When market is closed, price limit orders at ±offset bps from last price to improve fill odds.",
    ),
    dry_run: Optional[bool] = typer.Option(
        None,
        "--dry-run/--no-dry-run",
        help="If omitted: defaults to false for alpaca venue and true for coinbase venue.",
    ),
    max_loops: Optional[int] = typer.Option(None, help="Stop after N loops"),
) -> None:
    run_dir = Path("outputs") / "paper" / _run_id("paper")
    setup_logging(level=get_log_level(), log_file=run_dir / "run.log")

    if max_position_notional_usd is None:
        max_position_notional_usd = get_default_max_position_notional_usd(mode="paper")

    mkt = parse_market(market)
    data_source = str(data_source).strip().lower()
    if data_source not in {"alpaca", "coinbase"}:
        raise typer.BadParameter("data_source must be one of: alpaca|coinbase")
    execution_venue = str(execution_venue).strip().lower()
    if execution_venue not in {"alpaca", "coinbase"}:
        raise typer.BadParameter("execution_venue must be one of: alpaca|coinbase")

    if mkt == Market.DERIVATIVES and data_source != "coinbase":
        raise typer.BadParameter("market=derivatives requires --data-source coinbase")
    if mkt == Market.DERIVATIVES and execution_venue != "coinbase":
        raise typer.BadParameter("market=derivatives requires --execution-venue coinbase")
    if execution_venue == "coinbase" and mkt == Market.EQUITY:
        raise typer.BadParameter("execution-venue=coinbase supports crypto|derivatives only")
    if execution_venue == "coinbase" and data_source != "coinbase":
        raise typer.BadParameter("execution-venue=coinbase currently requires --data-source coinbase")

    effective_dry_run = (execution_venue == "coinbase") if dry_run is None else bool(dry_run)

    settings: Optional[AlpacaSettings] = None
    if data_source == "alpaca" or execution_venue == "alpaca":
        settings = get_alpaca_settings(require_keys=True)

    canonical_strategy = str(strategy).strip().lower().replace("-", "_")
    if canonical_strategy in {"nec_x", "nec_pdt"} and len(symbols) < 2:
        symbols = default_symbols(mkt, count=2)
    symbols = coerce_symbols_for_market(symbols, mkt)

    strat = build_strategy(
        name=strategy,
        params_path=strategy_params,
        symbols=symbols,
        fast_window=fast_window,
        slow_window=slow_window,
    )
    warmup_bars = int(max(1, int(strat.warmup_bars())))
    effective_lookback_bars = int(lookback_bars)
    recommended_paper_lookback = int(warmup_bars)
    if canonical_strategy == "perp_regime_adaptive_trend_capture":
        # RATC uses long horizons; keep extra bars beyond minimum warmup.
        recommended_paper_lookback = max(recommended_paper_lookback, 1000)
    if effective_lookback_bars < recommended_paper_lookback:
        typer.echo(
            f"paper lookback auto-raised {effective_lookback_bars} -> {recommended_paper_lookback} "
            f"(strategy warmup_bars={warmup_bars})"
        )
        effective_lookback_bars = int(recommended_paper_lookback)

    use_coinbase_fee_model = bool(
        coinbase_fee_model
        and mkt == Market.DERIVATIVES
        and str(data_source).strip().lower() == "coinbase"
        and str(execution_venue).strip().lower() == "coinbase"
    )
    default_fixed_fee_per_contract_usd = 0.0
    default_contract_size_units = 1.0
    if use_coinbase_fee_model:
        default_fixed_fee_per_contract_usd = 0.15
        default_contract_size_units = 0.01
    effective_fixed_fee_per_contract_usd = (
        float(default_fixed_fee_per_contract_usd)
        if fixed_fee_per_contract_usd is None
        else float(fixed_fee_per_contract_usd)
    )
    effective_contract_size_units = (
        float(default_contract_size_units) if contract_size_units is None else float(contract_size_units)
    )
    if not use_coinbase_fee_model:
        effective_fixed_fee_per_contract_usd = 0.0
        effective_contract_size_units = 1.0
    if effective_contract_size_units <= 0.0:
        effective_contract_size_units = 1.0

    cfg = PaperConfig(
        symbols=symbols,
        bar_timeframe=bar_timeframe,
        data_source=data_source,
        execution_venue=execution_venue,
        alpaca_feed=alpaca_feed,
        lookback_bars=int(effective_lookback_bars),
        poll_seconds=poll_seconds,
        initial_cash_usd=float(initial_cash),
        max_position_notional_usd=float(max_position_notional_usd),
        slippage_bps=float(
            (3.0 if mkt == Market.CRYPTO else 1.25 if mkt == Market.DERIVATIVES else 0.0)
            if slippage_bps is None
            else slippage_bps
        ),
        taker_fee_bps=float(
            (
                25.0
                if mkt == Market.CRYPTO
                else 10.0
                if (mkt == Market.DERIVATIVES and data_source == "coinbase" and execution_venue == "coinbase")
                else 3.0
                if mkt == Market.DERIVATIVES
                else 0.0
            )
            if taker_fee_bps is None
            else taker_fee_bps
        ),
        fixed_fee_per_contract_usd=float(effective_fixed_fee_per_contract_usd),
        contract_size_units=float(effective_contract_size_units),
        allow_short=allow_short,
        regular_hours_only=regular_hours_only,
        allow_trading_when_closed=allow_trading_when_closed,
        limit_offset_bps=float(limit_offset_bps),
        dry_run=bool(effective_dry_run),
        market=mkt.value,
    )

    if settings is not None:
        logger.info("paper=%s allow_live=%s", settings.paper, settings.allow_live)
    logger.info(
        "paper data_source=%s execution_venue=%s dry_run=%s",
        data_source,
        execution_venue,
        bool(effective_dry_run),
    )
    run_paper_loop(settings=settings, strategy=strat, cfg=cfg, run_dir=run_dir, max_loops=max_loops)


if __name__ == "__main__":
    app()
