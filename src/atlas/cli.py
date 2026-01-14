from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from atlas.backtest.engine import BacktestConfig, run_backtest
from atlas.backtest.derivatives_engine import run_derivatives_backtest
from atlas.config import (
    get_alpaca_settings,
    get_default_max_position_notional_usd,
    get_log_level,
)
from atlas.data.alpaca_data import download_stock_bars_to_csv
from atlas.data.bars import parse_bar_timeframe
from atlas.data.universe import load_universe_bars
from atlas.logging_utils import setup_logging
from atlas.market import coerce_symbols_for_market, default_symbols, parse_market
from atlas.paper.runner import PaperConfig, run_paper_loop
from atlas.strategies.registry import build_strategy
from atlas.tui.app import run_tui
from atlas.utils.time import now_ny, parse_iso_datetime

from atlas.ml.tune import (
    ObjectiveConfig,
    TuneConfig,
    WalkForwardConfig,
    parse_duration_spec,
    tune_walk_forward,
)

app = typer.Typer(add_completion=False)
logger = logging.getLogger(__name__)


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
            f"allow_short={cfg.allow_short}"
        )
        typer.echo(
            "results: "
            f"final_equity={final_equity:.2f} "
            f"total_return={metrics['total_return']:.4%} "
            f"max_drawdown={metrics['max_drawdown']:.4%} "
            f"sharpe={metrics['sharpe']:.2f} "
            f"fills={metrics['trades']} "
            f"gross_notional={gross_notional:.2f}"
        )
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
                f"allow_short={cfg.allow_short}",
            ]
        ),
    )
    table.add_row(
        "results",
        "  ".join(
            [
                f"final_equity={final_equity:.2f}",
                f"total_return={metrics['total_return']:.4%}",
                f"max_drawdown={metrics['max_drawdown']:.4%}",
                f"sharpe={metrics['sharpe']:.2f}",
                f"fills={metrics['trades']}",
                f"gross_notional={gross_notional:.2f}",
            ]
        ),
    )
    table.add_row("elapsed", f"{elapsed_s:.2f}s")

    console.print(table)


def _run_id(prefix: str) -> str:
    return datetime.now().strftime(f"{prefix}_%Y%m%d_%H%M%S")


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
    run_tui()


@app.command()
def download_bars(
    symbol: str = typer.Option(..., help="US equity symbol, e.g. SPY"),
    start: str = typer.Option(..., help="ISO datetime, e.g. 2024-01-02T09:30:00-05:00"),
    end: str = typer.Option(..., help="ISO datetime, e.g. 2024-01-02T16:00:00-05:00"),
    timeframe: str = typer.Option("1Min", help="Bar timeframe, e.g. 1Min, 5Min, 30Min, 1H, 4H"),
    feed: str = typer.Option(
        "delayed_sip",
        help="Alpaca data feed: iex, sip, delayed_sip (alias: uses sip but clamps end >=15m old).",
    ),
    out: Optional[Path] = typer.Option(None, help="Optional explicit output CSV path"),
) -> None:
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
    bar_timeframe: str = typer.Option("1Min", help="Bar timeframe, e.g. 1Min, 5Min, 30Min, 1H, 4H"),
    start: Optional[str] = typer.Option(
        None, help="ISO datetime (required for alpaca; optional filter otherwise)"
    ),
    end: Optional[str] = typer.Option(
        None, help="ISO datetime (required for alpaca; optional filter otherwise)"
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
        help="Fill cost per side in basis points (slippage/spread proxy). If omitted, nec_x/orb_trend default to 1.25 bps/side and nec_pdt defaults to 3.8 bps/side.",
    ),
    allow_short: bool = typer.Option(False, help="Allow negative exposure"),
) -> None:
    run_dir = Path("outputs") / "backtests" / _run_id("backtest")
    setup_logging(level=get_log_level(), log_file=run_dir / "run.log")

    if max_position_notional_usd is None:
        max_position_notional_usd = get_default_max_position_notional_usd(mode="backtest")

    mkt = parse_market(market)
    from atlas.market import Market
    # Ensure allow_short is True for derivatives unless explicitly forbidden?
    # Actually CLI args override. But usually perps allow short.
    
    tf = parse_bar_timeframe(bar_timeframe)
    start_dt = parse_iso_datetime(start) if start is not None else None
    end_dt = parse_iso_datetime(end) if end is not None else None

    canonical_strategy = str(strategy).strip().lower().replace("-", "_")
    if symbols is not None:
        raw_symbols = [s.strip() for s in (symbols or "").split(",") if s.strip()]
    else:
        raw_symbols = (
            default_symbols(mkt, count=2)
            if canonical_strategy in {"nec_x", "nec_pdt"}
            else [symbol.strip()]
        )
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

    strat = build_strategy(
        name=strategy,
        params_path=strategy_params,
        symbols=universe_symbols,
        fast_window=fast_window,
        slow_window=slow_window,
    )

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
                else 0.0
            )
            if slippage_bps is None
            else slippage_bps
        ),
        allow_short=allow_short,
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
        )
    else:
        run_backtest(bars_by_symbol=universe.bars_by_symbol, strategy=strat, cfg=cfg, run_dir=run_dir)
    elapsed_s = time.perf_counter() - t0

    if strategy_params is not None:
        strategy_params_hint = f"params={strategy_params}"
    elif strategy in {"ma_crossover", "ema_crossover"}:
        strategy_params_hint = f"fast={fast_window} slow={slow_window}"
    else:
        strategy_params_hint = "defaults"
    _print_backtest_summary(
        run_dir=run_dir,
        symbols=universe_symbols,
        data_source=data_source,
        data_hint=universe.hint,
        bar_index=common_index,
        strategy_name=strategy,
        strategy_params_hint=strategy_params_hint,
        warmup_bars=strat.warmup_bars(),
        cfg=cfg,
        elapsed_s=elapsed_s,
    )


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
    bar_timeframe: str = typer.Option("5Min", help="Bar timeframe, e.g. 1Min, 5Min, 30Min, 1H, 4H"),
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
    slippage_bps: float = typer.Option(
        1.25,
        help="Fill cost per side in bps (slippage/spread proxy).",
    ),
    allow_short: bool = typer.Option(True, help="Allow negative exposure"),
    trials_per_segment: int = typer.Option(60, help="Random trials per walk-forward segment"),
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

    backtest_cfg = BacktestConfig(
        symbols=universe_symbols,
        initial_cash=float(initial_cash),
        max_position_notional_usd=float(max_position_notional_usd),
        slippage_bps=float(slippage_bps),
        allow_short=bool(allow_short),
    )
    tune_cfg = TuneConfig(
        trials_per_segment=int(trials_per_segment),
        seed=int(seed),
        drift_frac=None if (drift_frac is None or float(drift_frac) == 0.0) else float(drift_frac),
        improvement_margin=float(improvement_margin),
        objective=ObjectiveConfig(),
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
def paper(
    market: str = typer.Option("equity", help="Market mode: equity|crypto|derivatives"),
    symbols: list[str] = typer.Option(["SPY"], help="Symbols to trade, repeatable"),
    bar_timeframe: str = typer.Option("1Min", help="Bar timeframe, e.g. 1Min, 5Min, 30Min, 1H, 4H"),
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
    max_position_notional_usd: Optional[float] = typer.Option(
        None, help="Max notional per symbol"
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
    dry_run: bool = typer.Option(False, help="Do not submit orders"),
    max_loops: Optional[int] = typer.Option(None, help="Stop after N loops"),
) -> None:
    settings = get_alpaca_settings(require_keys=True)
    run_dir = Path("outputs") / "paper" / _run_id("paper")
    setup_logging(level=get_log_level(), log_file=run_dir / "run.log")

    if max_position_notional_usd is None:
        max_position_notional_usd = get_default_max_position_notional_usd(mode="paper")

    mkt = parse_market(market)
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

    cfg = PaperConfig(
        symbols=symbols,
        bar_timeframe=bar_timeframe,
        alpaca_feed=alpaca_feed,
        lookback_bars=lookback_bars,
        poll_seconds=poll_seconds,
        max_position_notional_usd=float(max_position_notional_usd),
        allow_short=allow_short,
        regular_hours_only=regular_hours_only,
        allow_trading_when_closed=allow_trading_when_closed,
        limit_offset_bps=float(limit_offset_bps),
        dry_run=dry_run,
        market=mkt.value,
    )

    logger.info("paper=%s allow_live=%s", settings.paper, settings.allow_live)
    run_paper_loop(settings=settings, strategy=strat, cfg=cfg, run_dir=run_dir, max_loops=max_loops)


if __name__ == "__main__":
    app()
