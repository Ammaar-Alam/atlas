from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import typer

from atlas.backtest.engine import BacktestConfig, run_backtest
from atlas.config import (
    get_alpaca_settings,
    get_default_max_position_notional_usd,
    get_log_level,
)
from atlas.data.alpaca_data import download_stock_bars_to_csv, load_stock_bars_cached
from atlas.data.csv_loader import load_bars_csv
from atlas.logging_utils import setup_logging
from atlas.paper.runner import PaperConfig, run_paper_loop
from atlas.strategies.registry import build_strategy
from atlas.utils.time import parse_iso_datetime

app = typer.Typer(add_completion=False)
logger = logging.getLogger(__name__)


def _run_id(prefix: str) -> str:
    return datetime.now().strftime(f"{prefix}_%Y%m%d_%H%M%S")


@app.command()
def download_bars(
    symbol: str = typer.Option(..., help="US equity symbol, e.g. SPY"),
    start: str = typer.Option(..., help="ISO datetime, e.g. 2024-01-02T09:30:00-05:00"),
    end: str = typer.Option(..., help="ISO datetime, e.g. 2024-01-02T16:00:00-05:00"),
    timeframe: str = typer.Option("1Min", help="Only 1Min supported in this scaffold"),
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
    )


@app.command()
def backtest(
    symbol: str = typer.Option("SPY", help="Symbol to backtest"),
    data_source: str = typer.Option(
        "sample", help="sample|csv|alpaca", show_default=True
    ),
    csv_path: Optional[Path] = typer.Option(None, help="CSV path when data-source=csv"),
    start: Optional[str] = typer.Option(None, help="ISO datetime for alpaca data"),
    end: Optional[str] = typer.Option(None, help="ISO datetime for alpaca data"),
    strategy: str = typer.Option("ma_crossover", help="Strategy name"),
    strategy_params: Optional[Path] = typer.Option(
        None, help="JSON file with strategy parameters"
    ),
    fast_window: int = typer.Option(10, help="ma_crossover fast window"),
    slow_window: int = typer.Option(30, help="ma_crossover slow window"),
    initial_cash: float = typer.Option(100_000.0, help="Starting cash"),
    max_position_notional_usd: Optional[float] = typer.Option(
        None, help="Max notional per symbol"
    ),
    slippage_bps: float = typer.Option(0.0, help="Fill slippage in basis points"),
    allow_short: bool = typer.Option(False, help="Allow negative exposure"),
) -> None:
    run_dir = Path("outputs") / "backtests" / _run_id("backtest")
    setup_logging(level=get_log_level(), log_file=run_dir / "run.log")

    if max_position_notional_usd is None:
        max_position_notional_usd = get_default_max_position_notional_usd(mode="backtest")

    if data_source == "sample":
        sample_path = Path("data") / "sample" / f"{symbol}_1min_sample.csv"
        if not sample_path.exists():
            raise typer.BadParameter(f"missing sample data for {symbol}: {sample_path}")
        bars = load_bars_csv(sample_path)
    elif data_source == "csv":
        if csv_path is None:
            raise typer.BadParameter("csv-path is required when data-source=csv")
        bars = load_bars_csv(csv_path)
    elif data_source == "alpaca":
        if start is None or end is None:
            raise typer.BadParameter("start and end are required when data-source=alpaca")
        settings = get_alpaca_settings(require_keys=True)
        bars = load_stock_bars_cached(
            settings=settings,
            symbol=symbol,
            start=parse_iso_datetime(start),
            end=parse_iso_datetime(end),
            timeframe="1Min",
        )
    else:
        raise typer.BadParameter("data-source must be one of: sample, csv, alpaca")

    strat = build_strategy(
        name=strategy,
        params_path=strategy_params,
        fast_window=fast_window,
        slow_window=slow_window,
    )

    cfg = BacktestConfig(
        symbol=symbol,
        initial_cash=initial_cash,
        max_position_notional_usd=float(max_position_notional_usd),
        slippage_bps=slippage_bps,
        allow_short=allow_short,
    )

    run_backtest(bars=bars, strategy=strat, cfg=cfg, run_dir=run_dir)


@app.command()
def paper(
    symbols: list[str] = typer.Option(["SPY"], help="Symbols to trade, repeatable"),
    strategy: str = typer.Option("ma_crossover", help="Strategy name"),
    strategy_params: Optional[Path] = typer.Option(
        None, help="JSON file with strategy parameters"
    ),
    fast_window: int = typer.Option(10, help="ma_crossover fast window"),
    slow_window: int = typer.Option(30, help="ma_crossover slow window"),
    lookback_bars: int = typer.Option(200, help="Bars fetched each loop"),
    poll_seconds: int = typer.Option(60, help="Minimum seconds between loops"),
    max_position_notional_usd: Optional[float] = typer.Option(
        None, help="Max notional per symbol"
    ),
    allow_trading_when_closed: bool = typer.Option(
        False, help="Skip market-open check"
    ),
    dry_run: bool = typer.Option(False, help="Do not submit orders"),
    max_loops: Optional[int] = typer.Option(None, help="Stop after N loops"),
) -> None:
    settings = get_alpaca_settings(require_keys=True)
    run_dir = Path("outputs") / "paper" / _run_id("paper")
    setup_logging(level=get_log_level(), log_file=run_dir / "run.log")

    if max_position_notional_usd is None:
        max_position_notional_usd = get_default_max_position_notional_usd(mode="paper")

    strat = build_strategy(
        name=strategy,
        params_path=strategy_params,
        fast_window=fast_window,
        slow_window=slow_window,
    )

    cfg = PaperConfig(
        symbols=symbols,
        lookback_bars=lookback_bars,
        poll_seconds=poll_seconds,
        max_position_notional_usd=float(max_position_notional_usd),
        allow_trading_when_closed=allow_trading_when_closed,
        dry_run=dry_run,
    )

    logger.info("paper=%s allow_live=%s", settings.paper, settings.allow_live)
    run_paper_loop(settings=settings, strategy=strat, cfg=cfg, run_dir=run_dir, max_loops=max_loops)


if __name__ == "__main__":
    app()
