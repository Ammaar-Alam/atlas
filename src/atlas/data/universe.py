from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from atlas.config import AlpacaSettings
from atlas.data.alpaca_data import load_stock_bars_cached
from atlas.data.bars import BarTimeframe, filter_regular_hours, resample_ohlcv
from atlas.data.csv_loader import load_bars_csv


@dataclass(frozen=True)
class UniverseBars:
    bars_by_symbol: dict[str, pd.DataFrame]
    source: str
    hint: str
    timeframe: BarTimeframe


def _load_sample(symbol: str) -> pd.DataFrame:
    path = Path("data") / "sample" / f"{symbol}_1min_sample.csv"
    if not path.exists():
        if symbol.upper() == "QQQ":
            spy_path = Path("data") / "sample" / "SPY_1min_sample.csv"
            if spy_path.exists():
                spy = load_bars_csv(spy_path)
                out = spy.copy()
                scale = 0.87
                out[["open", "high", "low", "close"]] = (
                    out[["open", "high", "low", "close"]].astype(float) * scale
                )
                out["volume"] = (out["volume"].astype(float) * 0.6).round().astype(int)
                return out[["open", "high", "low", "close", "volume"]].copy()

        raise FileNotFoundError(f"missing sample data for {symbol}: {path}")
    return load_bars_csv(path)


def _load_csv_symbol(*, symbol: str, csv_path: Optional[Path], csv_dir: Optional[Path]) -> pd.DataFrame:
    if csv_path is not None:
        return load_bars_csv(csv_path)

    if csv_dir is None:
        raise ValueError("csv_dir is required when loading multiple symbols from csv")

    if not csv_dir.exists() or not csv_dir.is_dir():
        raise ValueError(f"csv_dir must be a directory: {csv_dir}")

    candidates = [
        csv_dir / f"{symbol}.csv",
        csv_dir / f"{symbol}_1min_sample.csv",
        csv_dir / f"{symbol}_bars.csv",
    ]
    for path in candidates:
        if path.exists():
            return load_bars_csv(path)

    raise FileNotFoundError(f"no csv found for {symbol} in {csv_dir}")


def load_universe_bars(
    *,
    symbols: list[str],
    data_source: str,
    timeframe: BarTimeframe,
    start: Optional[datetime],
    end: Optional[datetime],
    csv_path: Optional[Path] = None,
    csv_dir: Optional[Path] = None,
    alpaca_settings: Optional[AlpacaSettings] = None,
    alpaca_feed: str = "delayed_sip",
    regular_hours_only: bool = True,
) -> UniverseBars:
    if not symbols:
        raise ValueError("symbols must be non-empty")
    symbols = [s.strip().upper() for s in symbols if s.strip()]
    if not symbols:
        raise ValueError("symbols must be non-empty")

    bars_by_symbol: dict[str, pd.DataFrame] = {}

    if data_source == "sample":
        for symbol in symbols:
            bars_by_symbol[symbol] = _load_sample(symbol)
        hint = "data/sample/*_1min_sample.csv"
    elif data_source == "csv":
        for symbol in symbols:
            bars_by_symbol[symbol] = _load_csv_symbol(
                symbol=symbol, csv_path=csv_path, csv_dir=csv_dir
            )
        hint = str(csv_path or csv_dir or "")
    elif data_source == "alpaca":
        if alpaca_settings is None:
            raise ValueError("alpaca_settings is required when data_source=alpaca")
        if start is None or end is None:
            raise ValueError("start/end are required when data_source=alpaca")
        for symbol in symbols:
            bars_by_symbol[symbol] = load_stock_bars_cached(
                settings=alpaca_settings,
                symbol=symbol,
                start=start,
                end=end,
                timeframe=timeframe.name,
                feed=alpaca_feed,
            )
        hint = f"{start.isoformat()} -> {end.isoformat()} feed={alpaca_feed}"
    else:
        raise ValueError("data_source must be one of: sample, csv, alpaca")

    for symbol, bars in list(bars_by_symbol.items()):
        if start is not None:
            bars = bars[bars.index >= start]
        if end is not None:
            bars = bars[bars.index <= end]
        if regular_hours_only:
            bars = filter_regular_hours(bars)
        if timeframe.minutes > 1:
            bars = resample_ohlcv(bars, minutes=timeframe.minutes)
        bars_by_symbol[symbol] = bars

    for symbol, bars in bars_by_symbol.items():
        if len(bars) < 3:
            raise ValueError(f"too few bars for {symbol} after filtering: {len(bars)}")

    return UniverseBars(
        bars_by_symbol=bars_by_symbol,
        source=data_source,
        hint=hint,
        timeframe=timeframe,
    )
