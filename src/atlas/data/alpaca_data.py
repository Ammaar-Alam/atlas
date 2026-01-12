from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

from atlas.config import AlpacaSettings
from atlas.data.bars import parse_bar_timeframe
from atlas.utils.time import NY_TZ

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AlpacaBarsDownload:
    symbol: str
    start: datetime
    end: datetime
    timeframe: str


def _bars_cache_path(root: Path, req: AlpacaBarsDownload) -> Path:
    safe_symbol = req.symbol.replace("/", "_")
    start_s = req.start.isoformat().replace(":", "").replace("+", "")
    end_s = req.end.isoformat().replace(":", "").replace("+", "")
    return root / "data" / "alpaca" / safe_symbol / f"{safe_symbol}_{req.timeframe}_{start_s}_{end_s}.csv"


def _normalize_bars_df(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if isinstance(df.index, pd.MultiIndex):
        df = df.xs(symbol)

    df = df.sort_index()
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert(NY_TZ)

    cols = ["open", "high", "low", "close", "volume"]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"alpaca bars missing columns: {missing}")
    return df[cols].copy()


def download_stock_bars_to_csv(
    *,
    settings: AlpacaSettings,
    symbol: str,
    start: datetime,
    end: datetime,
    timeframe: str,
    out_path: Optional[Path],
) -> Path:
    tf = parse_bar_timeframe(timeframe)

    client = StockHistoricalDataClient(
        settings.api_key, settings.secret_key, url_override=settings.data_url_override
    )
    req = StockBarsRequest(
        symbol_or_symbols=[symbol],
        timeframe=TimeFrame(amount=tf.minutes, unit=TimeFrameUnit.Minute),
        start=start,
        end=end,
    )

    logger.info("downloading bars from alpaca: %s %s -> %s", symbol, start, end)
    res = client.get_stock_bars(req)
    bars = _normalize_bars_df(res.df, symbol)

    if out_path is None:
        out_path = _bars_cache_path(Path.cwd(), AlpacaBarsDownload(symbol, start, end, timeframe))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    export = bars.copy()
    export.insert(0, "timestamp", export.index.astype(str))
    export.to_csv(out_path, index=False)
    logger.info("saved bars to %s", out_path)
    return out_path


def load_stock_bars_cached(
    *,
    settings: AlpacaSettings,
    symbol: str,
    start: datetime,
    end: datetime,
    timeframe: str,
) -> pd.DataFrame:
    _ = parse_bar_timeframe(timeframe)
    path = _bars_cache_path(Path.cwd(), AlpacaBarsDownload(symbol, start, end, timeframe))
    if path.exists():
        logger.info("using cached bars: %s", path)
        df = pd.read_csv(path)
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="raise")
        df["timestamp"] = df["timestamp"].dt.tz_convert(NY_TZ)
        df = df.set_index("timestamp").sort_index()
        return df[["open", "high", "low", "close", "volume"]].copy()

    download_stock_bars_to_csv(
        settings=settings,
        symbol=symbol,
        start=start,
        end=end,
        timeframe=timeframe,
        out_path=path,
    )
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="raise")
    df["timestamp"] = df["timestamp"].dt.tz_convert(NY_TZ)
    df = df.set_index("timestamp").sort_index()
    return df[["open", "high", "low", "close", "volume"]].copy()
