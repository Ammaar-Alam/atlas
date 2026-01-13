from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

import pandas as pd
from alpaca.data.historical import CryptoHistoricalDataClient, StockHistoricalDataClient
from alpaca.data.enums import DataFeed
from alpaca.data.requests import CryptoBarsRequest, StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

from atlas.config import AlpacaSettings
from atlas.data.bars import BarTimeframe, parse_bar_timeframe
from atlas.utils.time import NY_TZ, now_ny

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AlpacaBarsDownload:
    symbol: str
    start: datetime
    end: datetime
    timeframe: str
    feed: str


def _bars_cache_path(root: Path, req: AlpacaBarsDownload) -> Path:
    safe_symbol = req.symbol.replace("/", "_")
    start_s = req.start.isoformat().replace(":", "").replace("+", "")
    end_s = req.end.isoformat().replace(":", "").replace("+", "")
    feed = (req.feed or "iex").replace("/", "_")
    return (
        root
        / "data"
        / "alpaca"
        / safe_symbol
        / f"{safe_symbol}_{req.timeframe}_{feed}_{start_s}_{end_s}.csv"
    )


@dataclass(frozen=True)
class AlpacaFeedConfig:
    api_feed: DataFeed
    cache_label: str
    min_end_delay_minutes: int = 0


def parse_alpaca_feed(value: str) -> AlpacaFeedConfig:
    """
    Alpaca-py documentation commonly references IEX and SIP feeds for stock data.
    Some accounts allow SIP only when end-time is sufficiently old (e.g. >= 15 minutes).
    To support that safely, we expose a "delayed_sip" alias that uses feed=sip but
    automatically clamps overly-recent end timestamps.
    """
    raw = value.strip()
    normalized = " ".join(raw.split()).lower().replace("-", "_").replace(" ", "_")

    if normalized == "iex":
        return AlpacaFeedConfig(api_feed=DataFeed.IEX, cache_label="iex", min_end_delay_minutes=0)
    if normalized == "sip":
        return AlpacaFeedConfig(api_feed=DataFeed.SIP, cache_label="sip", min_end_delay_minutes=0)
    if normalized in {"delayed_sip", "sip_delayed", "delayed"}:
        return AlpacaFeedConfig(api_feed=DataFeed.SIP, cache_label="delayed_sip", min_end_delay_minutes=16)

    raise ValueError("alpaca feed must be one of: iex, sip, delayed_sip")

def to_alpaca_timeframe(tf: BarTimeframe) -> TimeFrame:
    """
    Convert our minute-based BarTimeframe to an Alpaca TimeFrame.

    Alpaca supports hour-based timeframes natively; using Hour units for exact
    hour multiples avoids relying on "N minutes" support for larger windows.
    """
    minutes = int(tf.minutes)
    if minutes <= 0:
        raise ValueError("bar timeframe minutes must be > 0")
    if minutes % 1440 == 0:
        return TimeFrame(amount=minutes // 1440, unit=TimeFrameUnit.Day)
    if minutes % 60 == 0:
        return TimeFrame(amount=minutes // 60, unit=TimeFrameUnit.Hour)
    return TimeFrame(amount=minutes, unit=TimeFrameUnit.Minute)


def _clamp_end_for_feed(end: datetime, *, delay_minutes: int) -> datetime:
    if delay_minutes <= 0:
        return end
    latest = now_ny() - timedelta(minutes=delay_minutes)
    end_ny = end if end.tzinfo is not None else end.replace(tzinfo=NY_TZ)
    end_ny = end_ny.astimezone(NY_TZ)
    return min(end_ny, latest)


def _normalize_bars_df(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if isinstance(df.index, pd.MultiIndex):
        try:
            df = df.xs(symbol)
        except KeyError as exc:
            raise RuntimeError(f"alpaca returned no bars for symbol: {symbol}") from exc

    if "timestamp" in df.columns and not isinstance(df.index, pd.DatetimeIndex):
        ts = pd.to_datetime(df["timestamp"], errors="raise", utc=True)
        df = df.drop(columns=["timestamp"])
        df = df.copy()
        df.index = ts

    if not isinstance(df.index, pd.DatetimeIndex):
        if isinstance(df.index, pd.RangeIndex):
            raise RuntimeError(
                f"unexpected bars response for {symbol}: no datetime index (is the symbol valid?)"
            )
        try:
            df = df.copy()
            df.index = pd.to_datetime(df.index, errors="raise", utc=True)
        except Exception as exc:
            raise RuntimeError(
                f"unexpected bars index type from alpaca for {symbol}: {type(df.index).__name__}"
            ) from exc

    df = df.sort_index()
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert(NY_TZ)

    cols = ["open", "high", "low", "close", "volume"]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"alpaca bars missing columns: {missing}")
    out = df[cols].copy()
    if not len(out):
        raise RuntimeError(
            f"alpaca returned no bars for {symbol} (check symbol/pair and date range)"
        )
    return out


def _to_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=NY_TZ)
    return dt.astimezone(ZoneInfo("UTC"))


def _make_crypto_client(settings: AlpacaSettings) -> CryptoHistoricalDataClient:
    kwargs: dict[str, object] = {}
    if settings.api_key and settings.secret_key:
        kwargs["api_key"] = settings.api_key
        kwargs["secret_key"] = settings.secret_key
    if settings.data_url_override:
        kwargs["url_override"] = settings.data_url_override
    try:
        return CryptoHistoricalDataClient(**kwargs)
    except TypeError:
        kwargs.pop("url_override", None)
        return CryptoHistoricalDataClient(**kwargs)


def download_stock_bars_to_csv(
    *,
    settings: AlpacaSettings,
    symbol: str,
    start: datetime,
    end: datetime,
    timeframe: str,
    out_path: Optional[Path],
    feed: str = "delayed_sip",
) -> Path:
    tf = parse_bar_timeframe(timeframe)
    feed_cfg = parse_alpaca_feed(feed)
    end = _clamp_end_for_feed(end, delay_minutes=feed_cfg.min_end_delay_minutes)

    client = StockHistoricalDataClient(
        settings.api_key, settings.secret_key, url_override=settings.data_url_override
    )
    req = StockBarsRequest(
        symbol_or_symbols=[symbol],
        timeframe=to_alpaca_timeframe(tf),
        start=start,
        end=end,
        feed=feed_cfg.api_feed,
    )

    logger.info("downloading bars from alpaca: %s %s -> %s", symbol, start, end)
    try:
        res = client.get_stock_bars(req)
    except Exception as exc:
        msg = str(exc).lower()
        if "subscription" in msg and "sip" in msg and feed_cfg.api_feed == DataFeed.SIP:
            raise RuntimeError(
                "alpaca sip feed not available for this account. use feed=delayed_sip (sip with 15m delay) or feed=iex (live, limited)."
            ) from exc
        raise
    bars = _normalize_bars_df(res.df, symbol)

    if out_path is None:
        out_path = _bars_cache_path(
            Path.cwd(), AlpacaBarsDownload(symbol, start, end, timeframe, feed_cfg.cache_label)
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    export = bars.copy()
    export.insert(0, "timestamp", export.index.astype(str))
    export.to_csv(out_path, index=False)
    logger.info("saved bars to %s", out_path)
    return out_path


def download_crypto_bars_to_csv(
    *,
    settings: AlpacaSettings,
    symbol: str,
    start: datetime,
    end: datetime,
    timeframe: str,
    out_path: Optional[Path],
) -> Path:
    tf = parse_bar_timeframe(timeframe)
    start_utc = _to_utc(start)
    end_utc = _to_utc(end)
    client = _make_crypto_client(settings)
    req = CryptoBarsRequest(
        symbol_or_symbols=[symbol],
        timeframe=to_alpaca_timeframe(tf),
        start=start_utc,
        end=end_utc,
    )
    logger.info("downloading crypto bars from alpaca: %s %s -> %s", symbol, start_utc, end_utc)
    res = client.get_crypto_bars(req)
    bars = _normalize_bars_df(res.df, symbol)

    if out_path is None:
        out_path = _bars_cache_path(
            Path.cwd(), AlpacaBarsDownload(symbol, start_utc, end_utc, timeframe, "crypto")
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    export = bars.copy()
    export.insert(0, "timestamp", export.index.astype(str))
    export.to_csv(out_path, index=False)
    logger.info("saved crypto bars to %s", out_path)
    return out_path


def load_stock_bars_cached(
    *,
    settings: AlpacaSettings,
    symbol: str,
    start: datetime,
    end: datetime,
    timeframe: str,
    feed: str = "delayed_sip",
) -> pd.DataFrame:
    _ = parse_bar_timeframe(timeframe)
    feed_cfg = parse_alpaca_feed(feed)
    end = _clamp_end_for_feed(end, delay_minutes=feed_cfg.min_end_delay_minutes)
    path = _bars_cache_path(
        Path.cwd(), AlpacaBarsDownload(symbol, start, end, timeframe, feed_cfg.cache_label)
    )
    if path.exists():
        logger.info("using cached bars: %s", path)
        df = pd.read_csv(path)
        ts = pd.to_datetime(df["timestamp"], errors="raise", utc=True).dt.tz_convert(NY_TZ)
        df = df.drop(columns=["timestamp"])
        df.index = ts
        df = df.sort_index()
        return df[["open", "high", "low", "close", "volume"]].copy()

    download_stock_bars_to_csv(
        settings=settings,
        symbol=symbol,
        start=start,
        end=end,
        timeframe=timeframe,
        out_path=path,
        feed=feed,
    )
    df = pd.read_csv(path)
    ts = pd.to_datetime(df["timestamp"], errors="raise", utc=True).dt.tz_convert(NY_TZ)
    df = df.drop(columns=["timestamp"])
    df.index = ts
    df = df.sort_index()
    return df[["open", "high", "low", "close", "volume"]].copy()


def load_crypto_bars_cached(
    *,
    settings: AlpacaSettings,
    symbol: str,
    start: datetime,
    end: datetime,
    timeframe: str,
) -> pd.DataFrame:
    _ = parse_bar_timeframe(timeframe)
    start_utc = _to_utc(start)
    end_utc = _to_utc(end)
    path = _bars_cache_path(
        Path.cwd(), AlpacaBarsDownload(symbol, start_utc, end_utc, timeframe, "crypto")
    )
    if not path.exists():
        download_crypto_bars_to_csv(
            settings=settings,
            symbol=symbol,
            start=start_utc,
            end=end_utc,
            timeframe=timeframe,
            out_path=path,
        )
    df = pd.read_csv(path)
    ts = pd.to_datetime(df["timestamp"], errors="raise", utc=True).dt.tz_convert(NY_TZ)
    df = df.drop(columns=["timestamp"])
    df.index = ts
    df = df.sort_index()
    return df[["open", "high", "low", "close", "volume"]].copy()
