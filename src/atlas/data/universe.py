from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

import pandas as pd

from atlas.config import AlpacaSettings
from atlas.data.alpaca_data import load_crypto_bars_cached, load_stock_bars_cached
from atlas.data.bars import BarTimeframe, filter_regular_hours, resample_ohlcv
from atlas.data.csv_loader import load_bars_csv
from atlas.data.coinbase_data import load_coinbase_bars_cached
from atlas.market import Market, parse_market, safe_filename_symbol
from atlas.utils.time import NY_TZ


@dataclass(frozen=True)
class UniverseBars:
    bars_by_symbol: dict[str, pd.DataFrame]
    source: str
    hint: str
    timeframe: BarTimeframe


def _load_sample(symbol: str, *, assume_tz: ZoneInfo) -> pd.DataFrame:
    safe_symbol = safe_filename_symbol(symbol)
    path = Path("data") / "sample" / f"{safe_symbol}_1min_sample.csv"
    if not path.exists():
        if symbol.upper() == "QQQ":
            spy_path = Path("data") / "sample" / f"{safe_filename_symbol('SPY')}_1min_sample.csv"
            if spy_path.exists():
                spy = load_bars_csv(spy_path, assume_tz=assume_tz)
                out = spy.copy()
                scale = 0.87
                out[["open", "high", "low", "close"]] = (
                    out[["open", "high", "low", "close"]].astype(float) * scale
                )
                out["volume"] = (out["volume"].astype(float) * 0.6).round().astype(int)
                return out[["open", "high", "low", "close", "volume"]].copy()

        raise FileNotFoundError(f"missing sample data for {symbol}: {path}")
    return load_bars_csv(path, assume_tz=assume_tz)


def _load_csv_symbol(
    *, symbol: str, csv_path: Optional[Path], csv_dir: Optional[Path], assume_tz: ZoneInfo
) -> pd.DataFrame:
    if csv_path is not None:
        return load_bars_csv(csv_path, assume_tz=assume_tz)

    if csv_dir is None:
        raise ValueError("csv_dir is required when loading multiple symbols from csv")

    if not csv_dir.exists() or not csv_dir.is_dir():
        raise ValueError(f"csv_dir must be a directory: {csv_dir}")

    safe_symbol = safe_filename_symbol(symbol)
    flat_symbol = safe_symbol.replace("_", "")
    candidates = [
        csv_dir / f"{safe_symbol}.csv",
        csv_dir / f"{safe_symbol}_1min_sample.csv",
        csv_dir / f"{safe_symbol}_bars.csv",
        csv_dir / f"{flat_symbol}.csv",
        csv_dir / f"{flat_symbol}_bars.csv",
    ]
    for path in candidates:
        if path.exists():
            return load_bars_csv(path, assume_tz=assume_tz)

    raise FileNotFoundError(f"no csv found for {symbol} in {csv_dir}")


def _densify_crypto_bars(bars: pd.DataFrame, *, minutes: int) -> pd.DataFrame:
    if len(bars) < 2 or minutes <= 0:
        return bars
    if not isinstance(bars.index, pd.DatetimeIndex):
        raise ValueError("bars index must be a DatetimeIndex")
    if bars.index.tz is None:
        raise ValueError("bars index must be tz-aware")

    # Important: do NOT floor to the target frequency. Some data sources (notably Alpaca crypto)
    # can timestamp multi-hour bars on an exchange-specific offset (e.g. 6H bars at 01:00/07:00/...).
    # Flooring to the frequency would shift the grid and can erase all rows on reindex.
    bars = bars.sort_index()
    start = bars.index.min()
    end = bars.index.max()
    full_index = pd.date_range(start=start, end=end, freq=f"{int(minutes)}min", tz=bars.index.tz)

    out = bars.reindex(full_index)
    for col in ["open", "high", "low", "close"]:
        out[col] = out[col].ffill()
    out = out.dropna(subset=["open", "high", "low", "close"])
    out["volume"] = out["volume"].fillna(0.0)
    if "funding_rate" in out.columns:
        out["funding_rate"] = out["funding_rate"].ffill().fillna(0.0)
    cols = ["open", "high", "low", "close", "volume"]
    if "funding_rate" in out.columns:
        cols.append("funding_rate")
    return out[cols].copy()


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
    market: str = "equity",
) -> UniverseBars:
    if not symbols:
        raise ValueError("symbols must be non-empty")
    symbols = [s.strip().upper() for s in symbols if s.strip()]
    if not symbols:
        raise ValueError("symbols must be non-empty")

    mkt = parse_market(market)
    if mkt in {Market.CRYPTO, Market.DERIVATIVES} and regular_hours_only:
        # Crypto + crypto-derivatives trade (near) 24/7; don't drop overnight/weekend bars by default.
        regular_hours_only = False
    assume_tz = ZoneInfo("UTC") if mkt in {Market.CRYPTO, Market.DERIVATIVES} else NY_TZ

    bars_by_symbol: dict[str, pd.DataFrame] = {}

    def _infer_median_bar_minutes(index: pd.DatetimeIndex) -> float:
        if len(index) < 3:
            return 0.0
        diffs = index.to_series().diff().dropna().dt.total_seconds() / 60.0
        diffs = diffs[diffs > 0]
        if len(diffs) == 0:
            return 0.0
        median = float(diffs.median())
        return median if median > 0 else 0.0

    if data_source == "sample":
        for symbol in symbols:
            bars_by_symbol[symbol] = _load_sample(symbol, assume_tz=assume_tz)
        hint = "data/sample/*_1min_sample.csv"
    elif data_source == "csv":
        for symbol in symbols:
            bars_by_symbol[symbol] = _load_csv_symbol(
                symbol=symbol, csv_path=csv_path, csv_dir=csv_dir, assume_tz=assume_tz
            )
        hint = str(csv_path or csv_dir or "")
    elif data_source == "alpaca":
        if alpaca_settings is None:
            raise ValueError("alpaca_settings is required when data_source=alpaca")
        if start is None or end is None:
            raise ValueError("start/end are required when data_source=alpaca")
        fetch_tf = timeframe
        if (
            mkt == Market.CRYPTO
            and int(timeframe.minutes) >= 120
            and int(timeframe.minutes) % 60 == 0
        ):
            # Fetch 1H and resample locally for multi-hour crypto windows. Providers can
            # define higher-timeframe candle boundaries differently (e.g. 6H candles
            # starting at 00:00 UTC vs 06:00 UTC). Resampling from 1H makes the candle
            # grid deterministic and comparable across data sources.
            fetch_tf = BarTimeframe(name="1H", minutes=60)
        for symbol in symbols:
            if mkt == Market.CRYPTO:
                bars_by_symbol[symbol] = load_crypto_bars_cached(
                    settings=alpaca_settings,
                    symbol=symbol,
                    start=start,
                    end=end,
                    timeframe=fetch_tf.name,
                )
            else:
                bars_by_symbol[symbol] = load_stock_bars_cached(
                    settings=alpaca_settings,
                    symbol=symbol,
                    start=start,
                    end=end,
                    timeframe=timeframe.name,
                    feed=alpaca_feed,
                )
        hint = (
            f"{start.isoformat()} -> {end.isoformat()} crypto"
            if mkt == Market.CRYPTO
            else f"{start.isoformat()} -> {end.isoformat()} feed={alpaca_feed}"
        )
    elif data_source == "coinbase":
        if start is None or end is None:
            raise ValueError("start/end are required when data_source=coinbase")

        fetch_minutes = int(timeframe.minutes)
        if fetch_minutes >= 120 and fetch_minutes % 60 == 0:
            # Same rationale as Alpaca: use 1H candles and resample locally for
            # deterministic multi-hour candle boundaries.
            fetch_minutes = 60

        # Map timeframe to Coinbase granularity
        if fetch_minutes == 1:
            granularity = "ONE_MINUTE"
        elif fetch_minutes == 5:
            granularity = "FIVE_MINUTE"
        elif fetch_minutes == 15:
            granularity = "FIFTEEN_MINUTE"
        elif fetch_minutes == 30:
            # Coinbase does not support native 30-minute candles; fetch 15-minute and resample downstream.
            granularity = "FIFTEEN_MINUTE"
        elif fetch_minutes == 60:
            granularity = "ONE_HOUR"
        else:
             # Fallback: fetch 1 minute and let downstream resample
             granularity = "ONE_MINUTE"

        for symbol in symbols:
            bars_by_symbol[symbol] = load_coinbase_bars_cached(
                symbol=symbol,
                start=start,
                end=end,
                granularity=granularity,
            )
        hint = f"{start.isoformat()} -> {end.isoformat()} coinbase"
    else:
        raise ValueError("data_source must be one of: sample, csv, alpaca, coinbase")

    for symbol, bars in list(bars_by_symbol.items()):
        if bars is None or bars.empty:
            raise ValueError(f"no data found for {symbol} (check symbol or source)")
        
        if start is not None:
            bars = bars[bars.index >= start]
        if end is not None:
            bars = bars[bars.index <= end]
        if regular_hours_only:
            bars = filter_regular_hours(bars)
        if mkt in {Market.CRYPTO, Market.DERIVATIVES}:
            # Standardize crypto/derivatives bars to UTC to avoid DST artifacts and to
            # make resampling boundaries comparable across providers.
            idx = bars.index
            if idx.tz is None:
                bars = bars.copy()
                bars.index = idx.tz_localize("UTC")
            else:
                utc_tz = ZoneInfo("UTC")
                if str(idx.tz) != str(utc_tz):
                    bars = bars.copy()
                    bars.index = idx.tz_convert(utc_tz)
        if timeframe.minutes > 1:
            # Only resample when the raw data is not already at the requested cadence.
            # This avoids distorting provider-supplied multi-hour crypto bars that may be
            # timestamped on an offset relative to local midnight.
            observed = _infer_median_bar_minutes(bars.index)
            if observed <= 0.0 or abs(observed - float(timeframe.minutes)) > 1e-6:
                bars = resample_ohlcv(
                    bars,
                    minutes=timeframe.minutes,
                    drop_zero_volume=(mkt == Market.EQUITY),
                )
        if mkt in {Market.CRYPTO, Market.DERIVATIVES}:
            bars = _densify_crypto_bars(bars, minutes=timeframe.minutes)
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
