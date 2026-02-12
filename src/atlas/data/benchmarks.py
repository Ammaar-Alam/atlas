from __future__ import annotations

import io
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

from atlas.logging_utils import get_logger
from atlas.market import safe_filename_symbol

logger = get_logger(__name__)


@dataclass(frozen=True)
class BenchmarkTotalReturn:
    symbol: str
    source: str
    start_observed: str
    end_observed: str
    start_price: float
    end_price: float
    total_return: float

    def to_dict(self) -> dict[str, object]:
        return {
            "symbol": self.symbol,
            "source": self.source,
            "start_observed": self.start_observed,
            "end_observed": self.end_observed,
            "start_price": float(self.start_price),
            "end_price": float(self.end_price),
            "total_return": float(self.total_return),
        }


def _to_utc_timestamp(value: object) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tz is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _stooq_cache_path(*, stooq_symbol: str) -> Path:
    cache_dir = Path("outputs") / "cache" / "benchmarks"
    cache_dir.mkdir(parents=True, exist_ok=True)
    safe = safe_filename_symbol(stooq_symbol)
    return cache_dir / f"stooq_{safe}.csv"


def load_stooq_daily_ohlcv(
    *,
    stooq_symbol: str,
    cache_ttl_s: float = 6 * 3600.0,
) -> pd.DataFrame:
    """
    Load daily OHLCV from Stooq (free CSV endpoint) with a small local cache.

    Note: Stooq uses symbols like "spy.us" for US ETFs.
    """
    stooq_symbol = (stooq_symbol or "").strip().lower()
    if not stooq_symbol:
        raise ValueError("stooq_symbol is required")

    path = _stooq_cache_path(stooq_symbol=stooq_symbol)
    now = time.time()
    if path.exists():
        try:
            age_s = max(0.0, now - float(path.stat().st_mtime))
            if age_s <= float(cache_ttl_s):
                df = pd.read_csv(path)
                return df
        except Exception:
            pass

    url = "https://stooq.com/q/d/l/"
    params = {"s": stooq_symbol, "i": "d"}
    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text))
        if not df.empty:
            try:
                path.write_text(resp.text)
            except Exception as exc:
                logger.warning("Failed to write Stooq cache %s: %s", path, exc)
        return df
    except Exception as exc:
        if path.exists():
            logger.warning("Stooq fetch failed (%s); falling back to cache %s", exc, path)
            return pd.read_csv(path)
        raise


def stooq_total_return(
    *,
    stooq_symbol: str,
    start: datetime,
    end: datetime,
    cache_ttl_s: float = 6 * 3600.0,
) -> Optional[BenchmarkTotalReturn]:
    """
    Compute total return using daily close prices, with an exclusive end bound: [start, end).

    We map datetimes to midnight UTC for alignment, then:
    - start_observed = first trading day >= start_date
    - end_observed   = last trading day  < end_date
    """
    start_ts = _to_utc_timestamp(start).normalize()
    end_ts = _to_utc_timestamp(end).normalize()
    if end_ts <= start_ts:
        return None

    raw = load_stooq_daily_ohlcv(stooq_symbol=stooq_symbol, cache_ttl_s=cache_ttl_s)
    if raw.empty:
        return None

    date_col = "Date" if "Date" in raw.columns else "date"
    close_col = "Close" if "Close" in raw.columns else "close"
    if date_col not in raw.columns or close_col not in raw.columns:
        return None

    df = raw[[date_col, close_col]].copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce", utc=True)
    df[close_col] = pd.to_numeric(df[close_col], errors="coerce")
    df = df.dropna(subset=[date_col, close_col])
    if df.empty:
        return None

    df = df.sort_values(date_col).set_index(date_col)
    close = df[close_col].astype(float)

    start_obs = close.index[close.index >= start_ts].min()
    end_obs = close.index[close.index < end_ts].max()
    if start_obs is None or end_obs is None:
        return None
    if pd.Timestamp(end_obs) <= pd.Timestamp(start_obs):
        return None

    start_px = float(close.loc[start_obs])
    end_px = float(close.loc[end_obs])
    if start_px <= 0.0:
        return None

    total_return = float(end_px / start_px - 1.0)
    return BenchmarkTotalReturn(
        symbol=str(stooq_symbol),
        source="stooq",
        start_observed=pd.Timestamp(start_obs).isoformat(),
        end_observed=pd.Timestamp(end_obs).isoformat(),
        start_price=float(start_px),
        end_price=float(end_px),
        total_return=float(total_return),
    )


def spy_total_return(
    *,
    start: datetime,
    end: datetime,
    cache_ttl_s: float = 6 * 3600.0,
) -> Optional[BenchmarkTotalReturn]:
    """
    Convenience wrapper for S&P 500 via SPY ETF on Stooq ("spy.us").
    """
    return stooq_total_return(
        stooq_symbol="spy.us",
        start=start,
        end=end,
        cache_ttl_s=cache_ttl_s,
    )

