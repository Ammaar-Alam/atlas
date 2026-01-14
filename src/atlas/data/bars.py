from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import time

import pandas as pd

from atlas.utils.time import NY_TZ


@dataclass(frozen=True)
class BarTimeframe:
    name: str
    minutes: int


_TF_MIN_RE = re.compile(r"^(?P<n>\d+)\s*(?:min|m)$", re.IGNORECASE)
_TF_HOUR_RE = re.compile(r"^(?P<n>\d+)\s*(?:h|hr|hour|hours)$", re.IGNORECASE)


def parse_bar_timeframe(value: str) -> BarTimeframe:
    value = value.strip()
    if not value:
        raise ValueError("bar timeframe must be non-empty")

    m = _TF_MIN_RE.match(value)
    if m:
        minutes = int(m.group("n"))
        if minutes <= 0:
            raise ValueError("bar timeframe minutes must be > 0")
        if minutes >= 60 and minutes % 60 == 0:
            hours = minutes // 60
            return BarTimeframe(name=f"{hours}H", minutes=minutes)
        return BarTimeframe(name=f"{minutes}Min", minutes=minutes)

    m = _TF_HOUR_RE.match(value)
    if m:
        hours = int(m.group("n"))
        if hours <= 0:
            raise ValueError("bar timeframe hours must be > 0")
        minutes = hours * 60
        return BarTimeframe(name=f"{hours}H", minutes=minutes)

    raise ValueError("unsupported bar timeframe, expected like 1Min, 5Min, 30Min, 1H, 4H")


def resample_ohlcv(
    bars: pd.DataFrame, *, minutes: int, drop_zero_volume: bool = True
) -> pd.DataFrame:
    if len(bars) < 2 or minutes <= 1:
        return bars.copy()

    if not isinstance(bars.index, pd.DatetimeIndex):
        raise ValueError("bars index must be a DatetimeIndex")
    if bars.index.tz is None:
        raise ValueError("bars index must be tz-aware")

    rule = f"{int(minutes)}min"
    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    # Optional columns (kept when present). Funding rates are typically stepwise; mean is a simple
    # approximation over a larger resampled window.
    if "funding_rate" in bars.columns:
        agg["funding_rate"] = "mean"
    out = (
        bars.resample(rule, label="left", closed="left")
        .agg(agg)
        .dropna(subset=["open", "high", "low", "close"])
    )
    out["volume"] = out["volume"].fillna(0.0)
    if "funding_rate" in out.columns:
        out["funding_rate"] = out["funding_rate"].ffill().fillna(0.0)
    if drop_zero_volume:
        out = out[out["volume"] > 0]
    cols = ["open", "high", "low", "close", "volume"]
    if "funding_rate" in out.columns:
        cols.append("funding_rate")
    return out[cols].copy()


def filter_regular_hours(bars: pd.DataFrame, *, weekdays_only: bool = True) -> pd.DataFrame:
    if not isinstance(bars.index, pd.DatetimeIndex):
        raise ValueError("bars index must be a DatetimeIndex")
    if bars.index.tz is None:
        raise ValueError("bars index must be tz-aware")
    idx = bars.index.tz_convert(NY_TZ)
    bars = bars.copy()
    bars.index = idx
    bars = bars.between_time(time(9, 30), time(15, 59, 59))
    if weekdays_only:
        bars = bars[bars.index.dayofweek < 5]
    return bars
