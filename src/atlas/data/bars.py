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


_TF_RE = re.compile(r"^(?P<n>\\d+)\\s*min$", re.IGNORECASE)


def parse_bar_timeframe(value: str) -> BarTimeframe:
    value = value.strip()
    if value.lower() in {"1min", "1m"}:
        return BarTimeframe(name="1Min", minutes=1)
    if value.lower() in {"5min", "5m"}:
        return BarTimeframe(name="5Min", minutes=5)

    m = _TF_RE.match(value)
    if m:
        minutes = int(m.group("n"))
        if minutes <= 0:
            raise ValueError("bar timeframe minutes must be > 0")
        return BarTimeframe(name=f"{minutes}Min", minutes=minutes)

    raise ValueError("unsupported bar timeframe, expected like 1Min or 5Min")


def resample_ohlcv(bars: pd.DataFrame, *, minutes: int) -> pd.DataFrame:
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
    out = (
        bars.resample(rule, label="left", closed="left")
        .agg(agg)
        .dropna(subset=["open", "high", "low", "close"])
    )
    out["volume"] = out["volume"].fillna(0.0)
    out = out[out["volume"] > 0]
    return out[["open", "high", "low", "close", "volume"]].copy()


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
