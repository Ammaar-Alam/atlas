from __future__ import annotations

from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from atlas.utils.time import NY_TZ


def load_bars_csv(path: Path, *, assume_tz: ZoneInfo = NY_TZ) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "timestamp" not in df.columns:
        raise ValueError("csv must include a timestamp column")

    ts = pd.to_datetime(df["timestamp"], utc=False, errors="raise")
    if getattr(ts.dt, "tz", None) is None:
        ts = ts.dt.tz_localize(assume_tz)
    ts = ts.dt.tz_convert(NY_TZ)

    df = df.drop(columns=["timestamp"])
    df.index = ts
    df = df.sort_index()

    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"csv missing required columns: {sorted(missing)}")

    return df[["open", "high", "low", "close", "volume"]].copy()
