from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo


NY_TZ = ZoneInfo("America/New_York")


def now_ny() -> datetime:
    return datetime.now(tz=NY_TZ)


def parse_iso_datetime(value: str) -> datetime:
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=NY_TZ)
    return dt

