from __future__ import annotations

import csv
import math
import random
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo


def main() -> None:
    out_path = Path("data") / "sample" / "SPY_1min_sample.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tz = ZoneInfo("America/New_York")
    start = datetime(2024, 1, 2, 9, 30, tzinfo=tz)

    rng = random.Random(42)
    price = 470.0

    rows: list[dict] = []
    for i in range(390):
        ts = start + timedelta(minutes=i)

        drift = 0.00002 * math.sin(i / 35.0)
        shock = rng.gauss(0.0, 0.0006)
        ret = drift + shock

        open_px = price
        close_px = max(0.01, price * (1.0 + ret))
        high_px = max(open_px, close_px) * (1.0 + rng.random() * 0.0009)
        low_px = min(open_px, close_px) * (1.0 - rng.random() * 0.0009)
        volume = int(250_000 * (0.5 + rng.random() * 1.5))

        rows.append(
            {
                "timestamp": ts.isoformat(),
                "open": round(open_px, 4),
                "high": round(high_px, 4),
                "low": round(low_px, 4),
                "close": round(close_px, 4),
                "volume": volume,
            }
        )

        price = close_px

    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["timestamp", "open", "high", "low", "close", "volume"])
        w.writeheader()
        w.writerows(rows)

    print(f"wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()

