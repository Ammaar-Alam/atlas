from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from atlas.data.benchmarks import load_stooq_daily_ohlcv

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BenchmarkSeries:
    symbol: str
    close: pd.Series


def _load_equity_curve(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "equity_curve.csv"
    if not path.exists():
        raise FileNotFoundError(f"missing equity curve: {path}")
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df = df.dropna(subset=["timestamp"]).set_index("timestamp").sort_index()
    if "equity" not in df.columns:
        raise ValueError(f"equity_curve.csv missing 'equity' column: {path}")
    return df


def _load_daily_equity(run_dir: Path) -> pd.Series:
    equity_curve = _load_equity_curve(run_dir)
    daily = equity_curve["equity"].astype(float).resample("1D").last().dropna()
    if daily.empty:
        raise ValueError("no daily equity points found")
    return daily


def _load_stooq_close_series(*, stooq_symbol: str) -> BenchmarkSeries:
    raw = load_stooq_daily_ohlcv(stooq_symbol=stooq_symbol)
    if raw.empty:
        return BenchmarkSeries(symbol=stooq_symbol, close=pd.Series(dtype=float))

    date_col = "Date" if "Date" in raw.columns else "date"
    close_col = "Close" if "Close" in raw.columns else "close"
    if date_col not in raw.columns or close_col not in raw.columns:
        return BenchmarkSeries(symbol=stooq_symbol, close=pd.Series(dtype=float))

    df = raw[[date_col, close_col]].copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce", utc=True)
    df[close_col] = pd.to_numeric(df[close_col], errors="coerce")
    df = df.dropna(subset=[date_col, close_col])
    if df.empty:
        return BenchmarkSeries(symbol=stooq_symbol, close=pd.Series(dtype=float))

    df = df.sort_values(date_col).set_index(date_col)
    close = df[close_col].astype(float)
    close.index = close.index.normalize()
    close = close[~close.index.duplicated(keep="last")]
    return BenchmarkSeries(symbol=stooq_symbol, close=close.sort_index())


def write_equity_vs_benchmark_artifacts(
    *,
    run_dir: Path,
    benchmark: str = "spy.us",
    out_csv: Optional[Path] = None,
    out_png: Optional[Path] = None,
) -> tuple[Optional[Path], Optional[Path]]:
    """
    Write per-run artifacts comparing strategy equity vs a benchmark.

    Outputs:
    - CSV: daily normalized equity series (strategy vs benchmark)
    - PNG: plot of the same series (requires matplotlib)
    """
    run_dir = Path(run_dir)
    daily_equity = _load_daily_equity(run_dir)

    bench_symbol = (benchmark or "").strip().lower()
    if not bench_symbol:
        raise ValueError("benchmark symbol is required (use 'spy.us' or pass '' to disable upstream)")
    bench = _load_stooq_close_series(stooq_symbol=bench_symbol)
    if bench.close.empty:
        raise ValueError(f"no benchmark close data for {bench_symbol!r}")

    start = max(daily_equity.index.min(), bench.close.index.min())
    end = min(daily_equity.index.max(), bench.close.index.max())
    if start is pd.NaT or end is pd.NaT or pd.Timestamp(end) <= pd.Timestamp(start):
        raise ValueError("equity/benchmark series do not overlap")

    eq = daily_equity[(daily_equity.index >= start) & (daily_equity.index <= end)].copy()
    bc = bench.close[(bench.close.index >= start) & (bench.close.index <= end)].copy()

    combined_index = eq.index.intersection(bc.index)
    eq = eq.loc[combined_index]
    bc = bc.loc[combined_index]

    if len(eq) < 2 or len(bc) < 2:
        raise ValueError("too few overlapping daily points to plot")

    eq0 = float(eq.iloc[0])
    bc0 = float(bc.iloc[0])
    if eq0 <= 0 or bc0 <= 0:
        raise ValueError("invalid start value for normalization")

    df = pd.DataFrame(
        {
            "strategy_equity": eq.astype(float),
            "strategy_norm": (eq.astype(float) / eq0),
            "benchmark_symbol": str(bench.symbol),
            "benchmark_close": bc.astype(float),
            "benchmark_norm": (bc.astype(float) / bc0),
        },
        index=combined_index,
    )
    df.index.name = "date"

    out_csv = out_csv or (run_dir / f"equity_vs_{bench_symbol}.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv)

    if out_png is None:
        out_png = run_dir / f"equity_vs_{bench_symbol}.png"
    png_written: Optional[Path] = None

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        logger.info("Plot skipped (matplotlib unavailable): %s", exc)
        return out_csv, None

    try:
        fig = plt.figure(figsize=(12, 6), dpi=130)
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(df.index, df["strategy_norm"], label="Strategy (normalized)", linewidth=2.0)
        ax.plot(df.index, df["benchmark_norm"], label=f"{bench.symbol.upper()} (normalized)", linewidth=2.0)
        ax.set_title(f"Equity vs {bench.symbol.upper()} (daily, normalized)")
        ax.set_xlabel("Date (UTC)")
        ax.set_ylabel("Normalized value (start=1.0)")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")
        fig.tight_layout()
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png)
        png_written = out_png
    finally:
        try:
            plt.close("all")
        except Exception:
            pass

    return out_csv, png_written

