from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from atlas.data.benchmarks import load_stooq_daily_ohlcv


@dataclass(frozen=True)
class WindowSummary:
    window: str
    step: str
    windows: int
    windows_with_trades: int
    trade_window_frac: float
    mean_return: float
    median_return: float
    p05_return: float
    p95_return: float
    best_return: float
    worst_return: float
    benchmark: Optional[str] = None
    beat_benchmark_frac: Optional[float] = None
    mean_benchmark_return: Optional[float] = None
    median_benchmark_return: Optional[float] = None
    mean_alpha: Optional[float] = None
    median_alpha: Optional[float] = None
    p05_alpha: Optional[float] = None
    p95_alpha: Optional[float] = None
    best_alpha: Optional[float] = None
    worst_alpha: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "window": self.window,
            "step": self.step,
            "windows": int(self.windows),
            "windows_with_trades": int(self.windows_with_trades),
            "trade_window_frac": float(self.trade_window_frac),
            "mean_return": float(self.mean_return),
            "median_return": float(self.median_return),
            "p05_return": float(self.p05_return),
            "p95_return": float(self.p95_return),
            "best_return": float(self.best_return),
            "worst_return": float(self.worst_return),
        }
        if self.benchmark is not None:
            payload["benchmark"] = str(self.benchmark)
        if self.beat_benchmark_frac is not None:
            payload["beat_benchmark_frac"] = float(self.beat_benchmark_frac)
        if self.mean_benchmark_return is not None:
            payload["mean_benchmark_return"] = float(self.mean_benchmark_return)
        if self.median_benchmark_return is not None:
            payload["median_benchmark_return"] = float(self.median_benchmark_return)
        if self.mean_alpha is not None:
            payload["mean_alpha"] = float(self.mean_alpha)
        if self.median_alpha is not None:
            payload["median_alpha"] = float(self.median_alpha)
        if self.p05_alpha is not None:
            payload["p05_alpha"] = float(self.p05_alpha)
        if self.p95_alpha is not None:
            payload["p95_alpha"] = float(self.p95_alpha)
        if self.best_alpha is not None:
            payload["best_alpha"] = float(self.best_alpha)
        if self.worst_alpha is not None:
            payload["worst_alpha"] = float(self.worst_alpha)
        return payload


def _load_equity_curve(run_dir: Path) -> pd.Series:
    path = run_dir / "equity_curve.csv"
    if not path.exists():
        raise FileNotFoundError(f"missing equity curve: {path}")
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df = df.dropna(subset=["timestamp"]).set_index("timestamp").sort_index()
    if "equity" not in df.columns:
        raise ValueError(f"equity_curve.csv missing 'equity' column: {path}")
    return df["equity"].astype(float)


def _load_trades_timestamps(run_dir: Path) -> pd.DatetimeIndex:
    path = run_dir / "trades.csv"
    if not path.exists():
        return pd.DatetimeIndex([], tz="UTC")
    df = pd.read_csv(path)
    if "timestamp" not in df.columns:
        return pd.DatetimeIndex([], tz="UTC")
    ts = pd.to_datetime(df["timestamp"], errors="coerce", utc=True).dropna()
    return pd.DatetimeIndex(ts).sort_values()


def _to_utc_midnight(ts: pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    if ts.tz is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.normalize()


def _load_stooq_close_series(*, stooq_symbol: str) -> pd.Series:
    raw = load_stooq_daily_ohlcv(stooq_symbol=stooq_symbol)
    if raw.empty:
        return pd.Series(dtype=float)

    date_col = "Date" if "Date" in raw.columns else "date"
    close_col = "Close" if "Close" in raw.columns else "close"
    if date_col not in raw.columns or close_col not in raw.columns:
        return pd.Series(dtype=float)

    df = raw[[date_col, close_col]].copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce", utc=True)
    df[close_col] = pd.to_numeric(df[close_col], errors="coerce")
    df = df.dropna(subset=[date_col, close_col])
    if df.empty:
        return pd.Series(dtype=float)

    df = df.sort_values(date_col).set_index(date_col)
    close = df[close_col].astype(float)
    close.index = close.index.map(_to_utc_midnight)
    close = close[~close.index.duplicated(keep="last")]
    return close.sort_index()


def rolling_window_summary(
    *,
    run_dir: Path,
    window: timedelta,
    step: timedelta,
    min_windows: int = 3,
    benchmark: Optional[str] = "spy.us",
) -> tuple[WindowSummary, list[dict[str, Any]]]:
    """
    Compute rolling fixed-length window returns and trade counts from a completed run directory.

    - Windows are aligned on UTC midnight boundaries, but returns are computed using the
      full-resolution equity curve (first equity at/after window start, last equity before window end).
    - Each window is [start, start+window), advanced by `step`.
    """
    equity = _load_equity_curve(run_dir).copy()
    trades_ts = _load_trades_timestamps(run_dir)

    equity = equity.dropna()
    equity = equity[~equity.index.duplicated(keep="last")].sort_index()
    if equity.empty:
        raise ValueError("no equity points found")

    idx = pd.DatetimeIndex(equity.index).sort_values()
    values = equity.to_numpy(dtype=float, copy=False)
    start_ts = _to_utc_midnight(idx.min())
    end_ts = _to_utc_midnight(idx.max())

    bench_symbol: Optional[str] = None
    bench_close = pd.Series(dtype=float)
    if benchmark:
        try:
            bench_symbol = str(benchmark)
            bench_close = _load_stooq_close_series(stooq_symbol=bench_symbol)
        except Exception:
            bench_symbol = None
            bench_close = pd.Series(dtype=float)

    rows: list[dict[str, Any]] = []
    cur = start_ts
    while cur + window <= end_ts + timedelta(days=1):
        win_start = cur
        win_end = cur + window

        # Equity at start/end using full-resolution equity curve.
        i0 = int(idx.searchsorted(win_start, side="left"))
        i1 = int(idx.searchsorted(win_end, side="left")) - 1
        if i0 < 0 or i0 >= len(values) or i1 < i0:
            cur += step
            continue
        start_eq = float(values[i0])
        end_eq = float(values[i1])
        ret = (end_eq / start_eq - 1.0) if start_eq > 0 else 0.0

        trades_in_window = int(((trades_ts >= win_start) & (trades_ts < win_end)).sum())
        row: dict[str, Any] = {
            "start": win_start.isoformat(),
            "end": win_end.isoformat(),
            "return": float(ret),
            "trades": int(trades_in_window),
        }
        if bench_symbol is not None and not bench_close.empty:
            b_start = bench_close.index[bench_close.index >= win_start].min()
            b_end = bench_close.index[bench_close.index < win_end].max()
            if (
                b_start is not None
                and b_end is not None
                and pd.Timestamp(b_end) > pd.Timestamp(b_start)
            ):
                b_ret = float((bench_close.loc[b_end] / bench_close.loc[b_start]) - 1.0)
                row["benchmark"] = bench_symbol
                row["benchmark_return"] = float(b_ret)
                row["alpha"] = float(ret - b_ret)
        rows.append(row)

        cur += step

    if len(rows) < int(min_windows):
        raise ValueError(f"too few windows computed: {len(rows)} (min {min_windows})")

    returns = np.array([float(r["return"]) for r in rows], dtype=float)
    trades = np.array([int(r["trades"]) for r in rows], dtype=float)

    windows_with_trades = int((trades > 0).sum())

    bench_rets: Optional[np.ndarray] = None
    alphas: Optional[np.ndarray] = None
    if any("benchmark_return" in r for r in rows):
        bench_rets = np.array(
            [float(r.get("benchmark_return", 0.0) or 0.0) for r in rows], dtype=float
        )
        alphas = np.array([float(r.get("alpha", 0.0) or 0.0) for r in rows], dtype=float)

    beat_benchmark_frac: Optional[float] = None
    mean_bench: Optional[float] = None
    median_bench: Optional[float] = None
    mean_alpha: Optional[float] = None
    median_alpha: Optional[float] = None
    p05_alpha: Optional[float] = None
    p95_alpha: Optional[float] = None
    best_alpha: Optional[float] = None
    worst_alpha: Optional[float] = None
    if bench_rets is not None and alphas is not None:
        beat_benchmark_frac = float(np.mean(alphas > 0.0))
        mean_bench = float(np.mean(bench_rets))
        median_bench = float(np.median(bench_rets))
        mean_alpha = float(np.mean(alphas))
        median_alpha = float(np.median(alphas))
        p05_alpha = float(np.quantile(alphas, 0.05))
        p95_alpha = float(np.quantile(alphas, 0.95))
        best_alpha = float(np.max(alphas))
        worst_alpha = float(np.min(alphas))
    summary = WindowSummary(
        window=str(window),
        step=str(step),
        windows=int(len(rows)),
        windows_with_trades=int(windows_with_trades),
        trade_window_frac=float(windows_with_trades / max(1, len(rows))),
        mean_return=float(np.mean(returns)),
        median_return=float(np.median(returns)),
        p05_return=float(np.quantile(returns, 0.05)),
        p95_return=float(np.quantile(returns, 0.95)),
        best_return=float(np.max(returns)),
        worst_return=float(np.min(returns)),
        benchmark=bench_symbol,
        beat_benchmark_frac=beat_benchmark_frac,
        mean_benchmark_return=mean_bench,
        median_benchmark_return=median_bench,
        mean_alpha=mean_alpha,
        median_alpha=median_alpha,
        p05_alpha=p05_alpha,
        p95_alpha=p95_alpha,
        best_alpha=best_alpha,
        worst_alpha=worst_alpha,
    )
    return summary, rows


def write_window_analysis_json(
    *,
    run_dir: Path,
    window: timedelta,
    step: timedelta,
    out_path: Optional[Path] = None,
    benchmark: Optional[str] = "spy.us",
) -> Path:
    summary, rows = rolling_window_summary(
        run_dir=run_dir, window=window, step=step, benchmark=benchmark
    )
    payload = {"summary": summary.to_dict(), "windows": rows}
    out_path = out_path or (run_dir / "window_analysis.json")
    out_path.write_text(json.dumps(payload, indent=2, default=str))
    return out_path
