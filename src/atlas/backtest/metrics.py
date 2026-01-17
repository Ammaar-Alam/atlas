from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import pandas as pd

from atlas.utils.time import NY_TZ

@dataclass(frozen=True)
class BacktestMetrics:
    total_return: float
    max_drawdown: float
    sharpe: float
    sharpe_daily: float
    trades: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_return": self.total_return,
            "max_drawdown": self.max_drawdown,
            "sharpe": self.sharpe,
            "sharpe_daily": self.sharpe_daily,
            "trades": self.trades,
        }


def _infer_bar_minutes(index: pd.DatetimeIndex) -> float:
    if len(index) < 3:
        return 1.0
    diffs = index.to_series().diff().dropna().dt.total_seconds() / 60.0
    median = float(diffs.median())
    return median if median > 0 else 1.0


def _infer_periods_per_year(index: pd.DatetimeIndex) -> float:
    if not isinstance(index, pd.DatetimeIndex):
        raise ValueError("equity_curve index must be a DatetimeIndex")
    bar_minutes = _infer_bar_minutes(index)
    has_weekend_bars = bool((index.dayofweek >= 5).any())
    if has_weekend_bars:
        return (365.0 * 1440.0) / bar_minutes
    return (252.0 * 390.0) / bar_minutes


def _daily_eod_equity(equity_curve: pd.DataFrame) -> pd.Series:
    if "equity" not in equity_curve.columns:
        return pd.Series(dtype=float)
    if not isinstance(equity_curve.index, pd.DatetimeIndex):
        raise ValueError("equity_curve index must be a DatetimeIndex")

    eq = equity_curve[["equity"]].copy()
    idx = eq.index
    if idx.tz is None:
        idx = idx.tz_localize(NY_TZ)
    else:
        idx = idx.tz_convert(NY_TZ)
    eq.index = idx
    eq = eq.sort_index()

    eq = eq[eq.index.dayofweek < 5]
    if eq.empty:
        return pd.Series(dtype=float)

    try:
        session = eq.between_time("09:30", "16:00", include_start=True, include_end=True)
    except TypeError:
        session = eq.between_time("09:30", "16:00")

    if session.empty:
        daily = eq.resample("1D").last()
    else:
        daily = session.resample("1D").last()

    daily = daily.dropna(subset=["equity"])
    return daily["equity"].astype(float)


def _daily_sharpe(equity_curve: pd.DataFrame) -> float:
    daily_eq = _daily_eod_equity(equity_curve)
    daily_rets = daily_eq.pct_change().dropna()
    if len(daily_rets) < 2 or float(daily_rets.std()) == 0.0:
        return 0.0
    return float((daily_rets.mean() / daily_rets.std()) * np.sqrt(252.0))


def compute_metrics(equity_curve: pd.DataFrame, trades: pd.DataFrame) -> BacktestMetrics:
    eq = equity_curve["equity"].astype(float)
    total_return = float(eq.iloc[-1] / eq.iloc[0] - 1.0)

    running_max = eq.cummax()
    drawdown = (eq / running_max) - 1.0
    max_drawdown = float(drawdown.min())

    rets = eq.pct_change().dropna()
    if len(rets) < 2 or float(rets.std()) == 0.0:
        sharpe = 0.0
    else:
        periods_per_year = _infer_periods_per_year(equity_curve.index)
        sharpe = float((rets.mean() / rets.std()) * np.sqrt(periods_per_year))

    sharpe_daily = _daily_sharpe(equity_curve)

    return BacktestMetrics(
        total_return=total_return,
        max_drawdown=max_drawdown,
        sharpe=sharpe,
        sharpe_daily=sharpe_daily,
        trades=int(len(trades)),
    )
