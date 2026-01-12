from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BacktestMetrics:
    total_return: float
    max_drawdown: float
    sharpe: float
    trades: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_return": self.total_return,
            "max_drawdown": self.max_drawdown,
            "sharpe": self.sharpe,
            "trades": self.trades,
        }


def _infer_bar_minutes(index: pd.DatetimeIndex) -> float:
    if len(index) < 3:
        return 1.0
    diffs = index.to_series().diff().dropna().dt.total_seconds() / 60.0
    median = float(diffs.median())
    return median if median > 0 else 1.0


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
        bar_minutes = _infer_bar_minutes(equity_curve.index)
        periods_per_year = (252.0 * 390.0) / bar_minutes
        sharpe = float((rets.mean() / rets.std()) * np.sqrt(periods_per_year))

    return BacktestMetrics(
        total_return=total_return,
        max_drawdown=max_drawdown,
        sharpe=sharpe,
        trades=int(len(trades)),
    )

