from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd


@dataclass(frozen=True)
class StrategyState:
    timestamp: pd.Timestamp
    allow_short: bool
    cash: float
    positions: dict[str, float]
    equity: float
    day_start_equity: float
    day_pnl: float
    day_return: float
    holding_bars: dict[str, int]
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StrategyDecision:
    target_exposures: dict[str, float]
    reason: Optional[str] = None
    debug: Optional[dict[str, Any]] = None


class Strategy(ABC):
    name: str

    @abstractmethod
    def warmup_bars(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def target_exposures(
        self, bars_by_symbol: dict[str, pd.DataFrame], state: StrategyState
    ) -> StrategyDecision:
        raise NotImplementedError
