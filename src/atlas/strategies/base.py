from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass(frozen=True)
class StrategyDecision:
    target_exposure: float
    reason: Optional[str] = None


class Strategy(ABC):
    name: str

    @abstractmethod
    def warmup_bars(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def target_exposure(self, bars: pd.DataFrame) -> StrategyDecision:
        raise NotImplementedError

