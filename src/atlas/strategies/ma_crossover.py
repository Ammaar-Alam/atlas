from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from atlas.strategies.base import Strategy, StrategyDecision


@dataclass(frozen=True)
class MovingAverageCrossover(Strategy):
    name: str = "ma_crossover"
    fast_window: int = 10
    slow_window: int = 30

    def warmup_bars(self) -> int:
        return max(self.fast_window, self.slow_window) + 1

    def target_exposure(self, bars: pd.DataFrame) -> StrategyDecision:
        if len(bars) < self.warmup_bars():
            return StrategyDecision(target_exposure=0.0, reason="warmup")

        close = bars["close"]
        fast = close.rolling(self.fast_window).mean().iloc[-1]
        slow = close.rolling(self.slow_window).mean().iloc[-1]

        if fast > slow:
            return StrategyDecision(target_exposure=1.0, reason="fast_ma_above_slow_ma")
        return StrategyDecision(target_exposure=0.0, reason="fast_ma_below_slow_ma")

