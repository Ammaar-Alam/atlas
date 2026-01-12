from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from atlas.strategies.base import Strategy, StrategyDecision


@dataclass(frozen=True)
class MovingAverageCrossover(Strategy):
    name: str = "ma_crossover"
    fast_window: int = 10
    slow_window: int = 30
    symbol: str = "SPY"

    def warmup_bars(self) -> int:
        return max(self.fast_window, self.slow_window) + 1

    def target_exposures(self, bars_by_symbol: dict[str, pd.DataFrame], state) -> StrategyDecision:
        symbols = sorted(bars_by_symbol)
        exposures = {s: 0.0 for s in symbols}
        if not symbols:
            return StrategyDecision(target_exposures=exposures, reason="no_symbols")

        symbol = self.symbol.upper()
        if symbol not in bars_by_symbol:
            symbol = symbols[0]

        bars = bars_by_symbol[symbol]
        if len(bars) < self.warmup_bars():
            return StrategyDecision(target_exposures=exposures, reason="warmup")

        close = bars["close"]
        fast = close.rolling(self.fast_window).mean().iloc[-1]
        slow = close.rolling(self.slow_window).mean().iloc[-1]

        if fast > slow:
            exposures[symbol] = 1.0
            return StrategyDecision(
                target_exposures=exposures, reason="fast_ma_above_slow_ma"
            )
        return StrategyDecision(target_exposures=exposures, reason="fast_ma_below_slow_ma")
