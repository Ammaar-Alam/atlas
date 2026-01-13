from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from atlas.strategies.base import Strategy, StrategyDecision


@dataclass(frozen=True)
class EmaCrossover(Strategy):
    name: str = "ema_crossover"
    fast_window: int = 10
    slow_window: int = 30
    symbol: str = "SPY"

    def warmup_bars(self) -> int:
        return max(self.fast_window, self.slow_window) + 1

    def target_exposures(
        self, bars_by_symbol: dict[str, pd.DataFrame], state
    ) -> StrategyDecision:
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
        fast = close.ewm(span=self.fast_window, adjust=False).mean().iloc[-1]
        slow = close.ewm(span=self.slow_window, adjust=False).mean().iloc[-1]

        if fast > slow:
            exposures[symbol] = 1.0
            return StrategyDecision(
                target_exposures=exposures, reason="fast_ema_above_slow_ema"
            )
        return StrategyDecision(
            target_exposures=exposures, reason="fast_ema_below_slow_ema"
        )
