from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from atlas.strategies.base import Strategy, StrategyDecision


@dataclass(frozen=True)
class NoTrade(Strategy):
    name: str = "no_trade"

    def warmup_bars(self) -> int:
        return 1

    def target_exposures(
        self, bars_by_symbol: dict[str, pd.DataFrame], state
    ) -> StrategyDecision:
        exposures = {s: 0.0 for s in bars_by_symbol}
        return StrategyDecision(target_exposures=exposures, reason="no_trade")
