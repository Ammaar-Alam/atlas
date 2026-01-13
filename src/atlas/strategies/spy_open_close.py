from __future__ import annotations

from dataclasses import dataclass
from datetime import time

import pandas as pd

from atlas.strategies.base import Strategy, StrategyDecision
from atlas.utils.time import NY_TZ


@dataclass(frozen=True)
class SpyOpenClose(Strategy):
    name: str = "spy_open_close"
    symbol: str = "SPY"
    start_time: time = time(9, 30)
    end_time: time = time(16, 0)

    def warmup_bars(self) -> int:
        return 1

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

        ts = pd.Timestamp(state.timestamp)
        if ts.tzinfo is None:
            ts = ts.tz_localize(NY_TZ)
        else:
            ts = ts.tz_convert(NY_TZ)

        in_session = self.start_time <= ts.time() < self.end_time
        if in_session:
            exposures[symbol] = 1.0
            reason = "open_close_long"
        else:
            reason = "outside_session"

        return StrategyDecision(target_exposures=exposures, reason=reason)
