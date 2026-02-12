from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from atlas.strategies.base import Strategy, StrategyDecision, StrategyState
from atlas.utils.time import NY_TZ


def _to_ny(ts: pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    if ts.tz is None:
        return ts.tz_localize(NY_TZ)
    return ts.tz_convert(NY_TZ)


@dataclass
class CryptoMeta(Strategy):
    """
    Meta strategy for crypto spot: switches between two sub-strategies based on a simple regime signal.

    This is research scaffolding only; it does not guarantee profit.
    """

    name: str = "crypto_meta"

    # Universe
    symbols: tuple[str, ...] = ("BTC/USD", "ETH/USD")
    market_symbol: str = "BTC/USD"

    # Regime signal: log-momentum + efficiency ratio (ER) of market_symbol over N bars.
    regime_mom_bars: int = 168  # ~42d @ 6H bars
    regime_mom_threshold: float = 0.0
    regime_er_bars: int = 84  # ~21d @ 6H bars
    regime_er_min: float = 0.25

    # If the regime is "trending", blend toward rotation; otherwise blend toward ensemble.
    # 1.0 means "all rotation"; 0.0 means "all ensemble".
    rotation_weight_trending: float = 1.0
    rotation_weight_ranging: float = 0.0

    # Child presets (used by registry/TUI wiring; not consumed directly by this class).
    ensemble_params_file: str = "strategy_params/crypto_ensemble_ultra_6h_coinbase_heartbeat.json"
    rotation_params_file: str = "strategy_params/crypto_rotation_2022_candidate_r2_momfilter_v11_6h_coinbase_nohb.json"
    ensemble_symbols: str = "BTC/USD,ETH/USD"
    rotation_symbols: str = ""

    # Sub-strategies (injected by registry)
    ensemble: Strategy = field(default=None, repr=False)  # type: ignore[assignment]
    rotation: Strategy = field(default=None, repr=False)  # type: ignore[assignment]

    def warmup_bars(self) -> int:
        base = int(max(0, int(self.regime_mom_bars))) + 2
        a = int(self.ensemble.warmup_bars()) if self.ensemble is not None else 0
        b = int(self.rotation.warmup_bars()) if self.rotation is not None else 0
        return int(max(base, a, b))

    def _market_mom(self, bars_by_symbol: dict[str, pd.DataFrame]) -> float:
        mkt = (self.market_symbol or "").strip().upper()
        if not mkt:
            mkt = next(iter(bars_by_symbol.keys()), "")
        if mkt not in bars_by_symbol:
            mkt = next(iter(bars_by_symbol.keys()), "")
        if not mkt:
            return 0.0

        df = bars_by_symbol.get(mkt)
        if df is None or df.empty or "close" not in df.columns:
            return 0.0

        n = int(max(1, int(self.regime_mom_bars)))
        if len(df) <= n:
            return 0.0

        try:
            close = float(df["close"].iloc[-1])
            base = float(df["close"].iloc[-n - 1])
        except Exception:
            return 0.0
        if close <= 0 or base <= 0:
            return 0.0
        return float(math.log(close / base))

    def _market_er(self, bars_by_symbol: dict[str, pd.DataFrame]) -> float:
        mkt = (self.market_symbol or "").strip().upper()
        if not mkt:
            mkt = next(iter(bars_by_symbol.keys()), "")
        if mkt not in bars_by_symbol:
            mkt = next(iter(bars_by_symbol.keys()), "")
        if not mkt:
            return 0.0

        df = bars_by_symbol.get(mkt)
        if df is None or df.empty or "close" not in df.columns:
            return 0.0

        n = int(max(1, int(self.regime_er_bars)))
        if len(df) <= n:
            return 0.0

        closes = df["close"].iloc[-n - 1 :].astype(float)
        if len(closes) <= 1:
            return 0.0

        net = float(abs(closes.iloc[-1] - closes.iloc[0]))
        path = float(closes.diff().abs().iloc[1:].sum())
        if path <= 0 or not math.isfinite(path):
            return 0.0
        er = net / path
        if not math.isfinite(er):
            return 0.0
        return float(max(0.0, min(1.0, er)))

    def target_exposures(
        self, bars_by_symbol: dict[str, pd.DataFrame], state: StrategyState
    ) -> StrategyDecision:
        universe = sorted(bars_by_symbol)
        targets = {s: 0.0 for s in universe}

        mom = self._market_mom(bars_by_symbol)
        er = self._market_er(bars_by_symbol)

        trending = (float(mom) > float(self.regime_mom_threshold)) and (float(er) >= float(self.regime_er_min))

        w_rot = float(self.rotation_weight_trending if trending else self.rotation_weight_ranging)
        w_rot = max(0.0, min(1.0, w_rot))
        w_ens = 1.0 - w_rot

        rot = self.rotation
        ens = self.ensemble
        if rot is None and ens is None:
            return StrategyDecision(target_exposures=targets, reason="meta_missing_children")

        # Always advance child state (these strategies are stateful).
        rot_decision = (
            rot.target_exposures(bars_by_symbol, state)
            if rot is not None
            else StrategyDecision(target_exposures={s: 0.0 for s in universe}, reason="meta_rot_missing")
        )
        ens_decision = (
            ens.target_exposures(bars_by_symbol, state)
            if ens is not None
            else StrategyDecision(target_exposures={s: 0.0 for s in universe}, reason="meta_ens_missing")
        )

        for s in targets:
            targets[s] = w_rot * float((rot_decision.target_exposures or {}).get(s, 0.0)) + w_ens * float(
                (ens_decision.target_exposures or {}).get(s, 0.0)
            )

        debug = {
            "meta_ts": _to_ny(pd.Timestamp(state.timestamp)).isoformat(),
            "meta_market_symbol": str(self.market_symbol),
            "meta_regime_mom": float(mom),
            "meta_regime_mom_bars": int(self.regime_mom_bars),
            "meta_regime_mom_threshold": float(self.regime_mom_threshold),
            "meta_regime_er": float(er),
            "meta_regime_er_bars": int(self.regime_er_bars),
            "meta_regime_er_min": float(self.regime_er_min),
            "meta_trending": bool(trending),
            "meta_w_rotation": float(w_rot),
            "meta_w_ensemble": float(w_ens),
            "meta_rotation_reason": str(rot_decision.reason),
            "meta_ensemble_reason": str(ens_decision.reason),
        }
        if rot_decision.debug:
            debug["rotation"] = dict(rot_decision.debug)
        if ens_decision.debug:
            debug["ensemble"] = dict(ens_decision.debug)

        return StrategyDecision(
            target_exposures=targets,
            reason=f"meta_blend(trending={trending},w_rot={w_rot:.2f})",
            debug=debug,
            execution_hints=(rot_decision.execution_hints or ens_decision.execution_hints),
        )
