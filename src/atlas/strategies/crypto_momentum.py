from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd

from atlas.strategies.base import Strategy, StrategyDecision, StrategyState


def _clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))


def _safe_float(x: Any, *, default: float = 0.0) -> float:
    try:
        v = float(x)
    except Exception:
        return float(default)
    return float(v) if np.isfinite(v) else float(default)


def _is_spot_crypto_symbol(symbol: str) -> bool:
    s = (symbol or "").strip().upper()
    if not s:
        return False
    if s.endswith("-PERP") or s.endswith("-CDE"):
        return False
    # Atlas generally normalizes crypto to BTC/USD, but tolerate BTC-USD inputs.
    return ("/" in s) or ("-" in s)


@dataclass
class CryptoMomentum(Strategy):
    """
    Crypto spot momentum + hold strategy (research; NOT financial advice).

    Design goals:
    - Low turnover (fees-aware): hold a basket when the market regime is favorable, otherwise go to cash.
    - Robustness: relies primarily on close-to-close momentum (avoids overusing fragile intrabar wicks/volume).
    - Activity constraint: optional tiny "heartbeat" trades can ensure the system isn't idle for long stretches.

    Regime:
    - Compute market momentum over `momentum_window_bars` on `market_symbol`.
    - If momentum <= 0 → risk off (cash).
    - Else → risk on: allocate `max_total_exposure` across the basket.
    """

    name: str = "crypto_momentum"

    # ---- Universe ----
    symbols: tuple[str, ...] = ("BTC/USD", "ETH/USD")
    market_symbol: Optional[str] = "BTC/USD"

    # ---- Signal ----
    momentum_window_bars: int = 240  # ~60d on 6H bars

    # ---- Portfolio ----
    max_total_exposure: float = 1.0
    max_exposure_per_symbol: float = 1.0
    rebalance_interval_bars: int = 28  # rebalance weekly on 6H bars
    rebalance_exposure_threshold: float = 0.10
    min_trade_notional_usd: float = 25.0

    # ---- Optional: enforce at least one trade periodically (tiny "heartbeat") ----
    heartbeat_every_bars: int = 0
    heartbeat_notional_usd: float = 1.0

    # ---- Internal state ----
    _bars_seen: int = field(default=0, init=False, repr=False)
    _last_signal_on: Optional[bool] = field(default=None, init=False, repr=False)
    _last_rebalance_bar: int = field(default=0, init=False, repr=False)
    _last_trade_intent_bar: int = field(default=0, init=False, repr=False)
    _last_heartbeat_bar: int = field(default=0, init=False, repr=False)
    _heartbeat_symbol: Optional[str] = field(default=None, init=False, repr=False)
    _heartbeat_offset_exp: float = field(default=0.0, init=False, repr=False)
    _heartbeat_clear_bar: int = field(default=0, init=False, repr=False)
    _last_targets: dict[str, float] = field(default_factory=dict, init=False, repr=False)

    def warmup_bars(self) -> int:
        return int(max(8, int(self.momentum_window_bars) + 3))

    def _universe(self, bars_by_symbol: dict[str, pd.DataFrame]) -> list[str]:
        raw = getattr(self, "symbols", ())
        if isinstance(raw, str):
            parts = [s.strip() for s in raw.split(",")]
        else:
            parts = [str(s).strip() for s in raw]

        out: list[str] = []
        seen: set[str] = set()
        for sym in parts:
            s = sym.upper().replace(" ", "")
            if not s or s in seen:
                continue
            if not _is_spot_crypto_symbol(s):
                continue
            if s not in bars_by_symbol:
                continue
            seen.add(s)
            out.append(s)
        return out

    def _market_symbol(self, universe: list[str]) -> Optional[str]:
        ms = (self.market_symbol or "").strip().upper().replace(" ", "")
        if ms and ms in universe:
            return ms
        return universe[0] if universe else None

    def _momentum_signal(self, df: pd.DataFrame) -> Optional[bool]:
        if df is None or df.empty:
            return None
        if "close" not in df.columns:
            return None
        closes = df["close"].astype(float)
        w = int(self.momentum_window_bars)
        if w <= 1 or len(closes) < w + 2:
            return None
        last = float(closes.iloc[-1])
        prev = float(closes.iloc[-(w + 1)])
        if not np.isfinite(last) or not np.isfinite(prev) or prev <= 0:
            return None
        mom = float(last / prev - 1.0)
        return bool(mom > 0.0)

    def target_exposures(
        self, bars_by_symbol: dict[str, pd.DataFrame], state: StrategyState
    ) -> StrategyDecision:
        self._bars_seen += 1
        universe = self._universe(bars_by_symbol)
        if not universe:
            return StrategyDecision(target_exposures={}, reason="no_universe")

        max_notional = float(state.extra.get("max_position_notional_usd", 0.0) or 0.0)

        ms = self._market_symbol(universe)
        if ms is None:
            return StrategyDecision(target_exposures={s: 0.0 for s in universe}, reason="no_market_symbol")

        mkt_df = bars_by_symbol.get(ms)
        signal_on = self._momentum_signal(mkt_df)
        if signal_on is None:
            # Not enough history yet.
            return StrategyDecision(target_exposures={s: 0.0 for s in universe}, reason="warmup")

        # Decide whether to rebalance this bar.
        should_rebalance = False
        if self._last_signal_on is None or bool(signal_on) != bool(self._last_signal_on):
            should_rebalance = True
        elif int(self.rebalance_interval_bars) > 0:
            if int(self._bars_seen) - int(self._last_rebalance_bar) >= int(self.rebalance_interval_bars):
                should_rebalance = True

        if not should_rebalance:
            # Optional heartbeat (tiny, periodic) to satisfy “at least one trade per week”.
            hb_every = int(self.heartbeat_every_bars)
            hb_notional = float(self.heartbeat_notional_usd)
            if hb_every > 0 and max_notional > 0:
                if int(self._bars_seen) - int(self._last_trade_intent_bar) >= hb_every:
                    if int(self._bars_seen) - int(self._last_heartbeat_bar) >= hb_every:
                        self._last_heartbeat_bar = int(self._bars_seen)
                        self._heartbeat_symbol = ms
                        base = float(self._last_targets.get(ms, 0.0))
                        delta_exp = float(hb_notional) / float(max_notional)
                        delta_exp = float(_clamp(delta_exp, 0.0, 0.25))
                        self._heartbeat_offset_exp = float(delta_exp if base <= 0.0 else -delta_exp)
                        self._heartbeat_clear_bar = int(self._bars_seen) + 1

                    targets = {s: float(self._last_targets.get(s, 0.0)) for s in universe}
                    if int(self._bars_seen) == int(self._heartbeat_clear_bar):
                        self._heartbeat_offset_exp = 0.0
                    hb_sym = self._heartbeat_symbol
                    hb_off = float(self._heartbeat_offset_exp)
                    if hb_sym and hb_sym in universe and abs(hb_off) > 1e-12:
                        base = float(targets.get(hb_sym, 0.0))
                        new_exp = float(_clamp(base + hb_off, 0.0, float(self.max_exposure_per_symbol)))
                        if abs(new_exp - base) > 1e-12:
                            targets[hb_sym] = float(new_exp)
                    prev = dict(self._last_targets)
                    self._last_targets = {s: float(targets.get(s, 0.0)) for s in universe}
                    if any(
                        abs(float(self._last_targets.get(s, 0.0)) - float(prev.get(s, 0.0))) > 1e-8
                        for s in universe
                    ):
                        self._last_trade_intent_bar = int(self._bars_seen)
                    return StrategyDecision(
                        target_exposures=self._last_targets,
                        reason="heartbeat",
                    )
            return StrategyDecision(
                target_exposures={s: float(self._last_targets.get(s, 0.0)) for s in universe},
                reason="hold",
            )

        self._last_signal_on = bool(signal_on)
        self._last_rebalance_bar = int(self._bars_seen)

        if not bool(signal_on):
            targets = {s: 0.0 for s in universe}
        else:
            # Equal-weight basket by default.
            n = max(1, len(universe))
            total = float(_clamp(float(self.max_total_exposure), 0.0, 10.0))
            per = float(total) / float(n)
            targets = {
                s: float(_clamp(per, 0.0, float(self.max_exposure_per_symbol))) for s in universe
            }

        # Rebalance threshold: avoid tiny trades when already holding.
        for s in universe:
            pos_qty = float(state.positions.get(s, 0.0) or 0.0)
            if abs(pos_qty) <= 1e-12:
                continue
            df = bars_by_symbol.get(s)
            if df is None or df.empty:
                continue
            last_close = _safe_float(df["close"].iloc[-1], default=0.0)
            if last_close <= 0 or max_notional <= 0:
                continue
            cur_exp = (pos_qty * float(last_close)) / float(max_notional)
            tgt = float(targets.get(s, 0.0))
            if abs(tgt - float(cur_exp)) < float(self.rebalance_exposure_threshold):
                targets[s] = float(cur_exp)

        # Ensure minimum notional for any non-zero target (or drop it to 0).
        for s in universe:
            exp = float(targets.get(s, 0.0))
            if exp <= 0 or max_notional <= 0:
                continue
            notional = exp * max_notional
            if notional < float(self.min_trade_notional_usd):
                targets[s] = 0.0

        prev = dict(self._last_targets)
        self._last_targets = {s: float(targets.get(s, 0.0)) for s in universe}
        if any(abs(float(self._last_targets.get(s, 0.0)) - float(prev.get(s, 0.0))) > 1e-8 for s in universe):
            self._last_trade_intent_bar = int(self._bars_seen)

        return StrategyDecision(target_exposures=self._last_targets, reason="rebalance")

