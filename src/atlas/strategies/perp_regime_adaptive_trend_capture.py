from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd

from atlas.strategies.base import Strategy, StrategyDecision, StrategyState


def _clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))


def _sign(x: float, *, eps: float = 1e-9) -> int:
    if x > eps:
        return 1
    if x < -eps:
        return -1
    return 0


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=max(2, int(span)), adjust=False).mean()


def _to_utc(ts: pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    if ts.tz is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _utc_week_key(ts: pd.Timestamp) -> tuple[int, int]:
    ts_utc = _to_utc(ts)
    iso = ts_utc.isocalendar()
    return int(iso.year), int(iso.week)


@dataclass
class PerpRegimeAdaptiveTrendCapture(Strategy):
    """
    Regime-Adaptive Trend Capture (RATC):
    - Long-biased state machine designed to minimize churn under high microstructure costs.
    - Event-driven transitions with hysteresis/cooldown.
    - Contract-quantized sizing for derivatives lot realism.
    """

    name: str = "perp_regime_adaptive_trend_capture"
    symbols: tuple[str, ...] = ("BTC-PERP",)

    # Regime classifier horizons (1H bars)
    mom_horizon_a: int = 168
    mom_horizon_b: int = 504
    mom_horizon_c: int = 1008
    ema_fast_regime: int = 72
    ema_slow_regime: int = 504

    # Regime transition thresholds (bps)
    bear_exit_bps: float = 120.0
    short_entry_bps: float = 300.0

    # Hysteresis
    cooldown_bars: int = 168

    # Sizing
    long_base_exposure: float = 0.55
    short_base_exposure: float = 0.35
    extreme_vol_scale: float = 0.40
    high_vol_scale: float = 0.70
    extreme_vol_rank: float = 0.85
    high_vol_rank: float = 0.75
    vol_lookback_bars: int = 120
    vol_regime_window: int = 720

    # Crash override
    crash_threshold_bps: float = 350.0

    # Position lifecycle
    max_hold_bars: int = 2016
    rebalance_exposure_threshold: float = 0.02

    # Risk controls
    daily_loss_limit: float = 0.05
    weekly_loss_limit: float = 0.07
    kill_switch: float = 0.25

    # Internal state
    _current_state: str = field(default="long", init=False, repr=False)
    _bars_since_transition: int = field(default=9_999, init=False, repr=False)
    _bars_in_position: int = field(default=0, init=False, repr=False)
    _bars_seen: int = field(default=0, init=False, repr=False)
    _peak_equity: float = field(default=0.0, init=False, repr=False)
    _week_key: Optional[tuple[int, int]] = field(default=None, init=False, repr=False)
    _week_start_equity: float = field(default=0.0, init=False, repr=False)
    _risk_disabled_day: Optional[object] = field(default=None, init=False, repr=False)
    _risk_disabled_forever: bool = field(default=False, init=False, repr=False)
    _last_targets: dict[str, float] = field(default_factory=dict, init=False, repr=False)

    def warmup_bars(self) -> int:
        return int(
            max(
                int(self.mom_horizon_a),
                int(self.mom_horizon_b),
                int(self.mom_horizon_c),
                int(self.ema_slow_regime),
                int(self.vol_regime_window),
            )
            + 10
        )

    def _risk_off(self, symbols: list[str], *, reason: str, debug: dict[str, Any]) -> StrategyDecision:
        targets = {s: 0.0 for s in symbols}
        self._last_targets = dict(targets)
        return StrategyDecision(target_exposures=targets, reason=reason, debug=debug)

    def _safe_mom_bps(self, close: pd.Series, lookback: int) -> float:
        n = len(close)
        lb = int(max(2, lookback))
        if n <= lb + 1:
            return 0.0
        c_now = float(close.iloc[-1])
        c_prev = float(close.iloc[-lb - 1])
        if c_now <= 0.0 or c_prev <= 0.0:
            return 0.0
        return float(math.log(c_now / c_prev) * 10_000.0)

    def _classify_regime(self, close: pd.Series) -> dict[str, Any]:
        n = len(close)
        out: dict[str, Any] = {
            "bull_votes": 0,
            "bear_votes": 3,
            "mom_a": 0.0,
            "mom_b": 0.0,
            "mom_c": 0.0,
            "ema_trend": 0,
            "valid": False,
        }
        max_h = int(
            max(
                int(self.mom_horizon_a),
                int(self.mom_horizon_b),
                int(self.mom_horizon_c),
                int(self.ema_slow_regime),
            )
        )
        if n < max_h + 5:
            return out

        mom_a = self._safe_mom_bps(close, int(self.mom_horizon_a))
        mom_b = self._safe_mom_bps(close, int(self.mom_horizon_b))
        mom_c = self._safe_mom_bps(close, int(self.mom_horizon_c))
        bull_votes = int(mom_a > 0.0) + int(mom_b > 0.0) + int(mom_c > 0.0)

        ema_f = float(_ema(close, int(self.ema_fast_regime)).iloc[-1])
        ema_s = float(_ema(close, int(self.ema_slow_regime)).iloc[-1])
        ema_trend = _sign(ema_f - ema_s)

        out.update(
            {
                "bull_votes": int(bull_votes),
                "bear_votes": int(3 - bull_votes),
                "mom_a": float(mom_a),
                "mom_b": float(mom_b),
                "mom_c": float(mom_c),
                "ema_trend": int(ema_trend),
                "valid": True,
            }
        )
        return out

    def _vol_rank(self, close: pd.Series) -> Optional[float]:
        rets = np.log(close / close.shift(1)).dropna()
        if len(rets) < int(max(self.vol_lookback_bars, self.vol_regime_window, 20)):
            return None
        vol_now = float(rets.tail(int(self.vol_lookback_bars)).std())
        if not np.isfinite(vol_now) or vol_now <= 0.0:
            return None
        vol_hist = rets.rolling(int(self.vol_lookback_bars)).std().dropna()
        vol_tail = vol_hist.tail(int(self.vol_regime_window))
        if len(vol_tail) < 10:
            return None
        return float((vol_tail < vol_now).mean())

    def target_exposures(
        self, bars_by_symbol: dict[str, pd.DataFrame], state: StrategyState
    ) -> StrategyDecision:
        symbols = [s for s in self.symbols if s in bars_by_symbol]
        if not symbols:
            symbols = sorted(bars_by_symbol.keys())
        if not symbols:
            return StrategyDecision(target_exposures={}, reason="no_symbols")

        self._bars_seen += 1
        self._bars_since_transition += 1

        ts = pd.Timestamp(state.timestamp)
        ts_utc = _to_utc(ts)
        today = ts_utc.date()
        debug: dict[str, Any] = {
            "bars_seen": int(self._bars_seen),
            "state": str(self._current_state),
        }

        if self._risk_disabled_day is not None and self._risk_disabled_day != today:
            self._risk_disabled_day = None

        equity = float(state.equity)
        if self._peak_equity <= 0.0:
            self._peak_equity = equity
        self._peak_equity = max(self._peak_equity, equity)
        drawdown = float(equity / self._peak_equity - 1.0) if self._peak_equity > 0 else 0.0

        wk = _utc_week_key(ts_utc)
        if self._week_key is None or self._week_key != wk:
            self._week_key = wk
            self._week_start_equity = equity
        week_ret = float(equity / self._week_start_equity - 1.0) if self._week_start_equity > 0 else 0.0

        debug["drawdown"] = float(drawdown)
        debug["day_return"] = float(state.day_return)
        debug["week_return"] = float(week_ret)

        if self._risk_disabled_forever:
            return self._risk_off(symbols, reason="kill_switch", debug=debug)
        if drawdown <= -abs(float(self.kill_switch)):
            self._risk_disabled_forever = True
            return self._risk_off(symbols, reason="kill_switch", debug=debug)
        if float(state.day_return) <= -abs(float(self.daily_loss_limit)):
            self._risk_disabled_day = today
            return self._risk_off(symbols, reason="daily_loss_limit", debug=debug)
        if self._risk_disabled_day == today:
            return self._risk_off(symbols, reason="risk_disabled_day", debug=debug)
        if week_ret <= -abs(float(self.weekly_loss_limit)):
            return self._risk_off(symbols, reason="weekly_loss_limit", debug=debug)

        s = symbols[0]
        df = bars_by_symbol.get(s)
        if df is None or df.empty:
            return self._risk_off(symbols, reason="no_data", debug=debug)
        if not df.index.is_monotonic_increasing:
            df = df.sort_index()
        close = pd.to_numeric(df.get("close"), errors="coerce").dropna()
        if len(close) < self.warmup_bars():
            return self._risk_off(symbols, reason="warmup", debug=debug)

        regime = self._classify_regime(close)
        debug["regime"] = regime
        if not bool(regime.get("valid", False)):
            return self._risk_off(symbols, reason="regime_invalid", debug=debug)

        prev_state = str(self._current_state)
        new_state = prev_state
        can_transition = bool(self._bars_since_transition >= int(max(0, self.cooldown_bars)))

        bear_votes = int(regime["bear_votes"])
        bull_votes = int(regime["bull_votes"])
        ema_trend = int(regime["ema_trend"])
        mom_a = float(regime["mom_a"])
        mom_b = float(regime["mom_b"])

        if can_transition:
            if prev_state == "long":
                if bear_votes >= 3 and ema_trend < 0 and mom_b < -abs(float(self.bear_exit_bps)):
                    new_state = "flat"
            elif prev_state == "flat":
                if bull_votes >= 2 and ema_trend > 0:
                    new_state = "long"
                elif bear_votes >= 3 and mom_b < -abs(float(self.short_entry_bps)) and state.allow_short:
                    new_state = "short"
            elif prev_state == "short":
                if bear_votes <= 1 or ema_trend >= 0:
                    new_state = "flat"

        # Crash override can bypass cooldown.
        if prev_state == "long" and mom_a < -abs(float(self.crash_threshold_bps)):
            new_state = "flat"
        if prev_state == "short" and mom_a > abs(float(self.crash_threshold_bps)):
            new_state = "flat"

        if new_state != "flat" and int(self._bars_in_position) >= int(max(1, self.max_hold_bars)):
            new_state = "flat"

        transitioned = bool(new_state != prev_state)
        if transitioned:
            self._current_state = str(new_state)
            self._bars_since_transition = 0
            if new_state == "flat":
                self._bars_in_position = 0
            debug["transition"] = f"{prev_state}->{new_state}"

        if self._current_state != "flat":
            self._bars_in_position += 1

        extra = dict(state.extra or {})
        max_notional = float(extra.get("max_position_notional_usd", 0.0) or 0.0)
        if max_notional <= 0.0:
            max_notional = 1.0

        px = float(close.iloc[-1])
        contract_size = float(extra.get("contract_size_units", 0.01) or 0.01)
        if contract_size <= 0.0:
            contract_size = 0.01
        contract_notional = float(px * contract_size)

        if self._current_state == "long":
            base_exp = float(max(0.0, self.long_base_exposure))
            side = 1
        elif self._current_state == "short" and state.allow_short:
            base_exp = float(max(0.0, self.short_base_exposure))
            side = -1
        else:
            base_exp = 0.0
            side = 0

        vol_rank = self._vol_rank(close)
        if vol_rank is not None and base_exp > 0.0:
            if float(vol_rank) > float(self.extreme_vol_rank):
                base_exp *= float(self.extreme_vol_scale)
            elif float(vol_rank) > float(self.high_vol_rank):
                base_exp *= float(self.high_vol_scale)
        debug["vol_rank"] = float(vol_rank) if vol_rank is not None else None

        target_exp = 0.0
        n_contracts = 0
        if side != 0 and contract_notional > 0.0 and px > 0.0:
            target_notional = float(max(0.0, base_exp) * float(max_notional))
            n_contracts = int(math.floor(target_notional / contract_notional + 1e-12))
            if n_contracts >= 1:
                actual_notional = float(n_contracts * contract_notional)
                target_exp = float(_clamp(actual_notional / float(max_notional), 0.0, 1.0)) * float(side)

        current_q_raw = float(state.positions.get(s, 0.0) or 0.0)
        if contract_size > 0.0:
            q_steps = math.floor((abs(current_q_raw) + 1e-12) / contract_size)
            current_q = float(math.copysign(q_steps * contract_size, current_q_raw))
        else:
            current_q = float(current_q_raw)
        prev_exp = float((current_q * px) / max_notional) if px > 0.0 else 0.0
        # Event-driven behavior:
        # - On non-transition bars, keep current exposure to avoid churn from price drift.
        # - Only force rebalance if transitioning states, opening a new position, or flattening.
        if not transitioned:
            if self._current_state == "flat":
                target_exp = 0.0
            # Keep current sized position between transitions to avoid churn from
            # contract quantization and volatility bucket toggles.
            elif (
                self._current_state in {"long", "short"}
                and abs(prev_exp) > 1e-9
                and _sign(prev_exp) == side
            ):
                # Execution happens on next-bar open while this decision is based on the
                # current bar close. Add half-lot cushion so lot-floor quantization doesn't
                # repeatedly drop one-lot holdings to zero when open/close differ slightly.
                hold_qty = float(current_q + math.copysign(0.51 * contract_size, current_q))
                target_exp = float((hold_qty * px) / max_notional) if px > 0.0 else float(prev_exp)
            elif abs(float(target_exp) - float(prev_exp)) < float(self.rebalance_exposure_threshold):
                target_exp = float(prev_exp)

        targets = {sym: 0.0 for sym in symbols}
        targets[s] = float(target_exp)

        self._last_targets = dict(targets)
        debug["target_exposure"] = float(targets[s])
        debug["n_contracts"] = int(abs(n_contracts))
        debug["bars_since_transition"] = int(self._bars_since_transition)
        debug["bars_in_position"] = int(self._bars_in_position)

        return StrategyDecision(target_exposures=targets, reason=f"state={self._current_state}", debug=debug)
