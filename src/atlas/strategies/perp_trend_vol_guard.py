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


def _atr(df: pd.DataFrame, window: int) -> Optional[float]:
    if df is None or df.empty:
        return None
    close = pd.to_numeric(df.get("close"), errors="coerce")
    high = pd.to_numeric(df.get("high"), errors="coerce").fillna(close)
    low = pd.to_numeric(df.get("low"), errors="coerce").fillna(close)
    tmp = pd.DataFrame({"close": close, "high": high, "low": low}).dropna(subset=["close"])
    if tmp.empty:
        return None
    prev_close = tmp["close"].shift(1)
    tr = pd.concat(
        [
            (tmp["high"] - tmp["low"]).abs(),
            (tmp["high"] - prev_close).abs(),
            (tmp["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = float(tr.tail(max(2, int(window))).mean())
    if not np.isfinite(atr) or atr <= 0:
        return None
    return float(atr)


def _utc_week_key(ts: pd.Timestamp) -> tuple[int, int]:
    ts = pd.Timestamp(ts)
    if ts.tz is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    iso = ts.isocalendar()
    return int(iso.year), int(iso.week)


@dataclass
class PerpTrendVolGuard(Strategy):
    """
    Derivatives trend strategy with volatility and cost guards.

    Highlights:
    - EMA trend + momentum alignment.
    - Breakout confirmation to reduce churn.
    - Edge admission vs estimated round-trip costs.
    - ATR risk sizing with gross/per-symbol caps.
    - Weekly loss lock and daily kill controls.
    """

    name: str = "perp_trend_vol_guard"
    symbols: tuple[str, ...] = ("BTC-PERP", "ETH-PERP")

    # Signal parameters
    ema_fast: int = 18
    ema_slow: int = 72
    momentum_window_bars: int = 24
    breakout_window: int = 20
    breakout_buffer_bps: float = 2.0
    atr_window: int = 20
    trend_strength_min: float = 0.18
    min_atr_bps: float = 4.0

    # Cost-aware edge gating
    edge_floor_bps: float = 6.0
    k_cost: float = 1.8
    slippage_bps: float = 1.25
    taker_fee_bps: float = 3.0

    # Sizing
    risk_budget: float = 0.010
    stop_atr_mult: float = 2.2
    target_vol_bps_per_bar: float = 80.0
    max_positions: int = 2
    max_gross_exposure: float = 0.80
    max_per_symbol_exposure: float = 0.45
    rebalance_interval_bars: int = 2
    rebalance_exposure_threshold: float = 0.03
    min_trade_notional_usd: float = 20.0
    min_hold_bars: int = 6
    flip_confirm_bars: int = 2

    # Volatility regime guard (using first symbol as market proxy)
    market_vol_reduce_bps: float = 100.0
    market_vol_off_bps: float = 160.0

    # Portfolio protections
    weekly_loss_limit: float = 0.03
    enable_weekly_profit_lock: bool = True
    weekly_profit_target: float = 0.02
    weekly_lock_risk_scale: float = 0.35
    daily_loss_limit: float = 0.02
    kill_switch: float = 0.12

    # Internal state
    _bars_seen: int = field(default=0, init=False, repr=False)
    _last_rebalance_bar: int = field(default=0, init=False, repr=False)
    _last_targets: dict[str, float] = field(default_factory=dict, init=False, repr=False)
    _peak_equity: float = field(default=0.0, init=False, repr=False)
    _risk_disabled_day: Optional[object] = field(default=None, init=False, repr=False)
    _risk_disabled_forever: bool = field(default=False, init=False, repr=False)
    _week_key: Optional[tuple[int, int]] = field(default=None, init=False, repr=False)
    _week_start_equity: float = field(default=0.0, init=False, repr=False)
    _week_locked: bool = field(default=False, init=False, repr=False)
    _flip_counter: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _last_desired_side: dict[str, int] = field(default_factory=dict, init=False, repr=False)

    def warmup_bars(self) -> int:
        return int(
            max(
                int(self.ema_slow) + 3,
                int(self.momentum_window_bars) + 3,
                int(self.breakout_window) + 3,
                int(self.atr_window) + 3,
            )
            + 4
        )

    def _risk_off(
        self, symbols: list[str], *, reason: str, debug: dict[str, Any]
    ) -> StrategyDecision:
        targets = {s: 0.0 for s in symbols}
        self._last_targets = dict(targets)
        return StrategyDecision(target_exposures=targets, reason=reason, debug=debug)

    def _cost_round_trip_bps(self, state: StrategyState) -> float:
        extra = dict(state.extra or {})
        slip = float(extra.get("slippage_bps", self.slippage_bps) or self.slippage_bps)
        fee = float(extra.get("taker_fee_bps", self.taker_fee_bps) or self.taker_fee_bps)
        return float(2.0 * (abs(slip) + abs(fee)))

    def _symbol_signal(
        self,
        *,
        df: pd.DataFrame,
        cost_rt_bps: float,
    ) -> Optional[dict[str, float]]:
        if df is None or len(df) < self.warmup_bars():
            return None
        if not df.index.is_monotonic_increasing:
            df = df.sort_index()

        close = pd.to_numeric(df.get("close"), errors="coerce").dropna()
        if len(close) < self.warmup_bars():
            return None

        c = float(close.iloc[-1])
        if c <= 0:
            return None
        atr = _atr(df, int(self.atr_window))
        if atr is None or atr <= 0:
            return None
        atr_bps = float((atr / c) * 10_000.0)
        if atr_bps < float(self.min_atr_bps):
            return None

        ema_fast = float(_ema(close, int(self.ema_fast)).iloc[-1])
        ema_slow = float(_ema(close, int(self.ema_slow)).iloc[-1])
        trend_strength = float((ema_fast - ema_slow) / atr)
        side = _sign(trend_strength)
        if side == 0 or abs(trend_strength) < float(self.trend_strength_min):
            return None

        mom_base = float(close.iloc[-int(self.momentum_window_bars) - 1])
        if mom_base <= 0:
            return None
        mom = float(math.log(c / mom_base))
        if side > 0 and mom <= 0:
            return None
        if side < 0 and mom >= 0:
            return None

        w = int(max(2, int(self.breakout_window)))
        if len(close) < w + 2:
            return None
        highs = pd.to_numeric(df.get("high"), errors="coerce").fillna(close)
        lows = pd.to_numeric(df.get("low"), errors="coerce").fillna(close)
        hh = float(highs.iloc[-w - 1 : -1].max())
        ll = float(lows.iloc[-w - 1 : -1].min())
        buf = float(self.breakout_buffer_bps) / 10_000.0
        if side > 0 and c < hh * (1.0 + buf):
            return None
        if side < 0 and c > ll * (1.0 - buf):
            return None

        signal_bps = float(abs(trend_strength) * atr_bps)
        required_bps = float(self.edge_floor_bps) + float(self.k_cost) * float(cost_rt_bps)
        if signal_bps < required_bps:
            return None

        return {
            "close": float(c),
            "atr_bps": float(atr_bps),
            "trend_strength": float(trend_strength),
            "momentum": float(mom),
            "side": float(side),
            "signal_bps": float(signal_bps),
            "score": float(abs(trend_strength) * max(abs(mom), 1e-6)),
        }

    def target_exposures(
        self, bars_by_symbol: dict[str, pd.DataFrame], state: StrategyState
    ) -> StrategyDecision:
        symbols = [s for s in self.symbols if s in bars_by_symbol]
        if not symbols:
            symbols = sorted(bars_by_symbol.keys())
        if not symbols:
            return StrategyDecision(target_exposures={}, reason="no_symbols")

        self._bars_seen += 1
        ts = pd.Timestamp(state.timestamp)
        today = ts.date()

        if self._risk_disabled_day is not None and self._risk_disabled_day != today:
            self._risk_disabled_day = None

        equity = float(state.equity)
        if self._peak_equity <= 0:
            self._peak_equity = equity
        self._peak_equity = max(self._peak_equity, equity)
        drawdown = float(equity / self._peak_equity - 1.0) if self._peak_equity > 0 else 0.0

        wk = _utc_week_key(ts)
        if self._week_key is None or self._week_key != wk:
            self._week_key = wk
            self._week_start_equity = equity
            self._week_locked = False
        week_ret = (
            float(equity / self._week_start_equity - 1.0)
            if self._week_start_equity > 0
            else 0.0
        )
        if week_ret <= -abs(float(self.weekly_loss_limit)):
            self._week_locked = True
        if bool(self.enable_weekly_profit_lock) and week_ret >= abs(
            float(self.weekly_profit_target)
        ):
            self._week_locked = True

        debug: dict[str, Any] = {
            "bars_seen": int(self._bars_seen),
            "drawdown": float(drawdown),
            "day_return": float(state.day_return),
            "week_return": float(week_ret),
            "week_locked": bool(self._week_locked),
        }

        if self._risk_disabled_forever or drawdown <= -abs(float(self.kill_switch)):
            self._risk_disabled_forever = True
            return self._risk_off(symbols, reason="kill_switch", debug=debug)
        if float(state.day_return) <= -abs(float(self.daily_loss_limit)):
            self._risk_disabled_day = today
            return self._risk_off(symbols, reason="daily_loss_limit", debug=debug)
        if self._risk_disabled_day == today:
            return self._risk_off(symbols, reason="risk_disabled_day", debug=debug)

        risk_scale = 1.0
        if self._week_locked:
            risk_scale = float(self.weekly_lock_risk_scale) if bool(self.enable_weekly_profit_lock) else 0.0
        risk_scale = _clamp(risk_scale, 0.0, 1.0)
        if risk_scale <= 0.0:
            return self._risk_off(symbols, reason="week_locked", debug=debug)

        due = bool(
            (not self._last_targets)
            or (int(self._bars_seen) - int(self._last_rebalance_bar) >= int(max(1, self.rebalance_interval_bars)))
        )
        if not due:
            return StrategyDecision(
                target_exposures={s: float(self._last_targets.get(s, 0.0)) for s in symbols},
                reason="hold",
                debug=debug,
            )

        max_notional = float(state.extra.get("max_position_notional_usd", 0.0) or 0.0)
        if max_notional <= 0:
            return self._risk_off(symbols, reason="no_max_notional", debug=debug)

        cost_rt_bps = self._cost_round_trip_bps(state)
        candidates: dict[str, dict[str, float]] = {}
        for s in symbols:
            sig = self._symbol_signal(df=bars_by_symbol.get(s), cost_rt_bps=cost_rt_bps)
            if sig is not None:
                candidates[s] = sig
        if not candidates:
            return self._risk_off(symbols, reason="no_candidates", debug=debug)

        # Vol regime guard using first symbol.
        ms = symbols[0]
        m_atr_bps = float(candidates.get(ms, {}).get("atr_bps", 0.0))
        if m_atr_bps >= float(self.market_vol_off_bps):
            return self._risk_off(symbols, reason="market_vol_off", debug=debug)
        if m_atr_bps >= float(self.market_vol_reduce_bps):
            risk_scale *= 0.5

        scored = sorted(candidates.items(), key=lambda kv: float(kv[1]["score"]), reverse=True)
        selected = scored[: max(1, int(self.max_positions))]
        denom = float(sum(max(0.0, float(v["score"])) for _, v in selected))
        if denom <= 1e-12:
            return self._risk_off(symbols, reason="bad_scores", debug=debug)

        max_gross = float(self.max_gross_exposure) * float(risk_scale)
        max_gross = _clamp(max_gross, 0.0, float(self.max_gross_exposure))
        per_cap = float(self.max_per_symbol_exposure)
        targets = {s: 0.0 for s in symbols}

        for s, feat in selected:
            score_w = max(0.0, float(feat["score"])) / denom
            atr_bps = max(1e-6, float(feat["atr_bps"]))
            # ATR stop-risk sizing plus volatility target.
            risk_sizing = float(self.risk_budget) / max(
                1e-6, float(self.stop_atr_mult) * (atr_bps / 10_000.0)
            )
            vol_sizing = float(self.target_vol_bps_per_bar) / atr_bps
            raw = float(score_w) * min(float(risk_sizing), float(vol_sizing))
            exp_abs = _clamp(raw, 0.0, per_cap)
            side = _sign(float(feat["side"]))
            targets[s] = float(side * exp_abs)

        gross = float(sum(abs(v) for v in targets.values()))
        if gross > max_gross and gross > 1e-12:
            scale = float(max_gross / gross)
            for s in list(targets.keys()):
                targets[s] = float(targets[s] * scale)

        min_exp = float(self.min_trade_notional_usd) / float(max_notional)
        for s in list(targets.keys()):
            if abs(float(targets[s])) < min_exp:
                targets[s] = 0.0

        # Flip/hysteresis control and rebalance threshold.
        for s in symbols:
            cur_qty = float(state.positions.get(s, 0.0) or 0.0)
            cur_side = _sign(cur_qty)
            cur_exp = 0.0
            df = bars_by_symbol.get(s)
            if df is not None and not df.empty:
                close = pd.to_numeric(df.get("close"), errors="coerce").dropna()
                if not close.empty:
                    c = float(close.iloc[-1])
                    if c > 0:
                        cur_exp = float((cur_qty * c) / max_notional)

            desired = float(targets.get(s, 0.0))
            desired_side = _sign(desired)
            hold_bars = int(state.holding_bars.get(s, 0) or 0)

            if cur_side != 0 and desired_side != 0 and cur_side != desired_side:
                prev_side = int(self._last_desired_side.get(s, 0))
                if prev_side == desired_side:
                    self._flip_counter[s] = int(self._flip_counter.get(s, 0)) + 1
                else:
                    self._flip_counter[s] = 1
                self._last_desired_side[s] = desired_side

                if hold_bars < int(self.min_hold_bars) or int(self._flip_counter[s]) < int(
                    self.flip_confirm_bars
                ):
                    targets[s] = cur_exp
                    continue
            else:
                self._flip_counter[s] = 0
                self._last_desired_side[s] = desired_side

            if abs(float(desired) - float(cur_exp)) < float(self.rebalance_exposure_threshold):
                targets[s] = float(cur_exp)

        self._last_targets = {s: float(targets.get(s, 0.0)) for s in symbols}
        self._last_rebalance_bar = int(self._bars_seen)
        return StrategyDecision(target_exposures=self._last_targets, reason="rebalance", debug=debug)
