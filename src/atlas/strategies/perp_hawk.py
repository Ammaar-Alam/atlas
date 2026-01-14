from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd

from atlas.strategies.base import Strategy, StrategyDecision, StrategyState


def _sign(x: float, *, eps: float = 1e-9) -> int:
    if x > eps:
        return 1
    if x < -eps:
        return -1
    return 0


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=int(span), adjust=False).mean()


def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr1 = (high - low).abs()
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int) -> pd.Series:
    tr = _true_range(high, low, close)
    return tr.rolling(int(window)).mean()


def _efficiency_ratio(close: pd.Series, window: int) -> pd.Series:
    window = int(window)
    if window <= 1:
        return pd.Series(np.nan, index=close.index)
    direction = (close - close.shift(window)).abs()
    volatility = close.diff().abs().rolling(window).sum()
    return direction / volatility.replace(0.0, np.nan)


def _donchian(high: pd.Series, low: pd.Series, window: int) -> tuple[float, float]:
    """Highest high / lowest low over the previous `window` bars, excluding current bar."""

    window = int(window)
    if window <= 0 or len(high) < window + 1:
        return float("nan"), float("nan")
    hh = float(high.iloc[-window - 1 : -1].max())
    ll = float(low.iloc[-window - 1 : -1].min())
    return hh, ll


@dataclass
class PerpHawk(Strategy):
    """Volatility-targeted, regime-filtered trend strategy for perpetual futures.

    Design goals:
    - Low turnover (trend + breakout alignment)
    - Conservative risk sizing (ATR stop-distance) + portfolio leverage/margin caps
    - Optional funding-aware entry/hold filters when `funding_rate` data exists
    - Hard survival constraints (daily loss limit, global drawdown kill-switch)
    """

    name: str = "perp_hawk"

    # --- signal / regime ---
    atr_window: int = 14
    ema_fast: int = 20
    ema_slow: int = 60
    er_window: int = 20
    breakout_window: int = 20
    breakout_buffer_bps: float = 2.0
    er_min: float = 0.30
    trend_z_min: float = 0.25
    min_atr_bps: float = 5.0
    allow_trend_entry_without_breakout: bool = True

    # --- risk / sizing ---
    risk_budget: float = 0.010  # fraction of equity risked (worst-case to stop) across active positions
    stop_atr_mult: float = 2.2
    trail_atr_mult: float = 3.2
    max_positions: int = 2
    rebalance_exposure_threshold: float = 0.05

    # Portfolio-level caps (strategy-enforced; engine itself allows extreme leverage)
    max_leverage: float = 3.0
    max_margin_utilization: float = 0.35

    # --- funding filters (optional, bps/day) ---
    funding_entry_bps_per_day: float = 25.0
    funding_exit_bps_per_day: float = 60.0

    # --- risk-off controls ---
    daily_loss_limit: float = 0.02
    kill_switch: float = 0.10

    # --- execution hygiene ---
    min_hold_bars: int = 3
    flip_confirm_bars: int = 3
    cooldown_bars: int = 5

    # --- internal state ---
    _bars_seen: int = field(default=0, init=False, repr=False)
    _peak_equity: float = field(default=0.0, init=False, repr=False)
    _risk_disabled_day: Optional[object] = field(default=None, init=False, repr=False)
    _risk_disabled_forever: bool = field(default=False, init=False, repr=False)

    _cooldown_until: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _flip_counter: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _trail_extreme: dict[str, float] = field(default_factory=dict, init=False, repr=False)
    _last_pos_side: dict[str, int] = field(default_factory=dict, init=False, repr=False)

    def warmup_bars(self) -> int:
        return max(self.atr_window, self.ema_slow, self.er_window, self.breakout_window) + 2

    def _maybe_reset_daily_state(self, state: StrategyState) -> None:
        today = state.timestamp.date()
        if self._risk_disabled_day is not None and self._risk_disabled_day != today:
            self._risk_disabled_day = None

    def _risk_off(self, symbols: list[str], *, reason: str, debug: dict[str, Any]) -> StrategyDecision:
        return StrategyDecision(target_exposures={s: 0.0 for s in symbols}, reason=reason, debug=debug)

    def target_exposures(self, bars_by_symbol: dict[str, pd.DataFrame], state: StrategyState) -> StrategyDecision:
        symbols = sorted(s for s in bars_by_symbol.keys())
        exposures: dict[str, float] = {s: 0.0 for s in symbols}
        if not symbols:
            return StrategyDecision(target_exposures=exposures, reason="no_symbols")

        self._bars_seen += 1
        self._maybe_reset_daily_state(state)

        # Global kill-switch
        if self._peak_equity <= 0:
            self._peak_equity = float(state.equity)
        self._peak_equity = max(self._peak_equity, float(state.equity))
        drawdown = (float(state.equity) / self._peak_equity - 1.0) if self._peak_equity > 0 else 0.0
        if self._risk_disabled_forever or drawdown <= -abs(float(self.kill_switch)):
            self._risk_disabled_forever = True
            return self._risk_off(
                symbols,
                reason="kill_switch",
                debug={"drawdown": drawdown, "kill_switch": self.kill_switch},
            )

        # Daily loss limit
        if state.day_return <= -abs(float(self.daily_loss_limit)):
            self._risk_disabled_day = state.timestamp.date()
            return self._risk_off(
                symbols,
                reason="daily_loss_limit",
                debug={"day_return": state.day_return, "limit": self.daily_loss_limit},
            )

        if self._risk_disabled_day == state.timestamp.date():
            return self._risk_off(symbols, reason="risk_disabled_day", debug={})

        max_notional = float(state.extra.get("max_position_notional_usd", 0.0) or 0.0)
        if max_notional <= 0:
            max_notional = 1.0

        mmr = float(state.extra.get("maintenance_margin_rate", 0.05) or 0.05)
        funding_rates = state.extra.get("funding_rates", {}) or {}

        # --- build per-symbol signals ---
        per: dict[str, dict[str, Any]] = {}
        for s in symbols:
            bars = bars_by_symbol[s]
            if len(bars) < self.warmup_bars():
                continue

            close = bars["close"].astype(float)
            high = bars["high"].astype(float)
            low = bars["low"].astype(float)

            last_close = float(close.iloc[-1])
            if not np.isfinite(last_close) or last_close <= 0:
                continue

            atr_series = _atr(high, low, close, self.atr_window)
            atr = float(atr_series.iloc[-1]) if len(atr_series) else float("nan")
            if not np.isfinite(atr) or atr <= 0:
                continue

            er_series = _efficiency_ratio(close, self.er_window)
            er = float(er_series.iloc[-1]) if len(er_series) else float("nan")

            ema_f = float(_ema(close, self.ema_fast).iloc[-1])
            ema_s = float(_ema(close, self.ema_slow).iloc[-1])
            trend_strength = (ema_f - ema_s) / atr
            trend_dir = 1 if trend_strength > self.trend_z_min else -1 if trend_strength < -self.trend_z_min else 0

            hh, ll = _donchian(high, low, self.breakout_window)
            buf = float(self.breakout_buffer_bps) / 10_000.0
            breakout_dir = 0
            if np.isfinite(hh) and np.isfinite(ll):
                if last_close > hh * (1.0 + buf):
                    breakout_dir = 1
                elif last_close < ll * (1.0 - buf):
                    breakout_dir = -1

            atr_bps = (atr / last_close) * 10_000.0

            pos_qty = float(state.positions.get(s, 0.0) or 0.0)
            pos_side = _sign(pos_qty)
            hold_bars = int(state.holding_bars.get(s, 0) or 0)

            prev_side = int(self._last_pos_side.get(s, 0))
            if prev_side == 0 and pos_side != 0:
                # New position detected (filled at prior open). Initialize trailing anchor.
                self._trail_extreme[s] = last_close
                self._flip_counter[s] = 0
            if prev_side != 0 and pos_side == 0:
                # Position closed; start cooldown.
                self._cooldown_until[s] = max(self._cooldown_until.get(s, 0), self._bars_seen + int(self.cooldown_bars))
                self._flip_counter[s] = 0
                self._trail_extreme.pop(s, None)

            fr = float(funding_rates.get(s, 0.0) or 0.0)
            funding_bps_per_day = fr * 10_000.0 * 24.0
            pays_funding_bps = float(pos_side) * funding_bps_per_day

            desired_dir = 0
            reason = "flat"

            # --- in-position management ---
            if pos_side != 0:
                # Update trailing extreme
                prev_ext = float(self._trail_extreme.get(s, last_close))
                if pos_side > 0:
                    self._trail_extreme[s] = max(prev_ext, last_close)
                    trail_stop = self._trail_extreme[s] - float(self.trail_atr_mult) * atr
                    stop_hit = (last_close <= trail_stop)
                else:
                    self._trail_extreme[s] = min(prev_ext, last_close)
                    trail_stop = self._trail_extreme[s] + float(self.trail_atr_mult) * atr
                    stop_hit = (last_close >= trail_stop)

                # Exit if funding becomes extremely punitive.
                if pays_funding_bps > float(self.funding_exit_bps_per_day) and hold_bars >= self.min_hold_bars:
                    desired_dir = 0
                    reason = "funding_exit"
                elif stop_hit and hold_bars >= self.min_hold_bars:
                    desired_dir = 0
                    reason = "trail_stop"
                else:
                    # Flip only with confirmation to reduce chop.
                    opposite = (trend_dir == -pos_side) and (er >= self.er_min)
                    if opposite:
                        self._flip_counter[s] = int(self._flip_counter.get(s, 0)) + 1
                    else:
                        self._flip_counter[s] = 0

                    if self._flip_counter[s] >= int(self.flip_confirm_bars) and hold_bars >= self.min_hold_bars:
                        desired_dir = -pos_side
                        reason = "trend_flip"
                    else:
                        desired_dir = pos_side
                        reason = "hold"

            # --- entry logic ---
            else:
                if self._bars_seen < int(self._cooldown_until.get(s, 0)):
                    desired_dir = 0
                    reason = "cooldown"
                else:
                    if (er >= self.er_min) and (atr_bps >= self.min_atr_bps):
                        entry_dir = 0
                        if breakout_dir != 0 and breakout_dir == trend_dir and trend_dir != 0:
                            entry_dir = breakout_dir
                            reason = "breakout_entry"
                        elif self.allow_trend_entry_without_breakout and trend_dir != 0 and abs(trend_strength) >= 2.0 * self.trend_z_min:
                            entry_dir = trend_dir
                            reason = "trend_entry"

                        # Funding entry filter: avoid entering positions that are expected to pay outsized funding.
                        if entry_dir != 0 and (float(entry_dir) * funding_bps_per_day) > float(self.funding_entry_bps_per_day):
                            entry_dir = 0
                            reason = "funding_block"

                        desired_dir = entry_dir

            score = float(abs(trend_strength) * (er if np.isfinite(er) else 0.0))
            per[s] = {
                "desired_dir": int(desired_dir),
                "score": score,
                "atr": atr,
                "atr_bps": atr_bps,
                "er": er,
                "trend_strength": trend_strength,
                "trend_dir": int(trend_dir),
                "breakout_dir": int(breakout_dir),
                "pos_side": int(pos_side),
                "hold_bars": hold_bars,
                "funding_bps_per_day": funding_bps_per_day,
                "reason": reason,
                "last_close": last_close,
            }

            self._last_pos_side[s] = int(pos_side)

        if not per:
            return StrategyDecision(target_exposures=exposures, reason="warmup")

        # Active set and ranking
        active = [s for s, d in per.items() if int(d["desired_dir"]) != 0]
        if int(self.max_positions) > 0 and len(active) > int(self.max_positions):
            active_sorted = sorted(active, key=lambda s: float(per[s]["score"]), reverse=True)
            keep = set(active_sorted[: int(self.max_positions)])
            for s in active:
                if s not in keep:
                    per[s]["desired_dir"] = 0
                    per[s]["reason"] = "rank_cut"

        active = [s for s, d in per.items() if int(d["desired_dir"]) != 0]
        if not active:
            return StrategyDecision(target_exposures=exposures, reason="no_signal", debug={"per": per})

        equity = float(state.equity)
        if equity <= 0:
            return self._risk_off(symbols, reason="no_equity", debug={"equity": equity})

        # Risk budgets per symbol (proportional to score)
        scores = np.array([max(0.0, float(per[s]["score"])) for s in active], dtype=float)
        if float(scores.sum()) <= 0:
            weights = np.ones_like(scores) / max(1.0, float(len(scores)))
        else:
            weights = scores / scores.sum()

        total_risk_usd = max(0.0, float(self.risk_budget)) * equity
        targets: dict[str, float] = {}

        total_abs_notional = 0.0
        for s, w in zip(active, weights):
            d = per[s]
            side = int(d["desired_dir"])
            atr = float(d["atr"])
            px = float(d["last_close"])
            stop_dist = max(1e-9, float(self.stop_atr_mult) * atr)

            risk_usd = float(total_risk_usd) * float(w)
            qty = risk_usd / stop_dist
            notional = qty * px

            exposure = float(side) * (notional / max_notional)
            # Hard per-symbol cap (engine interprets exposure as fraction of max_notional)
            exposure = float(np.clip(exposure, -1.0, 1.0))
            # Use the *effective* capped notional for portfolio risk/margin aggregates.
            total_abs_notional += abs(exposure) * max_notional
            targets[s] = exposure

        # Portfolio scaling for leverage and margin utilization
        leverage = total_abs_notional / equity
        margin_util = (total_abs_notional * mmr) / equity
        scale = 1.0
        if leverage > float(self.max_leverage):
            scale = min(scale, float(self.max_leverage) / leverage)
        if margin_util > float(self.max_margin_utilization):
            scale = min(scale, float(self.max_margin_utilization) / margin_util)
        if scale < 1.0:
            for s in list(targets.keys()):
                targets[s] = float(targets[s]) * float(scale)

        # Rebalance gating to reduce churn
        for s in symbols:
            last_close = float(per.get(s, {}).get("last_close", np.nan))
            if not np.isfinite(last_close) or last_close <= 0:
                continue
            cur_qty = float(state.positions.get(s, 0.0) or 0.0)
            cur_exposure = (cur_qty * last_close) / max_notional
            tgt = float(targets.get(s, 0.0))

            # Always allow exits / flips.
            if _sign(cur_exposure) == _sign(tgt) and abs(tgt) > 0 and abs(cur_exposure) > 0:
                if abs(tgt - cur_exposure) < float(self.rebalance_exposure_threshold):
                    tgt = float(cur_exposure)
            exposures[s] = float(tgt)

        # Final safety clamp
        for s in symbols:
            exposures[s] = float(np.clip(float(exposures.get(s, 0.0)), -1.0, 1.0))

        reason = "active=" + ",".join(sorted([s for s in symbols if abs(exposures.get(s, 0.0)) > 1e-9]))
        debug = {
            "drawdown": drawdown,
            "scale": scale,
            "leverage": leverage,
            "margin_util": margin_util,
            "per": per,
        }
        return StrategyDecision(target_exposures=exposures, reason=reason, debug=debug)
