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
    window = int(window)
    if window <= 1:
        return pd.Series(np.nan, index=close.index)
    tr = _true_range(high, low, close)
    return tr.rolling(window).mean()


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


def _max_notional_for_liq_buffer(
    *,
    equity_alloc: float,
    entry_price: float,
    atr: float,
    mmr: float,
    stop_atr_mult: float,
    liq_buffer_atr: float,
    side: int,
) -> float:
    """Cap notional so a stop hits before liquidation (simple linear perp approximation).

    Uses a cross-margin style approximation for *one position* and treats `equity_alloc`
    as the collateral effectively available to this position.
    """

    if equity_alloc <= 0 or entry_price <= 0 or atr <= 0:
        return 0.0
    mmr = float(abs(mmr))
    if mmr <= 0 or mmr >= 0.5:
        return 0.0

    side = 1 if side >= 0 else -1
    stop_dist = (float(stop_atr_mult) + float(abs(liq_buffer_atr))) * float(atr)
    stop_frac = float(stop_dist) / float(entry_price)
    if stop_frac <= 0:
        return 0.0

    if side > 0:
        denom = mmr + (1.0 - mmr) * stop_frac
    else:
        denom = mmr + (1.0 + mmr) * stop_frac

    if denom <= 0:
        return 0.0
    return float(equity_alloc) / float(denom)


@dataclass
class PerpScalp(Strategy):
    """High-leverage BTC-perp scalping strategy (research/paper only).

    Notes:
    - Not "foolproof" and cannot guarantee profitability; designed to be conservative and
      failure-aware under leverage (daily loss limits + drawdown kill-switch + liquidation buffers).
    - Intended for 1–5 minute bars, per-symbol notional caps, and a derivatives engine that models
      funding + maintenance margin + liquidation.
    """

    name: str = "perp_scalp"
    symbols: tuple[str, ...] = ("BTC-PERP",)

    # --- signal / regime ---
    atr_window: int = 14
    ema_fast: int = 8
    ema_slow: int = 21
    er_window: int = 10
    breakout_window: int = 8
    breakout_buffer_bps: float = 1.0
    er_min: float = 0.25
    trend_z_min: float = 0.15
    min_atr_bps: float = 8.0

    # --- execution / admission ---
    edge_floor_bps: float = 3.0
    k_cost: float = 1.5
    taker_fee_bps: float = 3.0
    slippage_bps: float = 1.5  # per side (fallback; engines may override via StrategyState.extra)

    # --- funding filters (optional, bps/day) ---
    funding_entry_bps_per_day: float = 40.0
    funding_exit_bps_per_day: float = 80.0

    # --- risk / sizing ---
    risk_per_trade: float = 0.005
    stop_atr_mult: float = 1.2
    trail_atr_mult: float = 1.8
    take_profit_atr_mult: float = 1.5
    max_hold_bars: int = 12
    min_hold_bars: int = 2
    flip_confirm_bars: int = 2
    cooldown_bars: int = 4

    sizing_mode: str = "risk"  # "risk" (default) or "leverage"
    target_leverage: Optional[float] = None

    max_leverage: float = 5.0
    max_margin_utilization: float = 0.40
    maintenance_margin_rate: float = 0.05
    min_liq_buffer_atr: float = 2.5

    # --- risk-off controls ---
    daily_loss_limit: float = 0.02
    kill_switch: float = 0.10

    # --- internal state ---
    _bars_seen: int = field(default=0, init=False, repr=False)
    _peak_equity: float = field(default=0.0, init=False, repr=False)
    _risk_disabled_day: Optional[object] = field(default=None, init=False, repr=False)
    _risk_disabled_forever: bool = field(default=False, init=False, repr=False)

    _cooldown_until: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _flip_counter: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _trail_extreme: dict[str, float] = field(default_factory=dict, init=False, repr=False)
    _entry_price: dict[str, float] = field(default_factory=dict, init=False, repr=False)
    _last_pos_side: dict[str, int] = field(default_factory=dict, init=False, repr=False)

    def warmup_bars(self) -> int:
        return max(self.atr_window, self.ema_slow, self.er_window, self.breakout_window) + 3

    def _maybe_reset_daily_state(self, state: StrategyState) -> None:
        today = state.timestamp.date()
        if self._risk_disabled_day is not None and self._risk_disabled_day != today:
            self._risk_disabled_day = None

    def _risk_off(self, symbols: list[str], *, reason: str, debug: dict[str, Any]) -> StrategyDecision:
        return StrategyDecision(target_exposures={s: 0.0 for s in symbols}, reason=reason, debug=debug)

    def target_exposures(
        self, bars_by_symbol: dict[str, pd.DataFrame], state: StrategyState
    ) -> StrategyDecision:
        symbols = [s for s in self.symbols if s in bars_by_symbol]
        if not symbols:
            symbols = sorted(bars_by_symbol.keys())
        exposures: dict[str, float] = {s: 0.0 for s in symbols}
        if not symbols:
            return StrategyDecision(target_exposures=exposures, reason="no_symbols")

        self._bars_seen += 1
        self._maybe_reset_daily_state(state)

        # --- Global risk-off controls ---
        equity = float(state.equity)
        if self._peak_equity <= 0:
            self._peak_equity = equity
        self._peak_equity = max(self._peak_equity, equity)
        drawdown = (equity / self._peak_equity - 1.0) if self._peak_equity > 0 else 0.0
        if self._risk_disabled_forever or drawdown <= -abs(float(self.kill_switch)):
            self._risk_disabled_forever = True
            return self._risk_off(
                symbols,
                reason="kill_switch",
                debug={"drawdown": drawdown, "kill_switch": self.kill_switch},
            )

        if state.day_return <= -abs(float(self.daily_loss_limit)):
            self._risk_disabled_day = state.timestamp.date()
            return self._risk_off(
                symbols,
                reason="daily_loss_limit",
                debug={"day_return": float(state.day_return), "daily_loss_limit": self.daily_loss_limit},
            )
        if self._risk_disabled_day == state.timestamp.date():
            return self._risk_off(
                symbols,
                reason="risk_disabled_day",
                debug={"day_return": float(state.day_return), "daily_loss_limit": self.daily_loss_limit},
            )

        extra = dict(state.extra or {})
        max_notional = float(extra.get("max_position_notional_usd") or 0.0)
        if max_notional <= 0:
            return StrategyDecision(target_exposures=exposures, reason="no_max_position_notional")

        slippage_bps = float(
            extra.get("slippage_bps")
            if extra.get("slippage_bps") is not None
            else self.slippage_bps
        )
        taker_fee_bps = float(extra.get("taker_fee_bps") or self.taker_fee_bps)
        mmr = float(extra.get("maintenance_margin_rate") or self.maintenance_margin_rate)
        funding_rates = dict(extra.get("funding_rates") or {})

        sizing_mode = str(extra.get("sizing_mode") or self.sizing_mode).strip().lower()
        target_leverage = extra.get("target_leverage", self.target_leverage)
        if target_leverage is not None:
            try:
                target_leverage = float(target_leverage)
            except Exception:
                target_leverage = None

        # --- feature extraction ---
        per: dict[str, dict[str, float | int | str]] = {}
        for s in symbols:
            df = bars_by_symbol.get(s)
            if df is None or len(df) < self.warmup_bars():
                continue

            if not df.index.is_monotonic_increasing:
                df = df.sort_index()
            high = df["high"].astype(float)
            low = df["low"].astype(float)
            close = df["close"].astype(float)
            last_close = float(close.iloc[-1])
            if not np.isfinite(last_close) or last_close <= 0:
                continue

            atr_series = _atr(high, low, close, self.atr_window)
            atr = float(atr_series.iloc[-1]) if len(atr_series) else float("nan")
            if not np.isfinite(atr) or atr <= 0:
                continue

            ema_f = float(_ema(close, self.ema_fast).iloc[-1])
            ema_s = float(_ema(close, self.ema_slow).iloc[-1])
            trend_strength = float((ema_f - ema_s) / max(1e-12, atr))
            trend_dir = 1 if trend_strength >= float(self.trend_z_min) else -1 if trend_strength <= -float(self.trend_z_min) else 0

            er_series = _efficiency_ratio(close, self.er_window)
            er = float(er_series.iloc[-1]) if len(er_series) else float("nan")

            hh, ll = _donchian(high, low, self.breakout_window)
            buf = float(self.breakout_buffer_bps) / 10_000.0
            breakout_dir = 0
            if np.isfinite(hh) and last_close > hh * (1.0 + buf):
                breakout_dir = 1
            elif np.isfinite(ll) and last_close < ll * (1.0 - buf):
                breakout_dir = -1

            atr_bps = float((atr / last_close) * 10_000.0)

            # Simple edge proxy: breakout "excess" + trend term (both in bps).
            breakout_excess_bps = 0.0
            if breakout_dir > 0 and np.isfinite(hh) and hh > 0:
                breakout_excess_bps = float(((last_close - hh * (1.0 + buf)) / last_close) * 10_000.0)
            elif breakout_dir < 0 and np.isfinite(ll) and ll > 0:
                breakout_excess_bps = float(((ll * (1.0 - buf) - last_close) / last_close) * 10_000.0)

            edge_bps = float(max(0.0, breakout_excess_bps) + 0.35 * abs(trend_strength) * atr_bps)
            cost_rt_bps = float(2.0 * (abs(slippage_bps) + abs(taker_fee_bps)))
            required_edge = float(self.edge_floor_bps) + float(self.k_cost) * cost_rt_bps
            score = float(edge_bps - float(self.k_cost) * cost_rt_bps)

            per[s] = {
                "last_close": last_close,
                "atr": atr,
                "atr_bps": atr_bps,
                "ema_f": ema_f,
                "ema_s": ema_s,
                "trend_strength": trend_strength,
                "trend_dir": int(trend_dir),
                "er": er,
                "breakout_dir": int(breakout_dir),
                "breakout_excess_bps": breakout_excess_bps,
                "edge_bps": edge_bps,
                "cost_rt_bps": cost_rt_bps,
                "required_edge": required_edge,
                "score": score,
            }

        if not per:
            return StrategyDecision(target_exposures=exposures, reason="warmup")

        # --- decide desired direction per symbol ---
        desired_dir: dict[str, int] = {s: 0 for s in symbols}
        reason_tag: dict[str, str] = {s: "flat" for s in symbols}

        active_sides: dict[str, int] = {}
        for s in symbols:
            pos_qty = float(state.positions.get(s, 0.0) or 0.0)
            pos_side = _sign(pos_qty)
            active_sides[s] = int(pos_side)

        # Track position transitions for entry/trailing state.
        for s in symbols:
            if s not in per:
                continue
            last_close = float(per[s]["last_close"])
            pos_side = int(active_sides.get(s, 0))
            prev_side = int(self._last_pos_side.get(s, 0))

            if prev_side == 0 and pos_side != 0:
                # New position observed (filled at prior open); seed entry + trailing anchors.
                self._entry_price[s] = last_close
                self._trail_extreme[s] = last_close
                self._flip_counter[s] = 0
            elif prev_side != 0 and pos_side == 0:
                # Position closed; start cooldown (longer cooldown after a loss estimate).
                entry = float(self._entry_price.get(s, last_close))
                pnl_est = float((last_close - entry) * float(prev_side))
                extra_cooldown = int(self.cooldown_bars) * (2 if pnl_est < 0 else 1)
                self._cooldown_until[s] = max(
                    int(self._cooldown_until.get(s, 0)), int(self._bars_seen + extra_cooldown)
                )
                self._trail_extreme.pop(s, None)
                self._entry_price.pop(s, None)
                self._flip_counter[s] = 0

            self._last_pos_side[s] = int(pos_side)

        # Decision per symbol: manage open positions first.
        for s in symbols:
            if s not in per:
                continue

            info = per[s]
            last_close = float(info["last_close"])
            atr = float(info["atr"])
            er = float(info["er"]) if np.isfinite(float(info["er"])) else 0.0
            trend_dir = int(info["trend_dir"])
            breakout_dir = int(info["breakout_dir"])

            pos_qty = float(state.positions.get(s, 0.0) or 0.0)
            pos_side = _sign(pos_qty)
            hold_bars = int(state.holding_bars.get(s, 0) or 0)

            fr = float(funding_rates.get(s, 0.0) or 0.0)
            funding_bps_per_day = fr * 10_000.0 * 24.0
            pays_funding_bps = float(pos_side) * funding_bps_per_day

            if pos_side != 0:
                entry = float(self._entry_price.get(s, last_close))
                prev_ext = float(self._trail_extreme.get(s, last_close))
                if pos_side > 0:
                    self._trail_extreme[s] = max(prev_ext, last_close)
                    trail_stop = float(self._trail_extreme[s]) - float(self.trail_atr_mult) * atr
                    hard_stop = entry - float(self.stop_atr_mult) * atr
                    effective_stop = max(hard_stop, trail_stop)
                    stop_hit = bool(last_close <= effective_stop)
                    tp_hit = bool(last_close >= entry + float(self.take_profit_atr_mult) * atr)
                else:
                    self._trail_extreme[s] = min(prev_ext, last_close)
                    trail_stop = float(self._trail_extreme[s]) + float(self.trail_atr_mult) * atr
                    hard_stop = entry + float(self.stop_atr_mult) * atr
                    effective_stop = min(hard_stop, trail_stop)
                    stop_hit = bool(last_close >= effective_stop)
                    tp_hit = bool(last_close <= entry - float(self.take_profit_atr_mult) * atr)

                # Liquidation proximity check (uses current equity as collateral budget).
                liq_cap = _max_notional_for_liq_buffer(
                    equity_alloc=equity,
                    entry_price=entry,
                    atr=atr,
                    mmr=mmr,
                    stop_atr_mult=float(self.stop_atr_mult),
                    liq_buffer_atr=float(self.min_liq_buffer_atr),
                    side=int(pos_side),
                )
                cur_notional_est = abs(pos_qty) * last_close
                liq_buffer_breached = bool(cur_notional_est > (liq_cap * 1.02))  # small hysteresis

                if liq_buffer_breached and hold_bars >= int(self.min_hold_bars):
                    desired_dir[s] = 0
                    reason_tag[s] = "liq_buffer"
                elif pays_funding_bps > float(self.funding_exit_bps_per_day) and hold_bars >= int(self.min_hold_bars):
                    desired_dir[s] = 0
                    reason_tag[s] = "funding_exit"
                elif hold_bars >= int(self.max_hold_bars) and hold_bars >= int(self.min_hold_bars):
                    desired_dir[s] = 0
                    reason_tag[s] = "time_stop"
                elif stop_hit and hold_bars >= int(self.min_hold_bars):
                    desired_dir[s] = 0
                    reason_tag[s] = "stop"
                elif tp_hit and hold_bars >= int(self.min_hold_bars):
                    desired_dir[s] = 0
                    reason_tag[s] = "take_profit"
                else:
                    opposite = (breakout_dir == -pos_side or trend_dir == -pos_side) and (er >= float(self.er_min))
                    if opposite:
                        self._flip_counter[s] = int(self._flip_counter.get(s, 0)) + 1
                    else:
                        self._flip_counter[s] = 0

                    if self._flip_counter[s] >= int(self.flip_confirm_bars) and hold_bars >= int(self.min_hold_bars):
                        desired_dir[s] = -pos_side
                        reason_tag[s] = "flip"
                    else:
                        desired_dir[s] = pos_side
                        reason_tag[s] = "hold"
                continue

            # --- flat: evaluate entry ---
            if self._bars_seen < int(self._cooldown_until.get(s, 0)):
                desired_dir[s] = 0
                reason_tag[s] = "cooldown"
                continue

            if (not np.isfinite(float(info["er"]))) or (float(info["er"]) < float(self.er_min)):
                desired_dir[s] = 0
                reason_tag[s] = "gate_er"
                continue

            if float(info["atr_bps"]) < float(self.min_atr_bps):
                desired_dir[s] = 0
                reason_tag[s] = "gate_atr"
                continue

            if float(info["edge_bps"]) < float(info["required_edge"]):
                desired_dir[s] = 0
                reason_tag[s] = "gate_cost"
                continue

            entry_dir = int(info["breakout_dir"])
            if entry_dir == 0 or entry_dir != int(info["trend_dir"]):
                desired_dir[s] = 0
                reason_tag[s] = "no_entry"
                continue

            if (not state.allow_short) and entry_dir < 0:
                desired_dir[s] = 0
                reason_tag[s] = "short_blocked"
                continue

            if (float(entry_dir) * funding_bps_per_day) > float(self.funding_entry_bps_per_day):
                desired_dir[s] = 0
                reason_tag[s] = "funding_block"
                continue

            desired_dir[s] = entry_dir
            reason_tag[s] = "entry"

        active = [s for s in symbols if int(desired_dir.get(s, 0)) != 0 and s in per]
        if not active:
            debug = {"drawdown": drawdown, "per": per, "reason_tag": reason_tag}
            return StrategyDecision(target_exposures=exposures, reason="no_signal", debug=debug)

        # Rank-cut if multiple signals; prefer existing positions.
        max_positions = 1
        if len(active) > max_positions:
            active_sorted = sorted(
                active,
                key=lambda s: (
                    1 if int(active_sides.get(s, 0)) != 0 else 0,
                    float(per[s]["score"]),
                ),
                reverse=True,
            )
            keep = set(active_sorted[:max_positions])
            for s in list(active):
                if s not in keep:
                    desired_dir[s] = 0
                    reason_tag[s] = "rank_cut"
            active = [s for s in active_sorted if s in keep]

        equity_alloc = equity / max(1.0, float(len(active)))
        desired_notional_by_symbol: dict[str, float] = {}
        for s in active:
            side = int(desired_dir[s])
            price = float(per[s]["last_close"])
            atr = float(per[s]["atr"])
            if price <= 0 or atr <= 0:
                continue

            # Risk-based sizing by stop distance.
            stop_dist = max(1e-12, float(self.stop_atr_mult) * atr)
            risk_usd = max(0.0, float(self.risk_per_trade)) * equity_alloc
            risk_based_qty = risk_usd / stop_dist if stop_dist > 0 else 0.0
            risk_based_notional = risk_based_qty * price

            if sizing_mode == "leverage":
                if target_leverage is None or float(target_leverage) <= 0:
                    continue
                desired_notional = float(target_leverage) * float(equity_alloc)
            else:
                desired_notional = risk_based_notional

            # Hard caps: engine's per-symbol notional cap + liquidation-buffer-derived cap.
            liq_cap = _max_notional_for_liq_buffer(
                equity_alloc=float(equity_alloc),
                entry_price=price,
                atr=atr,
                mmr=mmr,
                stop_atr_mult=float(self.stop_atr_mult),
                liq_buffer_atr=float(self.min_liq_buffer_atr),
                side=side,
            )

            desired_notional = min(float(desired_notional), float(max_notional), float(liq_cap))
            desired_notional_by_symbol[s] = max(0.0, float(desired_notional))

        if not desired_notional_by_symbol:
            debug = {"drawdown": drawdown, "per": per, "reason_tag": reason_tag}
            return StrategyDecision(target_exposures=exposures, reason="sizing_blocked", debug=debug)

        # Convert to exposures and enforce portfolio-level leverage + margin utilization caps.
        total_abs_notional = float(sum(desired_notional_by_symbol.values()))
        leverage = (total_abs_notional / equity) if equity > 0 else 0.0
        margin_util = (total_abs_notional * float(abs(mmr)) / equity) if equity > 0 else 0.0
        scale = 1.0
        if leverage > float(self.max_leverage):
            scale = min(scale, float(self.max_leverage) / max(leverage, 1e-12))
        if margin_util > float(self.max_margin_utilization):
            scale = min(scale, float(self.max_margin_utilization) / max(margin_util, 1e-12))

        for s in symbols:
            if s not in per:
                continue
            side = int(desired_dir.get(s, 0))
            if side == 0:
                exposures[s] = 0.0
                continue
            notional = float(desired_notional_by_symbol.get(s, 0.0)) * float(scale)
            price = float(per[s]["last_close"])
            exposure = (float(side) * notional) / max_notional
            exposures[s] = float(np.clip(exposure, -1.0, 1.0))

        reason = "active=" + ",".join(sorted([s for s in symbols if abs(exposures.get(s, 0.0)) > 1e-9]))
        debug = {
            "drawdown": drawdown,
            "scale": scale,
            "leverage": leverage,
            "margin_util": margin_util,
            "reason_tag": reason_tag,
            "per": per,
        }
        return StrategyDecision(target_exposures=exposures, reason=reason, debug=debug)
