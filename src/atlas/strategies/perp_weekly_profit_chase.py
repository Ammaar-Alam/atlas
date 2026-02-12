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


def _infer_bar_minutes(index: pd.DatetimeIndex) -> float:
    if len(index) < 3:
        return 1.0
    try:
        diffs_ns = np.diff(index.asi8)
        diffs_min = diffs_ns.astype("float64") / (60.0 * 1e9)
        diffs_min = diffs_min[(diffs_min > 0.0) & (diffs_min <= 60.0)]
        if diffs_min.size == 0:
            return 1.0
        median = float(np.median(diffs_min))
        return median if median > 0 else 1.0
    except Exception:
        diffs = index.to_series().diff().dropna().dt.total_seconds() / 60.0
        diffs = diffs[(diffs > 0) & (diffs <= 60)]
        if len(diffs) == 0:
            return 1.0
        median = float(diffs.median())
        return median if median > 0 else 1.0


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


def _utc_week_key(ts: pd.Timestamp) -> tuple[int, int]:
    ts = pd.Timestamp(ts)
    if ts.tz is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    iso = ts.isocalendar()
    return int(iso.year), int(iso.week)


@dataclass
class PerpWeeklyProfitChase(Strategy):
    """
    Perps strategy explicitly designed to:
    - Ensure >= 1 fill per 7d window (weekly scheduled heartbeat/nudge).
    - Trade a daily "opening range breakout" (ORB) with leverage, but stop trading
      once the weekly profit target is hit.
    - Flatten once the weekly profit target is hit.

    This cannot guarantee future returns. It can, however, be evaluated ("verified") on
    historical data by looking at weekly window hit-rate via `atlas.cli analyze-run`.
    """

    name: str = "perp_weekly_profit_chase"
    symbols: tuple[str, ...] = ("BTC-PERP",)

    # --- weekly schedule (UTC) ---
    rebalance_weekday_utc: int = 0  # Monday
    rebalance_hour_utc: int = 0
    rebalance_minute_utc: int = 5

    # --- objective ---
    weekly_profit_target: float = 0.01  # 1%
    weekly_chase_k: float = 0.0  # optional leverage multiplier when behind

    # --- signal ---
    atr_window: int = 14
    opening_range_minutes: int = 60
    breakout_buffer_bps: float = 8.0
    lookback_short_days: float = 1.0  # fallback direction (if no breakout)
    lookback_long_days: float = 7.0   # fallback direction (if no breakout)
    momentum_threshold_bps: float = 0.0
    min_atr_bps: float = 5.0

    # --- sizing / leverage ---
    sizing_mode: str = "leverage"  # "leverage" or "risk"
    risk_per_trade: float = 0.03   # fraction of equity at the stop (risk sizing)
    base_leverage: float = 8.0
    max_leverage: float = 25.0
    max_margin_utilization: float = 0.95
    maintenance_margin_rate: float = 0.05
    stop_atr_mult: float = 2.0
    min_liq_buffer_atr: float = 3.0
    min_trade_notional_usd: float = 10.0
    weekly_heartbeat_exposure: float = 0.01
    weekly_heartbeat_hold_bars: int = 1
    weekly_nudge_exposure: float = 0.002
    max_flips_per_day: int = 3
    daily_loss_hard_stop: float = 0.0
    weekly_loss_hard_stop: float = 0.0
    cooldown_bars_after_exit: int = 0
    trailing_stop_atr_mult: float = 0.0
    break_even_trigger_atr: float = 0.0
    max_hold_bars: int = 0

    # --- internal state ---
    _bars_seen: int = field(default=0, init=False, repr=False)
    _week_key: Optional[tuple[int, int]] = field(default=None, init=False, repr=False)
    _week_start_equity: float = field(default=0.0, init=False, repr=False)
    _week_target_hit: bool = field(default=False, init=False, repr=False)
    _week_trade_requested: bool = field(default=False, init=False, repr=False)
    _rebalance_armed_for_week: bool = field(default=False, init=False, repr=False)
    _day_key: Optional[pd.Timestamp] = field(default=None, init=False, repr=False)
    _day_flips: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _entry_price: dict[str, float] = field(default_factory=dict, init=False, repr=False)
    _peak_price: dict[str, float] = field(default_factory=dict, init=False, repr=False)
    _trough_price: dict[str, float] = field(default_factory=dict, init=False, repr=False)
    _cooldown_bars_left: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _day_hard_stop_hit: bool = field(default=False, init=False, repr=False)
    _week_hard_stop_hit: bool = field(default=False, init=False, repr=False)

    _scheduled_stage: dict[str, str] = field(default_factory=dict, init=False, repr=False)
    _scheduled_exposure: dict[str, float] = field(default_factory=dict, init=False, repr=False)

    def warmup_bars(self) -> int:
        # ATR + enough history for momentum windows.
        return max(int(self.atr_window) + 3, 200)

    def _is_rebalance_time(self, ts: pd.Timestamp) -> bool:
        ts = pd.Timestamp(ts)
        if ts.tz is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        if int(ts.dayofweek) != int(self.rebalance_weekday_utc):
            return False
        if int(ts.hour) != int(self.rebalance_hour_utc):
            return False
        return int(ts.minute) >= int(self.rebalance_minute_utc)

    def _week_start_utc(self, ts: pd.Timestamp) -> pd.Timestamp:
        ts = pd.Timestamp(ts)
        if ts.tz is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        return (ts - pd.Timedelta(days=int(ts.dayofweek))).normalize()

    def _utc_day_key(self, ts: pd.Timestamp) -> pd.Timestamp:
        ts = pd.Timestamp(ts)
        if ts.tz is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        return ts.normalize()

    def _momentum_dir(
        self, close: pd.Series, *, bar_minutes: float
    ) -> tuple[int, float, float]:
        close = close.astype(float)
        if len(close) < 10:
            return 0, 0.0, 0.0

        bars_per_day = int(round((24.0 * 60.0) / max(float(bar_minutes), 1e-9)))
        bars_per_day = int(max(1, bars_per_day))

        lb_s = int(max(2, round(float(self.lookback_short_days) * bars_per_day)))
        lb_l = int(max(lb_s + 1, round(float(self.lookback_long_days) * bars_per_day)))
        lb_s = min(lb_s, len(close) - 2)
        lb_l = min(lb_l, len(close) - 2)
        if lb_s < 2 or lb_l < 2:
            return 0, 0.0, 0.0

        c1 = float(close.iloc[-1])
        c0s = float(close.iloc[-lb_s - 1])
        c0l = float(close.iloc[-lb_l - 1])
        if c1 <= 0 or c0s <= 0 or c0l <= 0:
            return 0, 0.0, 0.0

        mom_s_bps = float((c1 / c0s - 1.0) * 10_000.0)
        mom_l_bps = float((c1 / c0l - 1.0) * 10_000.0)
        score = float(0.65 * mom_s_bps + 0.35 * mom_l_bps)

        thr = float(abs(self.momentum_threshold_bps))
        if abs(score) <= thr:
            return 0, mom_s_bps, mom_l_bps
        return (1 if score > 0 else -1), mom_s_bps, mom_l_bps

    def _max_notional_for_liq_buffer(
        self,
        *,
        equity_alloc: float,
        entry_price: float,
        atr: float,
        mmr: float,
        side: int,
    ) -> float:
        if equity_alloc <= 0 or entry_price <= 0 or atr <= 0:
            return 0.0
        mmr = float(abs(mmr))
        if mmr <= 0 or mmr >= 0.5:
            return 0.0

        side = 1 if side >= 0 else -1
        stop_dist = (float(self.stop_atr_mult) + float(abs(self.min_liq_buffer_atr))) * float(atr)
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

        extra = dict(state.extra or {})
        max_notional = float(extra.get("max_position_notional_usd") or 0.0)
        if max_notional <= 0:
            return StrategyDecision(target_exposures=exposures, reason="no_max_position_notional")
        mmr = float(extra.get("maintenance_margin_rate") or self.maintenance_margin_rate)

        ts = pd.Timestamp(state.timestamp)
        wk = _utc_week_key(ts)
        if self._week_key is None or self._week_key != wk:
            self._week_key = wk
            self._week_start_equity = float(state.equity)
            self._week_target_hit = False
            self._week_hard_stop_hit = False
            self._week_trade_requested = False
            self._rebalance_armed_for_week = False
            self._day_flips.clear()
            self._day_key = None

        day_key = self._utc_day_key(ts)
        if self._day_key is None or self._day_key != day_key:
            self._day_key = day_key
            self._day_flips.clear()
            self._day_hard_stop_hit = False

        week_ret = (
            (float(state.equity) / float(self._week_start_equity) - 1.0)
            if self._week_start_equity > 0
            else 0.0
        )
        day_ret = float(state.day_return)
        if (not self._week_target_hit) and week_ret >= float(self.weekly_profit_target):
            self._week_target_hit = True
        if float(self.daily_loss_hard_stop) > 0 and day_ret <= -float(self.daily_loss_hard_stop):
            self._day_hard_stop_hit = True
        if float(self.weekly_loss_hard_stop) > 0 and week_ret <= -float(self.weekly_loss_hard_stop):
            self._week_hard_stop_hit = True

        # Arm the weekly rebalance when the time arrives.
        if (not self._rebalance_armed_for_week) and self._is_rebalance_time(ts):
            self._rebalance_armed_for_week = True

        debug: dict[str, Any] = {
            "week_key": wk,
            "week_start_equity": float(self._week_start_equity),
            "week_return": float(week_ret),
            "day_return": float(day_ret),
            "week_target_hit": bool(self._week_target_hit),
            "day_hard_stop_hit": bool(self._day_hard_stop_hit),
            "week_hard_stop_hit": bool(self._week_hard_stop_hit),
            "bars_seen": int(self._bars_seen),
        }

        for s in symbols:
            df = bars_by_symbol.get(s)
            if df is None or len(df) < 10:
                continue
            if not df.index.is_monotonic_increasing:
                df = df.sort_index()

            close = df["close"].astype(float)
            high = df["high"].astype(float)
            low = df["low"].astype(float)
            last_close = float(close.iloc[-1])
            if not np.isfinite(last_close) or last_close <= 0:
                continue

            bar_minutes = _infer_bar_minutes(df.index)
            atr = float("nan")
            if len(df) >= max(3, int(self.atr_window) + 2):
                atr_series = _atr(high, low, close, self.atr_window)
                atr = float(atr_series.iloc[-1]) if len(atr_series) else float("nan")
            atr_bps = float((atr / last_close) * 10_000.0) if np.isfinite(atr) and last_close > 0 else 0.0

            pos_qty = float(state.positions.get(s, 0.0) or 0.0)
            current_exposure = float((pos_qty * last_close) / max_notional) if max_notional > 0 else 0.0
            pos_side = _sign(pos_qty)

            if pos_side == 0:
                self._entry_price.pop(s, None)
                self._peak_price.pop(s, None)
                self._trough_price.pop(s, None)
            elif s not in self._entry_price:
                self._entry_price[s] = float(last_close)
                self._peak_price[s] = float(last_close)
                self._trough_price[s] = float(last_close)
            else:
                self._peak_price[s] = float(max(float(self._peak_price.get(s, last_close)), float(last_close)))
                self._trough_price[s] = float(min(float(self._trough_price.get(s, last_close)), float(last_close)))

            cooldown_left = int(self._cooldown_bars_left.get(s, 0))
            in_cooldown = cooldown_left > 0
            if in_cooldown:
                if cooldown_left > 1:
                    self._cooldown_bars_left[s] = cooldown_left - 1
                else:
                    self._cooldown_bars_left.pop(s, None)

            stage = str(self._scheduled_stage.get(s, "") or "")
            if stage == "nudge_return":
                exposures[s] = float(self._scheduled_exposure.get(s, 0.0))
                self._scheduled_stage.pop(s, None)
                self._scheduled_exposure.pop(s, None)
                debug[s] = {"stage": "nudge_return", "target": float(exposures[s])}
                continue

            min_exposure = float(self.min_trade_notional_usd) / float(max_notional) if max_notional > 0 else 0.0
            min_exposure = float(max(0.0, min(1.0, min_exposure)))

            if self._week_hard_stop_hit or self._day_hard_stop_hit:
                exposures[s] = 0.0
                if pos_side != 0 and int(self.cooldown_bars_after_exit) > 0:
                    self._cooldown_bars_left[s] = max(
                        int(self._cooldown_bars_left.get(s, 0)),
                        int(self.cooldown_bars_after_exit),
                    )
                debug[s] = {
                    "reason": "hard_stop_flat",
                    "day_hard_stop_hit": bool(self._day_hard_stop_hit),
                    "week_hard_stop_hit": bool(self._week_hard_stop_hit),
                }
                continue

            # If we've already hit the week's profit target, stay flat.
            if self._week_target_hit:
                exposures[s] = 0.0
                debug[s] = {"reason": "target_hit_flat"}
                continue

            dir_sig, mom_s_bps, mom_l_bps = self._momentum_dir(close, bar_minutes=bar_minutes)
            if atr_bps < float(self.min_atr_bps):
                dir_sig = 0

            # Compute leverage: optionally increase if behind for the week (very risky).
            sizing_mode = str(extra.get("sizing_mode") or self.sizing_mode).strip().lower()
            risk_per_trade = float(extra.get("risk_per_trade") or self.risk_per_trade)

            base = float(max(0.0, self.base_leverage))
            cap = float(max(base, self.max_leverage))
            target = float(max(0.0, self.weekly_profit_target))
            chase = float(self.weekly_chase_k)
            if target > 0 and week_ret < 0:
                base = base * (1.0 + chase * float(-week_ret) / float(target))
            lev = float(max(0.0, min(cap, base)))

            # Also cap by margin utilization.
            equity = float(state.equity)
            if equity > 0 and mmr > 0:
                margin_cap_lev = float(self.max_margin_utilization) / float(mmr)
                lev = float(min(lev, margin_cap_lev))

            # Daily ORB (UTC): opening range high/low from the day's start.
            day_start = self._utc_day_key(ts)
            day_end = day_start + pd.Timedelta(days=1)
            orb_end = day_start + pd.Timedelta(minutes=int(self.opening_range_minutes))
            df_today = df[(df.index >= day_start) & (df.index < day_end)]
            orb_window = df_today[df_today.index < orb_end]
            need = int(np.ceil(float(self.opening_range_minutes) / max(float(bar_minutes), 1e-9)))
            orb_ready = len(orb_window) >= max(1, need)
            orb_high = float(orb_window["high"].astype(float).max()) if orb_ready else float("nan")
            orb_low = float(orb_window["low"].astype(float).min()) if orb_ready else float("nan")

            # In-position exits: stop + weekly target hit.
            if pos_side != 0 and int(self.max_hold_bars) > 0:
                holding_bars = int(state.holding_bars.get(s, 0) or 0)
                if holding_bars >= int(self.max_hold_bars):
                    exposures[s] = 0.0
                    self._day_flips[s] = int(self._day_flips.get(s, 0)) + 1
                    if int(self.cooldown_bars_after_exit) > 0:
                        self._cooldown_bars_left[s] = max(
                            int(self._cooldown_bars_left.get(s, 0)),
                            int(self.cooldown_bars_after_exit),
                        )
                    debug[s] = {
                        "reason": "max_hold_exit",
                        "holding_bars": int(holding_bars),
                        "max_hold_bars": int(self.max_hold_bars),
                        "day_flips": int(self._day_flips[s]),
                    }
                    continue

            if pos_side != 0 and np.isfinite(atr) and atr > 0:
                entry_px = float(self._entry_price.get(s, last_close))
                stop_px = entry_px - float(self.stop_atr_mult) * float(atr) if pos_side > 0 else entry_px + float(
                    self.stop_atr_mult
                ) * float(atr)
                trail_mult = float(self.trailing_stop_atr_mult)
                if trail_mult > 0:
                    if pos_side > 0:
                        peak_px = float(self._peak_price.get(s, last_close))
                        trail_px = peak_px - trail_mult * float(atr)
                        stop_px = max(float(stop_px), float(trail_px))
                    else:
                        trough_px = float(self._trough_price.get(s, last_close))
                        trail_px = trough_px + trail_mult * float(atr)
                        stop_px = min(float(stop_px), float(trail_px))
                be_trigger = float(self.break_even_trigger_atr)
                if be_trigger > 0:
                    trigger_dist = be_trigger * float(atr)
                    if pos_side > 0 and (float(last_close) - float(entry_px)) >= trigger_dist:
                        stop_px = max(float(stop_px), float(entry_px))
                    elif pos_side < 0 and (float(entry_px) - float(last_close)) >= trigger_dist:
                        stop_px = min(float(stop_px), float(entry_px))
                stop_hit = (last_close <= stop_px) if pos_side > 0 else (last_close >= stop_px)
                if stop_hit:
                    exposures[s] = 0.0
                    self._day_flips[s] = int(self._day_flips.get(s, 0)) + 1
                    if int(self.cooldown_bars_after_exit) > 0:
                        self._cooldown_bars_left[s] = max(
                            int(self._cooldown_bars_left.get(s, 0)),
                            int(self.cooldown_bars_after_exit),
                        )
                    debug[s] = {
                        "reason": "stop_exit",
                        "pos_side": int(pos_side),
                        "entry_px": float(entry_px),
                        "stop_px": float(stop_px),
                        "cooldown_bars_after_exit": int(self.cooldown_bars_after_exit),
                        "day_flips": int(self._day_flips[s]),
                    }
                    continue

            # Default behavior: only enter/flip on weekly schedule or ORB breakout. This keeps
            # churn low vs continuously following the signal.
            desired_exposure = float(current_exposure)
            reason = "hold_or_wait"

            flips = int(self._day_flips.get(s, 0))
            if in_cooldown and pos_side == 0:
                desired_exposure = 0.0
                reason = "cooldown_wait"
            elif flips >= int(self.max_flips_per_day):
                # No more flips this week; keep position or stay flat.
                desired_exposure = float(current_exposure)
                reason = "flip_cap_hold"
            elif (ts >= orb_end) and (ts < day_end) and orb_ready and (not self._week_target_hit):
                buffer = float(self.breakout_buffer_bps) / 10_000.0
                up_level = float(orb_high) * (1.0 + buffer)
                dn_level = float(orb_low) * (1.0 - buffer)
                breakout_dir = 0
                if np.isfinite(up_level) and last_close > up_level:
                    breakout_dir = 1
                elif np.isfinite(dn_level) and last_close < dn_level:
                    breakout_dir = -1

                if breakout_dir != 0 and (pos_side == 0 or breakout_dir != pos_side):
                    # Size using leverage + liquidation buffer cap.
                    notional_target = 0.0
                    if sizing_mode == "risk":
                        if np.isfinite(atr) and atr > 0 and equity > 0:
                            stop_dist = float(self.stop_atr_mult) * float(atr)
                            risk_usd = float(max(0.0, risk_per_trade)) * float(equity)
                            qty = risk_usd / float(max(stop_dist, 1e-12))
                            notional_target = float(qty) * float(last_close)
                    else:
                        notional_target = float(lev) * float(equity)

                    notional_target = min(float(notional_target), float(max_notional))
                    liq_cap = self._max_notional_for_liq_buffer(
                        equity_alloc=float(self.max_margin_utilization) * float(equity),
                        entry_price=float(last_close),
                        atr=float(atr),
                        mmr=float(mmr),
                        side=int(breakout_dir),
                    )
                    if liq_cap > 0:
                        notional_target = min(notional_target, float(liq_cap), float(max_notional))

                    exp = float(max(0.0, min(1.0, notional_target / float(max_notional))))
                    exp = max(exp, float(min_exposure))
                    desired_exposure = float(exp) * float(breakout_dir)
                    reason = "orb_breakout_entry"
                    self._week_trade_requested = True
                    if pos_side != 0:
                        self._day_flips[s] = flips + 1
            elif self._rebalance_armed_for_week and (not self._week_trade_requested):
                # Fallback weekly heartbeat so every week has at least one fill.
                self._rebalance_armed_for_week = False
                d = dir_sig if dir_sig != 0 else (1 if (mom_s_bps + mom_l_bps) >= 0 else -1)
                exp = float(max(float(min_exposure), float(self.weekly_heartbeat_exposure)))
                desired_exposure = float(exp) * float(d)
                reason = "weekly_heartbeat_entry"
                self._scheduled_stage[s] = "nudge_return"
                self._scheduled_exposure[s] = 0.0
                self._week_trade_requested = True
            elif pos_side == 0 and ts >= orb_end and (not self._week_trade_requested):
                # If we somehow missed the scheduled heartbeat (gaps), do a tiny fallback entry.
                d = dir_sig if dir_sig != 0 else (1 if (mom_s_bps + mom_l_bps) >= 0 else -1)
                exp = float(max(float(min_exposure), float(self.weekly_heartbeat_exposure)))
                desired_exposure = float(exp) * float(d)
                reason = "weekly_fallback_entry"
                self._scheduled_stage[s] = "nudge_return"
                self._scheduled_exposure[s] = 0.0
                self._week_trade_requested = True

            exposures[s] = float(desired_exposure)
            debug[s] = {
                "reason": reason,
                "current_exposure": float(current_exposure),
                "desired_exposure": float(desired_exposure),
                "orb_ready": bool(orb_ready),
                "orb_high": float(orb_high) if np.isfinite(orb_high) else None,
                "orb_low": float(orb_low) if np.isfinite(orb_low) else None,
                "mom_s_bps": float(mom_s_bps),
                "mom_l_bps": float(mom_l_bps),
                "atr_bps": float(atr_bps),
                "lev": float(lev),
                "in_cooldown": bool(in_cooldown),
                "cooldown_bars_left": int(self._cooldown_bars_left.get(s, 0)),
                "week_trade_requested": bool(self._week_trade_requested),
            }

        return StrategyDecision(target_exposures=exposures, debug=debug)
