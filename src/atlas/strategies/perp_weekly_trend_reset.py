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


def _utc_week_key(ts: pd.Timestamp) -> tuple[int, int]:
    ts = pd.Timestamp(ts)
    if ts.tz is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    iso = ts.isocalendar()
    return int(iso.year), int(iso.week)


def _estimate_bars_per_day(index: pd.Index, *, default: int = 288) -> int:
    """
    Estimate bars/day from timestamp spacing so the strategy behaves consistently
    across 5m/15m/1h/4h datasets.
    """
    try:
        ts = pd.DatetimeIndex(index)
        if len(ts) < 4:
            return int(default)
        # Use recent spacing for robustness to older data gaps.
        deltas = ts.to_series().diff().dropna().dt.total_seconds()
        if deltas.empty:
            return int(default)
        step_s = float(deltas.tail(32).median())
        if not np.isfinite(step_s) or step_s <= 0:
            return int(default)
        bars = int(round(86_400.0 / step_s))
        return int(max(1, min(10_000, bars)))
    except Exception:
        return int(default)


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
class PerpWeeklyTrendReset(Strategy):
    """
    Weekly trend strategy for perps:
    - Compute a slow trend signal (lookback momentum + EMA confirmation).
    - Once per week, rebalance to the current trend target exposure.
    - If already at the weekly target, force a tiny "nudge" round-trip so the bot is verifiably trading weekly.
    - If trend is neutral, optionally run a tiny "heartbeat" position that exits after N bars.
    - Use leverage-aware sizing with margin caps and liquidation buffers.
    - In-position risk: ATR stop + trailing stop + optional time stop.
    """

    name: str = "perp_weekly_trend_reset"
    symbols: tuple[str, ...] = ("BTC-PERP",)

    # --- signal ---
    lookback_days: int = 14
    momentum_threshold_bps: float = 0.0
    ema_fast: int = 12
    ema_slow: int = 48
    require_ema_confirmation: bool = False

    # --- risk / sizing ---
    target_leverage: float = 8.0
    max_margin_utilization: float = 0.80
    maintenance_margin_rate: float = 0.05
    stop_atr_mult: float = 2.5
    trail_atr_mult: float = 4.0
    atr_window: int = 14
    min_liq_buffer_atr: float = 4.0
    use_stops: bool = False

    # --- weekly schedule (UTC) ---
    rebalance_weekday_utc: int = 0  # Monday
    rebalance_hour_utc: int = 0
    rebalance_minute_utc: int = 5
    weekly_nudge_exposure: float = 0.002
    min_trade_notional_usd: float = 10.0

    # --- heartbeat trade when trend is neutral ---
    heartbeat_exposure: float = 0.03
    heartbeat_hold_bars: int = 12

    # --- optional position time stop ---
    max_hold_bars: Optional[int] = None

    # --- internal state ---
    _bars_seen: int = field(default=0, init=False, repr=False)
    _week_key: Optional[tuple[int, int]] = field(default=None, init=False, repr=False)
    _rebalance_armed_for_week: bool = field(default=False, init=False, repr=False)
    _weekly_cycle_done: bool = field(default=False, init=False, repr=False)

    _entry_price: dict[str, float] = field(default_factory=dict, init=False, repr=False)
    _trail_extreme: dict[str, float] = field(default_factory=dict, init=False, repr=False)
    _scheduled_stage: dict[str, str] = field(default_factory=dict, init=False, repr=False)
    _scheduled_dir: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _scheduled_exposure: dict[str, float] = field(default_factory=dict, init=False, repr=False)
    _scheduled_exit_at_bar: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _last_pos_side: dict[str, int] = field(default_factory=dict, init=False, repr=False)

    def warmup_bars(self) -> int:
        # Need EMA slow + ATR warmup. Momentum lookback will clamp to available history early on.
        return max(self.ema_slow + 2, self.atr_window + 2, 50)

    def _is_rebalance_time(self, ts: pd.Timestamp) -> bool:
        ts = pd.Timestamp(ts)
        if ts.tz is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        if int(ts.dayofweek) != int(self.rebalance_weekday_utc):
            return False
        if int(ts.hour) < int(self.rebalance_hour_utc):
            return False
        if int(ts.hour) > int(self.rebalance_hour_utc):
            return True
        return int(ts.minute) >= int(self.rebalance_minute_utc)

    def _trend_dir(self, close: pd.Series) -> tuple[int, float, int]:
        """
        Returns (dir, momentum_bps, ema_dir).
        """
        close = close.astype(float)
        if len(close) < 3:
            return 0, 0.0, 0

        bars_per_day = _estimate_bars_per_day(close.index, default=288)
        lb = int(max(2, int(self.lookback_days) * bars_per_day))
        lb = min(lb, int(len(close) - 2))
        if lb < 2:
            return 0, 0.0, 0

        c0 = float(close.iloc[-lb - 1])
        c1 = float(close.iloc[-1])
        if c0 <= 0 or not np.isfinite(c0) or not np.isfinite(c1):
            return 0, 0.0, 0

        mom_bps = float((c1 / c0 - 1.0) * 10_000.0)
        ema_f = float(_ema(close, self.ema_fast).iloc[-1])
        ema_s = float(_ema(close, self.ema_slow).iloc[-1])
        ema_dir = 1 if ema_f > ema_s else -1 if ema_f < ema_s else 0

        thr = float(abs(self.momentum_threshold_bps))
        if abs(mom_bps) <= thr:
            return 0, mom_bps, int(ema_dir)
        d = 1 if mom_bps > 0 else -1
        if bool(self.require_ema_confirmation) and ema_dir != d:
            return 0, mom_bps, int(ema_dir)
        return int(d), mom_bps, int(ema_dir)

    def target_exposures(
        self, bars_by_symbol: dict[str, pd.DataFrame], state: StrategyState
    ) -> StrategyDecision:
        symbols = [s for s in self.symbols if s in bars_by_symbol]
        if not symbols:
            symbols = sorted(bars_by_symbol.keys())
        targets: dict[str, float] = {s: 0.0 for s in symbols}
        if not symbols:
            return StrategyDecision(target_exposures=targets, reason="no_symbols")

        self._bars_seen += 1

        extra = dict(state.extra or {})
        max_notional = float(extra.get("max_position_notional_usd") or 0.0)
        if max_notional <= 0:
            return StrategyDecision(target_exposures=targets, reason="no_max_position_notional")
        mmr = float(extra.get("maintenance_margin_rate") or self.maintenance_margin_rate)

        ts = pd.Timestamp(state.timestamp)
        wk = _utc_week_key(ts)
        if self._week_key is None or self._week_key != wk:
            self._week_key = wk
            self._rebalance_armed_for_week = False
            self._weekly_cycle_done = False

        debug: dict[str, Any] = {"week_key": wk, "bars_seen": int(self._bars_seen)}

        for s in symbols:
            df = bars_by_symbol.get(s)
            if df is None or len(df) < 3:
                debug[s] = {"reason": "too_few_bars"}
                continue
            if not df.index.is_monotonic_increasing:
                df = df.sort_index()

            close = df["close"].astype(float)
            high = df["high"].astype(float)
            low = df["low"].astype(float)

            last_close = float(close.iloc[-1])
            if not np.isfinite(last_close) or last_close <= 0:
                debug[s] = {"reason": "bad_price"}
                continue

            atr = float("nan")
            if len(df) >= max(3, int(self.atr_window) + 2):
                atr_series = _atr(high, low, close, self.atr_window)
                atr = float(atr_series.iloc[-1]) if len(atr_series) else float("nan")

            pos_qty = float(state.positions.get(s, 0.0) or 0.0)
            pos_side = _sign(pos_qty)
            hold_bars = int(state.holding_bars.get(s, 0) or 0)

            prev_side = int(self._last_pos_side.get(s, 0))
            if prev_side == 0 and pos_side != 0:
                self._entry_price[s] = last_close
                self._trail_extreme[s] = last_close
            if prev_side != 0 and pos_side == 0:
                self._entry_price.pop(s, None)
                self._trail_extreme.pop(s, None)
                self._scheduled_stage.pop(s, None)
                self._scheduled_exit_at_bar.pop(s, None)
            self._last_pos_side[s] = int(pos_side)

            trend_dir, mom_bps, ema_dir = self._trend_dir(close)
            if not np.isfinite(atr) or atr <= 0:
                # If ATR isn't warmed up yet, fall back to heartbeat mode.
                trend_dir = 0

            # Ensure we *do something* each week: arm the rebalance when the schedule time arrives.
            if (not self._weekly_cycle_done) and (not self._rebalance_armed_for_week) and self._is_rebalance_time(ts):
                self._rebalance_armed_for_week = True

            stage = str(self._scheduled_stage.get(s, "") or "")
            if stage == "nudge_return":
                targets[s] = float(self._scheduled_exposure.get(s, 0.0))
                self._scheduled_stage.pop(s, None)
                self._scheduled_exposure.pop(s, None)
                debug[s] = {"stage": "nudge_return", "target": float(targets[s])}
                continue

            # Heartbeat exit.
            exit_at = int(self._scheduled_exit_at_bar.get(s, 0) or 0)
            if pos_side != 0 and exit_at and self._bars_seen >= exit_at:
                targets[s] = 0.0
                self._scheduled_exit_at_bar.pop(s, None)
                debug[s] = {"reason": "heartbeat_exit"}
                continue

            # Weekly rebalance: set the new weekly target exposure. If already at target, do a
            # tiny "nudge" trade and then revert next bar to guarantee a weekly fill.
            if self._rebalance_armed_for_week and not self._weekly_cycle_done:
                self._weekly_cycle_done = True
                self._rebalance_armed_for_week = False

                current_exposure = float((pos_qty * last_close) / max_notional) if max_notional > 0 else 0.0
                min_exposure = float(self.min_trade_notional_usd) / float(max_notional) if max_notional > 0 else 0.0

                desired_exposure = 0.0
                desired_reason = "weekly_target_flat"
                desired_exit_at: Optional[int] = None

                if trend_dir == 0:
                    # Neutral regime: prefer being flat, but if already flat run a tiny heartbeat
                    # position that exits after N bars.
                    if pos_side == 0:
                        d = 1 if mom_bps >= 0 else -1
                        exp = float(max(0.0, min(1.0, float(self.heartbeat_exposure))))
                        exp = max(exp, float(min_exposure))
                        desired_exposure = float(exp) * float(d)
                        desired_reason = "weekly_heartbeat_entry"
                        desired_exit_at = self._bars_seen + int(self.heartbeat_hold_bars)
                    else:
                        desired_exposure = 0.0
                        desired_reason = "weekly_exit_neutral"
                else:
                    d = int(trend_dir)
                    exp = 0.0
                    equity = float(state.equity)
                    if equity > 0:
                        notional_target = float(self.target_leverage) * equity
                        margin_cap = float(self.max_margin_utilization) * equity / float(max(mmr, 1e-9))
                        notional_target = min(notional_target, margin_cap, float(max_notional))
                        liq_cap = _max_notional_for_liq_buffer(
                            equity_alloc=float(self.max_margin_utilization) * equity,
                            entry_price=last_close,
                            atr=atr,
                            mmr=mmr,
                            stop_atr_mult=float(self.stop_atr_mult),
                            liq_buffer_atr=float(self.min_liq_buffer_atr),
                            side=int(d),
                        )
                        if liq_cap > 0:
                            notional_target = min(notional_target, float(liq_cap), float(max_notional))
                        exp = float(max(0.0, min(1.0, notional_target / float(max_notional))))
                        exp = max(exp, float(min_exposure))
                    desired_exposure = float(exp) * float(d)
                    desired_reason = "weekly_target_trend"

                # Apply heartbeat exit schedule if relevant.
                if desired_exit_at is not None:
                    self._scheduled_exit_at_bar[s] = int(desired_exit_at)
                else:
                    self._scheduled_exit_at_bar.pop(s, None)

                # If we're already (approximately) at the desired exposure, force a small nudge
                # so the engine will emit at least one fill this week.
                delta_to_target = float(desired_exposure - current_exposure)
                if abs(delta_to_target) >= float(min_exposure):
                    targets[s] = float(desired_exposure)
                    debug[s] = {
                        "reason": desired_reason,
                        "current_exposure": float(current_exposure),
                        "desired_exposure": float(desired_exposure),
                        "mom_bps": float(mom_bps),
                        "ema_dir": int(ema_dir),
                    }
                    continue

                delta_exposure = float(abs(self.weekly_nudge_exposure))
                delta_exposure = max(delta_exposure, float(min_exposure))
                nudge_dir = _sign(desired_exposure) or (pos_side if pos_side != 0 else 1)
                nudge_exposure = float(desired_exposure + float(nudge_dir) * float(delta_exposure))
                nudge_exposure = max(-1.0, min(1.0, nudge_exposure))
                targets[s] = float(nudge_exposure)
                self._scheduled_stage[s] = "nudge_return"
                self._scheduled_exposure[s] = float(desired_exposure)
                debug[s] = {
                    "reason": "weekly_nudge",
                    "at_target": True,
                    "desired_reason": desired_reason,
                    "current_exposure": float(current_exposure),
                    "desired_exposure": float(desired_exposure),
                    "nudge_exposure": float(nudge_exposure),
                    "mom_bps": float(mom_bps),
                    "ema_dir": int(ema_dir),
                }
                continue

            # In-position risk management.
            if pos_side != 0:
                entry_px = float(self._entry_price.get(s, last_close))
                ext = float(self._trail_extreme.get(s, last_close))
                if bool(self.use_stops):
                    if pos_side > 0:
                        self._trail_extreme[s] = max(ext, last_close)
                        trail_stop = self._trail_extreme[s] - float(self.trail_atr_mult) * atr
                        hard_stop = entry_px - float(self.stop_atr_mult) * atr
                        stop_hit = last_close <= min(trail_stop, hard_stop)
                    else:
                        self._trail_extreme[s] = min(ext, last_close)
                        trail_stop = self._trail_extreme[s] + float(self.trail_atr_mult) * atr
                        hard_stop = entry_px + float(self.stop_atr_mult) * atr
                        stop_hit = last_close >= max(trail_stop, hard_stop)

                    if stop_hit:
                        targets[s] = 0.0
                        debug[s] = {"reason": "stop", "pos_side": pos_side}
                        continue

                if self.max_hold_bars is not None and hold_bars >= int(self.max_hold_bars):
                    targets[s] = 0.0
                    debug[s] = {"reason": "time_stop", "hold_bars": hold_bars}
                    continue

                # Hold constant quantity until scheduled weekly reset, but de-risk if margin
                # utilization breaches the configured cap (to avoid liquidation spirals).
                equity = float(state.equity)
                current_exposure = float((pos_qty * last_close) / max_notional) if max_notional > 0 else 0.0
                current_notional = float(abs(pos_qty) * last_close)
                margin_used = current_notional * float(max(mmr, 1e-12))
                util = (margin_used / equity) if equity > 0 else float("inf")

                if equity > 0 and util > float(self.max_margin_utilization) and mmr > 0:
                    target_notional = (float(self.max_margin_utilization) * equity) / float(mmr)
                    target_notional = min(target_notional, float(max_notional))
                    target_exposure = float(target_notional / float(max_notional)) if max_notional > 0 else 0.0
                    targets[s] = float(target_exposure) * float(pos_side)
                    debug[s] = {
                        "reason": "de_risk_margin",
                        "pos_side": pos_side,
                        "util": float(util),
                        "util_cap": float(self.max_margin_utilization),
                        "target_exposure": float(target_exposure),
                    }
                else:
                    targets[s] = float(current_exposure)
                    debug[s] = {"reason": "hold", "pos_side": pos_side, "trend_dir": trend_dir}
                continue

            # Otherwise stay flat until the next weekly schedule.
            targets[s] = 0.0
            debug[s] = {"reason": "flat_wait", "trend_dir": trend_dir, "mom_bps": mom_bps, "ema_dir": ema_dir}

        return StrategyDecision(target_exposures=targets, debug=debug)
