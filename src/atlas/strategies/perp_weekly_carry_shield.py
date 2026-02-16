from __future__ import annotations

import math
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


def _clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))


def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr1 = (high - low).abs()
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int) -> pd.Series:
    window = int(max(2, window))
    tr = _true_range(high, low, close)
    return tr.rolling(window).mean()


def _efficiency_ratio(close: pd.Series, window: int) -> pd.Series:
    window = int(max(2, window))
    direction = (close - close.shift(window)).abs()
    volatility = close.diff().abs().rolling(window).sum()
    return direction / volatility.replace(0.0, np.nan)


def _choppiness_index(
    high: pd.Series, low: pd.Series, close: pd.Series, window: int
) -> pd.Series:
    window = int(max(2, window))
    tr = _true_range(high, low, close)
    tr_sum = tr.rolling(window).sum()
    hh = high.rolling(window).max()
    ll = low.rolling(window).min()
    rng = (hh - ll).replace(0.0, np.nan)
    ratio = (tr_sum / rng).replace([np.inf, -np.inf], np.nan)
    return (100.0 * np.log10(ratio) / np.log10(float(window))).replace(
        [np.inf, -np.inf], np.nan
    )


def _utc_week_key(ts: pd.Timestamp) -> tuple[int, int]:
    ts = pd.Timestamp(ts)
    if ts.tz is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    iso = ts.isocalendar()
    return int(iso.year), int(iso.week)


@dataclass
class PerpWeeklyCarryShield(Strategy):
    """
    Low-turnover weekly derivatives strategy:
    - Trades once per week (unless risk-off exits trigger sooner).
    - Requires trend/regime confirmation and cost-aware edge.
    - Uses weekly profit lock and weekly loss quarantine to protect consistency.
    """

    name: str = "perp_weekly_carry_shield"
    symbols: tuple[str, ...] = ("BTC-PERP", "ETH-PERP")

    # ---- Weekly schedule ----
    rebalance_weekday_utc: int = 0  # Monday
    rebalance_hour_utc: int = 0
    rebalance_minute_utc: int = 0

    # ---- Regime/signal ----
    atr_window: int = 20
    ema_fast: int = 16
    ema_slow: int = 48
    er_window: int = 20
    choppiness_window: int = 20
    momentum_bars: int = 24
    trend_z_min: float = 0.20
    er_min: float = 0.28
    choppiness_max: float = 62.0
    momentum_threshold_bps: float = 10.0
    min_atr_bps: float = 5.0

    # ---- Cost-aware admission ----
    edge_floor_bps: float = 8.0
    k_cost: float = 2.2
    expected_move_atr_mult: float = 2.5
    slippage_bps: float = 1.25
    taker_fee_bps: float = 3.0

    # ---- Sizing ----
    risk_budget: float = 0.008
    stop_atr_mult: float = 2.8
    max_margin_utilization: float = 0.35
    max_leverage: float = 3.0
    max_positions: int = 2
    max_gross_exposure: float = 1.0
    max_per_symbol_exposure: float = 0.50
    min_trade_notional_usd: float = 20.0
    min_hold_bars: int = 6
    rebalance_exposure_threshold: float = 0.04

    # ---- Risk controls ----
    daily_loss_limit: float = 0.02
    kill_switch: float = 0.12
    weekly_profit_target: float = 0.006
    weekly_loss_limit: float = 0.006
    fallback_trend_floor_exposure: float = 0.0
    fallback_trend_floor_er_min: float = 0.0
    fallback_trend_floor_choppiness_max: float = 100.0
    fallback_trend_floor_min_momentum_bps: float = 0.0
    weekly_heartbeat_exposure: float = 0.01
    weekly_heartbeat_hold_bars: int = 1

    # ---- Internal state ----
    _bars_seen: int = field(default=0, init=False, repr=False)
    _peak_equity: float = field(default=0.0, init=False, repr=False)
    _risk_disabled_day: Optional[object] = field(default=None, init=False, repr=False)
    _risk_disabled_forever: bool = field(default=False, init=False, repr=False)
    _week_key: Optional[tuple[int, int]] = field(default=None, init=False, repr=False)
    _week_start_equity: float = field(default=0.0, init=False, repr=False)
    _week_locked_reason: Optional[str] = field(default=None, init=False, repr=False)
    _last_targets: dict[str, float] = field(default_factory=dict, init=False, repr=False)
    _last_rebalance_bar: int = field(default=0, init=False, repr=False)
    _entry_bar: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _heartbeat_symbol: Optional[str] = field(default=None, init=False, repr=False)
    _heartbeat_return_bar: int = field(default=0, init=False, repr=False)
    _heartbeat_return_exposure: float = field(default=0.0, init=False, repr=False)

    def warmup_bars(self) -> int:
        return (
            max(
                int(self.atr_window) + 3,
                int(self.ema_slow) + 3,
                int(self.er_window) + 3,
                int(self.choppiness_window) + 3,
                int(self.momentum_bars) + 3,
            )
            + 6
        )

    def _maybe_reset_daily_state(self, state: StrategyState) -> None:
        today = pd.Timestamp(state.timestamp).date()
        if self._risk_disabled_day is not None and self._risk_disabled_day != today:
            self._risk_disabled_day = None

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

    def _risk_off(
        self, symbols: list[str], *, reason: str, debug: dict[str, Any]
    ) -> StrategyDecision:
        targets = {s: 0.0 for s in symbols}
        self._last_targets = dict(targets)
        return StrategyDecision(target_exposures=targets, reason=reason, debug=debug)

    def _features(self, df: pd.DataFrame) -> Optional[dict[str, float]]:
        if df is None or df.empty:
            return None
        if not isinstance(df.index, pd.DatetimeIndex):
            return None
        df = df.sort_index()
        if len(df) < self.warmup_bars():
            return None

        close = pd.to_numeric(df["close"], errors="coerce")
        high = pd.to_numeric(df["high"], errors="coerce")
        low = pd.to_numeric(df["low"], errors="coerce")
        if close.isna().all():
            return None
        close = close.dropna()
        if len(close) < self.warmup_bars():
            return None

        c = float(close.iloc[-1])
        if c <= 0:
            return None

        ema_fast = float(
            close.ewm(span=max(2, int(self.ema_fast)), adjust=False).mean().iloc[-1]
        )
        ema_slow = float(
            close.ewm(span=max(2, int(self.ema_slow)), adjust=False).mean().iloc[-1]
        )

        atr_series = _atr(high, low, close, int(self.atr_window))
        atr_now = float(atr_series.iloc[-1]) if len(atr_series) else float("nan")
        atr_bps = float((atr_now / c) * 10_000.0) if np.isfinite(atr_now) and c > 0 else 0.0
        trend_z = float((ema_fast - ema_slow) / atr_now) if np.isfinite(atr_now) and atr_now > 0 else 0.0

        er_series = _efficiency_ratio(close, int(self.er_window))
        er_now = float(er_series.iloc[-1]) if len(er_series) else 0.0
        if not np.isfinite(er_now):
            er_now = 0.0
        er_now = _clamp(er_now, 0.0, 1.0)

        chop_series = _choppiness_index(high, low, close, int(self.choppiness_window))
        chop_now = float(chop_series.iloc[-1]) if len(chop_series) else 100.0
        if not np.isfinite(chop_now):
            chop_now = 100.0

        mb = int(max(2, self.momentum_bars))
        if len(close) <= mb:
            return None
        base = float(close.iloc[-mb - 1])
        mom_bps = float(math.log(c / base) * 10_000.0) if base > 0 and c > 0 else 0.0

        return {
            "close": float(c),
            "atr": float(atr_now) if np.isfinite(atr_now) else 0.0,
            "atr_bps": float(atr_bps),
            "trend_z": float(trend_z),
            "er": float(er_now),
            "choppiness": float(chop_now),
            "mom_bps": float(mom_bps),
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
        self._maybe_reset_daily_state(state)

        equity = float(state.equity)
        if self._peak_equity <= 0:
            self._peak_equity = equity
        self._peak_equity = max(self._peak_equity, equity)
        drawdown = (equity / self._peak_equity - 1.0) if self._peak_equity > 0 else 0.0

        today = pd.Timestamp(state.timestamp).date()
        wk = _utc_week_key(pd.Timestamp(state.timestamp))
        if self._week_key is None or self._week_key != wk:
            self._week_key = wk
            self._week_start_equity = float(equity)
            self._week_locked_reason = None

        week_ret = (
            (equity / float(self._week_start_equity) - 1.0)
            if self._week_start_equity > 0
            else 0.0
        )

        debug: dict[str, Any] = {
            "bars_seen": int(self._bars_seen),
            "drawdown": float(drawdown),
            "day_return": float(state.day_return),
            "week_return": float(week_ret),
            "week_locked_reason": self._week_locked_reason,
        }

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

        if self._week_locked_reason is None and week_ret >= float(self.weekly_profit_target):
            self._week_locked_reason = "profit_lock"
        if self._week_locked_reason is None and week_ret <= -abs(float(self.weekly_loss_limit)):
            self._week_locked_reason = "loss_quarantine"
        if self._week_locked_reason is not None:
            return self._risk_off(symbols, reason=self._week_locked_reason, debug=debug)

        extra = dict(state.extra or {})
        max_notional = float(extra.get("max_position_notional_usd", 0.0) or 0.0) or 1.0
        mmr = float(extra.get("maintenance_margin_rate", 0.05) or 0.05)
        slippage_bps = float(extra.get("slippage_bps", self.slippage_bps) or 0.0)
        taker_fee_bps = float(extra.get("taker_fee_bps", self.taker_fee_bps) or 0.0)
        cost_rt_bps = float(2.0 * (abs(slippage_bps) + abs(taker_fee_bps)))
        min_exp = float(self.min_trade_notional_usd) / float(max_notional)
        min_exp = _clamp(min_exp, 0.0, 1.0)

        features: dict[str, dict[str, float]] = {}
        for s in symbols:
            f = self._features(bars_by_symbol.get(s))
            if f is not None:
                features[s] = f
        if not features:
            return StrategyDecision(target_exposures={s: 0.0 for s in symbols}, reason="warmup", debug=debug)

        due = self._is_rebalance_time(pd.Timestamp(state.timestamp))
        if not self._last_targets:
            due = True

        prev_targets = {s: float(self._last_targets.get(s, 0.0)) for s in symbols}
        targets = dict(prev_targets)

        # Exit logic for held positions even off-cycle when regime fails.
        for s in symbols:
            cur_exp = float(prev_targets.get(s, 0.0))
            if abs(cur_exp) <= 1e-12:
                continue
            f = features.get(s)
            if f is None:
                targets[s] = 0.0
                self._entry_bar.pop(s, None)
                continue
            held = int(self._bars_seen - int(self._entry_bar.get(s, self._bars_seen)))
            can_exit = held >= int(max(1, self.min_hold_bars))
            trend_side = _sign(float(f.get("trend_z", 0.0)))
            now_side = _sign(cur_exp)
            floor_exp = _clamp(
                float(self.fallback_trend_floor_exposure),
                0.0,
                float(self.max_per_symbol_exposure),
            )
            is_floor_pos = floor_exp > 0.0 and abs(cur_exp) <= (floor_exp + 1e-12)
            regime_er_min = (
                float(self.fallback_trend_floor_er_min)
                if is_floor_pos
                else float(self.er_min)
            )
            regime_chop_max = (
                float(self.fallback_trend_floor_choppiness_max)
                if is_floor_pos
                else float(self.choppiness_max)
            )
            regime_ok = (
                float(f.get("er", 0.0)) >= float(self.er_min)
                and float(f.get("choppiness", 100.0)) <= float(self.choppiness_max)
                and float(f.get("atr_bps", 0.0)) >= float(self.min_atr_bps)
            )
            if is_floor_pos:
                regime_ok = (
                    float(f.get("er", 0.0)) >= float(regime_er_min)
                    and float(f.get("choppiness", 100.0)) <= float(regime_chop_max)
                    and float(f.get("atr_bps", 0.0)) >= float(self.min_atr_bps)
                )
                if abs(float(f.get("mom_bps", 0.0))) < float(
                    self.fallback_trend_floor_min_momentum_bps
                ):
                    regime_ok = False
            if can_exit and ((not regime_ok) or (trend_side != 0 and trend_side != now_side)):
                targets[s] = 0.0
                self._entry_bar.pop(s, None)

        if due:
            candidates: list[tuple[str, float, int, float]] = []
            for s, f in features.items():
                trend_z = float(f.get("trend_z", 0.0))
                er = float(f.get("er", 0.0))
                chop = float(f.get("choppiness", 100.0))
                atr_bps = float(f.get("atr_bps", 0.0))
                mom_bps = float(f.get("mom_bps", 0.0))
                if atr_bps < float(self.min_atr_bps):
                    continue
                if er < float(self.er_min):
                    continue
                if chop > float(self.choppiness_max):
                    continue
                if abs(trend_z) < float(self.trend_z_min):
                    continue
                if abs(mom_bps) < float(self.momentum_threshold_bps):
                    continue

                side_score = 0.7 * float(trend_z) + 0.3 * (float(mom_bps) / 100.0)
                side = _sign(side_score)
                if side == 0:
                    continue
                if not bool(state.allow_short) and side < 0:
                    continue

                expected_move_bps = float(self.expected_move_atr_mult) * float(atr_bps)
                required_edge_bps = float(self.edge_floor_bps) + float(self.k_cost) * float(cost_rt_bps)
                if expected_move_bps <= required_edge_bps:
                    continue

                score = abs(side_score) + 0.15 * er - 0.01 * chop
                candidates.append((s, float(score), int(side), float(atr_bps)))

            candidates.sort(key=lambda t: float(t[1]), reverse=True)
            candidates = candidates[: max(1, int(self.max_positions))]

            new_targets = {s: 0.0 for s in symbols}
            if candidates:
                raw = []
                for s, score, side, atr_bps in candidates:
                    raw.append((s, side, max(1e-9, float(score)) / max(1.0, float(atr_bps))))
                denom = float(sum(w for _, _, w in raw))
                gross_cap = float(self.max_gross_exposure)

                # Margin-derived leverage cap in exposure terms.
                margin_exp_cap = float(self.max_per_symbol_exposure)
                if equity > 0 and mmr > 0:
                    margin_cap_notional = float(self.max_margin_utilization) * float(equity) / float(mmr)
                    leverage_cap_notional = float(self.max_leverage) * float(equity)
                    margin_exp_cap = min(
                        float(self.max_per_symbol_exposure),
                        float(margin_cap_notional / max_notional),
                        float(leverage_cap_notional / max_notional),
                    )
                margin_exp_cap = _clamp(margin_exp_cap, 0.0, float(self.max_per_symbol_exposure))

                for s, side, w in raw:
                    f = features[s]
                    atr = float(max(1e-9, f.get("atr", 0.0)))
                    px = float(max(1e-9, f.get("close", 0.0)))
                    risk_frac = float(self.stop_atr_mult) * float(atr / px)
                    risk_frac = max(risk_frac, 1e-6)
                    exp_risk = float(self.risk_budget) / risk_frac
                    exp_weighted = float(gross_cap) * (float(w) / max(1e-9, denom))
                    exp = min(exp_risk, exp_weighted, margin_exp_cap)
                    exp = _clamp(exp, 0.0, float(self.max_per_symbol_exposure))
                    if exp < min_exp:
                        exp = 0.0
                    new_targets[s] = float(exp) * float(side)
                    if abs(new_targets[s]) > 1e-8 and s not in self._entry_bar:
                        self._entry_bar[s] = int(self._bars_seen)
            else:
                floor_exp = _clamp(
                    float(self.fallback_trend_floor_exposure),
                    0.0,
                    float(self.max_per_symbol_exposure),
                )
                floor_picked = False
                if floor_exp >= min_exp:
                    floor_ranked: list[tuple[str, float, int]] = []
                    for s in symbols:
                        f = features.get(s)
                        if f is None:
                            continue
                        er = float(f.get("er", 0.0))
                        chop = float(f.get("choppiness", 100.0))
                        mom = float(f.get("mom_bps", 0.0))
                        atr_bps = float(f.get("atr_bps", 0.0))
                        if atr_bps < float(self.min_atr_bps):
                            continue
                        if er < float(self.fallback_trend_floor_er_min):
                            continue
                        if chop > float(self.fallback_trend_floor_choppiness_max):
                            continue
                        if abs(mom) < float(self.fallback_trend_floor_min_momentum_bps):
                            continue
                        side = _sign(float(f.get("trend_z", 0.0)))
                        if side == 0:
                            side = _sign(mom)
                        if side == 0:
                            continue
                        if not bool(state.allow_short) and side < 0:
                            continue
                        # Favor cleaner directional states for fallback floor exposure.
                        rank = abs(float(f.get("trend_z", 0.0))) + 0.5 * er - 0.01 * chop
                        floor_ranked.append((s, float(rank), int(side)))
                    floor_ranked.sort(key=lambda t: float(t[1]), reverse=True)
                    if floor_ranked:
                        fs, _, fside = floor_ranked[0]
                        new_targets[fs] = float(floor_exp) * float(fside)
                        self._entry_bar[fs] = int(self._bars_seen)
                        self._heartbeat_symbol = None
                        self._heartbeat_return_exposure = 0.0
                        self._heartbeat_return_bar = 0
                        floor_picked = True

                if not floor_picked:
                    # Weekly heartbeat: ensure at least one small trade if no candidate passes.
                    hb_sym = symbols[0]
                    hb_feat = features.get(hb_sym)
                    if hb_feat is not None:
                        side = _sign(float(hb_feat.get("mom_bps", 0.0)))
                        if side == 0:
                            side = 1
                        if not bool(state.allow_short) and side < 0:
                            side = 1
                        hb_exp = _clamp(float(self.weekly_heartbeat_exposure), 0.0, float(self.max_per_symbol_exposure))
                        if hb_exp >= min_exp:
                            new_targets[hb_sym] = float(hb_exp) * float(side)
                            self._entry_bar[hb_sym] = int(self._bars_seen)
                            self._heartbeat_symbol = hb_sym
                            self._heartbeat_return_exposure = float(new_targets[hb_sym])
                            self._heartbeat_return_bar = int(self._bars_seen) + int(max(1, self.weekly_heartbeat_hold_bars))

            targets = new_targets
            self._last_rebalance_bar = int(self._bars_seen)

        # Revert one-bar heartbeat.
        if (
            self._heartbeat_symbol
            and int(self._bars_seen) >= int(self._heartbeat_return_bar)
            and abs(float(self._heartbeat_return_exposure)) > 1e-12
        ):
            hb = str(self._heartbeat_symbol)
            if hb in targets:
                targets[hb] = 0.0
                self._entry_bar.pop(hb, None)
            self._heartbeat_symbol = None
            self._heartbeat_return_exposure = 0.0
            self._heartbeat_return_bar = 0

        for s in symbols:
            prev = float(prev_targets.get(s, 0.0))
            now = float(targets.get(s, 0.0))
            if prev != 0.0 and now != 0.0 and _sign(prev) == _sign(now):
                if abs(now - prev) < float(self.rebalance_exposure_threshold):
                    targets[s] = prev

        self._last_targets = {s: float(targets.get(s, 0.0)) for s in symbols}
        debug["due"] = bool(due)
        debug["cost_rt_bps"] = float(cost_rt_bps)
        return StrategyDecision(
            target_exposures=self._last_targets,
            reason="rebalance" if due else "hold",
            debug=debug,
        )
