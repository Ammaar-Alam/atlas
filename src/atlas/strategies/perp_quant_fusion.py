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


def _clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))


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


def _choppiness_index(
    high: pd.Series, low: pd.Series, close: pd.Series, window: int
) -> pd.Series:
    window = int(window)
    if window <= 1:
        return pd.Series(np.nan, index=close.index)
    tr = _true_range(high, low, close)
    tr_sum = tr.rolling(window).sum()
    hh = high.rolling(window).max()
    ll = low.rolling(window).min()
    rng = (hh - ll).replace(0.0, np.nan)
    ratio = (tr_sum / rng).replace([np.inf, -np.inf], np.nan)
    return (100.0 * np.log10(ratio) / np.log10(float(window))).replace(
        [np.inf, -np.inf], np.nan
    )


def _donchian(high: pd.Series, low: pd.Series, window: int) -> tuple[float, float]:
    window = int(window)
    if window <= 0 or len(high) < window + 1:
        return float("nan"), float("nan")
    hh = float(high.iloc[-window - 1 : -1].max())
    ll = float(low.iloc[-window - 1 : -1].min())
    return hh, ll


def _utc_week_key(ts: pd.Timestamp) -> tuple[int, int]:
    ts = pd.Timestamp(ts)
    if ts.tz is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    iso = ts.isocalendar()
    return int(iso.year), int(iso.week)


@dataclass
class PerpQuantFusion(Strategy):
    """
    Multi-symbol perpetual futures trend strategy.

    Signal stack:
    - EMA trend direction / strength
    - ER + choppiness regime filters
    - Donchian breakout confirmation
    - Cost-aware edge admission using slippage + taker fee from `state.extra`

    Risk stack:
    - ATR stop-distance sizing (risk budget)
    - Per-symbol and gross exposure caps
    - Daily loss limit + drawdown kill switch
    - Optional weekly profit lock and optional weekly heartbeat
    """

    name: str = "perp_quant_fusion"
    symbols: tuple[str, ...] = ("BTC-PERP", "ETH-PERP")

    # --- Signal / regime ---
    atr_window: int = 14
    ema_fast: int = 20
    ema_slow: int = 60
    er_window: int = 20
    choppiness_window: int = 14
    breakout_window: int = 20
    breakout_buffer_bps: float = 2.0
    trend_z_min: float = 0.20
    er_min: float = 0.30
    er_exit_min: float = 0.18
    choppiness_max: float = 62.0
    choppiness_exit_max: float = 68.0
    min_atr_bps: float = 5.0

    # --- Cost-aware admission ---
    edge_floor_bps: float = 6.0
    k_cost: float = 1.8
    slippage_bps: float = 1.25  # fallback if engine does not provide `state.extra`
    taker_fee_bps: float = 3.0  # fallback if engine does not provide `state.extra`

    # --- Sizing / caps ---
    risk_budget: float = 0.02
    stop_atr_mult: float = 2.2
    max_positions: int = 3
    max_gross_exposure: float = 1.0
    max_per_symbol_exposure: float = 0.50
    rebalance_exposure_threshold: float = 0.03
    min_trade_notional_usd: float = 20.0
    min_hold_bars: int = 3
    flip_confirm_bars: int = 2

    # --- Risk-off controls ---
    daily_loss_limit: float = 0.025
    kill_switch: float = 0.12

    # --- Weekly profit lock (optional) ---
    enable_weekly_profit_lock: bool = False
    weekly_profit_target: float = 0.02
    weekly_lock_risk_scale: float = 0.25

    # --- Weekly heartbeat (optional; at-least-weekly activity nudge) ---
    enable_weekly_heartbeat: bool = False
    heartbeat_weekday_utc: int = 0  # Monday
    heartbeat_hour_utc: int = 0
    heartbeat_minute_utc: int = 5
    heartbeat_exposure: float = 0.01
    heartbeat_hold_bars: int = 1

    # --- Internal state ---
    _bars_seen: int = field(default=0, init=False, repr=False)
    _peak_equity: float = field(default=0.0, init=False, repr=False)
    _risk_disabled_day: Optional[object] = field(default=None, init=False, repr=False)
    _risk_disabled_forever: bool = field(default=False, init=False, repr=False)
    _flip_counter: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _last_pos_side: dict[str, int] = field(default_factory=dict, init=False, repr=False)

    _week_key: Optional[tuple[int, int]] = field(default=None, init=False, repr=False)
    _week_start_equity: float = field(default=0.0, init=False, repr=False)
    _week_target_hit: bool = field(default=False, init=False, repr=False)
    _week_activity_seen: bool = field(default=False, init=False, repr=False)
    _week_heartbeat_fired: bool = field(default=False, init=False, repr=False)
    _heartbeat_stage: Optional[str] = field(default=None, init=False, repr=False)
    _heartbeat_symbol: Optional[str] = field(default=None, init=False, repr=False)
    _heartbeat_return_exposure: float = field(default=0.0, init=False, repr=False)
    _heartbeat_return_at_bar: int = field(default=0, init=False, repr=False)

    def warmup_bars(self) -> int:
        return (
            max(
                int(self.atr_window) + 2,
                int(self.ema_slow) + 2,
                int(self.er_window) + 2,
                int(self.choppiness_window) + 2,
                int(self.breakout_window) + 2,
            )
            + 3
        )

    def _clear_heartbeat_schedule(self) -> None:
        self._heartbeat_stage = None
        self._heartbeat_symbol = None
        self._heartbeat_return_exposure = 0.0
        self._heartbeat_return_at_bar = 0

    def _maybe_reset_daily_state(self, state: StrategyState) -> None:
        today = pd.Timestamp(state.timestamp).date()
        if self._risk_disabled_day is not None and self._risk_disabled_day != today:
            self._risk_disabled_day = None

    def _is_heartbeat_time(self, ts: pd.Timestamp) -> bool:
        ts = pd.Timestamp(ts)
        if ts.tz is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        if int(ts.dayofweek) != int(self.heartbeat_weekday_utc):
            return False
        if int(ts.hour) != int(self.heartbeat_hour_utc):
            return False
        return int(ts.minute) >= int(self.heartbeat_minute_utc)

    def _risk_off(
        self, symbols: list[str], *, reason: str, debug: dict[str, Any]
    ) -> StrategyDecision:
        return StrategyDecision(
            target_exposures={s: 0.0 for s in symbols}, reason=reason, debug=debug
        )

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
        self._maybe_reset_daily_state(state)

        equity = float(state.equity)
        if self._peak_equity <= 0:
            self._peak_equity = equity
        self._peak_equity = max(self._peak_equity, equity)
        drawdown = (equity / self._peak_equity - 1.0) if self._peak_equity > 0 else 0.0

        if self._risk_disabled_forever or drawdown <= -abs(float(self.kill_switch)):
            self._risk_disabled_forever = True
            self._clear_heartbeat_schedule()
            return self._risk_off(
                symbols,
                reason="kill_switch",
                debug={"drawdown": float(drawdown), "kill_switch": float(self.kill_switch)},
            )

        today = pd.Timestamp(state.timestamp).date()
        if float(state.day_return) <= -abs(float(self.daily_loss_limit)):
            self._risk_disabled_day = today
            self._clear_heartbeat_schedule()
            return self._risk_off(
                symbols,
                reason="daily_loss_limit",
                debug={
                    "day_return": float(state.day_return),
                    "daily_loss_limit": float(self.daily_loss_limit),
                },
            )

        if self._risk_disabled_day == today:
            self._clear_heartbeat_schedule()
            return self._risk_off(
                symbols,
                reason="risk_disabled_day",
                debug={"risk_disabled_day": str(self._risk_disabled_day)},
            )

        extra = dict(state.extra or {})
        max_notional = float(extra.get("max_position_notional_usd") or 0.0)
        if max_notional <= 0:
            return StrategyDecision(
                target_exposures=targets, reason="no_max_position_notional"
            )

        slippage_bps = float(self.slippage_bps)
        if extra.get("slippage_bps") is not None:
            slippage_bps = float(extra.get("slippage_bps") or 0.0)
        taker_fee_bps = float(self.taker_fee_bps)
        if extra.get("taker_fee_bps") is not None:
            taker_fee_bps = float(extra.get("taker_fee_bps") or 0.0)

        cost_rt_bps = float(
            2.0 * (abs(float(slippage_bps)) + abs(float(taker_fee_bps)))
        )
        required_edge_bps = float(self.edge_floor_bps) + float(self.k_cost) * cost_rt_bps

        ts = pd.Timestamp(state.timestamp)
        wk = _utc_week_key(ts)
        if self._week_key is None or self._week_key != wk:
            self._week_key = wk
            self._week_start_equity = equity
            self._week_target_hit = False
            self._week_activity_seen = False
            self._week_heartbeat_fired = False
            self._clear_heartbeat_schedule()

        week_ret = (
            (equity / float(self._week_start_equity) - 1.0)
            if self._week_start_equity > 0
            else 0.0
        )
        if (
            bool(self.enable_weekly_profit_lock)
            and (not self._week_target_hit)
            and week_ret >= abs(float(self.weekly_profit_target))
        ):
            self._week_target_hit = True

        max_per_symbol_cap = _clamp(
            abs(float(self.max_per_symbol_exposure)), 0.0, 1.0
        )
        max_gross_cap = max(0.0, float(self.max_gross_exposure))

        per: dict[str, dict[str, float | int | str]] = {}
        desired_dir: dict[str, int] = {s: 0 for s in symbols}
        reason_tag: dict[str, str] = {s: "flat" for s in symbols}
        score: dict[str, float] = {s: 0.0 for s in symbols}
        current_exposure: dict[str, float] = {s: 0.0 for s in symbols}

        for s in symbols:
            df = bars_by_symbol.get(s)
            if df is None or len(df) < self.warmup_bars():
                reason_tag[s] = "warmup"
                continue

            if not df.index.is_monotonic_increasing:
                df = df.sort_index()

            close = df["close"].astype(float)
            high = df["high"].astype(float)
            low = df["low"].astype(float)
            last_close = float(close.iloc[-1])
            if not np.isfinite(last_close) or last_close <= 0:
                reason_tag[s] = "bad_price"
                continue

            pos_qty = float(state.positions.get(s, 0.0) or 0.0)
            pos_side = _sign(pos_qty)
            hold_bars = int(state.holding_bars.get(s, 0) or 0)
            cur_exp = (
                float(pos_qty * last_close / max_notional) if max_notional > 0 else 0.0
            )
            current_exposure[s] = float(cur_exp)

            prev_side = int(self._last_pos_side.get(s, 0))
            if prev_side != pos_side:
                self._flip_counter[s] = 0
            self._last_pos_side[s] = int(pos_side)

            atr_series = _atr(high, low, close, self.atr_window)
            atr = float(atr_series.iloc[-1]) if len(atr_series) else float("nan")
            if not np.isfinite(atr) or atr <= 0:
                reason_tag[s] = "bad_atr"
                continue

            ema_f = float(_ema(close, self.ema_fast).iloc[-1])
            ema_s = float(_ema(close, self.ema_slow).iloc[-1])
            trend_strength = float((ema_f - ema_s) / max(1e-12, atr))
            trend_dir = (
                1
                if trend_strength >= float(self.trend_z_min)
                else -1 if trend_strength <= -float(self.trend_z_min) else 0
            )

            er_series = _efficiency_ratio(close, self.er_window)
            er = float(er_series.iloc[-1]) if len(er_series) else float("nan")

            chop_series = _choppiness_index(high, low, close, self.choppiness_window)
            chop = float(chop_series.iloc[-1]) if len(chop_series) else float("nan")

            hh, ll = _donchian(high, low, self.breakout_window)
            buf = float(self.breakout_buffer_bps) / 10_000.0
            breakout_dir = 0
            if np.isfinite(hh) and last_close > hh * (1.0 + buf):
                breakout_dir = 1
            elif np.isfinite(ll) and last_close < ll * (1.0 - buf):
                breakout_dir = -1

            atr_bps = float((atr / last_close) * 10_000.0)
            breakout_excess_bps = 0.0
            if breakout_dir > 0 and np.isfinite(hh) and hh > 0:
                breakout_excess_bps = float(
                    ((last_close - hh * (1.0 + buf)) / last_close) * 10_000.0
                )
            elif breakout_dir < 0 and np.isfinite(ll) and ll > 0:
                breakout_excess_bps = float(
                    ((ll * (1.0 - buf) - last_close) / last_close) * 10_000.0
                )

            er_clean = float(er) if np.isfinite(er) else 0.0
            edge_bps = float(
                max(0.0, breakout_excess_bps)
                + 0.35 * abs(trend_strength) * atr_bps
                + 0.20 * max(0.0, er_clean) * atr_bps
            )
            net_edge_bps = float(edge_bps - float(self.k_cost) * cost_rt_bps)

            per[s] = {
                "last_close": float(last_close),
                "atr": float(atr),
                "atr_bps": float(atr_bps),
                "ema_f": float(ema_f),
                "ema_s": float(ema_s),
                "trend_strength": float(trend_strength),
                "trend_dir": int(trend_dir),
                "er": float(er) if np.isfinite(er) else float("nan"),
                "choppiness": float(chop) if np.isfinite(chop) else float("nan"),
                "breakout_dir": int(breakout_dir),
                "breakout_excess_bps": float(breakout_excess_bps),
                "edge_bps": float(edge_bps),
                "net_edge_bps": float(net_edge_bps),
            }

            er_ok = np.isfinite(er) and float(er) >= float(self.er_min)
            chop_ok = np.isfinite(chop) and float(chop) <= float(self.choppiness_max)
            atr_ok = atr_bps >= float(self.min_atr_bps)
            align_ok = (trend_dir != 0) and (breakout_dir == trend_dir)
            edge_ok = (edge_bps >= required_edge_bps) and (net_edge_bps > 0.0)

            if pos_side != 0:
                if (not bool(state.allow_short)) and pos_side < 0:
                    desired_dir[s] = 0
                    reason_tag[s] = "short_blocked"
                else:
                    opposite_signal = (
                        trend_dir == -pos_side
                        and breakout_dir == -pos_side
                        and er_ok
                        and chop_ok
                        and edge_ok
                    )
                    if opposite_signal:
                        self._flip_counter[s] = int(self._flip_counter.get(s, 0)) + 1
                    else:
                        self._flip_counter[s] = 0

                    exit_filter = (
                        (np.isfinite(er) and float(er) < float(self.er_exit_min))
                        or (
                            np.isfinite(chop)
                            and float(chop) > float(self.choppiness_exit_max)
                        )
                        or (
                            trend_dir != 0
                            and trend_dir != pos_side
                            and breakout_dir == -pos_side
                        )
                    )

                    if (
                        hold_bars >= int(self.min_hold_bars)
                        and int(self._flip_counter.get(s, 0))
                        >= int(self.flip_confirm_bars)
                    ):
                        desired_dir[s] = -pos_side
                        reason_tag[s] = "flip"
                    elif hold_bars >= int(self.min_hold_bars) and bool(exit_filter):
                        desired_dir[s] = 0
                        reason_tag[s] = "filter_exit"
                    else:
                        desired_dir[s] = pos_side
                        reason_tag[s] = "hold"
            else:
                self._flip_counter[s] = 0
                if not er_ok:
                    desired_dir[s] = 0
                    reason_tag[s] = "gate_er"
                elif not chop_ok:
                    desired_dir[s] = 0
                    reason_tag[s] = "gate_chop"
                elif not atr_ok:
                    desired_dir[s] = 0
                    reason_tag[s] = "gate_atr"
                elif not align_ok:
                    desired_dir[s] = 0
                    reason_tag[s] = "gate_breakout"
                elif not edge_ok:
                    desired_dir[s] = 0
                    reason_tag[s] = "gate_cost"
                elif trend_dir < 0 and (not bool(state.allow_short)):
                    desired_dir[s] = 0
                    reason_tag[s] = "short_blocked"
                else:
                    desired_dir[s] = int(trend_dir)
                    reason_tag[s] = "entry"

            if desired_dir[s] != 0:
                strength_term = max(0.25, abs(float(trend_strength)))
                er_term = max(0.10, min(1.0, er_clean))
                score[s] = float(max(0.0, net_edge_bps) * strength_term * er_term)

        active = [s for s in symbols if desired_dir.get(s, 0) != 0 and s in per]
        if int(self.max_positions) > 0 and len(active) > int(self.max_positions):
            active_sorted = sorted(
                active,
                key=lambda s: (
                    1 if _sign(float(state.positions.get(s, 0.0) or 0.0)) != 0 else 0,
                    float(score.get(s, 0.0)),
                ),
                reverse=True,
            )
            keep = set(active_sorted[: int(self.max_positions)])
            for s in active:
                if s not in keep:
                    desired_dir[s] = 0
                    reason_tag[s] = "rank_cut"
            active = [s for s in active_sorted if s in keep]

        if equity <= 0:
            return self._risk_off(
                symbols, reason="no_equity", debug={"equity": float(equity)}
            )

        if max_per_symbol_cap <= 0 or max_gross_cap <= 0:
            return StrategyDecision(
                target_exposures={s: 0.0 for s in symbols},
                reason="exposure_caps_zero",
                debug={
                    "max_per_symbol_cap": float(max_per_symbol_cap),
                    "max_gross_cap": float(max_gross_cap),
                },
            )

        base_targets: dict[str, float] = {s: 0.0 for s in symbols}
        if active and float(self.risk_budget) > 0:
            raw_scores = np.array(
                [max(0.0, float(score.get(s, 0.0))) for s in active], dtype=float
            )
            if float(raw_scores.sum()) <= 0:
                weights = np.ones_like(raw_scores) / max(1.0, float(len(raw_scores)))
            else:
                weights = raw_scores / raw_scores.sum()

            total_risk_usd = max(0.0, float(self.risk_budget)) * float(equity)
            min_trade_exp = (
                float(self.min_trade_notional_usd) / float(max_notional)
                if max_notional > 0
                else 0.0
            )

            for s, w in zip(active, weights):
                info = per[s]
                side = int(desired_dir[s])
                price = float(info["last_close"])
                atr = float(info["atr"])
                stop_dist = max(1e-12, float(self.stop_atr_mult) * atr)
                risk_usd = float(total_risk_usd) * float(w)
                qty = risk_usd / stop_dist
                notional = qty * price

                net_edge = max(0.0, float(info["net_edge_bps"]))
                conf = _clamp(net_edge / max(1e-9, float(required_edge_bps)), 0.15, 1.0)
                notional *= float(conf)

                if notional < float(self.min_trade_notional_usd):
                    cur_exp = float(current_exposure.get(s, 0.0))
                    if _sign(cur_exp) == side and abs(cur_exp) > 1e-9:
                        base_targets[s] = _clamp(
                            cur_exp, -float(max_per_symbol_cap), float(max_per_symbol_cap)
                        )
                        reason_tag[s] = "hold_min_notional"
                    continue

                exp = float(side) * float(notional) / float(max_notional)
                exp = _clamp(exp, -float(max_per_symbol_cap), float(max_per_symbol_cap))
                if abs(exp) < min_trade_exp and _sign(float(current_exposure.get(s, 0.0))) == side:
                    exp = float(current_exposure.get(s, 0.0))
                base_targets[s] = float(exp)

        # Keep same-direction positions unchanged when target delta is tiny.
        for s in symbols:
            cur_exp = float(current_exposure.get(s, 0.0))
            tgt = float(base_targets.get(s, 0.0))
            if _sign(cur_exp) != 0 and _sign(cur_exp) == _sign(tgt):
                if abs(tgt - cur_exp) < float(self.rebalance_exposure_threshold):
                    base_targets[s] = float(
                        _clamp(
                            cur_exp,
                            -float(max_per_symbol_cap),
                            float(max_per_symbol_cap),
                        )
                    )
                    if reason_tag.get(s) == "hold":
                        reason_tag[s] = "hold_threshold"

        gross_pre_scale = float(sum(abs(float(base_targets[s])) for s in symbols))
        gross_scale = 1.0
        if gross_pre_scale > float(max_gross_cap) and gross_pre_scale > 0:
            gross_scale = float(max_gross_cap) / gross_pre_scale
            base_targets = {
                s: float(base_targets[s]) * float(gross_scale) for s in symbols
            }

        lock_scale = 1.0
        if bool(self.enable_weekly_profit_lock) and bool(self._week_target_hit):
            lock_scale = _clamp(float(self.weekly_lock_risk_scale), 0.0, 1.0)
            base_targets = {s: float(base_targets[s]) * lock_scale for s in symbols}

        targets = dict(base_targets)
        heartbeat_returned = False
        heartbeat_triggered = False

        # Scheduled return leg of a prior heartbeat nudge.
        if (
            self._heartbeat_stage == "return"
            and self._heartbeat_symbol in targets
            and self._bars_seen >= int(self._heartbeat_return_at_bar)
        ):
            hb_sym = str(self._heartbeat_symbol)
            targets[hb_sym] = float(self._heartbeat_return_exposure)
            self._clear_heartbeat_schedule()
            heartbeat_returned = True

        if (
            bool(self.enable_weekly_heartbeat)
            and (not bool(self._week_activity_seen))
            and (not bool(self._week_heartbeat_fired))
            and self._is_heartbeat_time(ts)
        ):
            lock_blocks_heartbeat = (
                bool(self.enable_weekly_profit_lock)
                and bool(self._week_target_hit)
                and _clamp(float(self.weekly_lock_risk_scale), 0.0, 1.0) <= 0.0
            )
            if not lock_blocks_heartbeat:
                hb_candidates = [s for s in symbols if s in per]
                if hb_candidates:
                    hb_sym = sorted(
                        hb_candidates,
                        key=lambda s: float(score.get(s, 0.0)),
                        reverse=True,
                    )[0]
                    hb_base = float(targets.get(hb_sym, 0.0))
                    hb_dir = int(per[hb_sym]["trend_dir"])
                    if hb_dir == 0:
                        hb_dir = _sign(hb_base)
                    if hb_dir == 0:
                        hb_dir = 1
                    if hb_dir < 0 and (not bool(state.allow_short)):
                        hb_dir = 1

                    min_trade_exp = (
                        float(self.min_trade_notional_usd) / float(max_notional)
                        if max_notional > 0
                        else 0.0
                    )
                    hb_abs = max(float(self.heartbeat_exposure), float(min_trade_exp))
                    hb_abs = _clamp(hb_abs, 0.0, float(max_per_symbol_cap))

                    if hb_abs > 0:
                        hb_target = float(hb_dir) * hb_abs
                        if abs(hb_target - hb_base) < max(1e-4, 0.2 * hb_abs):
                            hb_step = _clamp(
                                abs(hb_base) + hb_abs, 0.0, float(max_per_symbol_cap)
                            )
                            hb_target = float(hb_dir) * hb_step

                        if abs(hb_target - hb_base) > 1e-9:
                            targets[hb_sym] = float(hb_target)
                            self._heartbeat_stage = "return"
                            self._heartbeat_symbol = str(hb_sym)
                            self._heartbeat_return_exposure = float(hb_base)
                            self._heartbeat_return_at_bar = int(
                                self._bars_seen + max(1, int(self.heartbeat_hold_bars))
                            )
                            self._week_heartbeat_fired = True
                            self._week_activity_seen = True
                            heartbeat_triggered = True

        # Final safety clamp after heartbeat overrides.
        for s in symbols:
            targets[s] = float(
                _clamp(
                    float(targets.get(s, 0.0)),
                    -float(max_per_symbol_cap),
                    float(max_per_symbol_cap),
                )
            )

        gross_after = float(sum(abs(float(targets[s])) for s in symbols))
        post_scale = 1.0
        if gross_after > float(max_gross_cap) and gross_after > 0:
            post_scale = float(max_gross_cap) / gross_after
            targets = {s: float(targets[s]) * float(post_scale) for s in symbols}

        activity_eps = max(
            1e-6,
            (float(self.min_trade_notional_usd) / float(max_notional) * 0.5)
            if max_notional > 0
            else 1e-6,
        )
        if any(
            abs(float(targets[s]) - float(current_exposure.get(s, 0.0))) > activity_eps
            for s in symbols
        ):
            self._week_activity_seen = True

        active_final = sorted([s for s in symbols if abs(float(targets[s])) > 1e-9])
        reason = (
            "active=" + ",".join(active_final)
            if active_final
            else ("heartbeat_return" if heartbeat_returned else "flat")
        )
        if heartbeat_triggered:
            reason = "heartbeat_entry"

        debug: dict[str, Any] = {
            "drawdown": float(drawdown),
            "cost_rt_bps": float(cost_rt_bps),
            "required_edge_bps": float(required_edge_bps),
            "gross_pre_scale": float(gross_pre_scale),
            "gross_scale": float(gross_scale),
            "gross_post_scale": float(post_scale),
            "week_key": self._week_key,
            "week_return": float(week_ret),
            "week_target_hit": bool(self._week_target_hit),
            "week_activity_seen": bool(self._week_activity_seen),
            "week_heartbeat_fired": bool(self._week_heartbeat_fired),
            "weekly_lock_scale": float(lock_scale),
            "heartbeat_stage": str(self._heartbeat_stage or ""),
            "reason_tag": reason_tag,
            "per": per,
        }
        return StrategyDecision(target_exposures=targets, reason=reason, debug=debug)
