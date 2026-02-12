from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, Optional

import numpy as np
import pandas as pd

from atlas.strategies.base import Strategy, StrategyDecision, StrategyState
from atlas.utils.time import NY_TZ


def _sign(x: float, *, eps: float = 1e-12) -> int:
    if x > eps:
        return 1
    if x < -eps:
        return -1
    return 0


def _clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))


def _alpha_from_span(span: int) -> float:
    span = int(span)
    if span <= 1:
        return 1.0
    return float(2.0 / (float(span) + 1.0))


def _decay_from_halflife(halflife_bars: int) -> float:
    h = int(halflife_bars)
    if h <= 0:
        return 1.0
    return float(math.exp(math.log(0.5) / float(h)))


def _to_ny(ts: pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    if ts.tz is None:
        return ts.tz_localize(NY_TZ)
    return ts.tz_convert(NY_TZ)


def _true_range(high: float, low: float, prev_close: Optional[float]) -> float:
    if prev_close is None or prev_close <= 0:
        return float(abs(high - low))
    return float(max(high - low, abs(high - prev_close), abs(low - prev_close)))


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
    return ("/" in s) or ("-" in s)


@dataclass
class _PerSymbolIndicators:
    prev_close: Optional[float] = None
    ema_fast: Optional[float] = None
    ema_slow: Optional[float] = None
    # ATR (simple moving average of TR)
    atr_trs: deque[float] = field(default_factory=deque)
    atr_sum: float = 0.0
    # Momentum horizon close
    mom_closes: deque[float] = field(default_factory=deque)
    # Liquidity proxy (EWMA dollar-volume)
    dv_ewma: Optional[float] = None
    # Last processed bar timestamp
    last_ts: Optional[pd.Timestamp] = None


@dataclass
class _PositionTracker:
    last_side: dict[str, int] = field(default_factory=dict)
    entry_price: dict[str, float] = field(default_factory=dict)
    trail_extreme: dict[str, float] = field(default_factory=dict)
    cooldown_until_bar: dict[str, int] = field(default_factory=dict)
    flip_counter: dict[str, int] = field(default_factory=dict)
    trend_confirm: dict[str, int] = field(default_factory=dict)
    exit_confirm: dict[str, int] = field(default_factory=dict)


@dataclass
class _MarketTracker:
    peak_price: float = 0.0
    last_price: float = 0.0
    ret_ewma_var: float = 0.0
    last_ts: Optional[pd.Timestamp] = None


@dataclass
class CryptoTSM(Strategy):
    """
    Crypto spot time-series momentum / trend strategy (research; NOT financial advice).

    High-level:
    - Direction: time-series momentum + EMA trend alignment (long-only by default).
    - Sizing: ATR-based position sizing with portfolio risk budget.
    - Controls: confirmation/hysteresis + cooldown + market risk-off overlay.

    Execution alignment:
    - Assumes the engine fills at NEXT bar OPEN and charges per-side slippage + taker fee.
    - This strategy uses engine-provided `state.extra['slippage_bps']` and `state.extra['taker_fee_bps']`
      for cost-aware admission.
    """

    name: str = "crypto_tsm"

    # ---- Universe ----
    symbols: tuple[str, ...] = ("BTC/USD", "ETH/USD")
    market_symbol: Optional[str] = "BTC/USD"

    # ---- Signals ----
    ema_fast: int = 24
    ema_slow: int = 120
    atr_window: int = 24
    min_atr_bps: float = 0.0
    momentum_window: int = 240
    confirm_bars: int = 3
    exit_confirm_bars: int = 3

    # ---- Portfolio / risk ----
    max_positions: int = 2
    max_gross_exposure: float = 1.0
    max_exposure_per_symbol: float = 1.0
    risk_budget: float = 0.05
    stop_atr_mult: float = 3.0
    trail_atr_mult: float = 5.0
    take_profit_atr_mult: float = 0.0
    max_hold_bars: int = 0
    min_hold_bars: int = 6
    cooldown_bars: int = 12
    rebalance_interval_bars: int = 4
    rebalance_exposure_threshold: float = 0.05

    # ---- Liquidity / hygiene ----
    min_dollar_volume_ewma: float = 100_000.0
    dv_ewm_span: int = 60
    min_trade_notional_usd: float = 25.0

    # ---- Cost model (bps) ----
    slippage_bps: float = 3.0
    taker_fee_bps: float = 25.0
    edge_floor_bps: float = 8.0
    k_cost: float = 2.0

    # ---- Risk-off controls ----
    daily_loss_limit: float = 0.05
    kill_switch: float = 0.20
    kill_switch_cooldown_days: int = 7
    market_drawdown_off: float = 0.20
    market_drawdown_reduce: float = 0.10
    market_vol_off_bps: float = 300.0
    market_vol_reduce_bps: float = 180.0
    market_peak_halflife_bars: int = 240

    # ---- Internal state ----
    _bars_seen: int = field(default=0, init=False, repr=False)
    _peak_equity: float = field(default=0.0, init=False, repr=False)
    _risk_disabled_day: Optional[object] = field(default=None, init=False, repr=False)
    _risk_disabled_until_day: Optional[object] = field(default=None, init=False, repr=False)
    _ind: dict[str, _PerSymbolIndicators] = field(default_factory=dict, init=False, repr=False)
    _pos: _PositionTracker = field(default_factory=_PositionTracker, init=False, repr=False)
    _mkt: _MarketTracker = field(default_factory=_MarketTracker, init=False, repr=False)

    def warmup_bars(self) -> int:
        return (
            max(
                int(self.ema_slow) + 2,
                int(self.atr_window) + 3,
                int(self.momentum_window) + 3,
                int(self.dv_ewm_span) + 3,
                int(self.confirm_bars) + 3,
            )
            + 10
        )

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

    def _ensure_symbol_state(self, symbol: str) -> _PerSymbolIndicators:
        st = self._ind.get(symbol)
        if st is not None:
            return st
        st = _PerSymbolIndicators()
        st.atr_trs = deque(maxlen=max(2, int(self.atr_window)))
        st.mom_closes = deque(maxlen=max(3, int(self.momentum_window) + 1))
        self._ind[symbol] = st
        return st

    def _maybe_reset_daily_state(self, state: StrategyState) -> None:
        today = _to_ny(pd.Timestamp(state.timestamp)).date()
        if self._risk_disabled_day is not None and self._risk_disabled_day != today:
            self._risk_disabled_day = None
        if self._risk_disabled_until_day is not None and today > self._risk_disabled_until_day:
            self._risk_disabled_until_day = None

    def _risk_off(self, universe: list[str], *, reason: str, debug: dict[str, Any]) -> StrategyDecision:
        return StrategyDecision(target_exposures={s: 0.0 for s in universe}, reason=reason, debug=debug)

    def _update_indicators_for_symbol(self, symbol: str, df: pd.DataFrame) -> None:
        if df is None or df.empty:
            return
        if not isinstance(df.index, pd.DatetimeIndex):
            return
        if not df.index.is_monotonic_increasing:
            df = df.sort_index()

        st = self._ensure_symbol_state(symbol)
        last_ts = st.last_ts
        if last_ts is None:
            new_df = df
        else:
            new_df = df[df.index > last_ts]
        if new_df.empty:
            return

        a_fast = _alpha_from_span(self.ema_fast)
        a_slow = _alpha_from_span(self.ema_slow)
        a_dv = _alpha_from_span(self.dv_ewm_span)

        for ts, row in new_df.iterrows():
            try:
                h = float(row["high"])
                l = float(row["low"])
                c = float(row["close"])
                v = float(row.get("volume", 0.0) or 0.0)
            except Exception:
                continue
            if not np.isfinite(c) or c <= 0:
                continue
            if not np.isfinite(h) or not np.isfinite(l) or h <= 0 or l <= 0:
                h = c
                l = c

            st.ema_fast = float(c) if st.ema_fast is None else float(a_fast * c + (1.0 - a_fast) * float(st.ema_fast))
            st.ema_slow = float(c) if st.ema_slow is None else float(a_slow * c + (1.0 - a_slow) * float(st.ema_slow))

            tr = _true_range(h, l, st.prev_close)
            if len(st.atr_trs) == st.atr_trs.maxlen:
                st.atr_sum -= float(st.atr_trs[0])
            st.atr_trs.append(float(tr))
            st.atr_sum += float(tr)

            st.mom_closes.append(float(c))

            dv = float(c) * max(0.0, float(v))
            if st.dv_ewma is None or not np.isfinite(float(st.dv_ewma)):
                st.dv_ewma = float(dv)
            else:
                st.dv_ewma = float(a_dv * dv + (1.0 - a_dv) * float(st.dv_ewma))

            st.prev_close = float(c)
            st.last_ts = pd.Timestamp(ts)

    def _update_market_tracker(self, symbol: str, df: pd.DataFrame) -> None:
        if df is None or df.empty or not isinstance(df.index, pd.DatetimeIndex):
            return
        df = df.sort_index()
        last_ts = self._mkt.last_ts
        if last_ts is None:
            new_df = df
        else:
            new_df = df[df.index > last_ts]
        if new_df.empty:
            return

        a = _alpha_from_span(max(8, int(self.atr_window)))
        peak_decay = _decay_from_halflife(int(self.market_peak_halflife_bars))
        for ts, row in new_df.iterrows():
            c = _safe_float(row.get("close"), default=0.0)
            if c <= 0:
                continue
            if self._mkt.last_price > 0:
                r = math.log(float(c) / float(self._mkt.last_price))
            else:
                r = 0.0
            self._mkt.ret_ewma_var = float((1.0 - a) * float(self._mkt.ret_ewma_var) + a * float(r * r))
            self._mkt.last_price = float(c)
            self._mkt.peak_price = float(self._mkt.peak_price) * float(peak_decay)
            self._mkt.peak_price = max(float(self._mkt.peak_price), float(c))
            self._mkt.last_ts = pd.Timestamp(ts)

    def _market_risk_scale(self) -> tuple[float, dict[str, float]]:
        dbg: dict[str, float] = {}
        last = float(self._mkt.last_price)
        peak = float(self._mkt.peak_price)
        dd = (last / peak - 1.0) if (peak > 0 and last > 0) else 0.0
        vol_bps = math.sqrt(max(0.0, float(self._mkt.ret_ewma_var))) * 10_000.0
        dbg["market_drawdown"] = float(dd)
        dbg["market_vol_bps"] = float(vol_bps)

        scale = 1.0
        if dd <= -abs(float(self.market_drawdown_off)) or vol_bps >= abs(float(self.market_vol_off_bps)):
            scale = 0.0
        elif dd <= -abs(float(self.market_drawdown_reduce)) or vol_bps >= abs(float(self.market_vol_reduce_bps)):
            scale = 0.5
        return float(scale), dbg

    def _features(self, symbol: str) -> Optional[dict[str, float]]:
        st = self._ind.get(symbol)
        if st is None:
            return None
        close = st.prev_close
        if close is None or close <= 0:
            return None

        atr = (float(st.atr_sum) / float(len(st.atr_trs))) if len(st.atr_trs) else 0.0
        atr_bps = float((atr / close) * 10_000.0) if close > 0 else 0.0

        ema_fast = _safe_float(st.ema_fast, default=close)
        ema_slow = _safe_float(st.ema_slow, default=close)
        trend_strength = float((ema_fast - ema_slow) / max(1e-12, atr)) if atr > 0 else 0.0

        mom = 0.0
        if len(st.mom_closes) >= max(3, int(self.momentum_window) + 1):
            base = float(list(st.mom_closes)[-int(self.momentum_window) - 1])
            if base > 0:
                mom = float(math.log(close / base))

        dv = _safe_float(st.dv_ewma, default=0.0)

        return {
            "close": float(close),
            "atr": float(atr),
            "atr_bps": float(atr_bps),
            "trend_strength": float(trend_strength),
            "mom": float(mom),
            "dv_ewma": float(dv),
        }

    def target_exposures(self, bars_by_symbol: dict[str, pd.DataFrame], state: StrategyState) -> StrategyDecision:
        universe = self._universe(bars_by_symbol)
        if not universe:
            return StrategyDecision(target_exposures={}, reason="no_crypto_symbols")

        self._bars_seen += 1
        self._maybe_reset_daily_state(state)

        equity = float(state.equity)
        extra = dict(state.extra or {})
        if self._peak_equity <= 0:
            self._peak_equity = equity
        self._peak_equity = max(self._peak_equity, equity)
        drawdown = (equity / self._peak_equity - 1.0) if self._peak_equity > 0 else 0.0

        debug: dict[str, Any] = {
            "bars_seen": int(self._bars_seen),
            "equity": float(equity),
            "drawdown": float(drawdown),
            "day_return": float(state.day_return),
            "allow_short": bool(state.allow_short),
        }

        # Defensive: if a caller accidentally passes future bars (>= decision timestamp),
        # trim them out to avoid lookahead. Backtest/paper should already be safe.
        decision_ts = pd.Timestamp(state.timestamp)
        trimmed = False
        for s in universe:
            df = bars_by_symbol.get(s)
            if df is None or len(df) == 0:
                continue
            if pd.Timestamp(df.index[-1]) >= decision_ts:
                if not trimmed:
                    bars_by_symbol = dict(bars_by_symbol)
                    trimmed = True
                bars_by_symbol[s] = df[df.index < decision_ts]

        today = _to_ny(pd.Timestamp(state.timestamp)).date()
        if self._risk_disabled_until_day is not None and today <= self._risk_disabled_until_day:
            return self._risk_off(universe, reason="risk_disabled_cooldown", debug=debug)

        if drawdown <= -abs(float(self.kill_switch)):
            self._peak_equity = float(equity)
            self._risk_disabled_until_day = today + timedelta(days=int(self.kill_switch_cooldown_days))
            return self._risk_off(universe, reason="kill_switch", debug=debug)

        if float(state.day_return) <= -abs(float(self.daily_loss_limit)):
            self._risk_disabled_day = _to_ny(pd.Timestamp(state.timestamp)).date()
            return self._risk_off(universe, reason="daily_loss_limit", debug=debug)

        if self._risk_disabled_day == _to_ny(pd.Timestamp(state.timestamp)).date():
            return self._risk_off(universe, reason="risk_disabled_day", debug=debug)

        for s in universe:
            self._update_indicators_for_symbol(s, bars_by_symbol.get(s))

        mkt_sym = (self.market_symbol or "").strip().upper()
        if not mkt_sym or mkt_sym not in universe:
            mkt_sym = universe[0]
        self._update_market_tracker(mkt_sym, bars_by_symbol.get(mkt_sym))
        risk_scale, market_dbg = self._market_risk_scale()
        debug["market_symbol"] = mkt_sym
        debug["market"] = market_dbg
        debug["risk_scale"] = float(risk_scale)
        if risk_scale <= 0.0:
            return self._risk_off(universe, reason="market_risk_off", debug=debug)

        max_notional = float(extra.get("max_position_notional_usd", 0.0) or 0.0)
        if max_notional <= 0:
            max_notional = 1.0

        slippage_bps = float(self.slippage_bps)
        if extra.get("slippage_bps") is not None:
            slippage_bps = float(extra.get("slippage_bps") or 0.0)
        taker_fee_bps = float(self.taker_fee_bps)
        if extra.get("taker_fee_bps") is not None:
            taker_fee_bps = float(extra.get("taker_fee_bps") or 0.0)

        feats: dict[str, dict[str, float]] = {}
        for s in universe:
            f = self._features(s)
            if f is not None:
                feats[s] = f
        if not feats:
            return StrategyDecision(target_exposures={s: 0.0 for s in universe}, reason="warmup", debug=debug)

        slip = abs(float(slippage_bps)) / 10_000.0

        # Position lifecycle tracking.
        for s in universe:
            pos_qty = float(state.positions.get(s, 0.0) or 0.0)
            pos_side = _sign(pos_qty)
            prev_side = int(self._pos.last_side.get(s, 0))
            last_close = float(feats[s]["close"])
            last_df = bars_by_symbol.get(s)
            last_row = last_df.iloc[-1] if last_df is not None and len(last_df) else None
            last_open = (
                _safe_float(last_row.get("open"), default=last_close) if last_row is not None else last_close
            )

            if prev_side == 0 and pos_side != 0:
                # Approximate entry price from the bar open where the fill occurred (backtest fills at open),
                # including configured slippage. Fees are handled separately by the engine.
                entry_fill = float(last_open) * (1.0 + slip) if pos_side > 0 else float(last_open) * (1.0 - slip)
                self._pos.entry_price[s] = float(entry_fill)
                self._pos.trail_extreme[s] = float(entry_fill)
                self._pos.flip_counter[s] = 0
            elif prev_side != 0 and pos_side == 0:
                self._pos.cooldown_until_bar[s] = max(
                    int(self._pos.cooldown_until_bar.get(s, 0)),
                    int(self._bars_seen + int(self.cooldown_bars)),
                )
                self._pos.entry_price.pop(s, None)
                self._pos.trail_extreme.pop(s, None)
                self._pos.flip_counter[s] = 0
                self._pos.trend_confirm[s] = 0
            self._pos.last_side[s] = int(pos_side)

        # Decide if we allow entries this bar (reduce churn).
        allow_entries = True
        allow_rebalance = True
        if int(self.rebalance_interval_bars) > 1:
            allow_entries = (int(self._bars_seen) % int(self.rebalance_interval_bars)) == 0
            allow_rebalance = allow_entries
        debug["allow_entries"] = bool(allow_entries)
        debug["allow_rebalance"] = bool(allow_rebalance)

        desired_dir: dict[str, int] = {s: 0 for s in universe}
        score: dict[str, float] = {s: -1e9 for s in universe}
        reason_tag: dict[str, str] = {s: "flat" for s in universe}
        execution_hints: dict[str, dict[str, Any]] = {}

        cost_rt_bps = float(2.0 * (abs(float(slippage_bps)) + abs(float(taker_fee_bps))))
        required_edge = float(self.edge_floor_bps) + float(self.k_cost) * cost_rt_bps
        debug["cost_rt_bps"] = float(cost_rt_bps)
        debug["required_edge_bps"] = float(required_edge)

        for s, f in feats.items():
            last_close = float(f["close"])
            atr = float(f["atr"])
            atr_bps = float(f["atr_bps"])
            trend_strength = float(f["trend_strength"])
            mom = float(f["mom"])
            last_df = bars_by_symbol.get(s)
            last_row = last_df.iloc[-1] if last_df is not None and len(last_df) else None
            bar_high = _safe_float(last_row.get("high"), default=last_close) if last_row is not None else last_close
            bar_low = _safe_float(last_row.get("low"), default=last_close) if last_row is not None else last_close

            if atr <= 0 or last_close <= 0:
                reason_tag[s] = "bad_prices"
                continue
            if atr_bps <= 0:
                reason_tag[s] = "bad_atr"
                continue
            if float(f.get("dv_ewma", 0.0)) < float(self.min_dollar_volume_ewma):
                reason_tag[s] = "gate_liquidity"
                continue

            pos_qty = float(state.positions.get(s, 0.0) or 0.0)
            pos_side = _sign(pos_qty)
            hold_bars = int(state.holding_bars.get(s, 0) or 0)

            # In-position management: stops and optional take-profit / time stop.
            if pos_side != 0:
                entry = float(self._pos.entry_price.get(s, last_close))
                prev_ext = float(self._pos.trail_extreme.get(s, last_close))
                if pos_side > 0:
                    trail_stop = float(prev_ext) - float(self.trail_atr_mult) * atr
                    hard_stop = entry - float(self.stop_atr_mult) * atr
                    effective_stop = max(hard_stop, trail_stop)
                    stop_hit = bool(float(bar_low) <= effective_stop)
                    tp_level = entry + float(self.take_profit_atr_mult) * atr
                    tp_hit = bool(float(self.take_profit_atr_mult) > 0 and float(bar_high) >= tp_level)
                else:
                    trail_stop = float(prev_ext) + float(self.trail_atr_mult) * atr
                    hard_stop = entry + float(self.stop_atr_mult) * atr
                    effective_stop = min(hard_stop, trail_stop)
                    stop_hit = bool(float(bar_high) >= effective_stop)
                    tp_level = entry - float(self.take_profit_atr_mult) * atr
                    tp_hit = bool(float(self.take_profit_atr_mult) > 0 and float(bar_low) <= tp_level)

                if stop_hit:
                    desired_dir[s] = 0
                    score[s] = 0.0
                    reason_tag[s] = "stop"
                    execution_hints[s] = {
                        "mode": "min" if pos_side > 0 else "max",
                        "price": float(effective_stop),
                    }
                    continue

                if tp_hit:
                    desired_dir[s] = 0
                    score[s] = 0.0
                    reason_tag[s] = "take_profit"
                    execution_hints[s] = {
                        "mode": "max" if pos_side > 0 else "min",
                        "price": float(tp_level),
                    }
                    continue

                # Update trailing extreme only after stop/TP checks (avoid within-bar ordering assumptions).
                if pos_side > 0:
                    self._pos.trail_extreme[s] = max(prev_ext, float(bar_high))
                else:
                    self._pos.trail_extreme[s] = min(prev_ext, float(bar_low))

                if hold_bars < int(self.min_hold_bars):
                    desired_dir[s] = pos_side
                    score[s] = 1e6
                    reason_tag[s] = "min_hold"
                    continue

                if int(self.max_hold_bars) > 0 and hold_bars >= int(self.max_hold_bars):
                    desired_dir[s] = 0
                    score[s] = 0.0
                    reason_tag[s] = "time_stop"
                    continue

                # Trend reversal exit: require persistence to reduce churn.
                if hold_bars >= int(self.min_hold_bars):
                    reverse = (float(trend_strength) * float(pos_side) < 0.0) and (float(mom) * float(pos_side) < 0.0)
                    if reverse:
                        self._pos.exit_confirm[s] = int(self._pos.exit_confirm.get(s, 0)) + 1
                    else:
                        self._pos.exit_confirm[s] = 0

                    if int(self._pos.exit_confirm.get(s, 0)) >= int(self.exit_confirm_bars):
                        desired_dir[s] = 0
                        score[s] = 0.0
                        reason_tag[s] = "trend_exit"
                        continue

                desired_dir[s] = pos_side
                score[s] = 10.0
                reason_tag[s] = "hold"
                continue

            if not allow_entries:
                desired_dir[s] = 0
                reason_tag[s] = "no_entry_bar"
                continue

            if self._bars_seen < int(self._pos.cooldown_until_bar.get(s, 0)):
                desired_dir[s] = 0
                reason_tag[s] = "cooldown"
                continue

            if float(atr_bps) < float(self.min_atr_bps):
                desired_dir[s] = 0
                reason_tag[s] = "gate_atr"
                continue

            trend_dir = _sign(trend_strength)
            mom_dir = _sign(mom)
            if trend_dir != 0 and trend_dir == mom_dir:
                self._pos.trend_confirm[s] = int(self._pos.trend_confirm.get(s, 0)) + 1
            else:
                self._pos.trend_confirm[s] = 0

            if int(self._pos.trend_confirm.get(s, 0)) < int(self.confirm_bars):
                desired_dir[s] = 0
                reason_tag[s] = "confirm"
                continue

            if (not bool(state.allow_short)) and trend_dir < 0:
                desired_dir[s] = 0
                reason_tag[s] = "short_blocked"
                continue

            # Edge proxy in bps: trend_strength is in ATR units, mom is log-return.
            edge_bps = float(0.60 * abs(trend_strength) * atr_bps + 0.30 * abs(mom) * 10_000.0)
            net = float(edge_bps - float(self.k_cost) * cost_rt_bps)
            if edge_bps < required_edge or net <= 0:
                desired_dir[s] = 0
                reason_tag[s] = "edge_gate"
                continue

            desired_dir[s] = int(trend_dir)
            score[s] = float(net)
            reason_tag[s] = "trend"

        active_syms = [s for s in universe if int(desired_dir.get(s, 0)) != 0]
        if not active_syms:
            debug["reason_tag"] = reason_tag
            return StrategyDecision(
                target_exposures={s: 0.0 for s in universe},
                reason="no_signal",
                debug=debug,
                execution_hints=(execution_hints or None),
            )

        max_positions = max(1, int(self.max_positions))
        def _rank_key(sym: str) -> tuple[int, float]:
            pos_side = _sign(float(state.positions.get(sym, 0.0) or 0.0))
            return (1 if pos_side != 0 else 0, float(score.get(sym, -1e9)))

        active_sorted = sorted(active_syms, key=_rank_key, reverse=True)
        keep = set(active_sorted[:max_positions])
        for s in active_syms:
            if s not in keep:
                desired_dir[s] = 0
                reason_tag[s] = "rank_cut"

        active = [s for s in active_sorted if s in keep and int(desired_dir.get(s, 0)) != 0]
        if not active:
            debug["reason_tag"] = reason_tag
            return StrategyDecision(
                target_exposures={s: 0.0 for s in universe},
                reason="rank_cut_all",
                debug=debug,
                execution_hints=(execution_hints or None),
            )

        total_risk_usd = float(max(0.0, float(self.risk_budget))) * float(equity)
        risk_per_pos = total_risk_usd / max(1.0, float(len(active)))

        desired_notional: dict[str, float] = {}
        for s in active:
            f = feats[s]
            price = float(f["close"])
            atr = float(f["atr"])
            if price <= 0 or atr <= 0:
                continue
            stop_dist = max(1e-12, float(self.stop_atr_mult) * atr)
            qty = float(risk_per_pos / stop_dist) if stop_dist > 0 else 0.0
            notional = qty * price

            net_edge = float(score.get(s, 0.0))
            conf = _clamp(net_edge / max(1e-9, float(required_edge)), 0.0, 1.0)
            notional *= float(conf)

            cap = float(max_notional) * float(self.max_exposure_per_symbol)
            notional = min(float(notional), float(cap))
            if notional < float(self.min_trade_notional_usd):
                pos_qty = float(state.positions.get(s, 0.0) or 0.0)
                if abs(pos_qty) > 1e-9:
                    # Don't force exits/resizes just because the computed target would be too small to trade.
                    desired_notional[s] = min(float(abs(pos_qty) * price), float(cap))
                continue
            desired_notional[s] = float(notional)

        if not desired_notional:
            debug["reason_tag"] = reason_tag
            return StrategyDecision(
                target_exposures={s: 0.0 for s in universe},
                reason="sizing_blocked",
                debug=debug,
                execution_hints=(execution_hints or None),
            )

        exposures: dict[str, float] = {s: 0.0 for s in universe}
        for s in universe:
            d = int(desired_dir.get(s, 0))
            if d == 0:
                continue
            notional = float(desired_notional.get(s, 0.0))
            exp = (float(d) * notional) / float(max_notional) if max_notional > 0 else 0.0
            exposures[s] = float(_clamp(exp, -float(self.max_exposure_per_symbol), float(self.max_exposure_per_symbol)))

        gross = float(sum(abs(exposures[s]) for s in universe))
        if gross > float(self.max_gross_exposure) + 1e-12:
            scale = float(self.max_gross_exposure) / max(gross, 1e-12)
            exposures = {s: float(exposures[s]) * scale for s in universe}
            debug["gross_scaled"] = True
            debug["gross_scale"] = float(scale)

        exposures = {s: float(exposures[s]) * float(risk_scale) for s in universe}

        # Rebalance threshold: keep current exposure when target is close (avoid churn).
        for s in universe:
            pos_qty = float(state.positions.get(s, 0.0) or 0.0)
            if abs(pos_qty) <= 1e-9:
                continue
            last_close = float(feats.get(s, {}).get("close", 0.0) or 0.0)
            if last_close <= 0:
                continue
            cur_exp = (float(pos_qty) * float(last_close)) / float(max_notional) if float(max_notional) > 0 else 0.0
            tgt = float(exposures.get(s, 0.0))
            if _sign(cur_exp) != 0 and _sign(tgt) == _sign(cur_exp):
                if abs(tgt - float(cur_exp)) < float(self.rebalance_exposure_threshold):
                    exposures[s] = float(cur_exp)
                    reason_tag[s] = "hold_threshold"

        # If not a rebalance bar, don't resize positions (but allow exits).
        if not allow_rebalance:
            for s in universe:
                pos_qty = float(state.positions.get(s, 0.0) or 0.0)
                if abs(pos_qty) <= 1e-9:
                    continue
                # If we are trying to exit, keep exit.
                if abs(float(exposures.get(s, 0.0))) <= 1e-9:
                    continue
                last_close = float(feats.get(s, {}).get("close", 0.0) or 0.0)
                if last_close <= 0:
                    continue
                cur_exp = (float(pos_qty) * float(last_close)) / float(max_notional) if float(max_notional) > 0 else 0.0
                if _sign(cur_exp) == _sign(float(exposures.get(s, 0.0))):
                    exposures[s] = float(cur_exp)
                    reason_tag[s] = "hold_schedule"

        debug["reason_tag"] = reason_tag
        debug["active"] = active
        debug["scores"] = {s: float(score.get(s, 0.0)) for s in universe}
        return StrategyDecision(
            target_exposures=exposures,
            reason="active=" + ",".join(sorted([s for s in universe if abs(exposures.get(s, 0.0)) > 1e-9])),
            debug=debug,
            execution_hints=(execution_hints or None),
        )
