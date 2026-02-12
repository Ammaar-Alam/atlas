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


def _to_ny(ts: pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    if ts.tz is None:
        return ts.tz_localize(NY_TZ)
    return ts.tz_convert(NY_TZ)


def _true_range(high: float, low: float, prev_close: Optional[float]) -> float:
    if prev_close is None or prev_close <= 0:
        return float(abs(high - low))
    return float(max(high - low, abs(high - prev_close), abs(low - prev_close)))


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


def _ewma_update(mean: Optional[float], var: Optional[float], x: float, alpha: float) -> tuple[float, float]:
    """
    Exponentially weighted mean/variance update.

    Returns (new_mean, new_var). `var` is the EWMA variance of x.
    """
    alpha = float(_clamp(alpha, 0.0, 1.0))
    if mean is None or (not np.isfinite(float(mean))):
        return float(x), 0.0
    if var is None or (not np.isfinite(float(var))):
        var = 0.0

    mean_f = float(mean)
    var_f = float(var)
    delta = float(x) - mean_f
    mean_new = mean_f + alpha * delta
    # This form keeps variance non-negative and behaves well for small alpha.
    var_new = (1.0 - alpha) * (var_f + alpha * delta * delta)
    if not np.isfinite(var_new) or var_new < 0:
        var_new = 0.0
    return float(mean_new), float(var_new)


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
    return "/" in s or "-" in s  # tolerate BTC-USD input; market module usually normalizes to BTC/USD


@dataclass
class _PerSymbolIndicators:
    prev_close: Optional[float] = None
    # Trend / momentum
    ema_fast: Optional[float] = None
    ema_slow: Optional[float] = None
    ret_ewma_mean: Optional[float] = None
    ret_ewma_var: Optional[float] = None
    meanrev_mean: Optional[float] = None
    meanrev_var: Optional[float] = None
    # ATR (simple moving average of TR)
    atr_trs: deque[float] = field(default_factory=deque)
    atr_sum: float = 0.0
    # Regime / ER
    er_closes: deque[float] = field(default_factory=deque)
    # Breakout
    breakout_highs: deque[float] = field(default_factory=deque)
    breakout_lows: deque[float] = field(default_factory=deque)
    # Momentum horizon close
    mom_closes: deque[float] = field(default_factory=deque)
    # Liquidity proxy (EWMA dollar-volume)
    dv_ewma: Optional[float] = None
    # RSI (Wilder-style smoothing)
    rsi_avg_gain: Optional[float] = None
    rsi_avg_loss: Optional[float] = None
    # Intraday VWAP reset (NY day boundary for deterministic daily reset)
    vwap_day: Optional[object] = None
    vwap_num: float = 0.0
    vwap_den: float = 0.0
    # Last processed bar timestamp
    last_ts: Optional[pd.Timestamp] = None


@dataclass
class _PositionTracker:
    last_side: dict[str, int] = field(default_factory=dict)
    last_qty: dict[str, float] = field(default_factory=dict)
    entry_price: dict[str, float] = field(default_factory=dict)
    trail_extreme: dict[str, float] = field(default_factory=dict)
    entry_reason: dict[str, str] = field(default_factory=dict)
    pending_entry_reason: dict[str, str] = field(default_factory=dict)
    cooldown_until_bar: dict[str, int] = field(default_factory=dict)
    flip_counter: dict[str, int] = field(default_factory=dict)
    trend_confirm: dict[str, int] = field(default_factory=dict)
    breakout_confirm: dict[str, int] = field(default_factory=dict)
    # Mean-reversion "setup" state: wait for reversal confirmation before entry.
    meanrev_setup_dir: dict[str, int] = field(default_factory=dict)
    meanrev_setup_until_bar: dict[str, int] = field(default_factory=dict)


@dataclass
class _MarketTracker:
    peak_price: float = 0.0
    last_price: float = 0.0
    ret_ewma_var: float = 0.0
    last_ts: Optional[pd.Timestamp] = None


@dataclass
class CryptoEnsemble(Strategy):
    """
    Crypto spot ensemble strategy (research; NOT financial advice).

    High-level design:
    - Regime-aware blend of:
      (1) time-series momentum/trend-following (EMA + ER gate),
      (2) breakout continuation (Donchian),
      (3) mean reversion to intraday VWAP / statistical mean (z-score + RSI).
    - Cost-aware admission: requires an edge proxy (bps) to exceed modeled friction.
    - Survival constraints: daily loss limit + drawdown kill-switch, plus per-position stops/trailing/time-stops.

    Notes:
    - This strategy is spot-only: it refuses perp-style symbols (e.g. BTC-PERP, *-CDE).
    - It can run long-only (default) or allow shorting if the engine supports it and allow_short=True.
    - No strategy can guarantee profits; backtests can be misleading due to regime change, slippage,
      fees, and data/venue differences.
    """

    name: str = "crypto_ensemble"

    # ---- Universe ----
    symbols: tuple[str, ...] = ("BTC/USD", "ETH/USD")
    market_symbol: Optional[str] = "BTC/USD"  # used for global risk regime (fallback: first symbol)

    # ---- Signal windows ----
    ema_fast: int = 20
    ema_slow: int = 80
    atr_window: int = 20
    er_window: int = 40
    breakout_window: int = 60
    momentum_window: int = 240

    # ---- Regime thresholds ----
    er_trend_min: float = 0.35
    er_range_max: float = 0.20
    trend_z_min: float = 0.20  # |(ema_fast-ema_slow)/atr|
    min_atr_bps: float = 6.0

    # ---- Mean reversion knobs ----
    meanrev_ewm_span: int = 120
    meanrev_entry_z: float = 1.75
    meanrev_exit_z: float = 0.50
    rsi_window: int = 14
    rsi_oversold: float = 35.0
    rsi_overbought: float = 65.0
    require_vwap_alignment_for_trend: bool = True
    meanrev_disable_cost_rt_bps: float = 30.0
    meanrev_allow_bear_trend_long_only: bool = False
    meanrev_setup_max_bars: int = 8
    meanrev_reversal_min_bps: float = 0.0
    meanrev_size_mult: float = 0.35
    meanrev_stop_atr_mult: float = 0.0  # 0 => use stop_atr_mult
    meanrev_trail_atr_mult: float = 0.0  # 0 => use trail_atr_mult
    meanrev_max_hold_bars: int = 0  # 0 disables

    # ---- Breakout knobs ----
    breakout_buffer_bps: float = 2.0
    confirm_bars: int = 2

    # ---- Portfolio / risk ----
    max_positions: int = 3
    max_gross_exposure: float = 1.0  # sum(abs(exposure)) cap across symbols
    max_exposure_per_symbol: float = 1.0  # per-symbol exposure cap (in units of max_position_notional_usd)
    risk_budget: float = 0.02  # fraction of equity risked to stop across active positions
    stop_atr_mult: float = 2.0
    trail_atr_mult: float = 3.0
    take_profit_atr_mult: float = 0.0  # 0 disables
    max_hold_bars: int = 0  # 0 disables
    min_hold_bars: int = 3
    cooldown_bars: int = 6
    flip_confirm_bars: int = 3

    # ---- Liquidity / hygiene ----
    min_dollar_volume_ewma: float = 50_000.0
    dv_ewm_span: int = 60
    rebalance_exposure_threshold: float = 0.05
    min_trade_notional_usd: float = 25.0

    # ---- Cost model (bps) ----
    slippage_bps: float = 3.0  # per side proxy (spread+impact)
    taker_fee_bps: float = 25.0  # per side proxy
    edge_floor_bps: float = 4.0
    k_cost: float = 2.0

    # ---- Risk-off controls ----
    daily_loss_limit: float = 0.03
    kill_switch: float = 0.12
    kill_switch_cooldown_days: int = 7
    market_drawdown_off: float = 0.15
    market_drawdown_reduce: float = 0.08
    market_vol_off_bps: float = 250.0  # EWMA vol in bps per bar (rough)
    market_vol_reduce_bps: float = 150.0
    market_peak_halflife_bars: int = 240  # decay the peak reference to avoid multi-year risk-off after crashes

    # ---- Optional: enforce periodic activity ("heartbeat") ----
    # If enabled, the strategy will apply a tiny, alternating exposure offset to a held symbol
    # to ensure the system is not idle for long stretches. This is intended for research/testing
    # and can reduce performance slightly due to costs.
    heartbeat_every_bars: int = 0
    heartbeat_notional_usd: float = 25.0

    # ---- Internal state ----
    _bars_seen: int = field(default=0, init=False, repr=False)
    _peak_equity: float = field(default=0.0, init=False, repr=False)
    _risk_disabled_day: Optional[object] = field(default=None, init=False, repr=False)
    _risk_disabled_until_day: Optional[object] = field(default=None, init=False, repr=False)
    _ind: dict[str, _PerSymbolIndicators] = field(default_factory=dict, init=False, repr=False)
    _pos: _PositionTracker = field(default_factory=_PositionTracker, init=False, repr=False)
    _mkt: _MarketTracker = field(default_factory=_MarketTracker, init=False, repr=False)
    _last_trade_bar: int = field(default=0, init=False, repr=False)
    _last_heartbeat_bar: int = field(default=0, init=False, repr=False)
    _heartbeat_symbol: Optional[str] = field(default=None, init=False, repr=False)
    _heartbeat_offset_exp: float = field(default=0.0, init=False, repr=False)

    def warmup_bars(self) -> int:
        return (
            max(
                int(self.ema_slow) + 2,
                int(self.atr_window) + 3,
                int(self.er_window) + 3,
                int(self.breakout_window) + 3,
                int(self.momentum_window) + 3,
                int(self.meanrev_ewm_span) + 3,
                int(self.dv_ewm_span) + 3,
                int(self.rsi_window) + 3,
            )
            + 5
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
                # Allow strategy to operate on available bars (paper loop can return missing symbols).
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
        st.er_closes = deque(maxlen=max(3, int(self.er_window) + 1))
        st.breakout_highs = deque(maxlen=max(3, int(self.breakout_window) + 1))
        st.breakout_lows = deque(maxlen=max(3, int(self.breakout_window) + 1))
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

    def _maybe_apply_heartbeat(
        self,
        *,
        exposures: dict[str, float],
        universe: list[str],
        state: StrategyState,
        market_symbol: str,
        max_notional_usd: float,
        debug: dict[str, Any],
    ) -> dict[str, float]:
        hb_every = int(getattr(self, "heartbeat_every_bars", 0) or 0)
        hb_notional = float(getattr(self, "heartbeat_notional_usd", 0.0) or 0.0)
        if hb_every <= 0 or hb_notional <= 0.0 or float(max_notional_usd) <= 0.0:
            return exposures

        # Respect the strategy's minimum trade size.
        if hb_notional < float(self.min_trade_notional_usd):
            return exposures

        hb_exp = float(hb_notional / float(max_notional_usd))
        hb_exp = float(_clamp(hb_exp, 0.0, 0.02))
        if hb_exp <= 1e-12:
            return exposures

        last_marker = max(int(self._last_trade_bar), int(self._last_heartbeat_bar))
        if (int(self._bars_seen) - int(last_marker)) >= int(hb_every):
            # Toggle: either apply an offset or revert it.
            if abs(float(self._heartbeat_offset_exp)) > 1e-12:
                self._heartbeat_offset_exp = 0.0
            else:
                # Choose a symbol to "nudge": prefer the largest absolute target exposure,
                # otherwise fall back to the market symbol.
                hb_sym: Optional[str] = None
                best_abs = 0.0
                for s in universe:
                    exp = float(exposures.get(s, 0.0))
                    if abs(exp) > best_abs + 1e-12:
                        best_abs = abs(exp)
                        hb_sym = s
                if hb_sym is None:
                    hb_sym = (market_symbol or "").strip().upper()
                    if not hb_sym or hb_sym not in universe:
                        hb_sym = universe[0]

                self._heartbeat_symbol = hb_sym
                base_exp = float(exposures.get(hb_sym, 0.0))
                if (not bool(state.allow_short)) and base_exp < 0.0:
                    base_exp = 0.0

                # If we already have meaningful exposure, reduce it slightly; otherwise add a micro position.
                if abs(base_exp) >= float(hb_exp) * 1.5:
                    if base_exp > 0:
                        self._heartbeat_offset_exp = -float(hb_exp)
                    elif base_exp < 0:
                        self._heartbeat_offset_exp = float(hb_exp)
                    else:
                        self._heartbeat_offset_exp = float(hb_exp)
                else:
                    if bool(state.allow_short) and base_exp < 0:
                        self._heartbeat_offset_exp = float(hb_exp)
                    else:
                        self._heartbeat_offset_exp = float(hb_exp)

            self._last_heartbeat_bar = int(self._bars_seen)

        hb_sym = (self._heartbeat_symbol or "").strip().upper()
        offset = float(self._heartbeat_offset_exp)
        if not hb_sym or hb_sym not in universe or abs(offset) <= 1e-12:
            return exposures

        base_exp = float(exposures.get(hb_sym, 0.0))
        lo = -float(self.max_exposure_per_symbol) if bool(state.allow_short) else 0.0
        hi = float(self.max_exposure_per_symbol)

        gross = float(sum(abs(float(exposures.get(s, 0.0))) for s in universe))
        adj = float(offset)
        if adj > 0.0:
            slack = float(self.max_gross_exposure) - gross
            if slack <= 1e-9:
                return exposures
            adj = min(float(adj), float(slack))

        new_exp = float(_clamp(float(base_exp) + float(adj), lo, hi))
        if abs(new_exp - base_exp) <= 1e-12:
            return exposures

        out = dict(exposures)
        out[hb_sym] = float(new_exp)
        debug["heartbeat"] = {
            "every_bars": int(hb_every),
            "notional_usd": float(hb_notional),
            "exp_delta": float(adj),
            "symbol": str(hb_sym),
        }
        return out

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
        a_meanrev = _alpha_from_span(self.meanrev_ewm_span)
        a_dv = _alpha_from_span(self.dv_ewm_span)
        a_rsi = _alpha_from_span(self.rsi_window)
        a_ret = _alpha_from_span(max(4, int(self.atr_window)))

        for ts, row in new_df.iterrows():
            try:
                o = float(row["open"])
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

            # VWAP (daily reset at NY day boundary for deterministic behavior across timezones)
            ny_day = _to_ny(pd.Timestamp(ts)).date()
            if st.vwap_day != ny_day:
                st.vwap_day = ny_day
                st.vwap_num = 0.0
                st.vwap_den = 0.0
            tp = (h + l + c) / 3.0
            vol = max(0.0, float(v))
            st.vwap_num += float(tp) * vol
            st.vwap_den += vol

            # Returns (for volatility regime)
            if st.prev_close is not None and st.prev_close > 0:
                r = math.log(float(c) / float(st.prev_close))
            else:
                r = 0.0
            st.ret_ewma_mean, st.ret_ewma_var = _ewma_update(st.ret_ewma_mean, st.ret_ewma_var, float(r), float(a_ret))

            # Trend EMAs (on close)
            st.ema_fast = float(c) if st.ema_fast is None else float(a_fast * c + (1.0 - a_fast) * float(st.ema_fast))
            st.ema_slow = float(c) if st.ema_slow is None else float(a_slow * c + (1.0 - a_slow) * float(st.ema_slow))

            # Mean reversion EWM mean/var
            st.meanrev_mean, st.meanrev_var = _ewma_update(st.meanrev_mean, st.meanrev_var, float(c), float(a_meanrev))

            # ATR (SMA of TR)
            tr = _true_range(h, l, st.prev_close)
            if st.atr_trs.maxlen is None or st.atr_trs.maxlen <= 0:
                st.atr_trs = deque(maxlen=max(2, int(self.atr_window)))
                st.atr_sum = 0.0
            if len(st.atr_trs) == st.atr_trs.maxlen:
                st.atr_sum -= float(st.atr_trs[0])
            st.atr_trs.append(float(tr))
            st.atr_sum += float(tr)

            # ER close deque
            st.er_closes.append(float(c))

            # Breakout deques
            st.breakout_highs.append(float(h))
            st.breakout_lows.append(float(l))

            # Momentum close deque
            st.mom_closes.append(float(c))

            # Dollar volume EWMA
            dv = float(c) * vol
            if st.dv_ewma is None or not np.isfinite(float(st.dv_ewma)):
                st.dv_ewma = float(dv)
            else:
                st.dv_ewma = float(a_dv * dv + (1.0 - a_dv) * float(st.dv_ewma))

            # RSI smoothing
            if st.prev_close is not None and st.prev_close > 0:
                chg = float(c - st.prev_close)
                gain = max(0.0, chg)
                loss = max(0.0, -chg)
            else:
                gain = 0.0
                loss = 0.0

            if st.rsi_avg_gain is None or st.rsi_avg_loss is None:
                st.rsi_avg_gain = float(gain)
                st.rsi_avg_loss = float(loss)
            else:
                st.rsi_avg_gain = float((1.0 - a_rsi) * float(st.rsi_avg_gain) + a_rsi * float(gain))
                st.rsi_avg_loss = float((1.0 - a_rsi) * float(st.rsi_avg_loss) + a_rsi * float(loss))

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
            # EWMA of squared log-returns (per-bar volatility proxy)
            self._mkt.ret_ewma_var = float((1.0 - a) * float(self._mkt.ret_ewma_var) + a * float(r * r))
            self._mkt.last_price = float(c)
            self._mkt.peak_price = float(self._mkt.peak_price) * float(peak_decay)
            self._mkt.peak_price = max(float(self._mkt.peak_price), float(c))
            self._mkt.last_ts = pd.Timestamp(ts)

    def _compute_features(self, symbol: str) -> Optional[dict[str, float]]:
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

        # Efficiency ratio
        er = 0.0
        if len(st.er_closes) >= max(3, int(self.er_window) + 1):
            closes = list(st.er_closes)[-(int(self.er_window) + 1) :]
            change = abs(closes[-1] - closes[0])
            vol = 0.0
            for i in range(1, len(closes)):
                vol += abs(closes[i] - closes[i - 1])
            er = float(change / vol) if vol > 0 else 0.0

        # VWAP
        vwap = float(st.vwap_num / st.vwap_den) if st.vwap_den > 0 else float(close)
        vwap_dev = float((close - vwap) / vwap) if vwap > 0 else 0.0

        # Mean-reversion z-score vs EWMA mean/var on price
        mean_p = _safe_float(st.meanrev_mean, default=close)
        var_p = _safe_float(st.meanrev_var, default=0.0)
        std_p = float(math.sqrt(max(0.0, var_p)))
        z = float((close - mean_p) / std_p) if std_p > 1e-12 else 0.0

        # Momentum (log return over window)
        mom = 0.0
        if len(st.mom_closes) >= max(3, int(self.momentum_window) + 1):
            base = float(list(st.mom_closes)[-int(self.momentum_window) - 1])
            if base > 0:
                mom = float(math.log(close / base))

        # RSI
        ag = _safe_float(st.rsi_avg_gain, default=0.0)
        al = _safe_float(st.rsi_avg_loss, default=0.0)
        rs = float(ag / al) if al > 1e-12 else float("inf") if ag > 0 else 0.0
        rsi = float(100.0 - (100.0 / (1.0 + rs))) if rs > 0 else 0.0

        dv = _safe_float(st.dv_ewma, default=0.0)

        # Donchian breakout bounds excluding current bar.
        hh = float("nan")
        ll = float("nan")
        if len(st.breakout_highs) >= max(3, int(self.breakout_window) + 1):
            highs = list(st.breakout_highs)[-(int(self.breakout_window) + 1) : -1]
            lows = list(st.breakout_lows)[-(int(self.breakout_window) + 1) : -1]
            if highs and lows:
                hh = float(max(highs))
                ll = float(min(lows))

        return {
            "close": float(close),
            "atr": float(atr),
            "atr_bps": float(atr_bps),
            "ema_fast": float(ema_fast),
            "ema_slow": float(ema_slow),
            "trend_strength": float(trend_strength),
            "er": float(er),
            "vwap": float(vwap),
            "vwap_dev": float(vwap_dev),
            "z": float(z),
            "mom": float(mom),
            "rsi": float(rsi),
            "dv_ewma": float(dv),
            "hh": float(hh) if np.isfinite(hh) else float("nan"),
            "ll": float(ll) if np.isfinite(ll) else float("nan"),
        }

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

        # --- risk-off controls ---
        today = _to_ny(pd.Timestamp(state.timestamp)).date()
        if self._risk_disabled_until_day is not None and today <= self._risk_disabled_until_day:
            return self._risk_off(universe, reason="risk_disabled_cooldown", debug=debug)

        if drawdown <= -abs(float(self.kill_switch)):
            # Reset the equity peak so we can re-enable after cooldown without immediately re-triggering.
            self._peak_equity = float(equity)
            self._risk_disabled_until_day = today + timedelta(days=int(self.kill_switch_cooldown_days))
            return self._risk_off(universe, reason="kill_switch", debug=debug)

        if float(state.day_return) <= -abs(float(self.daily_loss_limit)):
            self._risk_disabled_day = _to_ny(pd.Timestamp(state.timestamp)).date()
            return self._risk_off(universe, reason="daily_loss_limit", debug=debug)

        if self._risk_disabled_day == _to_ny(pd.Timestamp(state.timestamp)).date():
            return self._risk_off(universe, reason="risk_disabled_day", debug=debug)

        # --- update indicator state ---
        for s in universe:
            self._update_indicators_for_symbol(s, bars_by_symbol.get(s))

        # Market regime symbol for global risk scaling.
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
            # In paper mode, this is always present, but guard anyway.
            max_notional = 1.0

        slippage_bps = float(self.slippage_bps)
        if extra.get("slippage_bps") is not None:
            slippage_bps = float(extra.get("slippage_bps") or 0.0)

        taker_fee_bps = float(self.taker_fee_bps)
        if extra.get("taker_fee_bps") is not None:
            taker_fee_bps = float(extra.get("taker_fee_bps") or 0.0)

        slip = abs(float(slippage_bps)) / 10_000.0

        # --- per-symbol feature extraction ---
        feats: dict[str, dict[str, float]] = {}
        for s in universe:
            f = self._compute_features(s)
            if f is not None:
                feats[s] = f

        if not feats or any("close" not in feats[s] for s in feats):
            exposures = {s: 0.0 for s in universe}
            exposures = self._maybe_apply_heartbeat(
                exposures=exposures,
                universe=universe,
                state=state,
                market_symbol=mkt_sym,
                max_notional_usd=max_notional,
                debug=debug,
            )
            reason = "warmup"
            if "heartbeat" in debug:
                reason = "heartbeat_warmup"
            return StrategyDecision(target_exposures=exposures, reason=reason, debug=debug)

        hb_ignore_notional_usd = 0.0
        hb_every = int(getattr(self, "heartbeat_every_bars", 0) or 0)
        hb_notional = float(getattr(self, "heartbeat_notional_usd", 0.0) or 0.0)
        if hb_every > 0 and hb_notional > 0.0:
            hb_ignore_notional_usd = max(
                float(self.min_trade_notional_usd), 1.25 * float(hb_notional)
            )

        pos_qty_eff: dict[str, float] = {}
        for s in universe:
            qty = float(state.positions.get(s, 0.0) or 0.0)
            close = float(feats.get(s, {}).get("close", 0.0) or 0.0)
            if hb_ignore_notional_usd > 0.0 and close > 0.0 and abs(qty) * close < hb_ignore_notional_usd:
                pos_qty_eff[s] = 0.0
            else:
                pos_qty_eff[s] = qty
        if hb_ignore_notional_usd > 0.0:
            debug["heartbeat_ignore_notional_usd"] = float(hb_ignore_notional_usd)

        # --- position lifecycle tracking (entry price / trailing anchors / cooldown) ---
        for s in universe:
            pos_qty_raw = float(state.positions.get(s, 0.0) or 0.0)
            pos_qty = float(pos_qty_eff.get(s, 0.0) or 0.0)
            pos_side = _sign(pos_qty)
            prev_side = int(self._pos.last_side.get(s, 0))
            prev_qty = float(self._pos.last_qty.get(s, 0.0) or 0.0)
            if abs(pos_qty_raw - prev_qty) > 1e-8:
                self._last_trade_bar = int(self._bars_seen)
            self._pos.last_qty[s] = float(pos_qty_raw)
            last_close = float(feats[s]["close"])
            last_df = bars_by_symbol.get(s)
            last_row = last_df.iloc[-1] if last_df is not None and len(last_df) else None
            last_open = (
                _safe_float(last_row.get("open"), default=last_close) if last_row is not None else last_close
            )

            if prev_side == 0 and pos_side != 0:
                entry_fill = float(last_open) * (1.0 + slip) if pos_side > 0 else float(last_open) * (1.0 - slip)
                self._pos.entry_price[s] = float(entry_fill)
                self._pos.trail_extreme[s] = float(entry_fill)
                self._pos.flip_counter[s] = 0
                pending = self._pos.pending_entry_reason.pop(s, None)
                if pending:
                    self._pos.entry_reason[s] = str(pending)
                # Clear any pending mean-reversion setup once we're in a position.
                self._pos.meanrev_setup_dir.pop(s, None)
                self._pos.meanrev_setup_until_bar.pop(s, None)
            elif prev_side != 0 and pos_side == 0:
                self._pos.cooldown_until_bar[s] = max(
                    int(self._pos.cooldown_until_bar.get(s, 0)),
                    int(self._bars_seen + int(self.cooldown_bars)),
                )
                self._pos.entry_price.pop(s, None)
                self._pos.trail_extreme.pop(s, None)
                self._pos.entry_reason.pop(s, None)
                self._pos.pending_entry_reason.pop(s, None)
                self._pos.flip_counter[s] = 0
                self._pos.meanrev_setup_dir.pop(s, None)
                self._pos.meanrev_setup_until_bar.pop(s, None)
            self._pos.last_side[s] = int(pos_side)

        # --- compute desired directions + scores ---
        desired_dir: dict[str, int] = {s: 0 for s in universe}
        score: dict[str, float] = {s: -1e9 for s in universe}
        reason_tag: dict[str, str] = {s: "flat" for s in universe}
        execution_hints: dict[str, dict[str, Any]] = {}

        cost_rt_bps = float(2.0 * (abs(float(slippage_bps)) + abs(float(taker_fee_bps))))
        required_edge = float(self.edge_floor_bps) + float(self.k_cost) * cost_rt_bps
        confirm_required = max(1, int(self.confirm_bars))
        allow_mean_reversion = True
        # Heuristic: short-horizon mean-reversion tends to be dominated by fees/spread once
        # round-trip costs get large. Trend legs can better amortize friction.
        if cost_rt_bps > float(self.meanrev_disable_cost_rt_bps):
            allow_mean_reversion = False
        # Heuristic: require more persistence under higher friction to reduce churn.
        if cost_rt_bps > 50.0:
            confirm_required = max(confirm_required, 3)
        debug["cost_rt_bps"] = float(cost_rt_bps)
        debug["required_edge_bps"] = float(required_edge)
        debug["confirm_required"] = int(confirm_required)
        debug["allow_mean_reversion"] = bool(allow_mean_reversion)

        for s, f in feats.items():
            last_close = float(f["close"])
            atr = float(f["atr"])
            atr_bps = float(f["atr_bps"])
            er = float(f["er"])
            z = float(f["z"])
            rsi = float(f["rsi"])
            trend_strength = float(f["trend_strength"])
            trend_dir = _sign(trend_strength, eps=1e-12)
            mom = float(f["mom"])
            mom_dir = _sign(mom, eps=1e-12)
            last_df = bars_by_symbol.get(s)
            last_row = last_df.iloc[-1] if last_df is not None and len(last_df) else None
            bar_high = _safe_float(last_row.get("high"), default=last_close) if last_row is not None else last_close
            bar_low = _safe_float(last_row.get("low"), default=last_close) if last_row is not None else last_close
            prev_close = (
                _safe_float(last_df["close"].iloc[-2], default=last_close)
                if last_df is not None and len(last_df) >= 2
                else last_close
            )
            ret_bps = (
                float(((last_close / prev_close) - 1.0) * 10_000.0)
                if prev_close > 0
                else 0.0
            )

            if atr_bps < float(self.min_atr_bps):
                reason_tag[s] = "gate_atr"
                continue
            if float(f.get("dv_ewma", 0.0)) < float(self.min_dollar_volume_ewma):
                reason_tag[s] = "gate_liquidity"
                continue

            # Breakout direction
            hh = float(f["hh"])
            ll = float(f["ll"])
            buf = float(self.breakout_buffer_bps) / 10_000.0
            breakout_dir = 0
            breakout_excess_bps = 0.0
            if np.isfinite(hh) and hh > 0 and last_close > hh * (1.0 + buf):
                breakout_dir = 1
                breakout_excess_bps = float(((last_close - hh * (1.0 + buf)) / last_close) * 10_000.0)
            elif np.isfinite(ll) and ll > 0 and last_close < ll * (1.0 - buf):
                breakout_dir = -1
                breakout_excess_bps = float(((ll * (1.0 - buf) - last_close) / last_close) * 10_000.0)

            # Regime flags
            is_trend = (er >= float(self.er_trend_min)) and (abs(trend_strength) >= float(self.trend_z_min))
            is_range = er <= float(self.er_range_max)

            # VWAP alignment (trend trades only)
            vwap = float(f["vwap"])
            vwap_ok = True
            if bool(self.require_vwap_alignment_for_trend) and vwap > 0:
                vwap_ok = (last_close > vwap) if trend_dir > 0 else (last_close < vwap) if trend_dir < 0 else False

            # --- in-position management ---
            pos_qty = float(pos_qty_eff.get(s, 0.0) or 0.0)
            pos_side = _sign(pos_qty)
            hold_bars = int(state.holding_bars.get(s, 0) or 0)
            if pos_side != 0:
                entry_mode = str(self._pos.entry_reason.get(s, "") or "")
                entry = float(self._pos.entry_price.get(s, last_close))
                prev_ext = float(self._pos.trail_extreme.get(s, last_close))
                stop_mult = float(self.stop_atr_mult)
                trail_mult = float(self.trail_atr_mult)
                if entry_mode == "mean_reversion":
                    if float(self.meanrev_stop_atr_mult) > 0:
                        stop_mult = float(self.meanrev_stop_atr_mult)
                    if float(self.meanrev_trail_atr_mult) > 0:
                        trail_mult = float(self.meanrev_trail_atr_mult)
                if pos_side > 0:
                    trail_stop = float(prev_ext) - float(trail_mult) * atr
                    hard_stop = entry - float(stop_mult) * atr
                    effective_stop = max(hard_stop, trail_stop)
                    stop_hit = bool(float(bar_low) <= effective_stop)
                    tp_level = entry + float(self.take_profit_atr_mult) * atr
                    tp_hit = bool(float(self.take_profit_atr_mult) > 0 and float(bar_high) >= tp_level)
                else:
                    trail_stop = float(prev_ext) + float(trail_mult) * atr
                    hard_stop = entry + float(stop_mult) * atr
                    effective_stop = min(hard_stop, trail_stop)
                    stop_hit = bool(float(bar_high) >= effective_stop)
                    tp_level = entry - float(self.take_profit_atr_mult) * atr
                    tp_hit = bool(float(self.take_profit_atr_mult) > 0 and float(bar_low) <= tp_level)

                if stop_hit:
                    desired_dir[s] = 0
                    reason_tag[s] = "stop"
                    score[s] = 0.0
                    execution_hints[s] = {
                        "mode": "min" if pos_side > 0 else "max",
                        "price": float(effective_stop),
                    }
                    continue

                if tp_hit:
                    desired_dir[s] = 0
                    reason_tag[s] = "take_profit"
                    score[s] = 0.0
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
                    reason_tag[s] = "min_hold"
                    score[s] = 1e6  # force keep
                    continue

                if int(self.max_hold_bars) > 0 and hold_bars >= int(self.max_hold_bars):
                    desired_dir[s] = 0
                    reason_tag[s] = "time_stop"
                    score[s] = 0.0
                    continue

                if (
                    entry_mode == "mean_reversion"
                    and int(self.meanrev_max_hold_bars) > 0
                    and hold_bars >= int(self.meanrev_max_hold_bars)
                ):
                    desired_dir[s] = 0
                    reason_tag[s] = "meanrev_time_stop"
                    score[s] = 0.0
                    continue

                if entry_mode == "mean_reversion" and abs(z) <= abs(float(self.meanrev_exit_z)):
                    desired_dir[s] = 0
                    reason_tag[s] = "meanrev_exit"
                    score[s] = 0.0
                    continue

                # Optional flip in strong opposite trend regime.
                flip_ok = bool(state.allow_short) and is_trend and vwap_ok and (trend_dir == -pos_side)
                if flip_ok:
                    self._pos.flip_counter[s] = int(self._pos.flip_counter.get(s, 0)) + 1
                else:
                    self._pos.flip_counter[s] = 0

                if self._pos.flip_counter[s] >= int(self.flip_confirm_bars):
                    desired_dir[s] = -pos_side
                    reason_tag[s] = "flip"
                    score[s] = 10.0  # keep but allow rank-cut
                    continue

                desired_dir[s] = pos_side
                reason_tag[s] = "hold"
                score[s] = 10.0
                continue

            # --- flat: respect cooldown ---
            if self._bars_seen < int(self._pos.cooldown_until_bar.get(s, 0)):
                desired_dir[s] = 0
                reason_tag[s] = "cooldown"
                continue

            # --- signal persistence (confirmation) ---
            if is_trend and vwap_ok and mom_dir != 0 and mom_dir == trend_dir:
                self._pos.trend_confirm[s] = int(self._pos.trend_confirm.get(s, 0)) + 1
            else:
                self._pos.trend_confirm[s] = 0

            if is_trend and vwap_ok and breakout_dir != 0 and breakout_dir == trend_dir:
                self._pos.breakout_confirm[s] = int(self._pos.breakout_confirm.get(s, 0)) + 1
            else:
                self._pos.breakout_confirm[s] = 0

            # --- entry candidates ---
            candidates: list[tuple[int, float, str]] = []

            # Trend / breakout continuation (requires trend regime + breakout alignment)
            if (
                is_trend
                and vwap_ok
                and breakout_dir != 0
                and breakout_dir == trend_dir
                and int(self._pos.breakout_confirm.get(s, 0)) >= int(confirm_required)
            ):
                edge_bps = float(max(0.0, breakout_excess_bps) + 0.35 * abs(trend_strength) * atr_bps)
                edge_bps += float(max(0.0, abs(mom)) * 0.10 * 10_000.0)
                net = float(edge_bps - float(self.k_cost) * cost_rt_bps)
                if edge_bps >= required_edge and net > 0:
                    candidates.append((int(breakout_dir), float(net), "trend_breakout"))

            # Pure momentum continuation (trend regime but no strict breakout)
            if (
                is_trend
                and vwap_ok
                and mom_dir != 0
                and mom_dir == trend_dir
                and int(self._pos.trend_confirm.get(s, 0)) >= int(confirm_required)
                and breakout_dir == 0
            ):
                edge_bps = float(0.25 * abs(trend_strength) * atr_bps + 0.15 * abs(mom) * 10_000.0)
                net = float(edge_bps - float(self.k_cost) * cost_rt_bps)
                if edge_bps >= required_edge and net > 0:
                    candidates.append((int(trend_dir), float(net), "trend_momentum"))

            # Mean reversion (range regime): fade extreme z-score with RSI confirmation.
            #
            # Optional: in long-only mode, allow mean reversion even during bearish trend regimes
            # (buy dips) to avoid fully-idle periods when short signals are blocked.
            allow_mr_here = bool(allow_mean_reversion) and (
                bool(is_range)
                or (
                    bool(self.meanrev_allow_bear_trend_long_only)
                    and (not bool(state.allow_short))
                    and int(trend_dir) < 0
                )
            )
            if allow_mr_here:
                setup_dir = 0
                if z <= -abs(float(self.meanrev_entry_z)) and rsi <= float(self.rsi_oversold):
                    setup_dir = 1
                elif bool(state.allow_short) and z >= abs(float(self.meanrev_entry_z)) and rsi >= float(self.rsi_overbought):
                    setup_dir = -1

                mr_dir = 0
                if bool(is_range):
                    mr_dir = int(setup_dir)
                else:
                    # In bearish trend regimes (long-only), avoid catching a falling knife by arming a
                    # setup at extremes and then requiring a reversal confirmation to enter.
                    until = int(self._pos.meanrev_setup_until_bar.get(s, 0) or 0)
                    if until > 0 and int(self._bars_seen) > until:
                        self._pos.meanrev_setup_dir.pop(s, None)
                        self._pos.meanrev_setup_until_bar.pop(s, None)

                    if int(setup_dir) > 0:
                        self._pos.meanrev_setup_dir[s] = int(setup_dir)
                        self._pos.meanrev_setup_until_bar[s] = int(
                            int(self._bars_seen) + max(1, int(self.meanrev_setup_max_bars))
                        )

                    armed = int(self._pos.meanrev_setup_dir.get(s, 0) or 0)
                    until = int(self._pos.meanrev_setup_until_bar.get(s, 0) or 0)
                    if armed != 0 and int(self._bars_seen) <= until:
                        rev_bps = float(self.meanrev_reversal_min_bps)
                        if armed > 0 and float(ret_bps) >= rev_bps and z <= -abs(float(self.meanrev_exit_z)):
                            mr_dir = 1
                        elif armed < 0 and float(ret_bps) <= -rev_bps and z >= abs(float(self.meanrev_exit_z)):
                            mr_dir = -1

                if mr_dir != 0:
                    # Edge proxy: z magnitude scaled by ATR (bps).
                    edge_bps = float(min(6.0, abs(z)) * 0.50 * atr_bps)
                    net = float(edge_bps - float(self.k_cost) * cost_rt_bps)
                    if edge_bps >= required_edge and net > 0:
                        candidates.append((int(mr_dir), float(net), "mean_reversion"))

            if not candidates:
                reason_tag[s] = "no_entry"
                continue

            # Best candidate for this symbol (respect long-only mode by selecting the best long
            # candidate rather than rejecting the symbol outright if the best overall candidate is short).
            allowed = (
                candidates
                if bool(state.allow_short)
                else [c for c in candidates if int(c[0]) > 0]
            )
            if not allowed:
                reason_tag[s] = "short_blocked"
                continue
            best_dir, best_score, best_reason = max(allowed, key=lambda t: float(t[1]))

            desired_dir[s] = int(best_dir)
            score[s] = float(best_score)
            reason_tag[s] = str(best_reason)

        # --- rank-cut / portfolio selection ---
        active_syms = [s for s in universe if int(desired_dir.get(s, 0)) != 0]
        if not active_syms:
            if len(universe) <= 6:
                debug["features_all"] = {
                    s: {
                        k: float(v)
                        for k, v in feats.get(s, {}).items()
                        if k
                        in {
                            "atr_bps",
                            "er",
                            "trend_strength",
                            "z",
                            "mom",
                            "rsi",
                            "dv_ewma",
                        }
                    }
                    for s in universe
                    if s in feats
                }
            debug["reason_tag"] = reason_tag
            exposures = {s: 0.0 for s in universe}
            exposures = self._maybe_apply_heartbeat(
                exposures=exposures,
                universe=universe,
                state=state,
                market_symbol=mkt_sym,
                max_notional_usd=max_notional,
                debug=debug,
            )
            reason = "no_signal"
            if "heartbeat" in debug:
                reason = "heartbeat_no_signal"
            return StrategyDecision(
                target_exposures=exposures,
                reason=reason,
                debug=debug,
                execution_hints=(execution_hints or None),
            )

        max_positions = max(1, int(self.max_positions))
        # Prefer existing positions first, then by score.
        def _rank_key(sym: str) -> tuple[int, float]:
            pos_side = _sign(float(pos_qty_eff.get(sym, 0.0) or 0.0))
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

        # --- sizing (risk-based with confidence scaling) ---
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

            # Confidence scaling based on score (net edge bps).
            net_edge = float(score.get(s, 0.0))
            conf = _clamp(net_edge / max(1e-9, float(required_edge)), 0.0, 1.0)
            notional *= float(conf)

            # Reduce sizing for mean-reversion entries (tends to have lower SNR and higher adverse selection).
            pos_qty = float(pos_qty_eff.get(s, 0.0) or 0.0)
            mode = (
                str(self._pos.entry_reason.get(s, "") or "")
                if abs(pos_qty) > 1e-9
                else str(reason_tag.get(s, "") or "")
            )
            if mode == "mean_reversion":
                notional *= float(_clamp(float(self.meanrev_size_mult), 0.0, 1.0))

            # Per-symbol cap in exposure units.
            cap = float(max_notional) * float(self.max_exposure_per_symbol)
            notional = min(float(notional), float(cap))
            if notional < float(self.min_trade_notional_usd):
                pos_qty = float(pos_qty_eff.get(s, 0.0) or 0.0)
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

        # Convert to exposures and enforce gross exposure cap.
        exposures: dict[str, float] = {s: 0.0 for s in universe}
        for s in universe:
            d = int(desired_dir.get(s, 0))
            if d == 0:
                exposures[s] = 0.0
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

        # Apply global risk scale (market regime)
        exposures = {s: float(exposures[s]) * float(risk_scale) for s in universe}

        # Record pending entry reasons for fills that will occur at the next bar open.
        for s in universe:
            pos_qty = float(pos_qty_eff.get(s, 0.0) or 0.0)
            if abs(pos_qty) > 1e-9:
                continue
            if abs(float(exposures.get(s, 0.0))) <= 1e-9:
                self._pos.pending_entry_reason.pop(s, None)
                continue
            self._pos.pending_entry_reason[s] = str(reason_tag.get(s, "") or "")

        # Rebalance threshold: avoid churn for tiny exposure diffs when holding.
        for s in universe:
            pos_qty = float(pos_qty_eff.get(s, 0.0) or 0.0)
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

        exposures = self._maybe_apply_heartbeat(
            exposures=exposures,
            universe=universe,
            state=state,
            market_symbol=mkt_sym,
            max_notional_usd=max_notional,
            debug=debug,
        )

        debug["reason_tag"] = reason_tag
        debug["active"] = active
        debug["scores"] = {s: float(score.get(s, 0.0)) for s in universe}
        if len(universe) <= 6:
            debug["features_all"] = {
                s: {
                    k: float(v)
                    for k, v in feats.get(s, {}).items()
                    if k
                    in {
                        "atr_bps",
                        "er",
                        "trend_strength",
                        "z",
                        "mom",
                        "rsi",
                        "dv_ewma",
                    }
                }
                for s in universe
                if s in feats
            }
        debug["features"] = {
            s: {k: float(v) for k, v in feats[s].items() if k in {"atr_bps", "er", "trend_strength", "z", "mom", "rsi", "dv_ewma"}}
            for s in active
        }

        reason = "active=" + ",".join(sorted([s for s in universe if abs(exposures.get(s, 0.0)) > 1e-9]))
        return StrategyDecision(
            target_exposures=exposures,
            reason=reason,
            debug=debug,
            execution_hints=(execution_hints or None),
        )
