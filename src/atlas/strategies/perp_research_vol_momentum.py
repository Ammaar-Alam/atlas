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
    val = float(tr.tail(max(2, int(window))).mean())
    if not np.isfinite(val) or val <= 0.0:
        return None
    return float(val)


def _ewma_rms(rets: pd.Series, span: int) -> Optional[float]:
    x = pd.to_numeric(rets, errors="coerce").dropna()
    if len(x) < 2:
        return None
    span = max(2, int(span))
    v = float((x * x).ewm(span=span, adjust=False).mean().iloc[-1])
    if not np.isfinite(v) or v < 0.0:
        return None
    return float(math.sqrt(v))


def _efficiency_ratio(close: pd.Series, window: int) -> Optional[float]:
    c = pd.to_numeric(close, errors="coerce").dropna()
    window = int(max(2, window))
    if len(c) < window + 2:
        return None
    c_tail = c.iloc[-(window + 1) :]
    net = float(abs(c_tail.iloc[-1] - c_tail.iloc[0]))
    diffs = c_tail.diff().abs().dropna()
    denom = float(diffs.sum())
    if not np.isfinite(denom) or denom <= 0.0:
        return 0.0
    return float(_clamp(net / denom, 0.0, 1.0))


def _linreg_tstat(y: np.ndarray) -> Optional[float]:
    if y is None:
        return None
    y = np.asarray(y, dtype=float)
    n = int(y.shape[0])
    if n < 8:
        return None
    x = np.arange(n, dtype=float)
    x_mean = float(x.mean())
    y_mean = float(y.mean())
    xx = x - x_mean
    yy = y - y_mean
    sxx = float(np.dot(xx, xx))
    if not np.isfinite(sxx) or sxx <= 1e-12:
        return None
    b = float(np.dot(xx, yy) / sxx)
    a = float(y_mean - b * x_mean)
    resid = y - (a + b * x)
    sse = float(np.dot(resid, resid))
    dof = n - 2
    if not np.isfinite(sse) or dof <= 2:
        return None
    s2 = sse / float(dof)
    se_b = math.sqrt(max(s2 / sxx, 1e-18))
    t = float(b / se_b) if se_b > 0 else 0.0
    if not np.isfinite(t):
        return None
    return float(t)


def _fixed_fee_bps_side(
    price: float, contract_size_units: float, fixed_fee_per_contract_usd: float
) -> float:
    p = float(price)
    cs = float(contract_size_units)
    ff = float(fixed_fee_per_contract_usd)
    if p <= 0.0 or cs <= 0.0 or ff <= 0.0:
        return 0.0
    notional_per_contract = p * cs
    if notional_per_contract <= 0.0:
        return 0.0
    return float((ff / notional_per_contract) * 10_000.0)


def _quantize_qty_to_contracts(qty_btc: float, contract_size_units: float, mode: str) -> float:
    cs = float(contract_size_units)
    if cs <= 0.0:
        return float(qty_btc)
    q = float(qty_btc)
    sign = 1.0 if q >= 0.0 else -1.0
    n = abs(q) / cs
    if mode == "ceil":
        k = int(math.ceil(n - 1e-12))
    elif mode == "round":
        k = int(round(n))
    else:
        k = int(math.floor(n + 1e-12))
    return float(sign * k * cs)


def _utc_week_key(ts: pd.Timestamp) -> tuple[int, int]:
    ts = pd.Timestamp(ts)
    if ts.tz is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    iso = ts.isocalendar()
    return int(iso.year), int(iso.week)


def _to_utc(ts: pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    if ts.tz is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


@dataclass
class PerpResearchVolMomentum(Strategy):
    """
    Research-driven derivatives strategy:
    - Time-series momentum direction (Moskowitz/Ooi/Pedersen, 2012).
    - Volatility-managed sizing (Moreira/Muir, 2017 style scaling).
    - Crash-state de-risking inspired by momentum crash literature.
    - Weekly rebalance cadence to reduce turnover/fees.
    """

    name: str = "perp_research_vol_momentum"
    symbols: tuple[str, ...] = ("BTC-PERP",)

    # Rebalance schedule (UTC)
    rebalance_weekday_utc: int = 0
    rebalance_days_utc: tuple[int, ...] = (0,)
    rebalance_hour_utc: int = 0
    rebalance_minute_utc: int = 0

    # Signal windows
    long_momentum_bars: int = 24 * 14
    short_momentum_bars: int = 24 * 2
    ema_fast: int = 24
    ema_slow: int = 24 * 7
    atr_window: int = 24 * 2
    vol_lookback_bars: int = 24 * 5
    vol_regime_window: int = 24 * 30

    # Signal/edge gates
    min_abs_long_momentum_bps: float = 45.0
    min_atr_bps: float = 8.0
    trend_strength_min: float = 0.10
    edge_floor_bps: float = 8.0
    k_cost: float = 2.6
    expected_hold_bars: int = 120
    signal_decay_factor: float = 0.55
    min_net_edge_bps: float = 18.0
    trend_consistency_min: float = 0.75
    trend_consistency_subwindows: int = 4

    # Volatility-managed sizing
    target_vol_per_bar: float = 0.0065
    vol_floor: float = 0.0020
    max_leverage: float = 4.0
    max_margin_utilization: float = 0.40
    max_gross_exposure: float = 0.95
    max_per_symbol_exposure: float = 0.95
    max_positions: int = 1
    min_trade_notional_usd: float = 25.0
    rebalance_exposure_threshold: float = 0.04
    vol_pctl_low: float = 0.15
    vol_pctl_high: float = 0.82

    # Momentum crash / stress-state controls
    crash_vol_z: float = 1.25
    crash_reversal_bps: float = 55.0
    crash_risk_scale: float = 0.30
    vol_off_z: float = 2.4

    # Position management
    stop_atr_mult: float = 3.2
    trail_atr_mult: float = 4.2
    min_hold_bars: int = 24
    max_hold_bars: int = 24 * 10
    max_loss_per_trade_pct: float = 0.015
    weekly_loss_limit: float = 0.03
    daily_loss_limit: float = 0.02
    kill_switch: float = 0.20

    # New signal stack
    mom_h1_bars: int = 48
    mom_h2_bars: int = 168
    mom_h3_bars: int = 504
    mom_h4_bars: int = 1512
    mom_w1: float = 0.15
    mom_w2: float = 0.25
    mom_w3: float = 0.30
    mom_w4: float = 0.30
    mom_z_scale: float = 2.0
    mom_score_min: float = 0.20

    trend_regression_bars: int = 504
    trend_tstat_entry: float = 2.2
    trend_tstat_full: float = 4.0
    trend_tstat_exit: float = 1.0

    er_window_bars: int = 168
    er_min: float = 0.28
    er_full: float = 0.45

    vol_short_span: int = 48
    vol_long_span: int = 336
    vol_ratio_delever: float = 1.25
    vol_ratio_off: float = 1.80
    vol_ratio_power: float = 1.5

    # Contract and cost mechanics
    min_contracts: int = 1
    qty_rounding: str = "floor"
    include_fixed_fee_in_cost: bool = True

    # Exits and cooldown
    mom_exit_score: float = 0.12
    flip_exit_mom_score: float = 0.22
    cooldown_bars: int = 24

    # Optional lockouts
    use_daily_loss_lockout: bool = False
    use_weekly_loss_lockout: bool = False

    # Internal state
    _bars_seen: int = field(default=0, init=False, repr=False)
    _last_rebalance_slot: Optional[tuple[int, int, int]] = field(default=None, init=False, repr=False)
    _last_targets: dict[str, float] = field(default_factory=dict, init=False, repr=False)
    _peak_equity: float = field(default=0.0, init=False, repr=False)
    _risk_disabled_day: Optional[object] = field(default=None, init=False, repr=False)
    _risk_disabled_forever: bool = field(default=False, init=False, repr=False)
    _week_key: Optional[tuple[int, int]] = field(default=None, init=False, repr=False)
    _week_start_equity: float = field(default=0.0, init=False, repr=False)
    _entry_bar: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _entry_price: dict[str, float] = field(default_factory=dict, init=False, repr=False)
    _peak_price: dict[str, float] = field(default_factory=dict, init=False, repr=False)
    _trough_price: dict[str, float] = field(default_factory=dict, init=False, repr=False)
    _cooldown_until_bar: dict[str, int] = field(default_factory=dict, init=False, repr=False)

    def warmup_bars(self) -> int:
        return int(
            max(
                int(self.mom_h1_bars) + 3,
                int(self.mom_h2_bars) + 3,
                int(self.mom_h3_bars) + 3,
                int(self.mom_h4_bars) + 3,
                int(self.trend_regression_bars) + 3,
                int(self.er_window_bars) + 3,
                int(self.atr_window) + 3,
                int(self.vol_long_span) + 3,
                int(self.vol_short_span) + 3,
            )
            + 8
        )

    def _is_rebalance_time(self, ts: pd.Timestamp) -> bool:
        ts_utc = _to_utc(ts)
        raw_days = tuple(self.rebalance_days_utc or ())
        if not raw_days:
            raw_days = (int(self.rebalance_weekday_utc),)
        allowed_days = {int(d) % 7 for d in raw_days}
        if int(ts_utc.dayofweek) not in allowed_days:
            return False
        if int(ts_utc.hour) != int(self.rebalance_hour_utc):
            return False
        return int(ts_utc.minute) >= int(self.rebalance_minute_utc)

    def _risk_off(self, symbols: list[str], *, reason: str, debug: dict[str, Any]) -> StrategyDecision:
        targets = {s: 0.0 for s in symbols}
        self._last_targets = dict(targets)
        return StrategyDecision(target_exposures=targets, reason=reason, debug=debug)

    def _features(self, df: pd.DataFrame) -> Optional[dict[str, float]]:
        if df is None or df.empty:
            return None
        if not df.index.is_monotonic_increasing:
            df = df.sort_index()
        if len(df) < self.warmup_bars():
            return None

        close = pd.to_numeric(df.get("close"), errors="coerce").dropna()
        if len(close) < self.warmup_bars():
            return None
        c = float(close.iloc[-1])
        if not np.isfinite(c) or c <= 0.0:
            return None

        atr = _atr(df, int(self.atr_window))
        if atr is None or atr <= 0.0:
            return None
        atr_bps = float((atr / c) * 10_000.0)

        rets = np.log(close / close.shift(1)).dropna()
        if len(rets) < max(int(self.vol_long_span), int(self.vol_short_span), 20):
            return None

        vol_short = _ewma_rms(rets, int(self.vol_short_span))
        vol_long = _ewma_rms(rets, int(self.vol_long_span))
        if vol_short is None or vol_long is None or vol_short <= 0.0 or vol_long <= 0.0:
            return None
        vol_ratio = float(vol_short / max(vol_long, 1e-12))

        er = _efficiency_ratio(close, int(self.er_window_bars))
        if er is None:
            return None

        L = int(max(8, self.trend_regression_bars))
        if len(close) < L + 2:
            return None
        y = np.log(close.tail(L).to_numpy(dtype=float))
        trend_tstat = _linreg_tstat(y)
        if trend_tstat is None:
            return None

        horizons = [
            int(self.mom_h1_bars),
            int(self.mom_h2_bars),
            int(self.mom_h3_bars),
            int(self.mom_h4_bars),
        ]
        weights = [float(self.mom_w1), float(self.mom_w2), float(self.mom_w3), float(self.mom_w4)]
        raw_weight_sum = float(sum(weights))
        w_sum = raw_weight_sum if abs(raw_weight_sum) > 1e-12 else 1.0
        weights = [w / w_sum for w in weights]

        def _mom_bps(h: int) -> Optional[float]:
            h = int(max(2, h))
            if len(close) <= h:
                return None
            c_prev = float(close.iloc[-h - 1])
            if not np.isfinite(c_prev) or c_prev <= 0.0:
                return None
            return float(math.log(c / c_prev) * 10_000.0)

        mom_bps_list: list[float] = []
        mom_signal = 0.0
        for h, w in zip(horizons, weights):
            mb = _mom_bps(h)
            if mb is None:
                return None
            mom_bps_list.append(float(mb))
            z = (mb / 10_000.0) / (float(vol_long) * math.sqrt(float(h)))
            contrib = math.tanh(float(z) / max(float(self.mom_z_scale), 1e-9))
            mom_signal += float(w) * float(contrib)
        mom_score = float(abs(mom_signal))
        mom_h1_bps, mom_h2_bps, mom_h3_bps, mom_h4_bps = [float(x) for x in mom_bps_list]

        return {
            "close": float(c),
            "atr": float(atr),
            "atr_bps": float(atr_bps),
            "vol_short": float(vol_short),
            "vol_long": float(vol_long),
            "vol_ratio": float(vol_ratio),
            "er": float(er),
            "trend_tstat": float(trend_tstat),
            "mom_signal": float(mom_signal),
            "mom_score": float(mom_score),
            "mom_h1_bps": float(mom_h1_bps),
            "mom_h2_bps": float(mom_h2_bps),
            "mom_h3_bps": float(mom_h3_bps),
            "mom_h4_bps": float(mom_h4_bps),
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
        ts_utc = _to_utc(ts)
        today = ts.date()
        debug: dict[str, Any] = {"bars_seen": int(self._bars_seen)}

        if self._risk_disabled_day is not None and self._risk_disabled_day != today:
            self._risk_disabled_day = None

        equity = float(state.equity)
        if self._peak_equity <= 0.0:
            self._peak_equity = equity
        self._peak_equity = max(self._peak_equity, equity)
        drawdown = float(equity / self._peak_equity - 1.0) if self._peak_equity > 0 else 0.0
        debug["drawdown"] = float(drawdown)
        debug["day_return"] = float(state.day_return)

        wk = _utc_week_key(ts_utc)
        if self._week_key is None or self._week_key != wk:
            self._week_key = wk
            self._week_start_equity = equity
        week_ret = float(equity / self._week_start_equity - 1.0) if self._week_start_equity > 0 else 0.0
        debug["week_return"] = float(week_ret)

        if self._risk_disabled_forever:
            return self._risk_off(symbols, reason="kill_switch", debug=debug)
        if drawdown <= -abs(float(self.kill_switch)):
            self._risk_disabled_forever = True
            return self._risk_off(symbols, reason="kill_switch", debug=debug)

        if self.use_daily_loss_lockout:
            if float(state.day_return) <= -abs(float(self.daily_loss_limit)):
                self._risk_disabled_day = today
                return self._risk_off(symbols, reason="daily_loss_limit", debug=debug)
            if self._risk_disabled_day == today:
                return self._risk_off(symbols, reason="risk_disabled_day", debug=debug)

        if self.use_weekly_loss_lockout:
            if week_ret <= -abs(float(self.weekly_loss_limit)):
                return self._risk_off(symbols, reason="weekly_loss_limit", debug=debug)

        extra = dict(state.extra or {})
        max_notional = float(extra.get("max_position_notional_usd", 0.0) or 0.0)
        if max_notional <= 0.0:
            max_notional = 1.0
        mmr = float(extra.get("maintenance_margin_rate", 0.05) or 0.05)
        slippage_bps = float(extra.get("slippage_bps", 1.25) or 0.0)
        taker_fee_bps = float(extra.get("taker_fee_bps", 3.0) or 0.0)
        fixed_fee_usd_default = float(extra.get("fixed_fee_per_contract_usd", 0.0) or 0.0)
        contract_size_units_default = float(extra.get("contract_size_units", 0.0) or 0.0)
        if contract_size_units_default <= 0.0:
            contract_size_units_default = 0.01
        fixed_fee_map_raw = dict(extra.get("fixed_fee_per_contract_usd_by_symbol", {}) or {})
        contract_size_map_raw = dict(extra.get("contract_size_units_by_symbol", {}) or {})

        def _symbol_fixed_fee(symbol: str) -> float:
            raw = fixed_fee_map_raw.get(
                symbol, fixed_fee_map_raw.get(str(symbol).upper(), fixed_fee_usd_default)
            )
            return float(raw or 0.0)

        def _symbol_contract_size(symbol: str) -> float:
            raw = contract_size_map_raw.get(
                symbol, contract_size_map_raw.get(str(symbol).upper(), contract_size_units_default)
            )
            value = float(raw or 0.0)
            return value if value > 0.0 else float(contract_size_units_default)

        features: dict[str, dict[str, float]] = {}
        last_prices: dict[str, float] = {}
        current_exp: dict[str, float] = {}
        current_side: dict[str, int] = {}

        for s in symbols:
            df = bars_by_symbol.get(s)
            f = self._features(df)
            if f is not None:
                features[s] = f
                last_prices[s] = float(f["close"])
            elif df is not None and len(df):
                last_prices[s] = float(pd.to_numeric(df["close"], errors="coerce").iloc[-1])
            else:
                last_prices[s] = 0.0

            px = float(last_prices[s])
            qty = float(state.positions.get(s, 0.0) or 0.0)
            exp = float((qty * px) / max_notional) if px > 0 else 0.0
            current_exp[s] = float(exp)
            current_side[s] = _sign(float(qty))

            if current_side[s] != 0 and px > 0:
                if s not in self._entry_price:
                    self._entry_price[s] = float(px)
                    self._peak_price[s] = float(px)
                    self._trough_price[s] = float(px)
                    self._entry_bar[s] = int(self._bars_seen)
                else:
                    self._peak_price[s] = float(max(float(self._peak_price.get(s, px)), px))
                    self._trough_price[s] = float(min(float(self._trough_price.get(s, px)), px))
            else:
                self._entry_price.pop(s, None)
                self._peak_price.pop(s, None)
                self._trough_price.pop(s, None)
                self._entry_bar.pop(s, None)

        targets = dict(current_exp)

        # Off-cycle exits.
        for s in symbols:
            side = int(current_side.get(s, 0))
            if side == 0:
                continue
            f = features.get(s)
            if f is None:
                targets[s] = 0.0
                continue

            atr = float(f["atr"])
            close = float(f["close"])
            if atr <= 0.0 or close <= 0.0:
                targets[s] = 0.0
                continue

            entry = float(self._entry_price.get(s, close))
            held = int(self._bars_seen - int(self._entry_bar.get(s, self._bars_seen)))
            if held >= int(max(1, self.max_hold_bars)):
                targets[s] = 0.0
                self._cooldown_until_bar[s] = int(self._bars_seen + int(max(0, self.cooldown_bars)))
                continue

            stop_mult_effective = float(self.stop_atr_mult)
            position_notional = abs(float(current_exp.get(s, 0.0))) * float(max_notional)
            atr_frac = float(atr / close) if close > 0.0 else 0.0
            if position_notional > 0.0 and atr_frac > 0.0 and equity > 0.0:
                max_loss_usd = float(max(0.0, self.max_loss_per_trade_pct) * equity)
                denom = float(position_notional * atr_frac)
                if denom > 0.0:
                    stop_cap = float(max_loss_usd / denom)
                    if np.isfinite(stop_cap) and stop_cap > 0.0:
                        stop_mult_effective = float(max(0.5, min(float(self.stop_atr_mult), stop_cap)))

            hard = (
                float(entry - stop_mult_effective * atr)
                if side > 0
                else float(entry + stop_mult_effective * atr)
            )
            if side > 0:
                trail = float(self._peak_price.get(s, close) - self.trail_atr_mult * atr)
                stop_px = float(max(hard, trail))
                stop_hit = bool(close <= stop_px)
            else:
                trail = float(self._trough_price.get(s, close) + self.trail_atr_mult * atr)
                stop_px = float(min(hard, trail))
                stop_hit = bool(close >= stop_px)

            if stop_hit and held >= int(max(1, self.min_hold_bars)):
                targets[s] = 0.0
                self._cooldown_until_bar[s] = int(self._bars_seen + int(max(0, self.cooldown_bars)))
                continue

            if held >= int(max(1, self.min_hold_bars)):
                mom_signal = float(f.get("mom_signal", 0.0))
                mom_score = float(abs(mom_signal))
                trend_t = float(f.get("trend_tstat", 0.0))
                flip = (_sign(mom_signal) != side) and (mom_score >= float(self.flip_exit_mom_score))
                collapse = (mom_score < float(self.mom_exit_score)) or (
                    abs(trend_t) < float(self.trend_tstat_exit)
                )
                if flip or collapse:
                    targets[s] = 0.0
                    self._cooldown_until_bar[s] = int(self._bars_seen + int(max(0, self.cooldown_bars)))

        due = bool(self._is_rebalance_time(ts_utc))
        slot_key = (int(wk[0]), int(wk[1]), int(ts_utc.dayofweek), int(ts_utc.hour))
        if due and self._last_rebalance_slot == slot_key:
            due = False

        if not due:
            self._last_targets = dict(targets)
            debug["due"] = False
            debug["targets"] = {k: float(v) for k, v in targets.items()}
            return StrategyDecision(target_exposures=targets, reason="hold_until_rebalance", debug=debug)

        self._last_rebalance_slot = slot_key
        targets = {s: float(targets.get(s, 0.0)) for s in symbols}

        scored: list[tuple[str, float, float, dict[str, float], dict[str, float]]] = []
        cost_debug: dict[str, dict[str, float]] = {}
        mmr_cap_lev = float(self.max_margin_utilization) / float(max(mmr, 1e-9))
        lev_cap = float(min(float(self.max_leverage), float(max(0.0, mmr_cap_lev))))

        for s in symbols:
            f = features.get(s)
            if f is None:
                continue

            cd_until = int(self._cooldown_until_bar.get(s, 0) or 0)
            if int(self._bars_seen) < cd_until:
                continue

            mom_signal = float(f["mom_signal"])
            mom_score = float(f["mom_score"])
            trend_t = float(f["trend_tstat"])
            er = float(f["er"])
            atr_bps = float(f["atr_bps"])
            vol_short = float(f["vol_short"])
            vol_ratio = float(f["vol_ratio"])
            mom_primary_bps = float(f["mom_h3_bps"])

            if mom_score < float(self.mom_score_min):
                continue
            side = _sign(mom_signal)
            if side == 0:
                continue
            if _sign(trend_t) != side:
                continue
            if abs(trend_t) < float(self.trend_tstat_entry):
                continue
            if er < float(self.er_min):
                continue
            if atr_bps < float(self.min_atr_bps):
                continue
            if vol_ratio >= float(self.vol_ratio_off):
                continue
            if (not state.allow_short) and side < 0:
                continue

            px = float(f["close"])
            contract_size_units = float(_symbol_contract_size(s))
            fixed_fee_usd = float(_symbol_fixed_fee(s))
            fixed_bps_side = (
                _fixed_fee_bps_side(px, contract_size_units, fixed_fee_usd)
                if self.include_fixed_fee_in_cost
                else 0.0
            )
            cost_side_bps = float(abs(slippage_bps) + abs(taker_fee_bps) + abs(fixed_bps_side))
            cost_rt_bps_sym = float(2.0 * cost_side_bps)
            required_mom_bps = float(self.min_abs_long_momentum_bps) + float(self.k_cost) * float(
                cost_rt_bps_sym
            )
            if abs(mom_primary_bps) < required_mom_bps:
                continue

            lev = float(self.target_vol_per_bar) / float(max(float(self.vol_floor), vol_short))
            if vol_ratio > float(self.vol_ratio_delever):
                lev *= float(
                    (float(self.vol_ratio_delever) / float(max(vol_ratio, 1e-12)))
                    ** float(self.vol_ratio_power)
                )
            lev = float(_clamp(lev, 0.0, lev_cap))
            if lev <= 0.0:
                continue

            mom_conf = float(
                _clamp(
                    (mom_score - float(self.mom_score_min))
                    / max(1.0 - float(self.mom_score_min), 1e-9),
                    0.0,
                    1.0,
                )
            )
            trend_conf = float(
                _clamp(
                    (abs(trend_t) - float(self.trend_tstat_entry))
                    / max(float(self.trend_tstat_full) - float(self.trend_tstat_entry), 1e-9),
                    0.0,
                    1.0,
                )
            )
            er_conf = float(
                _clamp(
                    (er - float(self.er_min)) / max(float(self.er_full) - float(self.er_min), 1e-9),
                    0.0,
                    1.0,
                )
            )
            confidence = float(
                _clamp((0.5 * mom_conf + 0.5 * trend_conf) * (0.5 + 0.5 * er_conf), 0.0, 1.0)
            )
            if confidence <= 0.0:
                continue

            notional_target = float(lev) * float(equity) * float(confidence)
            notional_target = float(
                min(notional_target, float(max_notional) * float(self.max_per_symbol_exposure))
            )

            contract_notional = float(px * contract_size_units)
            if contract_notional <= 0.0:
                continue
            min_contracts = int(max(1, int(self.min_contracts)))
            min_notional = float(min_contracts * contract_notional)
            min_notional_effective = float(max(min_notional, float(self.min_trade_notional_usd)))
            if notional_target < min_notional_effective:
                continue

            qty_target = float((notional_target / px) * float(side))
            qty_quant = _quantize_qty_to_contracts(
                qty_target, contract_size_units, str(self.qty_rounding or "floor").lower()
            )
            min_qty = float(min_contracts) * float(contract_size_units)
            if abs(qty_quant) < min_qty:
                qty_quant = float(side) * min_qty

            notional_quant = float(abs(qty_quant) * px)
            exp = float(
                _clamp(
                    notional_quant / float(max_notional),
                    0.0,
                    float(self.max_per_symbol_exposure),
                )
            )
            if exp <= 0.0:
                continue

            score = float(abs(trend_t) * mom_score * confidence)
            score_meta = {
                "cost_rt_bps": float(cost_rt_bps_sym),
                "cost_side_bps": float(cost_side_bps),
                "fixed_bps_side": float(fixed_bps_side),
                "confidence": float(confidence),
                "lev": float(lev),
                "mom_score": float(mom_score),
                "trend_tstat": float(trend_t),
                "er": float(er),
                "contracts": float(abs(qty_quant) / max(contract_size_units, 1e-12)),
            }
            scored.append((s, float(exp * side), score, f, score_meta))
            cost_debug[s] = {
                "fixed_bps_side": float(fixed_bps_side),
                "cost_side_bps": float(cost_side_bps),
                "cost_rt_bps": float(cost_rt_bps_sym),
            }

        scored.sort(key=lambda x: float(x[2]), reverse=True)
        selected = scored[: int(max(0, self.max_positions))]
        targets = {s: 0.0 for s in symbols}
        gross = 0.0
        for s, t, _score, _f, _meta in selected:
            if gross >= float(self.max_gross_exposure):
                break
            avail = float(max(0.0, float(self.max_gross_exposure) - gross))
            sized = float(_clamp(abs(t), 0.0, min(float(self.max_per_symbol_exposure), avail)))
            if sized <= 0.0:
                continue
            targets[s] = float(math.copysign(sized, t))
            gross += float(abs(targets[s]))

        for s in symbols:
            prev = float(current_exp.get(s, 0.0))
            tgt = float(targets.get(s, 0.0))
            if abs(tgt - prev) < float(self.rebalance_exposure_threshold):
                targets[s] = float(prev)

        self._last_targets = dict(targets)
        debug["due"] = True
        debug["selected"] = [s for s, _, _, _, _ in selected]
        debug["selected_meta"] = {s: meta for s, _, _, _, meta in selected}
        debug["targets"] = {k: float(v) for k, v in targets.items()}
        debug["cost_bps"] = cost_debug
        return StrategyDecision(target_exposures=targets, reason="weekly_rebalance", debug=debug)
