from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, Optional

import numpy as np
import pandas as pd

from atlas.strategies.base import Strategy, StrategyDecision, StrategyState
from atlas.utils.time import NY_TZ


def _clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))


def _safe_float(x: Any, *, default: float = 0.0) -> float:
    try:
        v = float(x)
    except Exception:
        return float(default)
    return float(v) if np.isfinite(v) else float(default)


def _to_ny(ts: pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    if ts.tz is None:
        return ts.tz_localize(NY_TZ)
    return ts.tz_convert(NY_TZ)


def _is_spot_crypto_symbol(symbol: str) -> bool:
    s = (symbol or "").strip().upper()
    if not s:
        return False
    if s.endswith("-PERP") or s.endswith("-CDE"):
        return False
    return ("/" in s) or ("-" in s)


def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr1 = (high - low).abs()
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int) -> pd.Series:
    w = int(max(2, window))
    tr = _true_range(high, low, close)
    return tr.rolling(w).mean()


@dataclass
class _MarketTracker:
    peak_price: float = 0.0
    last_price: float = 0.0
    ret_ewma_var: float = 0.0
    last_ts: Optional[pd.Timestamp] = None


@dataclass
class CryptoVolSqueeze(Strategy):
    """
    Long-only crypto breakout strategy:
    - Requires volatility compression (Bollinger bandwidth squeeze).
    - Enters only on cost-aware Donchian breakouts.
    - Rebalances on sparse cadence and exits on momentum fade / time stop.
    """

    name: str = "crypto_vol_squeeze"

    # ---- Universe ----
    symbols: tuple[str, ...] = ("BTC/USD", "ETH/USD", "SOL/USD", "AVAX/USD")
    market_symbol: Optional[str] = "BTC/USD"

    # ---- Schedule / trade hygiene ----
    rebalance_interval_bars: int = 28
    rebalance_exposure_threshold: float = 0.05
    min_trade_notional_usd: float = 25.0

    # ---- Squeeze + breakout ----
    bb_window: int = 20
    bb_k: float = 2.0
    squeeze_lookback: int = 120
    squeeze_percentile: float = 0.20
    donchian_window: int = 40
    atr_window: int = 20
    min_atr_bps: float = 12.0
    entry_breakout_buffer_bps: float = 8.0
    expected_move_atr_mult: float = 2.0
    cost_k: float = 2.5
    edge_floor_bps: float = 4.0
    slippage_bps: float = 3.0
    taker_fee_bps: float = 25.0

    # ---- Sizing ----
    max_total_exposure: float = 1.0
    max_exposure_per_symbol: float = 0.55
    vol_target_bps_per_bar: float = 70.0
    exposure_scale_on_squeeze: float = 1.0

    # ---- Exit ----
    min_hold_bars: int = 12
    max_hold_bars: int = 56
    exit_mom_bars: int = 24
    exit_mom_threshold: float = 0.0

    # ---- Global risk overlay ----
    daily_loss_limit: float = 0.03
    kill_switch: float = 0.12
    kill_switch_cooldown_days: int = 4
    market_drawdown_off: float = 0.20
    market_drawdown_reduce: float = 0.10
    market_vol_off_bps: float = 260.0
    market_vol_reduce_bps: float = 160.0

    # ---- Internal ----
    _bars_seen: int = field(default=0, init=False, repr=False)
    _last_rebalance_bar: int = field(default=0, init=False, repr=False)
    _last_targets: dict[str, float] = field(default_factory=dict, init=False, repr=False)
    _entry_bar: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _peak_equity: float = field(default=0.0, init=False, repr=False)
    _risk_disabled_day: Optional[object] = field(default=None, init=False, repr=False)
    _risk_disabled_until_day: Optional[object] = field(default=None, init=False, repr=False)
    _mkt: _MarketTracker = field(default_factory=_MarketTracker, init=False, repr=False)

    def warmup_bars(self) -> int:
        return int(
            max(
                int(self.bb_window) + 3,
                int(self.squeeze_lookback) + 3,
                int(self.donchian_window) + 3,
                int(self.atr_window) + 3,
                int(self.exit_mom_bars) + 3,
            )
            + 8
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

    def _maybe_reset_daily_state(self, state: StrategyState) -> None:
        today = _to_ny(pd.Timestamp(state.timestamp)).date()
        if self._risk_disabled_day is not None and self._risk_disabled_day != today:
            self._risk_disabled_day = None
        if self._risk_disabled_until_day is not None and today > self._risk_disabled_until_day:
            self._risk_disabled_until_day = None

    def _risk_off(
        self, universe: list[str], *, reason: str, debug: dict[str, Any]
    ) -> StrategyDecision:
        targets = {s: 0.0 for s in universe}
        self._last_targets = dict(targets)
        return StrategyDecision(target_exposures=targets, reason=reason, debug=debug)

    def _update_market_tracker(self, df: pd.DataFrame) -> None:
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

        alpha = float(2.0 / (40.0 + 1.0))
        for ts, row in new_df.iterrows():
            c = _safe_float(row.get("close"), default=0.0)
            if c <= 0:
                continue
            if self._mkt.last_price > 0:
                r = math.log(c / self._mkt.last_price)
            else:
                r = 0.0
            self._mkt.ret_ewma_var = float(
                (1.0 - alpha) * float(self._mkt.ret_ewma_var) + alpha * float(r * r)
            )
            self._mkt.last_price = float(c)
            self._mkt.peak_price = max(float(self._mkt.peak_price), float(c))
            self._mkt.last_ts = pd.Timestamp(ts)

    def _market_risk_scale(self) -> tuple[float, dict[str, float]]:
        last = float(self._mkt.last_price)
        peak = float(self._mkt.peak_price)
        dd = (last / peak - 1.0) if (peak > 0 and last > 0) else 0.0
        vol_bps = float(math.sqrt(max(0.0, float(self._mkt.ret_ewma_var))) * 10_000.0)

        scale = 1.0
        if dd <= -abs(float(self.market_drawdown_off)) or vol_bps >= abs(float(self.market_vol_off_bps)):
            scale = 0.0
        elif dd <= -abs(float(self.market_drawdown_reduce)) or vol_bps >= abs(float(self.market_vol_reduce_bps)):
            scale = 0.5
        return float(scale), {
            "market_drawdown": float(dd),
            "market_vol_bps": float(vol_bps),
        }

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
        close = close.dropna()
        if len(close) < self.warmup_bars():
            return None

        c = float(close.iloc[-1])
        if c <= 0:
            return None

        ma = close.rolling(int(self.bb_window)).mean()
        sd = close.rolling(int(self.bb_window)).std(ddof=0)
        upper = ma + float(self.bb_k) * sd
        lower = ma - float(self.bb_k) * sd
        bw = (upper - lower) / ma.replace(0.0, np.nan)

        bw_tail = bw.dropna().tail(int(self.squeeze_lookback))
        if len(bw_tail) < max(10, int(self.squeeze_lookback // 3)):
            return None
        bw_now = float(bw_tail.iloc[-1])
        bw_thr = float(bw_tail.quantile(float(_clamp(self.squeeze_percentile, 0.01, 0.99))))
        in_squeeze = bw_now <= bw_thr

        dc_w = int(self.donchian_window)
        if len(high) < dc_w + 2 or len(low) < dc_w + 2:
            return None
        hh = float(high.iloc[-dc_w - 1 : -1].max())
        ll = float(low.iloc[-dc_w - 1 : -1].min())

        atr_series = _atr(high, low, close, int(self.atr_window))
        atr_now = float(atr_series.iloc[-1]) if len(atr_series) > 0 else float("nan")
        atr_bps = float((atr_now / c) * 10_000.0) if np.isfinite(atr_now) and c > 0 else 0.0

        mb = int(max(2, self.exit_mom_bars))
        if len(close) <= mb:
            return None
        base = float(close.iloc[-mb - 1])
        mom = float(math.log(c / base)) if base > 0 and c > 0 else 0.0

        up_level = float(hh) * (1.0 + float(self.entry_breakout_buffer_bps) / 10_000.0)
        dn_level = float(ll) * (1.0 - float(self.entry_breakout_buffer_bps) / 10_000.0)
        breakout_up = c > up_level
        breakout_down = c < dn_level

        return {
            "close": float(c),
            "atr": float(atr_now) if np.isfinite(atr_now) else 0.0,
            "atr_bps": float(atr_bps),
            "mom": float(mom),
            "bw": float(bw_now),
            "bw_thr": float(bw_thr),
            "in_squeeze": float(1.0 if in_squeeze else 0.0),
            "breakout_up": float(1.0 if breakout_up else 0.0),
            "breakout_down": float(1.0 if breakout_down else 0.0),
            "score": float(max(0.0, mom * 10_000.0) + max(0.0, atr_bps)),
        }

    def target_exposures(
        self, bars_by_symbol: dict[str, pd.DataFrame], state: StrategyState
    ) -> StrategyDecision:
        universe = self._universe(bars_by_symbol)
        if not universe:
            return StrategyDecision(target_exposures={}, reason="no_crypto_symbols")

        self._bars_seen += 1
        self._maybe_reset_daily_state(state)

        equity = float(state.equity)
        if self._peak_equity <= 0:
            self._peak_equity = equity
        self._peak_equity = max(self._peak_equity, equity)
        drawdown = (equity / self._peak_equity - 1.0) if self._peak_equity > 0 else 0.0
        today = _to_ny(pd.Timestamp(state.timestamp)).date()

        debug: dict[str, Any] = {
            "bars_seen": int(self._bars_seen),
            "equity": float(equity),
            "drawdown": float(drawdown),
            "day_return": float(state.day_return),
        }

        if self._risk_disabled_until_day is not None and today <= self._risk_disabled_until_day:
            return self._risk_off(universe, reason="risk_disabled_cooldown", debug=debug)
        if drawdown <= -abs(float(self.kill_switch)):
            self._peak_equity = float(equity)
            self._risk_disabled_until_day = today + timedelta(days=int(self.kill_switch_cooldown_days))
            return self._risk_off(universe, reason="kill_switch", debug=debug)
        if float(state.day_return) <= -abs(float(self.daily_loss_limit)):
            self._risk_disabled_day = today
            return self._risk_off(universe, reason="daily_loss_limit", debug=debug)
        if self._risk_disabled_day == today:
            return self._risk_off(universe, reason="risk_disabled_day", debug=debug)

        market_symbol = (self.market_symbol or "").strip().upper()
        if not market_symbol or market_symbol not in universe:
            market_symbol = universe[0]
        self._update_market_tracker(bars_by_symbol.get(market_symbol))
        risk_scale, mdbg = self._market_risk_scale()
        debug["market"] = mdbg
        debug["risk_scale"] = float(risk_scale)

        extra = dict(state.extra or {})
        max_notional = float(extra.get("max_position_notional_usd", 0.0) or 0.0) or 1.0
        slippage_bps = float(extra.get("slippage_bps", self.slippage_bps) or 0.0)
        taker_fee_bps = float(extra.get("taker_fee_bps", self.taker_fee_bps) or 0.0)
        cost_rt_bps = float(2.0 * (abs(slippage_bps) + abs(taker_fee_bps)))
        min_exp = float(self.min_trade_notional_usd) / float(max_notional)
        min_exp = float(_clamp(min_exp, 0.0, 1.0))
        debug["cost_rt_bps"] = float(cost_rt_bps)

        due = (self._bars_seen - int(self._last_rebalance_bar)) >= max(1, int(self.rebalance_interval_bars))
        if not self._last_targets:
            due = True

        features: dict[str, dict[str, float]] = {}
        for s in universe:
            f = self._features(bars_by_symbol.get(s))
            if f is not None:
                features[s] = f
        if not features:
            return StrategyDecision(target_exposures={s: 0.0 for s in universe}, reason="warmup", debug=debug)

        prev_targets = {s: float(self._last_targets.get(s, 0.0)) for s in universe}
        targets = dict(prev_targets)
        active: list[str] = [s for s in universe if float(targets.get(s, 0.0)) > 1e-8]

        exits: set[str] = set()
        entries: list[str] = []
        for s in active:
            f = features.get(s)
            if f is None:
                exits.add(s)
                continue
            held = int(self._bars_seen - int(self._entry_bar.get(s, self._bars_seen)))
            can_exit = held >= int(max(1, self.min_hold_bars))
            mom_fade = float(f.get("mom", 0.0)) <= float(self.exit_mom_threshold)
            breakout_down = float(f.get("breakout_down", 0.0)) > 0.5
            timed_out = int(self.max_hold_bars) > 0 and held >= int(self.max_hold_bars)
            if (can_exit and (mom_fade or breakout_down)) or timed_out:
                exits.add(s)

        for s in exits:
            targets[s] = 0.0
            if s in active:
                active.remove(s)
            self._entry_bar.pop(s, None)

        if due:
            for s, f in features.items():
                if s in active:
                    continue
                if float(f.get("in_squeeze", 0.0)) <= 0.5:
                    continue
                if float(f.get("breakout_up", 0.0)) <= 0.5:
                    continue
                atr_bps = float(f.get("atr_bps", 0.0))
                if atr_bps < float(self.min_atr_bps):
                    continue
                expected_move_bps = float(self.expected_move_atr_mult) * float(atr_bps)
                required_edge_bps = float(self.edge_floor_bps) + float(self.cost_k) * float(cost_rt_bps)
                if expected_move_bps <= required_edge_bps:
                    continue
                entries.append(s)

            selected = active + entries
            selected = list(dict.fromkeys(selected))
            if selected:
                mkt_vol_bps = float(mdbg.get("market_vol_bps", 0.0) or 0.0)
                vol_scale = 1.0
                if mkt_vol_bps > 1e-9 and float(self.vol_target_bps_per_bar) > 0:
                    vol_scale = float(self.vol_target_bps_per_bar) / float(mkt_vol_bps)
                vol_scale = _clamp(vol_scale, 0.0, 1.0)

                total_exposure = float(self.max_total_exposure) * float(risk_scale)
                total_exposure *= float(_clamp(self.exposure_scale_on_squeeze, 0.0, 1.5))
                total_exposure *= float(vol_scale)
                total_exposure = _clamp(total_exposure, 0.0, float(self.max_total_exposure))

                raw_w: list[tuple[str, float]] = []
                for s in selected:
                    f = features.get(s)
                    if f is None:
                        continue
                    score = float(f.get("score", 0.0))
                    atr_bps = float(max(1.0, f.get("atr_bps", 1.0)))
                    raw_w.append((s, max(0.0, score) / atr_bps))
                denom = float(sum(w for _, w in raw_w))
                if denom <= 1e-12:
                    targets = {s: 0.0 for s in universe}
                    self._entry_bar.clear()
                else:
                    new_targets = {s: 0.0 for s in universe}
                    for s, w in raw_w:
                        exp = float(total_exposure) * (float(w) / float(denom))
                        exp = _clamp(exp, 0.0, float(self.max_exposure_per_symbol))
                        if exp < min_exp:
                            exp = 0.0
                        new_targets[s] = float(exp)
                    targets = new_targets
                    for s in selected:
                        if float(targets.get(s, 0.0)) > 1e-8 and s not in self._entry_bar:
                            self._entry_bar[s] = int(self._bars_seen)
            else:
                targets = {s: 0.0 for s in universe}
                self._entry_bar.clear()

            self._last_rebalance_bar = int(self._bars_seen)

        for s in universe:
            prev = float(prev_targets.get(s, 0.0))
            now = float(targets.get(s, 0.0))
            if prev > 0 and now > 0 and abs(now - prev) < float(self.rebalance_exposure_threshold):
                targets[s] = prev

        self._last_targets = {s: float(targets.get(s, 0.0)) for s in universe}
        debug["due"] = bool(due)
        debug["active"] = active
        debug["entries"] = entries
        return StrategyDecision(target_exposures=self._last_targets, reason="rebalance" if due else "hold", debug=debug)
