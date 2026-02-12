from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd

from atlas.strategies.base import Strategy, StrategyDecision, StrategyState


def _clamp(value: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, value)))


def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr1 = (high - low).abs()
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int) -> pd.Series:
    tr = _true_range(high, low, close)
    return tr.rolling(int(max(2, window))).mean()


def _bars_for_days(index: pd.DatetimeIndex, *, days: float) -> int:
    if len(index) < 2:
        return 28
    diffs = index.to_series().diff().dropna().dt.total_seconds().to_numpy(dtype=float)
    if diffs.size == 0:
        return 28
    dt_sec = float(np.nanmedian(diffs))
    if not np.isfinite(dt_sec) or dt_sec <= 0:
        return 28
    bars = int(round((float(days) * 24.0 * 3600.0) / dt_sec))
    return int(max(2, min(bars, 10000)))


@dataclass
class Crypto7DPositiveGate(Strategy):
    """
    Long-biased, cost-aware crypto strategy optimized for weekly consistency.
    """

    name: str = "crypto_7d_positive_gate"
    symbols: tuple[str, ...] = ("BTC/USD", "ETH/USD")
    market_symbol: Optional[str] = "BTC/USD"

    pos7_lookback_windows: int = 120
    pos7_on: float = 0.60
    pos7_off: float = 0.52
    pos7_reset_floor: float = 0.50

    trend_ema_bars: int = 96
    trend_on: float = 0.005
    trend_off: float = -0.002
    donchian_bars: int = 36
    entry_buffer_bps: float = 4.0

    atr_bars: int = 20
    min_atr_bps: float = 6.0
    expected_move_atr_mult: float = 2.2
    edge_floor_bps: float = 8.0
    taker_fee_bps: float = 25.0
    slippage_bps: float = 3.0

    min_reentry_bars: int = 28
    max_hold_bars: int = 280
    cooldown_bars: int = 32

    max_total_exposure: float = 0.70
    max_exposure_per_symbol: float = 0.40
    vol_target_bps_per_bar: float = 75.0
    edge_scale_bps: float = 40.0
    min_trade_notional_usd: float = 20.0

    stop_atr_mult: float = 2.2
    take_profit_atr_mult: float = 4.0
    rebalance_exposure_threshold: float = 0.06

    daily_loss_limit: float = 0.03
    kill_switch: float = 0.20
    mkt_peak_lookback: int = 120
    mkt_vol_bars: int = 40
    mkt_dd_off: float = 0.20
    mkt_vol_off_bps: float = 200.0

    _bars_seen: int = field(default=0, init=False, repr=False)
    _mode: str = field(default="FLAT", init=False, repr=False)
    _cooldown_until: int = field(default=0, init=False, repr=False)
    _last_trade_bar: int = field(default=-10**9, init=False, repr=False)
    _active_symbol: Optional[str] = field(default=None, init=False, repr=False)
    _entry_price: float = field(default=0.0, init=False, repr=False)
    _entry_atr_bps: float = field(default=0.0, init=False, repr=False)
    _last_target_exposure: float = field(default=0.0, init=False, repr=False)
    _peak_equity: float = field(default=0.0, init=False, repr=False)

    def warmup_bars(self) -> int:
        return (
            max(
                int(self.trend_ema_bars) + 3,
                int(self.atr_bars) + 3,
                int(self.donchian_bars) + 3,
                int(self.pos7_lookback_windows) + 32,
            )
            + 12
        )

    def _flat_decision(self, symbols: list[str], *, reason: str) -> StrategyDecision:
        targets = {sym: 0.0 for sym in symbols}
        return StrategyDecision(target_exposures=targets, reason=reason)

    def _risk_hard_off(self, state: StrategyState) -> bool:
        if float(state.day_return) <= -abs(float(self.daily_loss_limit)):
            return True
        if self._peak_equity <= 0.0:
            return False
        dd = float(state.equity / self._peak_equity - 1.0)
        return dd <= -abs(float(self.kill_switch))

    def _regime_ok(self, bars_by_symbol: dict[str, pd.DataFrame]) -> bool:
        if not self.market_symbol or self.market_symbol not in bars_by_symbol:
            return True
        df = bars_by_symbol[self.market_symbol]
        if len(df) < max(3, int(self.mkt_peak_lookback), int(self.mkt_vol_bars)):
            return True
        close = pd.to_numeric(df["close"], errors="coerce").dropna()
        if len(close) < max(3, int(self.mkt_peak_lookback), int(self.mkt_vol_bars)):
            return True
        c = float(close.iloc[-1])
        peak = float(close.iloc[-int(self.mkt_peak_lookback) :].max())
        if peak <= 0:
            return True
        mkt_dd = c / peak - 1.0
        rets = np.log(close / close.shift(1)).dropna().to_numpy(dtype=float)
        if rets.size < 2:
            return mkt_dd > -abs(float(self.mkt_dd_off))
        vol_bps = float(np.nanstd(rets[-int(self.mkt_vol_bars) :], ddof=0) * 10_000.0)
        return bool(
            (mkt_dd > -abs(float(self.mkt_dd_off)))
            and (vol_bps < float(self.mkt_vol_off_bps))
        )

    def _symbol_features(
        self,
        sym: str,
        df: pd.DataFrame,
        *,
        cost_rt_bps: float,
    ) -> Optional[dict[str, float]]:
        if df is None or df.empty:
            return None
        if not isinstance(df.index, pd.DatetimeIndex):
            return None
        df = df.sort_index()

        close = pd.to_numeric(df["close"], errors="coerce")
        high = pd.to_numeric(df["high"], errors="coerce")
        low = pd.to_numeric(df["low"], errors="coerce")
        if close.isna().all() or high.isna().all() or low.isna().all():
            return None
        close = close.dropna()
        high = high.reindex(close.index).ffill()
        low = low.reindex(close.index).ffill()
        if len(close) < self.warmup_bars():
            return None

        bars_7d = _bars_for_days(close.index, days=7.0)
        min_len = max(
            self.warmup_bars(),
            int(self.donchian_bars) + 3,
            int(self.trend_ema_bars) + 3,
            bars_7d + int(self.pos7_lookback_windows) + 3,
        )
        if len(close) < min_len:
            return None

        c = float(close.iloc[-1])
        if not np.isfinite(c) or c <= 0:
            return None

        ema = close.ewm(span=max(2, int(self.trend_ema_bars)), adjust=False).mean()
        trend = float(math.log(c / float(ema.iloc[-1]))) if ema.iloc[-1] > 0 else 0.0

        atr_series = _atr(high, low, close, int(self.atr_bars))
        atr_now = float(atr_series.iloc[-1]) if len(atr_series) else float("nan")
        if not np.isfinite(atr_now) or atr_now <= 0:
            return None
        atr_bps = float((atr_now / c) * 10_000.0)

        ret_7d = np.log(close / close.shift(bars_7d))
        tail = ret_7d.dropna().tail(int(self.pos7_lookback_windows))
        if tail.empty:
            return None
        pos7_frac = float((tail > 0.0).mean())

        upper = close.shift(1).rolling(int(max(2, self.donchian_bars))).max()
        upper_now = float(upper.iloc[-1]) if len(upper) else float("nan")
        if not np.isfinite(upper_now) or upper_now <= 0:
            return None
        breakout_lvl = upper_now * (1.0 + float(self.entry_buffer_bps) / 10_000.0)
        breakout = bool(c > breakout_lvl)

        edge_bps = float(float(self.expected_move_atr_mult) * atr_bps - float(cost_rt_bps))
        score = float(max(0.0, trend) * max(0.0, pos7_frac - 0.50) * max(0.0, edge_bps))

        return {
            "symbol": sym,
            "close": c,
            "atr_bps": atr_bps,
            "trend": trend,
            "pos7_frac": pos7_frac,
            "breakout": 1.0 if breakout else 0.0,
            "edge_bps": edge_bps,
            "score": score,
        }

    def _exposure_for(self, feat: dict[str, float], state: StrategyState) -> float:
        edge_scale = max(1e-6, float(self.edge_scale_bps))
        risk_scale = _clamp(float(feat["edge_bps"]) / edge_scale, 0.0, 1.0)
        vol_scale = _clamp(
            float(self.vol_target_bps_per_bar) / max(1e-6, float(feat["atr_bps"])),
            0.0,
            1.0,
        )
        exposure = float(self.max_total_exposure) * risk_scale * vol_scale
        exposure = min(exposure, float(self.max_exposure_per_symbol))
        exposure = _clamp(exposure, 0.0, 1.0)

        max_notional = float(state.extra.get("max_position_notional_usd") or 0.0)
        if max_notional > 0 and exposure * max_notional < float(self.min_trade_notional_usd):
            return 0.0
        return exposure

    def target_exposures(
        self, bars_by_symbol: dict[str, pd.DataFrame], state: StrategyState
    ) -> StrategyDecision:
        symbols = [s for s in self.symbols if s in bars_by_symbol] or sorted(bars_by_symbol.keys())
        if not symbols:
            return StrategyDecision(target_exposures={}, reason="no_symbols")

        self._bars_seen += 1
        if self._peak_equity <= 0:
            self._peak_equity = float(state.equity)
        self._peak_equity = max(self._peak_equity, float(state.equity))

        fee_bps = float(state.extra.get("taker_fee_bps", self.taker_fee_bps) or self.taker_fee_bps)
        slip_bps = float(state.extra.get("slippage_bps", self.slippage_bps) or self.slippage_bps)
        cost_rt_bps = 2.0 * (fee_bps + slip_bps)

        feats: list[dict[str, float]] = []
        for sym in symbols:
            row = self._symbol_features(sym, bars_by_symbol[sym], cost_rt_bps=cost_rt_bps)
            if row is not None:
                feats.append(row)
        if not feats:
            return self._flat_decision(symbols, reason="insufficient_features")

        by_sym = {row["symbol"]: row for row in feats}
        regime_ok = self._regime_ok(bars_by_symbol)

        hard_off = self._risk_hard_off(state) or (not regime_ok)
        if hard_off:
            self._mode = "COOLDOWN"
            self._cooldown_until = self._bars_seen + int(max(1, self.cooldown_bars))
            self._active_symbol = None
            self._last_target_exposure = 0.0
            return self._flat_decision(symbols, reason="risk_off")

        best = max(feats, key=lambda row: float(row["score"]))
        best_ok = bool(
            float(best["pos7_frac"]) >= float(self.pos7_on)
            and float(best["trend"]) >= float(self.trend_on)
            and float(best["edge_bps"]) >= float(self.edge_floor_bps)
            and float(best["atr_bps"]) >= float(self.min_atr_bps)
        )

        if self._mode == "COOLDOWN":
            if self._bars_seen < self._cooldown_until:
                return self._flat_decision(symbols, reason="cooldown")
            if float(best["pos7_frac"]) < float(self.pos7_reset_floor):
                self._cooldown_until = self._bars_seen + int(max(1, self.cooldown_bars // 2))
                return self._flat_decision(symbols, reason="cooldown_reset")
            self._mode = "FLAT"

        if self._mode in {"FLAT", "ARMED"}:
            if not best_ok:
                self._mode = "FLAT"
                return self._flat_decision(symbols, reason="flat_no_edge")
            self._mode = "ARMED"
            if self._bars_seen - self._last_trade_bar < int(max(1, self.min_reentry_bars)):
                return self._flat_decision(symbols, reason="reentry_cooldown")
            if bool(best["breakout"]) is False:
                return self._flat_decision(symbols, reason="armed_wait_breakout")

            exposure = self._exposure_for(best, state)
            if exposure <= 0:
                return self._flat_decision(symbols, reason="armed_min_notional")

            targets = {sym: 0.0 for sym in symbols}
            targets[str(best["symbol"])] = float(exposure)
            self._mode = "LONG"
            self._active_symbol = str(best["symbol"])
            self._entry_price = float(best["close"])
            self._entry_atr_bps = float(best["atr_bps"])
            self._last_target_exposure = float(exposure)
            self._last_trade_bar = self._bars_seen
            return StrategyDecision(target_exposures=targets, reason=f"enter:{self._active_symbol}")

        if self._mode == "LONG":
            sym = str(self._active_symbol or "")
            if sym not in by_sym:
                self._mode = "COOLDOWN"
                self._cooldown_until = self._bars_seen + int(max(1, self.cooldown_bars))
                self._active_symbol = None
                self._last_target_exposure = 0.0
                return self._flat_decision(symbols, reason="exit_missing_symbol")

            feat = by_sym[sym]
            hold_bars = int(state.holding_bars.get(sym, 0) or 0)
            close_now = float(feat["close"])
            entry = float(self._entry_price) if self._entry_price > 0 else close_now
            pnl = (close_now / entry - 1.0) if entry > 0 else 0.0
            atr_ref_bps = max(float(feat["atr_bps"]), float(self._entry_atr_bps), 1e-6)
            stop_ret = -float(self.stop_atr_mult) * atr_ref_bps / 10_000.0
            tp_ret = float(self.take_profit_atr_mult) * atr_ref_bps / 10_000.0

            exit_now = bool(
                float(feat["pos7_frac"]) < float(self.pos7_off)
                or float(feat["trend"]) < float(self.trend_off)
                or float(feat["edge_bps"]) < float(self.edge_floor_bps)
                or hold_bars >= int(max(1, self.max_hold_bars))
                or pnl <= stop_ret
                or pnl >= tp_ret
            )
            if exit_now:
                self._mode = "COOLDOWN"
                self._cooldown_until = self._bars_seen + int(max(1, self.cooldown_bars))
                self._active_symbol = None
                self._last_target_exposure = 0.0
                self._last_trade_bar = self._bars_seen
                return self._flat_decision(symbols, reason="exit_rules")

            exposure = self._exposure_for(feat, state)
            if exposure <= 0:
                self._mode = "COOLDOWN"
                self._cooldown_until = self._bars_seen + int(max(1, self.cooldown_bars))
                self._active_symbol = None
                self._last_target_exposure = 0.0
                self._last_trade_bar = self._bars_seen
                return self._flat_decision(symbols, reason="exit_min_notional")

            if abs(exposure - self._last_target_exposure) < float(self.rebalance_exposure_threshold):
                exposure = float(self._last_target_exposure)
            else:
                self._last_target_exposure = float(exposure)

            targets = {s: 0.0 for s in symbols}
            targets[sym] = float(exposure)
            return StrategyDecision(target_exposures=targets, reason=f"hold:{sym}")

        self._mode = "FLAT"
        return self._flat_decision(symbols, reason="fallback_flat")
