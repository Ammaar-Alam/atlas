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
    return "/" in s or "-" in s


@dataclass
class _PerSymbolState:
    prev_close: Optional[float] = None
    last_ts: Optional[pd.Timestamp] = None
    # Price history for momentum horizons.
    closes: deque[float] = field(default_factory=deque)
    # Return history for volatility.
    rets: deque[float] = field(default_factory=deque)
    # Liquidity proxy: EWMA dollar volume.
    dv_ewma: Optional[float] = None


@dataclass
class _MarketTracker:
    peak_price: float = 0.0
    last_price: float = 0.0
    ret_ewma_var: float = 0.0
    last_ts: Optional[pd.Timestamp] = None


@dataclass
class _PositionTracker:
    last_side: dict[str, int] = field(default_factory=dict)
    last_qty: dict[str, float] = field(default_factory=dict)
    entry_price: dict[str, float] = field(default_factory=dict)


@dataclass
class CryptoRotation(Strategy):
    """
    Crypto spot cross-sectional rotation strategy (research; NOT financial advice).

    Core idea:
    - Rebalance on a fixed cadence (e.g. weekly) into the top-K symbols by a
      volatility-adjusted, multi-horizon momentum score.
    - Use a global BTC risk overlay (drawdown/vol) to scale risk down or go to cash.
    - Long-only by default; shorting optional if the engine supports it.

    This is designed to trade regularly (rotation) without needing perps/leverage.
    """

    name: str = "crypto_rotation"

    # ---- Universe ----
    symbols: tuple[str, ...] = ("BTC/USD", "ETH/USD", "SOL/USD", "XRP/USD", "ADA/USD")
    market_symbol: Optional[str] = "BTC/USD"

    # ---- Rebalance ----
    rebalance_interval_bars: int = 28  # 7d @ 6H bars
    min_trade_notional_usd: float = 25.0
    rebalance_exposure_threshold: float = 0.02  # skip tiny reallocations when holding

    # ---- Score (multi-horizon momentum) ----
    mom_short_bars: int = 28
    mom_med_bars: int = 120
    mom_long_bars: int = 360
    w_mom_short: float = 0.20
    w_mom_med: float = 0.30
    w_mom_long: float = 0.50

    # ---- Vol targeting ----
    vol_window_bars: int = 120
    vol_target_bps_per_bar: float = 80.0  # rough; tuned per timeframe
    max_total_exposure: float = 1.0
    max_exposure_per_symbol: float = 0.60
    top_k: int = 2
    score_floor: float = 0.0  # require score > floor to allocate

    # ---- Liquidity filter ----
    dv_ewm_span: int = 60
    min_dollar_volume_ewma: float = 50_000.0

    # ---- Costs (bps, used for conservative entry gating) ----
    slippage_bps: float = 3.0
    taker_fee_bps: float = 25.0
    k_cost: float = 1.0
    edge_floor_bps: float = 0.0

    # ---- Risk limits ----
    daily_loss_limit: float = 0.05
    kill_switch: float = 0.25
    kill_switch_cooldown_days: int = 7
    market_drawdown_off: float = 0.25
    market_drawdown_reduce: float = 0.12
    market_vol_off_bps: float = 300.0
    market_vol_reduce_bps: float = 180.0
    market_peak_halflife_bars: int = 240
    # Optional: market trend filter (uses market_symbol momentum).
    # Disabled if market_mom_bars <= 0. If enabled:
    # - if market_mom <= market_mom_off  -> risk off (scale=0)
    # - elif market_mom <= market_mom_reduce -> risk reduce (scale=min(scale,0.5))
    market_mom_bars: int = 0
    market_mom_off: float = 0.0
    market_mom_reduce: float = 0.0

    # ---- Optional: enforce at least one trade periodically (tiny "heartbeat") ----
    # Set to 0 to disable. If >0, the strategy will place a tiny rotation trade at least
    # every N bars, even if scores are unchanged. This is *not* recommended for live trading.
    heartbeat_every_bars: int = 0
    heartbeat_notional_usd: float = 25.0

    # ---- Internal state ----
    _bars_seen: int = field(default=0, init=False, repr=False)
    _last_rebalance_bar: int = field(default=0, init=False, repr=False)
    _last_trade_intent_bar: int = field(default=0, init=False, repr=False)
    _last_trade_bar: int = field(default=0, init=False, repr=False)
    _last_heartbeat_bar: int = field(default=0, init=False, repr=False)
    _heartbeat_symbol: Optional[str] = field(default=None, init=False, repr=False)
    _heartbeat_offset_exp: float = field(default=0.0, init=False, repr=False)
    _heartbeat_clear_bar: int = field(default=0, init=False, repr=False)
    _peak_equity: float = field(default=0.0, init=False, repr=False)
    _risk_disabled_day: Optional[object] = field(default=None, init=False, repr=False)
    _risk_disabled_until_day: Optional[object] = field(default=None, init=False, repr=False)
    _state: dict[str, _PerSymbolState] = field(default_factory=dict, init=False, repr=False)
    _mkt: _MarketTracker = field(default_factory=_MarketTracker, init=False, repr=False)
    _pos: _PositionTracker = field(default_factory=_PositionTracker, init=False, repr=False)
    _last_targets: dict[str, float] = field(default_factory=dict, init=False, repr=False)

    def warmup_bars(self) -> int:
        return (
            max(
                int(self.mom_long_bars) + 2,
                int(self.vol_window_bars) + 3,
                int(self.dv_ewm_span) + 3,
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

    def _ensure_symbol_state(self, symbol: str) -> _PerSymbolState:
        st = self._state.get(symbol)
        if st is not None:
            return st
        st = _PerSymbolState()
        st.closes = deque(
            maxlen=max(
                16,
                int(max(self.mom_long_bars, self.vol_window_bars)) + 2,
            )
        )
        st.rets = deque(maxlen=max(16, int(self.vol_window_bars) + 2))
        self._state[symbol] = st
        return st

    def _update_symbol(self, symbol: str, df: pd.DataFrame) -> None:
        if df is None or df.empty or not isinstance(df.index, pd.DatetimeIndex):
            return
        df = df.sort_index()

        st = self._ensure_symbol_state(symbol)
        last_ts = st.last_ts
        if last_ts is None:
            new_df = df
        else:
            new_df = df[df.index > last_ts]
        if new_df.empty:
            return

        a_dv = _alpha_from_span(self.dv_ewm_span)

        for ts, row in new_df.iterrows():
            c = _safe_float(row.get("close"), default=0.0)
            if c <= 0:
                continue
            v = _safe_float(row.get("volume"), default=0.0)

            if st.prev_close is not None and st.prev_close > 0:
                r = math.log(c / st.prev_close)
            else:
                r = 0.0
            st.rets.append(float(r))
            st.closes.append(float(c))

            dv = float(c) * max(0.0, float(v))
            if st.dv_ewma is None or not np.isfinite(float(st.dv_ewma)):
                st.dv_ewma = float(dv)
            else:
                st.dv_ewma = float(a_dv * dv + (1.0 - a_dv) * float(st.dv_ewma))

            st.prev_close = float(c)
            st.last_ts = pd.Timestamp(ts)

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

        a = _alpha_from_span(max(8, int(self.vol_window_bars // 4)))
        peak_decay = _decay_from_halflife(int(self.market_peak_halflife_bars))
        for ts, row in new_df.iterrows():
            c = _safe_float(row.get("close"), default=0.0)
            if c <= 0:
                continue
            if self._mkt.last_price > 0:
                r = math.log(float(c) / float(self._mkt.last_price))
            else:
                r = 0.0
            self._mkt.ret_ewma_var = float(
                (1.0 - a) * float(self._mkt.ret_ewma_var) + a * float(r * r)
            )
            self._mkt.last_price = float(c)
            self._mkt.peak_price = float(self._mkt.peak_price) * float(peak_decay)
            self._mkt.peak_price = max(float(self._mkt.peak_price), float(c))
            self._mkt.last_ts = pd.Timestamp(ts)

    def _market_risk_scale(self, market_symbol: str) -> tuple[float, dict[str, float]]:
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

        mom_bars = int(getattr(self, "market_mom_bars", 0) or 0)
        if mom_bars > 0:
            mkt_sym = (market_symbol or "").strip().upper()
            st = self._state.get(mkt_sym)
            if st is not None and len(st.closes) > mom_bars:
                base = float(st.closes[-mom_bars - 1])
                last_close = float(st.closes[-1])
                mom = float(math.log(last_close / base)) if base > 0 and last_close > 0 else 0.0
                dbg["market_mom"] = float(mom)
                dbg["market_mom_bars"] = float(mom_bars)

                mom_off = float(getattr(self, "market_mom_off", 0.0) or 0.0)
                mom_reduce = float(getattr(self, "market_mom_reduce", 0.0) or 0.0)
                if mom_off > mom_reduce:
                    mom_off, mom_reduce = mom_reduce, mom_off
                dbg["market_mom_off"] = float(mom_off)
                dbg["market_mom_reduce"] = float(mom_reduce)

                if mom <= mom_off:
                    scale = 0.0
                elif mom <= mom_reduce:
                    scale = min(float(scale), 0.5)
        return float(scale), dbg

    def _maybe_reset_daily_state(self, state: StrategyState) -> None:
        today = _to_ny(pd.Timestamp(state.timestamp)).date()
        if self._risk_disabled_day is not None and self._risk_disabled_day != today:
            self._risk_disabled_day = None
        if self._risk_disabled_until_day is not None and today > self._risk_disabled_until_day:
            self._risk_disabled_until_day = None

    def _risk_off(self, universe: list[str], *, reason: str, debug: dict[str, Any]) -> StrategyDecision:
        return StrategyDecision(target_exposures={s: 0.0 for s in universe}, reason=reason, debug=debug)

    def _score_symbol(self, symbol: str) -> Optional[dict[str, float]]:
        st = self._state.get(symbol)
        if st is None or len(st.closes) < int(max(self.mom_long_bars, self.vol_window_bars)) + 2:
            return None

        close = float(st.closes[-1])
        if close <= 0:
            return None
        if float(st.dv_ewma or 0.0) < float(self.min_dollar_volume_ewma):
            return None

        def _mom(bars: int) -> float:
            b = int(bars)
            if b <= 1 or len(st.closes) <= b:
                return 0.0
            base = float(st.closes[-b - 1])
            if base <= 0:
                return 0.0
            return float(math.log(close / base))

        m_s = _mom(self.mom_short_bars)
        m_m = _mom(self.mom_med_bars)
        m_l = _mom(self.mom_long_bars)

        rets = np.array(list(st.rets)[-int(self.vol_window_bars) :], dtype=float)
        rets = rets[np.isfinite(rets)]
        vol = float(np.std(rets, ddof=1)) if rets.size >= 2 else 0.0
        vol_bps = float(vol * 10_000.0)

        score_raw = float(self.w_mom_short) * m_s + float(self.w_mom_med) * m_m + float(self.w_mom_long) * m_l
        score = float(score_raw / max(1e-9, vol)) if vol > 0 else float(score_raw)
        return {
            "close": float(close),
            "score": float(score),
            "score_raw": float(score_raw),
            "vol_bps": float(vol_bps),
            "dv_ewma": float(st.dv_ewma or 0.0),
        }

    def target_exposures(self, bars_by_symbol: dict[str, pd.DataFrame], state: StrategyState) -> StrategyDecision:
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

        debug: dict[str, Any] = {
            "bars_seen": int(self._bars_seen),
            "equity": float(equity),
            "drawdown": float(drawdown),
            "day_return": float(state.day_return),
            "allow_short": bool(state.allow_short),
        }

        today = _to_ny(pd.Timestamp(state.timestamp)).date()
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

        # Update per-symbol incremental state.
        for s in universe:
            self._update_symbol(s, bars_by_symbol.get(s))

        # Detect actual trades by tracking position quantity changes. This supports heartbeat scheduling
        # even during extended risk-off periods.
        for s in universe:
            pos_qty = float(state.positions.get(s, 0.0) or 0.0)
            prev_qty = float(self._pos.last_qty.get(s, 0.0) or 0.0)
            if abs(pos_qty - prev_qty) > 1e-8:
                self._last_trade_bar = int(self._bars_seen)
            self._pos.last_qty[s] = float(pos_qty)

        # Global market tracker (BTC by default).
        mkt_sym = (self.market_symbol or "").strip().upper()
        if not mkt_sym or mkt_sym not in universe:
            mkt_sym = universe[0]
        self._update_market_tracker(bars_by_symbol.get(mkt_sym))
        risk_scale, market_dbg = self._market_risk_scale(mkt_sym)
        debug["market_symbol"] = mkt_sym
        debug["market"] = market_dbg
        debug["risk_scale"] = float(risk_scale)

        extra = dict(state.extra or {})
        max_notional = float(extra.get("max_position_notional_usd", 0.0) or 0.0) or 1.0
        slippage_bps = float(extra.get("slippage_bps", self.slippage_bps) or 0.0)
        taker_fee_bps = float(extra.get("taker_fee_bps", self.taker_fee_bps) or 0.0)
        cost_rt_bps = float(2.0 * (abs(slippage_bps) + abs(taker_fee_bps)))
        required_edge_bps = float(self.edge_floor_bps) + float(self.k_cost) * cost_rt_bps
        debug["cost_rt_bps"] = float(cost_rt_bps)
        debug["required_edge_bps"] = float(required_edge_bps)

        # Track entry prices for optional debug / future stop logic.
        slip = abs(float(slippage_bps)) / 10_000.0
        for s in universe:
            pos_qty = float(state.positions.get(s, 0.0) or 0.0)
            pos_side = _sign(pos_qty)
            prev_side = int(self._pos.last_side.get(s, 0))
            last_df = bars_by_symbol.get(s)
            if last_df is None or len(last_df) == 0:
                continue
            last_open = _safe_float(last_df.iloc[-1].get("open"), default=_safe_float(last_df.iloc[-1].get("close"), default=0.0))
            if prev_side == 0 and pos_side != 0:
                entry_fill = float(last_open) * (1.0 + slip) if pos_side > 0 else float(last_open) * (1.0 - slip)
                self._pos.entry_price[s] = float(entry_fill)
            if prev_side != 0 and pos_side == 0:
                self._pos.entry_price.pop(s, None)
            self._pos.last_side[s] = int(pos_side)

        # Only recompute targets on rebalance cadence. Heartbeat trades are applied as a tiny overlay
        # on the most recent targets to avoid changing the rotation schedule.
        due = (self._bars_seen - int(self._last_rebalance_bar)) >= max(1, int(self.rebalance_interval_bars))
        if not self._last_targets:
            due = True

        heartbeat_due = False
        hb_every = int(self.heartbeat_every_bars)
        if hb_every > 0:
            last_marker = max(int(self._last_trade_bar), int(self._last_heartbeat_bar))
            heartbeat_due = (self._bars_seen - int(last_marker)) >= int(hb_every)

        if not due and self._last_targets:
            # If a heartbeat offset is active, revert it as soon as scheduled (even if no new heartbeat is due).
            if abs(float(self._heartbeat_offset_exp)) > 1e-12 and int(self._bars_seen) >= int(self._heartbeat_clear_bar):
                hb_sym = (self._heartbeat_symbol or "").strip().upper()
                off = float(self._heartbeat_offset_exp)
                if hb_sym and hb_sym in universe:
                    targets = dict(self._last_targets)
                    base = float(targets.get(hb_sym, 0.0))
                    new_exp = float(_clamp(base - off, 0.0, float(self.max_exposure_per_symbol)))
                    if abs(new_exp - base) > 1e-12:
                        targets[hb_sym] = float(new_exp)
                    prev = dict(self._last_targets)
                    self._last_targets = {s: float(targets.get(s, 0.0)) for s in universe}
                    if any(
                        abs(float(self._last_targets.get(s, 0.0)) - float(prev.get(s, 0.0))) > 1e-8
                        for s in universe
                    ):
                        self._last_trade_intent_bar = int(self._bars_seen)
                    self._heartbeat_offset_exp = 0.0
                    self._heartbeat_symbol = None
                    debug["heartbeat"] = {"action": "revert", "symbol": str(hb_sym), "exp_delta": float(new_exp - base)}
                    return StrategyDecision(
                        target_exposures=self._last_targets,
                        reason="heartbeat_revert",
                        debug=debug,
                    )

            if heartbeat_due:
                targets = dict(self._last_targets)
                hb_notional = float(self.heartbeat_notional_usd)
                if hb_notional > 0.0 and max_notional > 0:
                    hb_exp = float(hb_notional / max_notional)
                    hb_exp = float(_clamp(hb_exp, 0.0, 0.01))
                    if hb_exp > 1e-12:
                        # Toggle offset on/off so a trade is guaranteed.
                        if abs(float(self._heartbeat_offset_exp)) > 1e-12:
                            self._heartbeat_offset_exp = 0.0
                        else:
                            hb_sym: Optional[str] = None
                            for s in universe:
                                if abs(float(state.positions.get(s, 0.0) or 0.0)) > 1e-12:
                                    hb_sym = s
                                    break
                            if hb_sym is None:
                                hb_sym = (mkt_sym or "").strip().upper()
                                if not hb_sym or hb_sym not in universe:
                                    hb_sym = universe[0]

                            gross = float(sum(abs(float(targets.get(s, 0.0))) for s in universe))
                            slack = float(self.max_total_exposure) - gross
                            self._heartbeat_symbol = str(hb_sym)
                            self._heartbeat_offset_exp = float(hb_exp) if slack > (hb_exp + 1e-12) else -float(hb_exp)

                        self._last_heartbeat_bar = int(self._bars_seen)
                        if abs(float(self._heartbeat_offset_exp)) > 1e-12:
                            self._heartbeat_clear_bar = int(self._bars_seen) + 1

                        hb_sym = (self._heartbeat_symbol or "").strip().upper()
                        hb_off = float(self._heartbeat_offset_exp)
                        if hb_sym and hb_sym in universe and abs(hb_off) > 1e-12:
                            base = float(targets.get(hb_sym, 0.0))
                            new_exp = float(_clamp(base + hb_off, 0.0, float(self.max_exposure_per_symbol)))
                            if abs(new_exp - base) > 1e-12:
                                targets[hb_sym] = float(new_exp)
                                debug["heartbeat"] = {
                                    "every_bars": int(hb_every),
                                    "notional_usd": float(hb_notional),
                                    "symbol": str(hb_sym),
                                    "exp_delta": float(new_exp - base),
                                }

                prev = dict(self._last_targets)
                self._last_targets = {s: float(targets.get(s, 0.0)) for s in universe}
                if any(
                    abs(float(self._last_targets.get(s, 0.0)) - float(prev.get(s, 0.0))) > 1e-8
                    for s in universe
                ):
                    self._last_trade_intent_bar = int(self._bars_seen)
                return StrategyDecision(
                    target_exposures=self._last_targets,
                    reason="heartbeat",
                    debug=debug,
                )
            return StrategyDecision(
                target_exposures={s: float(self._last_targets.get(s, 0.0)) for s in universe},
                reason="hold",
                debug=debug,
            )

        # Score all symbols and pick top-K.
        feats: dict[str, dict[str, float]] = {}
        for s in universe:
            f = self._score_symbol(s)
            if f is not None:
                feats[s] = f
        if not feats:
            return StrategyDecision(target_exposures={s: 0.0 for s in universe}, reason="warmup", debug=debug)

        ranked = sorted(feats.items(), key=lambda kv: float(kv[1]["score"]), reverse=True)
        top_k = max(1, int(self.top_k))
        selected: list[tuple[str, dict[str, float]]] = []
        for sym, f in ranked:
            if float(f["score"]) <= float(self.score_floor):
                continue
            selected.append((sym, f))
            if len(selected) >= top_k:
                break

        if not selected:
            # No positive-score assets: go to cash (but optional heartbeat may still trade).
            targets = {s: 0.0 for s in universe}
        else:
            # Vol targeting: scale total exposure by target_vol / realized market vol.
            mkt_vol_bps = float(market_dbg.get("market_vol_bps", 0.0) or 0.0)
            vol_scale = 1.0
            if mkt_vol_bps > 1e-9 and float(self.vol_target_bps_per_bar) > 0:
                vol_scale = float(self.vol_target_bps_per_bar) / float(mkt_vol_bps)
            vol_scale = _clamp(vol_scale, 0.0, 1.0)

            total_exposure = float(self.max_total_exposure) * float(risk_scale) * float(vol_scale)
            total_exposure = _clamp(total_exposure, 0.0, float(self.max_total_exposure))

            # Weights ∝ max(0, score_raw) / vol.
            raw_w: list[tuple[str, float]] = []
            for sym, f in selected:
                vol_bps = float(f.get("vol_bps", 0.0) or 0.0)
                score_raw = float(f.get("score_raw", 0.0) or 0.0)
                w = max(0.0, score_raw) / max(1e-9, vol_bps / 10_000.0)
                raw_w.append((sym, float(w)))
            denom = float(sum(w for _, w in raw_w))
            if denom <= 0.0:
                targets = {s: 0.0 for s in universe}
            else:
                targets = {s: 0.0 for s in universe}
                for sym, w in raw_w:
                    exp = float(total_exposure) * (float(w) / float(denom))
                    exp = _clamp(exp, 0.0, float(self.max_exposure_per_symbol))
                    targets[sym] = float(exp)

        # Entry gating: if the universe is entirely negative momentum, avoid churn at high costs.
        # We approximate expected edge using the best score_raw among selected names.
        best_edge_bps = 0.0
        if selected:
            best_edge_bps = float(max(0.0, float(selected[0][1].get("score_raw", 0.0))) * 10_000.0)
        if best_edge_bps < float(required_edge_bps) and all(v <= 0.0 for v in targets.values()):
            targets = {s: 0.0 for s in universe}

        # Rebalance threshold: avoid tiny trades when holding.
        for s in universe:
            pos_qty = float(state.positions.get(s, 0.0) or 0.0)
            if abs(pos_qty) <= 1e-12:
                continue
            last_close = float(feats.get(s, {}).get("close", 0.0) or 0.0)
            if last_close <= 0:
                continue
            cur_exp = (pos_qty * last_close) / max_notional if max_notional > 0 else 0.0
            tgt = float(targets.get(s, 0.0))
            if _sign(cur_exp) == _sign(tgt) and abs(tgt - float(cur_exp)) < float(self.rebalance_exposure_threshold):
                targets[s] = float(cur_exp)

        # Ensure minimum notional for any non-zero target (or drop it to 0).
        for s in universe:
            exp = float(targets.get(s, 0.0))
            if exp <= 0:
                continue
            notional = exp * max_notional
            if notional < float(self.min_trade_notional_usd):
                targets[s] = 0.0

        self._last_rebalance_bar = int(self._bars_seen)

        # Track whether we likely caused a trade (intent) to support heartbeat scheduling.
        prev = dict(self._last_targets)
        self._last_targets = {s: float(targets.get(s, 0.0)) for s in universe}
        if any(abs(float(self._last_targets.get(s, 0.0)) - float(prev.get(s, 0.0))) > 1e-8 for s in universe):
            self._last_trade_intent_bar = int(self._bars_seen)

        debug["selected"] = [s for s, _ in selected]
        debug["features"] = {
            s: {k: float(v) for k, v in feats[s].items() if k in {"score", "score_raw", "vol_bps", "dv_ewma"}}
            for s in feats
            if s in feats
        }

        reason = "rebalance=" + ",".join([s for s in universe if float(self._last_targets.get(s, 0.0)) > 1e-9])
        return StrategyDecision(target_exposures=self._last_targets, reason=reason, debug=debug)
