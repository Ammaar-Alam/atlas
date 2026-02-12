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


def _to_ny(ts: pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    if ts.tz is None:
        return ts.tz_localize(NY_TZ)
    return ts.tz_convert(NY_TZ)


def _utc_week_key(ts: pd.Timestamp) -> tuple[int, int]:
    ts = pd.Timestamp(ts)
    if ts.tz is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    iso = ts.isocalendar()
    return int(iso.year), int(iso.week)


def _is_spot_crypto_symbol(symbol: str) -> bool:
    s = (symbol or "").strip().upper()
    if not s:
        return False
    if s.endswith("-PERP") or s.endswith("-CDE"):
        return False
    return ("/" in s) or ("-" in s)


def _atr_from_df(df: pd.DataFrame, window: int) -> Optional[float]:
    if df is None or df.empty:
        return None
    if "close" not in df.columns:
        return None
    tmp = pd.DataFrame(
        {
            "high": pd.to_numeric(df.get("high"), errors="coerce"),
            "low": pd.to_numeric(df.get("low"), errors="coerce"),
            "close": pd.to_numeric(df.get("close"), errors="coerce"),
        }
    ).dropna(subset=["close"])
    if tmp.empty:
        return None
    tmp["high"] = tmp["high"].fillna(tmp["close"])
    tmp["low"] = tmp["low"].fillna(tmp["close"])
    prev_close = tmp["close"].shift(1)
    tr = pd.concat(
        [
            (tmp["high"] - tmp["low"]).abs(),
            (tmp["high"] - prev_close).abs(),
            (tmp["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = float(tr.tail(max(2, int(window))).mean())
    if not np.isfinite(atr) or atr <= 0:
        return None
    return float(atr)


@dataclass
class CryptoRegimeVolTarget(Strategy):
    """
    Long-only crypto strategy with:
    - higher-timeframe regime gate,
    - cross-sectional momentum ranking,
    - ATR volatility targeting,
    - weekly lock + kill-switch controls.
    """

    name: str = "crypto_regime_vol_target"

    symbols: tuple[str, ...] = ("BTC/USD", "ETH/USD", "SOL/USD", "AVAX/USD")
    market_symbol: Optional[str] = "BTC/USD"

    # Signal stack
    fast_window: int = 20
    slow_window: int = 80
    regime_window: int = 200
    regime_slope_bars: int = 10
    momentum_window_bars: int = 120
    atr_window: int = 20
    top_k: int = 2

    # Exposure stack
    target_vol_bps_per_bar: float = 70.0
    max_total_exposure: float = 1.0
    max_exposure_per_symbol: float = 0.70
    rebalance_interval_bars: int = 8
    rebalance_exposure_threshold: float = 0.04
    min_trade_notional_usd: float = 25.0

    # Regime drawdown scaling
    market_drawdown_reduce: float = 0.08
    market_drawdown_off: float = 0.16
    market_peak_lookback_bars: int = 240

    # Portfolio controls
    weekly_loss_limit: float = 0.04
    enable_weekly_profit_lock: bool = True
    weekly_profit_target: float = 0.03
    daily_loss_limit: float = 0.03
    kill_switch: float = 0.15
    kill_switch_cooldown_days: int = 5

    # Trade management
    trailing_stop_pct: float = 0.10
    min_hold_bars: int = 6

    # Internal state
    _bars_seen: int = field(default=0, init=False, repr=False)
    _last_rebalance_bar: int = field(default=0, init=False, repr=False)
    _last_targets: dict[str, float] = field(default_factory=dict, init=False, repr=False)
    _peak_equity: float = field(default=0.0, init=False, repr=False)
    _risk_disabled_day: Optional[object] = field(default=None, init=False, repr=False)
    _risk_disabled_until_day: Optional[object] = field(default=None, init=False, repr=False)
    _week_key: Optional[tuple[int, int]] = field(default=None, init=False, repr=False)
    _week_start_equity: float = field(default=0.0, init=False, repr=False)
    _week_locked: bool = field(default=False, init=False, repr=False)
    _symbol_peak_close: dict[str, float] = field(default_factory=dict, init=False, repr=False)

    def warmup_bars(self) -> int:
        return int(
            max(
                int(self.slow_window) + 3,
                int(self.regime_window) + int(self.regime_slope_bars) + 3,
                int(self.momentum_window_bars) + 3,
                int(self.atr_window) + 3,
                int(self.market_peak_lookback_bars) + 3,
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
                continue
            seen.add(s)
            out.append(s)
        return out

    def _market_symbol(self, universe: list[str]) -> Optional[str]:
        ms = (self.market_symbol or "").strip().upper().replace(" ", "")
        if ms and ms in universe:
            return ms
        return universe[0] if universe else None

    def _risk_off(
        self, universe: list[str], *, reason: str, debug: dict[str, Any]
    ) -> StrategyDecision:
        targets = {s: 0.0 for s in universe}
        self._last_targets = dict(targets)
        return StrategyDecision(target_exposures=targets, reason=reason, debug=debug)

    def _maybe_reset_daily_state(self, state: StrategyState) -> None:
        today = _to_ny(pd.Timestamp(state.timestamp)).date()
        if self._risk_disabled_day is not None and self._risk_disabled_day != today:
            self._risk_disabled_day = None
        if self._risk_disabled_until_day is not None and today > self._risk_disabled_until_day:
            self._risk_disabled_until_day = None

    def _symbol_features(self, df: pd.DataFrame) -> Optional[dict[str, float]]:
        if df is None or df.empty:
            return None
        close = pd.to_numeric(df.get("close"), errors="coerce").dropna()
        if len(close) < self.warmup_bars():
            return None

        c = float(close.iloc[-1])
        base = float(close.iloc[-int(self.momentum_window_bars) - 1])
        if c <= 0 or base <= 0:
            return None
        mom = float(math.log(c / base))
        atr = _atr_from_df(df, int(self.atr_window))
        if atr is None or atr <= 0:
            return None
        atr_bps = float((atr / c) * 10_000.0)
        score = float(mom / max(atr_bps, 1.0))
        return {
            "close": float(c),
            "mom": float(mom),
            "atr_bps": float(atr_bps),
            "score": float(score),
        }

    def _regime_scale(self, df: pd.DataFrame) -> tuple[float, dict[str, float]]:
        if df is None or df.empty:
            return 0.0, {}
        close = pd.to_numeric(df.get("close"), errors="coerce").dropna()
        if len(close) < self.warmup_bars():
            return 0.0, {}

        c = float(close.iloc[-1])
        ema_fast = float(close.ewm(span=max(2, int(self.fast_window)), adjust=False).mean().iloc[-1])
        ema_slow = float(close.ewm(span=max(2, int(self.slow_window)), adjust=False).mean().iloc[-1])
        ema_regime = float(close.ewm(span=max(2, int(self.regime_window)), adjust=False).mean().iloc[-1])
        slope_idx = int(max(1, int(self.regime_slope_bars)))
        if len(close) <= slope_idx:
            return 0.0, {}
        ema_hist = close.ewm(span=max(2, int(self.regime_window)), adjust=False).mean()
        ema_prev = float(ema_hist.iloc[-slope_idx - 1])
        slope = float(ema_regime / ema_prev - 1.0) if ema_prev > 0 else 0.0

        peak = float(close.tail(max(2, int(self.market_peak_lookback_bars))).max())
        drawdown = float(c / peak - 1.0) if peak > 0 else 0.0

        trend_on = bool(c > ema_regime and ema_fast > ema_slow and slope > 0.0)
        scale = 1.0 if trend_on else 0.0
        if drawdown <= -abs(float(self.market_drawdown_off)):
            scale = 0.0
        elif drawdown <= -abs(float(self.market_drawdown_reduce)):
            scale *= 0.5

        atr = _atr_from_df(df, int(self.atr_window))
        atr_bps = float((atr / c) * 10_000.0) if atr is not None and c > 0 else 0.0
        return float(scale), {
            "market_close": float(c),
            "ema_fast": float(ema_fast),
            "ema_slow": float(ema_slow),
            "ema_regime": float(ema_regime),
            "regime_slope": float(slope),
            "market_drawdown": float(drawdown),
            "market_atr_bps": float(atr_bps),
        }

    def _apply_trailing_stops(
        self,
        *,
        universe: list[str],
        bars_by_symbol: dict[str, pd.DataFrame],
        state: StrategyState,
        targets: dict[str, float],
    ) -> tuple[dict[str, float], bool]:
        changed = False
        hold_min = int(max(0, int(self.min_hold_bars)))
        stop_pct = float(max(0.0, float(self.trailing_stop_pct)))
        if stop_pct <= 0:
            return targets, changed

        out = {s: float(targets.get(s, 0.0)) for s in universe}
        for s in universe:
            df = bars_by_symbol.get(s)
            if df is None or df.empty:
                continue
            close = pd.to_numeric(df.get("close"), errors="coerce").dropna()
            if close.empty:
                continue
            c = float(close.iloc[-1])
            prev_peak = float(self._symbol_peak_close.get(s, c))
            peak = max(prev_peak, c)
            self._symbol_peak_close[s] = float(peak)

            if float(out.get(s, 0.0)) <= 0.0:
                continue
            if int(state.holding_bars.get(s, 0) or 0) < hold_min:
                continue

            stop_price = float(peak * (1.0 - stop_pct))
            if c < stop_price:
                out[s] = 0.0
                self._symbol_peak_close[s] = float(c)
                changed = True

        return out, changed

    def target_exposures(
        self, bars_by_symbol: dict[str, pd.DataFrame], state: StrategyState
    ) -> StrategyDecision:
        universe = self._universe(bars_by_symbol)
        if not universe:
            return StrategyDecision(target_exposures={}, reason="no_universe")

        self._bars_seen += 1
        self._maybe_reset_daily_state(state)

        wk = _utc_week_key(pd.Timestamp(state.timestamp))
        if self._week_key is None or self._week_key != wk:
            self._week_key = wk
            self._week_start_equity = float(state.equity)
            self._week_locked = False

        equity = float(state.equity)
        week_ret = (
            float(equity / self._week_start_equity - 1.0)
            if self._week_start_equity > 0
            else 0.0
        )
        if week_ret <= -abs(float(self.weekly_loss_limit)):
            self._week_locked = True
        if bool(self.enable_weekly_profit_lock) and week_ret >= abs(
            float(self.weekly_profit_target)
        ):
            self._week_locked = True

        if self._peak_equity <= 0:
            self._peak_equity = equity
        self._peak_equity = max(self._peak_equity, equity)
        drawdown = float(equity / self._peak_equity - 1.0) if self._peak_equity > 0 else 0.0
        today = _to_ny(pd.Timestamp(state.timestamp)).date()

        debug: dict[str, Any] = {
            "bars_seen": int(self._bars_seen),
            "day_return": float(state.day_return),
            "drawdown": float(drawdown),
            "week_return": float(week_ret),
            "week_locked": bool(self._week_locked),
        }

        if self._risk_disabled_until_day is not None and today <= self._risk_disabled_until_day:
            return self._risk_off(universe, reason="risk_disabled_cooldown", debug=debug)
        if drawdown <= -abs(float(self.kill_switch)):
            self._risk_disabled_until_day = today + timedelta(days=int(self.kill_switch_cooldown_days))
            return self._risk_off(universe, reason="kill_switch", debug=debug)
        if float(state.day_return) <= -abs(float(self.daily_loss_limit)):
            self._risk_disabled_day = today
            return self._risk_off(universe, reason="daily_loss_limit", debug=debug)
        if self._risk_disabled_day == today:
            return self._risk_off(universe, reason="risk_disabled_day", debug=debug)
        if self._week_locked:
            return self._risk_off(universe, reason="week_locked", debug=debug)

        ms = self._market_symbol(universe)
        if ms is None:
            return self._risk_off(universe, reason="no_market_symbol", debug=debug)

        regime_scale, regime_dbg = self._regime_scale(bars_by_symbol.get(ms))
        debug["regime"] = regime_dbg
        if regime_scale <= 0.0:
            return self._risk_off(universe, reason="regime_off", debug=debug)

        due = bool(
            (not self._last_targets)
            or (int(self._bars_seen) - int(self._last_rebalance_bar) >= int(max(1, self.rebalance_interval_bars)))
        )

        if not due:
            held = {s: float(self._last_targets.get(s, 0.0)) for s in universe}
            held, changed = self._apply_trailing_stops(
                universe=universe,
                bars_by_symbol=bars_by_symbol,
                state=state,
                targets=held,
            )
            self._last_targets = dict(held)
            return StrategyDecision(
                target_exposures=held,
                reason="trailing_stop" if changed else "hold",
                debug=debug,
            )

        feats: dict[str, dict[str, float]] = {}
        for s in universe:
            f = self._symbol_features(bars_by_symbol.get(s))
            if f is not None:
                feats[s] = f
        if not feats:
            return self._risk_off(universe, reason="warmup", debug=debug)

        ranked = sorted(feats.items(), key=lambda kv: float(kv[1]["score"]), reverse=True)
        selected: list[tuple[str, dict[str, float]]] = []
        for s, f in ranked:
            if float(f.get("mom", 0.0)) <= 0.0:
                continue
            if float(f.get("score", 0.0)) <= 0.0:
                continue
            selected.append((s, f))
            if len(selected) >= max(1, int(self.top_k)):
                break
        if not selected:
            return self._risk_off(universe, reason="no_positive_momentum", debug=debug)

        max_notional = float(state.extra.get("max_position_notional_usd", 0.0) or 0.0)
        min_exp = 0.0
        if max_notional > 0:
            min_exp = float(self.min_trade_notional_usd) / float(max_notional)

        market_atr_bps = float(regime_dbg.get("market_atr_bps", 0.0) or 0.0)
        vol_scale = 1.0
        if market_atr_bps > 0:
            vol_scale = float(self.target_vol_bps_per_bar) / float(market_atr_bps)
            vol_scale = _clamp(vol_scale, 0.20, 1.50)

        total_exp = _clamp(
            float(self.max_total_exposure) * float(regime_scale) * float(vol_scale),
            0.0,
            float(self.max_total_exposure),
        )

        denom = float(sum(max(0.0, float(f["score"])) for _, f in selected))
        if denom <= 1e-12:
            return self._risk_off(universe, reason="bad_weights", debug=debug)

        targets = {s: 0.0 for s in universe}
        for s, f in selected:
            w = max(0.0, float(f["score"])) / denom
            exp = float(total_exp * w)
            exp = _clamp(exp, 0.0, float(self.max_exposure_per_symbol))
            if exp < float(min_exp):
                exp = 0.0
            targets[s] = float(exp)

        for s in universe:
            cur_qty = float(state.positions.get(s, 0.0) or 0.0)
            if abs(cur_qty) <= 1e-12 or max_notional <= 0:
                continue
            df = bars_by_symbol.get(s)
            if df is None or df.empty:
                continue
            close = pd.to_numeric(df.get("close"), errors="coerce").dropna()
            if close.empty:
                continue
            c = float(close.iloc[-1])
            if c <= 0:
                continue
            cur_exp = float((cur_qty * c) / max_notional)
            tgt = float(targets.get(s, 0.0))
            if abs(tgt - cur_exp) < float(self.rebalance_exposure_threshold):
                targets[s] = cur_exp

        targets, _ = self._apply_trailing_stops(
            universe=universe,
            bars_by_symbol=bars_by_symbol,
            state=state,
            targets=targets,
        )
        self._last_targets = {s: float(targets.get(s, 0.0)) for s in universe}
        self._last_rebalance_bar = int(self._bars_seen)
        return StrategyDecision(
            target_exposures=self._last_targets,
            reason="rebalance",
            debug=debug,
        )
