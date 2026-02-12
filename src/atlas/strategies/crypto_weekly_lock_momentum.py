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


def _utc_week_key(ts: pd.Timestamp) -> tuple[int, int]:
    ts = pd.Timestamp(ts)
    if ts.tz is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    iso = ts.isocalendar()
    return int(iso.year), int(iso.week)


@dataclass
class CryptoWeeklyLockMomentum(Strategy):
    """
    Regime-filtered cross-sectional momentum with weekly lock controls:
    - Weekly profit target => flat for rest of week (lock gains).
    - Weekly loss limit => flat for rest of week (quarantine).
    - Market regime filter gates risk-on.
    """

    name: str = "crypto_weekly_lock_momentum"

    symbols: tuple[str, ...] = ("BTC/USD", "ETH/USD", "SOL/USD", "AVAX/USD")
    market_symbol: Optional[str] = "BTC/USD"

    # Rebalance / score
    rebalance_interval_bars: int = 8
    rebalance_exposure_threshold: float = 0.05
    mom_short_bars: int = 28
    mom_med_bars: int = 84
    mom_long_bars: int = 252
    w_mom_short: float = 0.20
    w_mom_med: float = 0.35
    w_mom_long: float = 0.45
    vol_window_bars: int = 84
    top_k: int = 2
    score_floor: float = 0.0

    # Exposure
    max_total_exposure: float = 1.0
    max_exposure_per_symbol: float = 0.70
    vol_target_bps_per_bar: float = 90.0
    min_trade_notional_usd: float = 25.0

    # Regime filter on market symbol
    regime_ema_bars: int = 168
    regime_mom_bars: int = 84
    regime_mom_off: float = 0.0
    regime_dd_off: float = 0.15
    regime_dd_reduce: float = 0.08
    regime_peak_lookback_bars: int = 252

    # Weekly controls
    weekly_profit_target: float = 0.010
    weekly_loss_limit: float = 0.012

    # Risk controls
    daily_loss_limit: float = 0.03
    kill_switch: float = 0.15
    kill_switch_cooldown_days: int = 5

    # Internal
    _bars_seen: int = field(default=0, init=False, repr=False)
    _last_rebalance_bar: int = field(default=0, init=False, repr=False)
    _last_targets: dict[str, float] = field(default_factory=dict, init=False, repr=False)
    _peak_equity: float = field(default=0.0, init=False, repr=False)
    _risk_disabled_day: Optional[object] = field(default=None, init=False, repr=False)
    _risk_disabled_until_day: Optional[object] = field(default=None, init=False, repr=False)
    _week_key: Optional[tuple[int, int]] = field(default=None, init=False, repr=False)
    _week_start_equity: float = field(default=0.0, init=False, repr=False)
    _week_locked: bool = field(default=False, init=False, repr=False)

    def warmup_bars(self) -> int:
        return int(
            max(
                int(self.mom_long_bars) + 4,
                int(self.vol_window_bars) + 4,
                int(self.regime_ema_bars) + 4,
                int(self.regime_peak_lookback_bars) + 4,
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

    def _score_symbol(self, df: pd.DataFrame) -> Optional[dict[str, float]]:
        if df is None or df.empty or not isinstance(df.index, pd.DatetimeIndex):
            return None
        df = df.sort_index()
        if len(df) < self.warmup_bars():
            return None

        close = pd.to_numeric(df["close"], errors="coerce").dropna()
        if len(close) < self.warmup_bars():
            return None
        c = float(close.iloc[-1])
        if c <= 0:
            return None

        def _mom(lb: int) -> float:
            k = int(max(2, lb))
            if len(close) <= k:
                return 0.0
            base = float(close.iloc[-k - 1])
            if base <= 0:
                return 0.0
            return float(math.log(c / base))

        m_s = _mom(self.mom_short_bars)
        m_m = _mom(self.mom_med_bars)
        m_l = _mom(self.mom_long_bars)
        score_raw = (
            float(self.w_mom_short) * m_s
            + float(self.w_mom_med) * m_m
            + float(self.w_mom_long) * m_l
        )

        rets = close.pct_change().dropna().tail(int(self.vol_window_bars))
        vol = float(rets.std(ddof=1)) if len(rets) >= 2 else 0.0
        vol_bps = float(vol * 10_000.0) if np.isfinite(vol) else 0.0
        score = float(score_raw / max(1e-9, vol)) if vol > 0 else float(score_raw)
        return {
            "close": float(c),
            "score": float(score),
            "score_raw": float(score_raw),
            "vol_bps": float(vol_bps),
        }

    def _market_regime(self, df: pd.DataFrame) -> tuple[float, dict[str, float]]:
        if df is None or df.empty or not isinstance(df.index, pd.DatetimeIndex):
            return 0.0, {}
        df = df.sort_index()
        close = pd.to_numeric(df["close"], errors="coerce").dropna()
        if len(close) < int(max(self.regime_ema_bars, self.regime_peak_lookback_bars)) + 3:
            return 0.0, {}
        c = float(close.iloc[-1])
        ema = float(close.ewm(span=max(2, int(self.regime_ema_bars)), adjust=False).mean().iloc[-1])
        if len(close) <= int(self.regime_mom_bars):
            return 0.0, {}
        base = float(close.iloc[-int(self.regime_mom_bars) - 1])
        mom = float(math.log(c / base)) if c > 0 and base > 0 else 0.0
        peak = float(close.tail(int(self.regime_peak_lookback_bars)).max())
        dd = (c / peak - 1.0) if peak > 0 and c > 0 else 0.0

        scale = 1.0
        if c <= ema or mom <= float(self.regime_mom_off) or dd <= -abs(float(self.regime_dd_off)):
            scale = 0.0
        elif dd <= -abs(float(self.regime_dd_reduce)):
            scale = 0.5
        return float(scale), {
            "market_close": float(c),
            "market_ema": float(ema),
            "market_mom": float(mom),
            "market_drawdown": float(dd),
        }

    def target_exposures(
        self, bars_by_symbol: dict[str, pd.DataFrame], state: StrategyState
    ) -> StrategyDecision:
        universe = self._universe(bars_by_symbol)
        if not universe:
            return StrategyDecision(target_exposures={}, reason="no_crypto_symbols")

        self._bars_seen += 1
        self._maybe_reset_daily_state(state)

        wk = _utc_week_key(pd.Timestamp(state.timestamp))
        if self._week_key is None or self._week_key != wk:
            self._week_key = wk
            self._week_start_equity = float(state.equity)
            self._week_locked = False

        week_ret = (
            float(state.equity) / float(self._week_start_equity) - 1.0
            if self._week_start_equity > 0
            else 0.0
        )
        if (not self._week_locked) and week_ret >= float(self.weekly_profit_target):
            self._week_locked = True
        if (not self._week_locked) and week_ret <= -abs(float(self.weekly_loss_limit)):
            self._week_locked = True

        equity = float(state.equity)
        if self._peak_equity <= 0:
            self._peak_equity = equity
        self._peak_equity = max(self._peak_equity, equity)
        drawdown = (equity / self._peak_equity - 1.0) if self._peak_equity > 0 else 0.0
        today = _to_ny(pd.Timestamp(state.timestamp)).date()

        debug: dict[str, Any] = {
            "bars_seen": int(self._bars_seen),
            "drawdown": float(drawdown),
            "day_return": float(state.day_return),
            "week_return": float(week_ret),
            "week_locked": bool(self._week_locked),
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
        if self._week_locked:
            return self._risk_off(universe, reason="week_locked", debug=debug)

        market_symbol = (self.market_symbol or "").strip().upper()
        if not market_symbol or market_symbol not in universe:
            market_symbol = universe[0]
        regime_scale, regime_dbg = self._market_regime(bars_by_symbol.get(market_symbol))
        debug["regime"] = regime_dbg
        debug["regime_scale"] = float(regime_scale)
        if regime_scale <= 0.0:
            return self._risk_off(universe, reason="market_regime_off", debug=debug)

        due = (self._bars_seen - int(self._last_rebalance_bar)) >= max(1, int(self.rebalance_interval_bars))
        if not self._last_targets:
            due = True
        if not due:
            return StrategyDecision(
                target_exposures={s: float(self._last_targets.get(s, 0.0)) for s in universe},
                reason="hold",
                debug=debug,
            )

        feats: dict[str, dict[str, float]] = {}
        for s in universe:
            f = self._score_symbol(bars_by_symbol.get(s))
            if f is not None:
                feats[s] = f
        if not feats:
            return self._risk_off(universe, reason="warmup", debug=debug)

        ranked = sorted(feats.items(), key=lambda kv: float(kv[1]["score"]), reverse=True)
        selected: list[tuple[str, dict[str, float]]] = []
        for s, f in ranked:
            if float(f.get("score", 0.0)) <= float(self.score_floor):
                continue
            selected.append((s, f))
            if len(selected) >= max(1, int(self.top_k)):
                break
        if not selected:
            return self._risk_off(universe, reason="no_positive_scores", debug=debug)

        extra = dict(state.extra or {})
        max_notional = float(extra.get("max_position_notional_usd", 0.0) or 0.0) or 1.0
        min_exp = float(self.min_trade_notional_usd) / float(max_notional)
        min_exp = _clamp(min_exp, 0.0, 1.0)

        mkt_vol_bps = float(regime_dbg.get("market_drawdown", 0.0))
        # Use inverse drawdown pressure as a mild additional scale factor.
        dd_pressure = 1.0 - abs(min(0.0, float(mkt_vol_bps)))
        dd_pressure = _clamp(dd_pressure, 0.3, 1.0)
        total_exposure = float(self.max_total_exposure) * float(regime_scale) * float(dd_pressure)
        total_exposure = _clamp(total_exposure, 0.0, float(self.max_total_exposure))

        raw_w: list[tuple[str, float]] = []
        for s, f in selected:
            score = max(0.0, float(f.get("score_raw", 0.0)))
            vol_bps = max(1.0, float(f.get("vol_bps", 0.0)))
            raw_w.append((s, score / vol_bps))
        denom = float(sum(w for _, w in raw_w))
        if denom <= 1e-12:
            return self._risk_off(universe, reason="bad_weights", debug=debug)

        targets = {s: 0.0 for s in universe}
        for s, w in raw_w:
            exp = float(total_exposure) * (float(w) / float(denom))
            exp = _clamp(exp, 0.0, float(self.max_exposure_per_symbol))
            if exp < min_exp:
                exp = 0.0
            targets[s] = float(exp)

        prev = {s: float(self._last_targets.get(s, 0.0)) for s in universe}
        for s in universe:
            if prev.get(s, 0.0) > 0 and targets.get(s, 0.0) > 0:
                if abs(float(targets[s]) - float(prev[s])) < float(self.rebalance_exposure_threshold):
                    targets[s] = float(prev[s])

        self._last_targets = {s: float(targets.get(s, 0.0)) for s in universe}
        self._last_rebalance_bar = int(self._bars_seen)
        debug["selected"] = [s for s, _ in selected]
        return StrategyDecision(target_exposures=self._last_targets, reason="rebalance", debug=debug)
