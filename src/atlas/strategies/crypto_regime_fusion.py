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


@dataclass
class CryptoRegimeFusion(Strategy):
    """
    Long-only crypto spot strategy that fuses:
    - Trend regime: multi-symbol risk-adjusted momentum allocation.
    - Range regime: market-symbol mean reversion.
    - Neutral regime: reduced trend exposure.

    This is research code only and does not guarantee profitability.
    """

    name: str = "crypto_regime_fusion"

    # ---- Universe ----
    symbols: tuple[str, ...] = ("BTC/USD", "ETH/USD", "SOL/USD", "AVAX/USD")
    market_symbol: Optional[str] = "BTC/USD"

    # ---- Regime detection on market_symbol ----
    regime_momentum_bars: int = 120
    regime_er_bars: int = 80
    regime_atr_bars: int = 40
    regime_ema_fast: int = 24
    regime_ema_slow: int = 96
    regime_trend_mom: float = 0.02
    regime_trend_er_min: float = 0.30
    regime_trend_strength_min: float = 0.20
    regime_range_abs_mom_max: float = 0.015
    regime_range_er_max: float = 0.18

    # ---- Trend allocation ----
    momentum_window_bars: int = 120
    vol_window_bars: int = 80
    trend_top_k: int = 3
    trend_score_floor: float = 0.0
    max_total_exposure: float = 1.0
    max_exposure_per_symbol: float = 0.55
    neutral_exposure_scale: float = 0.25

    # ---- Range-mode mean reversion (market symbol only) ----
    meanrev_window_bars: int = 72
    meanrev_entry_z: float = 1.5
    meanrev_exit_z: float = 0.5
    meanrev_max_z: float = 4.0
    range_min_exposure: float = 0.15
    range_max_exposure: float = 0.45

    # ---- Rebalance / trade hygiene ----
    rebalance_interval_bars: int = 8
    rebalance_exposure_threshold: float = 0.03
    min_trade_notional_usd: float = 25.0

    # ---- Risk controls ----
    daily_loss_limit: float = 0.04
    kill_switch: float = 0.12
    kill_switch_cooldown_days: int = 2

    # ---- Optional periodic tiny trade overlay ----
    heartbeat_every_bars: int = 0
    heartbeat_notional_usd: float = 1.0
    heartbeat_max_exposure_delta: float = 0.02
    heartbeat_respect_min_trade_notional: bool = False

    # ---- Internal state ----
    _bars_seen: int = field(default=0, init=False, repr=False)
    _last_rebalance_bar: int = field(default=0, init=False, repr=False)
    _last_regime: Optional[str] = field(default=None, init=False, repr=False)
    _last_targets: dict[str, float] = field(default_factory=dict, init=False, repr=False)
    _last_trade_intent_bar: int = field(default=0, init=False, repr=False)
    _last_heartbeat_bar: int = field(default=0, init=False, repr=False)
    _heartbeat_symbol: Optional[str] = field(default=None, init=False, repr=False)
    _heartbeat_offset_exp: float = field(default=0.0, init=False, repr=False)
    _heartbeat_clear_bar: int = field(default=0, init=False, repr=False)
    _peak_equity: float = field(default=0.0, init=False, repr=False)
    _risk_disabled_day: Optional[object] = field(default=None, init=False, repr=False)
    _risk_disabled_until_day: Optional[object] = field(default=None, init=False, repr=False)

    def warmup_bars(self) -> int:
        return int(
            max(
                int(self.regime_momentum_bars) + 2,
                int(self.regime_er_bars) + 2,
                int(self.regime_atr_bars) + 3,
                int(self.regime_ema_slow) + 3,
                int(self.momentum_window_bars) + 3,
                int(self.vol_window_bars) + 3,
                int(self.meanrev_window_bars) + 3,
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
        return StrategyDecision(target_exposures=targets, reason=reason, debug=debug)

    def _close_series(self, df: pd.DataFrame) -> Optional[pd.Series]:
        if df is None or df.empty or "close" not in df.columns:
            return None
        closes = pd.to_numeric(df["close"], errors="coerce").dropna()
        if len(closes) < 2:
            return None
        return closes

    def _market_regime(self, df: pd.DataFrame) -> Optional[dict[str, float | str]]:
        closes = self._close_series(df)
        if closes is None:
            return None

        need = max(
            int(self.regime_momentum_bars) + 1,
            int(self.regime_er_bars) + 1,
            int(self.regime_ema_slow) + 1,
            int(self.regime_atr_bars) + 1,
        )
        if len(closes) < need:
            return None

        last = float(closes.iloc[-1])
        base = float(closes.iloc[-int(self.regime_momentum_bars) - 1])
        if last <= 0 or base <= 0:
            return None
        mom = float(math.log(last / base))

        er_closes = closes.iloc[-int(self.regime_er_bars) - 1 :]
        net = float(abs(er_closes.iloc[-1] - er_closes.iloc[0]))
        path = float(er_closes.diff().abs().iloc[1:].sum())
        er = float(net / path) if path > 1e-12 else 0.0
        er = _clamp(er, 0.0, 1.0)

        ema_fast = float(
            closes.ewm(span=max(2, int(self.regime_ema_fast)), adjust=False).mean().iloc[-1]
        )
        ema_slow = float(
            closes.ewm(span=max(2, int(self.regime_ema_slow)), adjust=False).mean().iloc[-1]
        )

        atr = 0.0
        if df is not None and ("high" in df.columns) and ("low" in df.columns):
            tmp = pd.DataFrame(
                {
                    "high": pd.to_numeric(df["high"], errors="coerce"),
                    "low": pd.to_numeric(df["low"], errors="coerce"),
                    "close": pd.to_numeric(df["close"], errors="coerce"),
                }
            ).dropna(subset=["close"])
            if not tmp.empty:
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
                atr = float(tr.tail(max(2, int(self.regime_atr_bars))).mean())
        if atr <= 1e-12:
            atr = float(closes.diff().abs().tail(max(2, int(self.regime_atr_bars))).mean())

        trend_strength = float((ema_fast - ema_slow) / atr) if atr > 1e-12 else 0.0

        trend_cond = (
            abs(mom) >= abs(float(self.regime_trend_mom))
            and er >= float(self.regime_trend_er_min)
            and abs(trend_strength) >= abs(float(self.regime_trend_strength_min))
        )
        range_cond = (
            abs(mom) <= abs(float(self.regime_range_abs_mom_max))
            and er <= float(self.regime_range_er_max)
        )

        if trend_cond:
            regime = "trend"
        elif range_cond:
            regime = "range"
        else:
            regime = "neutral"

        trend_dir = 0
        if mom > 0 and trend_strength > 0:
            trend_dir = 1
        elif mom < 0 and trend_strength < 0:
            trend_dir = -1
        else:
            tilt = float(mom + 0.10 * trend_strength)
            if tilt > 0:
                trend_dir = 1
            elif tilt < 0:
                trend_dir = -1

        return {
            "regime": regime,
            "mom": float(mom),
            "er": float(er),
            "trend_strength": float(trend_strength),
            "trend_dir": float(trend_dir),
            "atr": float(atr),
            "close": float(last),
        }

    def _risk_adjusted_momentum(self, df: pd.DataFrame) -> Optional[dict[str, float]]:
        closes = self._close_series(df)
        if closes is None:
            return None

        w_mom = int(self.momentum_window_bars)
        w_vol = int(self.vol_window_bars)
        if len(closes) < max(w_mom, w_vol) + 2:
            return None

        last = float(closes.iloc[-1])
        base = float(closes.iloc[-w_mom - 1])
        if last <= 0 or base <= 0:
            return None

        mom = float(math.log(last / base))
        log_rets = np.log(closes).diff().dropna().tail(w_vol).astype(float)
        if len(log_rets) < 2:
            return None

        vol = float(np.std(log_rets, ddof=1))
        vol = max(vol, 1e-6)
        score = float(mom / vol)

        return {
            "close": float(last),
            "mom": float(mom),
            "vol": float(vol),
            "score": float(score),
        }

    def _trend_targets(
        self,
        universe: list[str],
        bars_by_symbol: dict[str, pd.DataFrame],
        *,
        exposure_scale: float,
    ) -> tuple[dict[str, float], dict[str, Any]]:
        targets = {s: 0.0 for s in universe}
        feats: dict[str, dict[str, float]] = {}
        for s in universe:
            f = self._risk_adjusted_momentum(bars_by_symbol.get(s))
            if f is not None:
                feats[s] = f

        debug: dict[str, Any] = {
            "feature_count": int(len(feats)),
            "selected": [],
            "features": {
                s: {
                    "mom": float(v["mom"]),
                    "vol": float(v["vol"]),
                    "score": float(v["score"]),
                }
                for s, v in feats.items()
            },
        }
        if not feats:
            return targets, debug

        ranked = sorted(feats.items(), key=lambda kv: float(kv[1]["score"]), reverse=True)

        selected: list[tuple[str, dict[str, float]]] = []
        for sym, f in ranked:
            if float(f["mom"]) <= 0.0:
                continue
            if float(f["score"]) <= float(self.trend_score_floor):
                continue
            selected.append((sym, f))
            if len(selected) >= max(1, int(self.trend_top_k)):
                break

        debug["selected"] = [s for s, _ in selected]
        if not selected:
            return targets, debug

        total_exposure = float(self.max_total_exposure) * float(exposure_scale)
        total_exposure = _clamp(total_exposure, 0.0, float(self.max_total_exposure))
        if total_exposure <= 0.0:
            return targets, debug

        raw_weights = [max(0.0, float(f["score"])) for _, f in selected]
        denom = float(sum(raw_weights))
        if denom <= 1e-12:
            raw_weights = [1.0 for _ in selected]
            denom = float(len(raw_weights))

        for (sym, _), w in zip(selected, raw_weights):
            exp = float(total_exposure) * float(w / denom)
            exp = _clamp(exp, 0.0, float(self.max_exposure_per_symbol))
            targets[sym] = float(exp)

        debug["total_exposure"] = float(sum(targets.values()))
        return targets, debug

    def _range_target(
        self,
        market_df: pd.DataFrame,
        *,
        current_market_exp: float,
    ) -> tuple[float, dict[str, float]]:
        closes = self._close_series(market_df)
        if closes is None or len(closes) < int(self.meanrev_window_bars) + 1:
            return 0.0, {"z": 0.0, "mean": 0.0, "std": 0.0}

        window = closes.iloc[-int(self.meanrev_window_bars) :]
        mean_p = float(window.mean())
        std_p = float(window.std(ddof=1))
        last = float(window.iloc[-1])
        z = float((last - mean_p) / std_p) if std_p > 1e-12 else 0.0

        entry_z = abs(float(self.meanrev_entry_z))
        exit_z = abs(float(self.meanrev_exit_z))
        max_z = max(entry_z + 1e-6, abs(float(self.meanrev_max_z)))

        if z <= -entry_z:
            z_abs = min(-z, max_z)
            strength = (z_abs - entry_z) / max(1e-9, max_z - entry_z)
            strength = _clamp(strength, 0.0, 1.0)
            target = float(self.range_min_exposure) + strength * float(
                max(0.0, float(self.range_max_exposure) - float(self.range_min_exposure))
            )
        elif z >= -exit_z:
            target = 0.0
        else:
            target = _clamp(float(current_market_exp), 0.0, float(self.range_max_exposure))

        target = _clamp(target, 0.0, min(float(self.range_max_exposure), float(self.max_exposure_per_symbol)))
        return float(target), {"z": float(z), "mean": float(mean_p), "std": float(std_p)}

    def _apply_rebalance_threshold(
        self,
        targets: dict[str, float],
        universe: list[str],
        bars_by_symbol: dict[str, pd.DataFrame],
        state: StrategyState,
        *,
        max_notional: float,
    ) -> dict[str, float]:
        if max_notional <= 0.0:
            return targets

        for s in universe:
            df = bars_by_symbol.get(s)
            if df is None or df.empty or "close" not in df.columns:
                continue
            close = _safe_float(df["close"].iloc[-1], default=0.0)
            if close <= 0.0:
                continue

            qty = float(state.positions.get(s, 0.0) or 0.0)
            cur_exp = float((qty * close) / max_notional)
            if cur_exp < 0.0:
                continue

            tgt = float(targets.get(s, 0.0))
            if abs(tgt - cur_exp) < float(self.rebalance_exposure_threshold):
                targets[s] = float(cur_exp)

        return targets

    def _apply_min_trade_notional(
        self,
        targets: dict[str, float],
        universe: list[str],
        *,
        max_notional: float,
        allow_small_heartbeat: bool,
    ) -> dict[str, float]:
        if max_notional <= 0.0:
            return targets

        for s in universe:
            exp = float(targets.get(s, 0.0))
            if exp <= 0.0:
                continue
            notional = exp * max_notional
            if notional >= float(self.min_trade_notional_usd):
                continue
            if allow_small_heartbeat and s == (self._heartbeat_symbol or ""):
                continue
            targets[s] = 0.0

        return targets

    def _maybe_apply_heartbeat(
        self,
        *,
        targets: dict[str, float],
        universe: list[str],
        market_symbol: str,
        max_notional: float,
        state: StrategyState,
    ) -> tuple[dict[str, float], Optional[str]]:
        hb_every = int(self.heartbeat_every_bars)
        hb_notional = float(self.heartbeat_notional_usd)
        if hb_every <= 0 or hb_notional <= 0.0 or max_notional <= 0.0:
            return targets, None

        if bool(self.heartbeat_respect_min_trade_notional) and (
            hb_notional < float(self.min_trade_notional_usd)
        ):
            return targets, None

        hb_exp = hb_notional / max_notional
        hb_exp = _clamp(hb_exp, 0.0, float(self.heartbeat_max_exposure_delta))
        if hb_exp <= 1e-12:
            return targets, None

        # Revert prior heartbeat offset on schedule.
        if (
            abs(float(self._heartbeat_offset_exp)) > 1e-12
            and int(self._bars_seen) >= int(self._heartbeat_clear_bar)
        ):
            sym = (self._heartbeat_symbol or "").strip().upper()
            if sym and sym in universe:
                base = float(targets.get(sym, 0.0))
                new_exp = _clamp(
                    base - float(self._heartbeat_offset_exp),
                    0.0,
                    float(self.max_exposure_per_symbol),
                )
                if abs(new_exp - base) > 1e-12:
                    targets[sym] = float(new_exp)
            self._heartbeat_offset_exp = 0.0
            self._heartbeat_symbol = None
            self._heartbeat_clear_bar = 0
            return targets, "heartbeat_revert"

        last_marker = max(int(self._last_trade_intent_bar), int(self._last_heartbeat_bar))
        if int(self._bars_seen) - last_marker < hb_every:
            return targets, None

        hb_sym: Optional[str] = None

        biggest = 0.0
        for s in universe:
            e = float(targets.get(s, 0.0))
            if e > biggest:
                biggest = e
                hb_sym = s

        if hb_sym is None:
            for s in universe:
                if abs(float(state.positions.get(s, 0.0) or 0.0)) > 1e-12:
                    hb_sym = s
                    break

        if hb_sym is None:
            hb_sym = market_symbol if market_symbol in universe else universe[0]

        gross = float(sum(max(0.0, float(targets.get(s, 0.0))) for s in universe))
        base = float(targets.get(hb_sym, 0.0))
        direction = -1.0 if base > hb_exp * 1.5 else 1.0
        if direction > 0.0 and gross + hb_exp > float(self.max_total_exposure) + 1e-12:
            direction = -1.0

        new_exp = _clamp(base + direction * hb_exp, 0.0, float(self.max_exposure_per_symbol))
        if abs(new_exp - base) <= 1e-12:
            return targets, None

        targets[hb_sym] = float(new_exp)
        self._heartbeat_symbol = str(hb_sym)
        self._heartbeat_offset_exp = float(new_exp - base)
        self._heartbeat_clear_bar = int(self._bars_seen) + 1
        self._last_heartbeat_bar = int(self._bars_seen)
        return targets, "heartbeat"

    def target_exposures(
        self, bars_by_symbol: dict[str, pd.DataFrame], state: StrategyState
    ) -> StrategyDecision:
        universe = self._universe(bars_by_symbol)
        if not universe:
            return StrategyDecision(target_exposures={}, reason="no_crypto_symbols")

        self._bars_seen += 1
        self._maybe_reset_daily_state(state)

        equity = float(state.equity)
        if self._peak_equity <= 0.0:
            self._peak_equity = equity
        self._peak_equity = max(float(self._peak_equity), equity)
        drawdown = (equity / self._peak_equity - 1.0) if self._peak_equity > 0 else 0.0

        extra = dict(state.extra or {})
        max_notional = float(extra.get("max_position_notional_usd", 0.0) or 0.0)

        debug: dict[str, Any] = {
            "bars_seen": int(self._bars_seen),
            "equity": float(equity),
            "drawdown": float(drawdown),
            "day_return": float(state.day_return),
        }

        today = _to_ny(pd.Timestamp(state.timestamp)).date()

        if self._risk_disabled_until_day is not None and today <= self._risk_disabled_until_day:
            return self._risk_off(universe, reason="risk_disabled_cooldown", debug=debug)

        if drawdown <= -abs(float(self.kill_switch)):
            self._peak_equity = float(equity)
            self._risk_disabled_until_day = today + timedelta(
                days=max(0, int(self.kill_switch_cooldown_days))
            )
            return self._risk_off(universe, reason="kill_switch", debug=debug)

        if float(state.day_return) <= -abs(float(self.daily_loss_limit)):
            self._risk_disabled_day = today
            return self._risk_off(universe, reason="daily_loss_limit", debug=debug)

        if self._risk_disabled_day == today:
            return self._risk_off(universe, reason="risk_disabled_day", debug=debug)

        if max_notional <= 0.0:
            return self._risk_off(universe, reason="no_max_position_notional", debug=debug)

        market_symbol = self._market_symbol(universe)
        if market_symbol is None:
            return self._risk_off(universe, reason="no_market_symbol", debug=debug)

        regime_info = self._market_regime(bars_by_symbol.get(market_symbol))
        debug["market_symbol"] = str(market_symbol)
        if regime_info is None:
            return self._risk_off(universe, reason="warmup", debug=debug)

        regime = str(regime_info.get("regime", "neutral"))
        trend_dir = int(regime_info.get("trend_dir", 0) or 0)
        debug["regime"] = regime
        debug["regime_metrics"] = {
            "mom": float(regime_info.get("mom", 0.0) or 0.0),
            "er": float(regime_info.get("er", 0.0) or 0.0),
            "trend_strength": float(regime_info.get("trend_strength", 0.0) or 0.0),
            "trend_dir": int(trend_dir),
        }

        due = False
        if not self._last_targets:
            due = True
        elif self._last_regime != regime:
            due = True
        elif (int(self._bars_seen) - int(self._last_rebalance_bar)) >= max(
            1, int(self.rebalance_interval_bars)
        ):
            due = True

        if not due and self._last_targets:
            hold_targets = {s: float(self._last_targets.get(s, 0.0)) for s in universe}
            hold_targets, hb_reason = self._maybe_apply_heartbeat(
                targets=hold_targets,
                universe=universe,
                market_symbol=market_symbol,
                max_notional=max_notional,
                state=state,
            )
            hold_targets = self._apply_min_trade_notional(
                hold_targets,
                universe,
                max_notional=max_notional,
                allow_small_heartbeat=not bool(self.heartbeat_respect_min_trade_notional),
            )

            prev = dict(self._last_targets)
            self._last_targets = {s: float(hold_targets.get(s, 0.0)) for s in universe}
            changed = any(
                abs(float(self._last_targets.get(s, 0.0)) - float(prev.get(s, 0.0))) > 1e-8
                for s in universe
            )
            if changed:
                self._last_trade_intent_bar = int(self._bars_seen)

            if hb_reason is not None:
                return StrategyDecision(
                    target_exposures=self._last_targets,
                    reason=hb_reason,
                    debug=debug,
                )

            return StrategyDecision(
                target_exposures=self._last_targets,
                reason="hold",
                debug=debug,
            )

        targets = {s: 0.0 for s in universe}
        mode_debug: dict[str, Any] = {}

        if regime == "trend":
            if trend_dir > 0:
                targets, mode_debug = self._trend_targets(
                    universe,
                    bars_by_symbol,
                    exposure_scale=1.0,
                )
            else:
                mode_debug = {"note": "trend_down_no_long"}

        elif regime == "neutral":
            if trend_dir > 0 and float(self.neutral_exposure_scale) > 0.0:
                targets, mode_debug = self._trend_targets(
                    universe,
                    bars_by_symbol,
                    exposure_scale=float(self.neutral_exposure_scale),
                )
            else:
                mode_debug = {"note": "neutral_flat"}

        else:  # range
            market_df = bars_by_symbol.get(market_symbol)
            market_close = _safe_float(
                market_df["close"].iloc[-1], default=0.0
            ) if market_df is not None and not market_df.empty and "close" in market_df.columns else 0.0
            market_qty = float(state.positions.get(market_symbol, 0.0) or 0.0)
            current_market_exp = (
                (market_qty * market_close) / max_notional
                if max_notional > 0.0 and market_close > 0.0
                else 0.0
            )
            current_market_exp = _clamp(current_market_exp, 0.0, float(self.range_max_exposure))

            market_target, mr_dbg = self._range_target(
                market_df,
                current_market_exp=current_market_exp,
            )
            targets[market_symbol] = float(market_target)
            mode_debug = {"range": mr_dbg}

        targets = self._apply_rebalance_threshold(
            targets,
            universe,
            bars_by_symbol,
            state,
            max_notional=max_notional,
        )
        targets = self._apply_min_trade_notional(
            targets,
            universe,
            max_notional=max_notional,
            allow_small_heartbeat=False,
        )

        self._last_rebalance_bar = int(self._bars_seen)
        self._last_regime = regime
        prev = dict(self._last_targets)
        self._last_targets = {s: float(targets.get(s, 0.0)) for s in universe}
        changed = any(
            abs(float(self._last_targets.get(s, 0.0)) - float(prev.get(s, 0.0))) > 1e-8
            for s in universe
        )
        if changed:
            self._last_trade_intent_bar = int(self._bars_seen)

        debug["mode"] = mode_debug

        invested = [s for s in universe if float(self._last_targets.get(s, 0.0)) > 1e-9]
        reason = f"rebalance_{regime}"
        if invested:
            reason = reason + ":" + ",".join(invested)

        return StrategyDecision(
            target_exposures=self._last_targets,
            reason=reason,
            debug=debug,
        )
