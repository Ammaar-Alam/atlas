from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import time
from typing import Any, Optional

import pandas as pd

from atlas.strategies.base import Strategy, StrategyDecision, StrategyState
from atlas.utils.time import NY_TZ


def _sign(value: float) -> int:
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def _to_ny(ts: pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    if ts.tz is None:
        return ts.tz_localize(NY_TZ)
    return ts.tz_convert(NY_TZ)


def _infer_bar_minutes(index: pd.DatetimeIndex) -> float:
    """
    Infer bar size in minutes, ignoring overnight gaps.
    """
    if len(index) < 3:
        return 1.0
    diffs = index.to_series().diff().dropna().dt.total_seconds() / 60.0
    diffs = diffs[(diffs > 0) & (diffs < 60)]  # drop session gaps
    if len(diffs) == 0:
        return 1.0
    median = float(diffs.median())
    return median if median > 0 else 1.0


def _true_range(high: pd.Series, low: pd.Series, prev_close: pd.Series) -> pd.Series:
    hl = (high - low).abs()
    hc = (high - prev_close).abs()
    lc = (low - prev_close).abs()
    return pd.concat([hl, hc, lc], axis=1).max(axis=1)


def _efficiency_ratio(close: pd.Series, window: int) -> float:
    """
    Kaufman Efficiency Ratio over `window` bars in [0,1].
    Uses only close prices; robust for intraday chop vs trend.
    """
    window = int(window)
    if window <= 1 or len(close) < window + 1:
        return 0.0
    segment = close.iloc[-(window + 1) :].astype(float)
    change = float(abs(segment.iloc[-1] - segment.iloc[0]))
    volatility = float(segment.diff().abs().sum())
    if volatility <= 0:
        return 0.0
    return float(change / volatility)


@dataclass
class OrbTrend(Strategy):
    """
    Intraday ORB + VWAP trend strategy for a universe of symbols (1–5m bars, RTH only).

    New edge source (vs NEC-X):
      - Opening Range Breakout (ORB) continuation, confirmed on closes
      - VWAP alignment (price must be on the "right" side of VWAP)
      - Trend-quality gate via Efficiency Ratio (ER) to abstain in chop

    Controls to target "many small trades can't clear costs + whipsaw exits":
      - Cost-aware admission: expected_edge_bps must exceed k_cost * (round_trip_cost_bps)
      - Regime abstention: ORB-only entries + ER gate
      - Anti-whipsaw: confirmation bars + minimum hold + hysteresis exits

    Notes on execution alignment:
      - This strategy assumes the engine fills at NEXT bar OPEN with a per-side `slippage_bps`.
      - Set this strategy's `slippage_bps` to match BacktestConfig.slippage_bps for consistent gating.
    """

    name: str = "orb_trend"

    # Universe (default)
    symbols: tuple[str, ...] = ("SPY", "QQQ")

    # ---- Tunable parameters (exactly 12) ----
    orb_minutes: int = 30
    orb_breakout_bps: float = 4.0
    confirm_bars: int = 2

    atr_window: int = 20
    er_window: int = 12
    er_min: float = 0.35

    expected_hold_bars: int = 12  # only used for the edge proxy scaling

    k_cost: float = 2.0
    slippage_bps: float = 1.25  # per side, should match engine

    min_hold_bars: int = 3

    daily_loss_limit: float = 0.010
    kill_switch: float = 0.025

    # ---- Internal state ----
    _risk_disabled_day: Optional[object] = field(default=None, init=False, repr=False)

    def warmup_bars(self) -> int:
        return int(max(self.atr_window + 2, self.er_window + 2, self.confirm_bars + 10))

    def _universe(self) -> list[str]:
        raw = getattr(self, "symbols", ())
        if isinstance(raw, str):
            parts = [s.strip() for s in raw.split(",")]
        else:
            parts = [str(s).strip() for s in raw]

        out: list[str] = []
        seen: set[str] = set()
        for symbol in parts:
            sym = symbol.upper()
            if not sym or sym in seen:
                continue
            seen.add(sym)
            out.append(sym)
        return out

    def _session_start(self, ts_ny: pd.Timestamp) -> pd.Timestamp:
        # Equity ORB uses NYSE open; crypto runs 24/7 so we anchor the "session"
        # to the NY day boundary for a consistent daily reset.
        if any("/" in s for s in self._universe()):
            return ts_ny.normalize()
        return ts_ny.normalize() + pd.Timedelta(hours=9, minutes=30)

    def _compute_intraday_vwap(self, df_today: pd.DataFrame) -> float:
        if len(df_today) == 0:
            return float("nan")
        tp = (
            df_today["high"].astype(float)
            + df_today["low"].astype(float)
            + df_today["close"].astype(float)
        ) / 3.0
        vol = df_today["volume"].astype(float).clip(lower=0.0)
        den = float(vol.sum())
        if den <= 0:
            return float(df_today["close"].astype(float).iloc[-1])
        return float((tp * vol).sum() / den)

    def _compute_orb(
        self,
        df_today: pd.DataFrame,
        *,
        bar_minutes: float,
        session_start: pd.Timestamp,
    ) -> tuple[bool, float, float, pd.Timestamp]:
        """
        Returns (orb_ready, orb_high, orb_low, orb_end_ts).
        ORB uses bars whose OPEN timestamp is < orb_end_ts.
        """
        orb_end = session_start + pd.Timedelta(minutes=int(self.orb_minutes))
        if len(df_today) == 0:
            return (False, float("nan"), float("nan"), orb_end)

        orb_window = df_today[df_today.index < orb_end]
        need = int(
            math.ceil(float(self.orb_minutes) / max(float(bar_minutes), 1e-9))
        )
        if len(orb_window) < max(1, need):
            return (False, float("nan"), float("nan"), orb_end)

        orb_high = float(orb_window["high"].astype(float).max())
        orb_low = float(orb_window["low"].astype(float).min())
        return (True, orb_high, orb_low, orb_end)

    def _atr_bps(self, df: pd.DataFrame) -> float:
        if len(df) < 3:
            return 0.0
        w = max(int(self.atr_window), 2)
        tail = df.iloc[-(w + 1) :].copy()
        high = tail["high"].astype(float)
        low = tail["low"].astype(float)
        close = tail["close"].astype(float)
        prev_close = close.shift(1)
        tr = _true_range(high, low, prev_close).dropna()
        if len(tr) == 0:
            return 0.0
        atr = float(tr.iloc[-w:].mean())
        last_close = float(close.iloc[-1]) if float(close.iloc[-1]) > 0 else 0.0
        if last_close <= 0:
            return 0.0
        return float((atr / last_close) * 10_000.0)

    def _entry_candidate(
        self,
        *,
        symbol: str,
        df: pd.DataFrame,
        decision_ts_ny: pd.Timestamp,
        allow_short: bool,
    ) -> tuple[bool, dict[str, Any]]:
        """
        Compute an entry candidate for one symbol at the current decision time.
        Returns (ok, info). If ok=True, info contains keys:
          dir, edge_bps, net_edge_bps, cost_rt_bps, orb_high, orb_low, vwap, er, atr_bps, reason_tag
        """
        info: dict[str, Any] = {"symbol": symbol}

        if df is None or len(df) < 10:
            info["reason_tag"] = "insufficient_bars"
            return False, info

        df = df.sort_index()
        idx_ny = df.index
        if idx_ny.tz is None:
            idx_ny = idx_ny.tz_localize(NY_TZ)
        else:
            idx_ny = idx_ny.tz_convert(NY_TZ)
        df = df.copy()
        df.index = idx_ny

        bar_minutes = _infer_bar_minutes(df.index)
        info["bar_minutes"] = float(bar_minutes)

        session_start = self._session_start(decision_ts_ny)
        df_today = df[df.index >= session_start]
        if len(df_today) < 5:
            info["reason_tag"] = "too_few_today"
            return False, info

        vwap = self._compute_intraday_vwap(df_today)
        info["vwap"] = float(vwap)

        orb_ready, orb_high, orb_low, orb_end = self._compute_orb(
            df_today, bar_minutes=bar_minutes, session_start=session_start
        )
        info["orb_ready"] = bool(orb_ready)
        info["orb_high"] = float(orb_high) if orb_ready else None
        info["orb_low"] = float(orb_low) if orb_ready else None
        info["orb_end"] = _to_ny(orb_end).isoformat()
        if not orb_ready:
            info["reason_tag"] = "orb_not_ready"
            return False, info

        # Breakout checks should use bars after ORB end.
        df_after_orb = df_today[df_today.index >= orb_end]
        if len(df_after_orb) < int(self.confirm_bars):
            info["reason_tag"] = "confirm_wait"
            return False, info

        last_close = float(df["close"].astype(float).iloc[-1])
        info["close"] = float(last_close)

        er = _efficiency_ratio(df["close"].astype(float), int(self.er_window))
        atr_bps = self._atr_bps(df)
        info["er"] = float(er)
        info["atr_bps"] = float(atr_bps)

        if er < float(self.er_min):
            info["reason_tag"] = "gate_er"
            return False, info

        buf = float(self.orb_breakout_bps) / 10_000.0
        th_up = float(orb_high) * (1.0 + buf)
        th_dn = float(orb_low) * (1.0 - buf)
        info["th_up"] = float(th_up)
        info["th_dn"] = float(th_dn)

        closes_after = df_after_orb["close"].astype(float)
        recent = closes_after.iloc[-int(self.confirm_bars) :]

        long_ok = bool((recent > th_up).all()) and (last_close > vwap)
        short_ok = bool((recent < th_dn).all()) and (last_close < vwap)

        if not allow_short:
            short_ok = False

        if not long_ok and not short_ok:
            info["reason_tag"] = "no_breakout"
            return False, info

        if long_ok:
            dir_ = 1
            breakout_bps = (
                float((last_close - orb_high) / last_close * 10_000.0)
                if last_close > 0
                else 0.0
            )
            info["side"] = "LONG"
        else:
            dir_ = -1
            breakout_bps = (
                float((orb_low - last_close) / last_close * 10_000.0)
                if last_close > 0
                else 0.0
            )
            info["side"] = "SHORT"

        trend_edge_bps = (
            float(er)
            * float(atr_bps)
            * math.sqrt(max(float(self.expected_hold_bars), 1.0))
        )
        edge_bps = float(max(breakout_bps, 0.0) + max(trend_edge_bps, 0.0))

        cost_rt_bps = float(2.0 * float(self.slippage_bps))
        net_edge_bps = float(edge_bps) - float(self.k_cost) * float(cost_rt_bps)

        info["dir"] = int(dir_)
        info["breakout_bps"] = float(breakout_bps)
        info["trend_edge_bps"] = float(trend_edge_bps)
        info["edge_bps"] = float(edge_bps)
        info["cost_rt_bps"] = float(cost_rt_bps)
        info["net_edge_bps"] = float(net_edge_bps)

        if net_edge_bps <= 0.0:
            info["reason_tag"] = "net_edge_not_positive"
            return False, info

        info["reason_tag"] = "candidate_ok"
        return True, info

    def _exit_signal(
        self,
        *,
        symbol: str,
        df: pd.DataFrame,
        decision_ts_ny: pd.Timestamp,
        held_dir: int,
        holding_bars: int,
    ) -> tuple[bool, dict[str, Any]]:
        """
        Determine whether to exit an existing position.
        Returns (should_exit, info).
        """
        info: dict[str, Any] = {
            "symbol": symbol,
            "held_dir": int(held_dir),
            "holding_bars": int(holding_bars),
        }
        if df is None or len(df) < 5:
            info["reason_tag"] = "missing_bars_exit"
            return True, info

        df = df.sort_index()
        idx_ny = df.index
        if idx_ny.tz is None:
            idx_ny = idx_ny.tz_localize(NY_TZ)
        else:
            idx_ny = idx_ny.tz_convert(NY_TZ)
        df = df.copy()
        df.index = idx_ny

        bar_minutes = _infer_bar_minutes(df.index)
        session_start = self._session_start(decision_ts_ny)
        df_today = df[df.index >= session_start]

        vwap = self._compute_intraday_vwap(df_today)
        info["vwap"] = float(vwap)

        orb_ready, orb_high, orb_low, _orb_end = self._compute_orb(
            df_today, bar_minutes=bar_minutes, session_start=session_start
        )
        info["orb_ready"] = bool(orb_ready)
        info["orb_high"] = float(orb_high) if orb_ready else None
        info["orb_low"] = float(orb_low) if orb_ready else None

        last_close = float(df["close"].astype(float).iloc[-1])
        info["close"] = float(last_close)

        if not orb_ready:
            info["reason_tag"] = "orb_unavailable_exit"
            return True, info

        # Hysteresis: require a meaningful move back into the range.
        buf = float(self.orb_breakout_bps) / 10_000.0
        if held_dir > 0:
            fail_level = float(orb_high) * (1.0 - buf)
            should_exit = (last_close < fail_level) or (last_close < vwap)
        else:
            fail_level = float(orb_low) * (1.0 + buf)
            should_exit = (last_close > fail_level) or (last_close > vwap)

        info["fail_level"] = float(fail_level)

        # Anti-whipsaw: min-hold bars before acting on these exits.
        if holding_bars < int(self.min_hold_bars):
            info["reason_tag"] = "min_hold"
            return False, info

        info["reason_tag"] = "breakout_fail" if should_exit else "hold_ok"
        return bool(should_exit), info

    def target_exposures(
        self, bars_by_symbol: dict[str, pd.DataFrame], state: StrategyState
    ) -> StrategyDecision:
        universe = self._universe()
        if not universe:
            raise ValueError("orb_trend requires at least 1 symbol")
        decision_ts_ny = _to_ny(pd.Timestamp(state.timestamp))

        if self._risk_disabled_day is not None and decision_ts_ny.date() != self._risk_disabled_day:
            self._risk_disabled_day = None

        targets = {sym: 0.0 for sym in universe}
        debug: dict[str, Any] = {
            "ts": decision_ts_ny.isoformat(),
            "day_return": float(state.day_return),
            "allow_short": bool(state.allow_short),
        }

        crypto_mode = any("/" in s for s in universe)
        debug["crypto_mode"] = bool(crypto_mode)

        if not crypto_mode:
            # Session/time constraints (equities only).
            if decision_ts_ny.time() < time(9, 30) or decision_ts_ny.time() > time(16, 0):
                return StrategyDecision(
                    target_exposures=targets, reason="outside_rth", debug=debug
                )

            # Hard exit: must be flat by/after 15:55 ET.
            if decision_ts_ny.time() >= time(15, 55):
                return StrategyDecision(
                    target_exposures=targets, reason="forced_flat", debug=debug
                )

        # Risk controls (daily).
        if float(state.day_return) <= -float(self.kill_switch):
            self._risk_disabled_day = decision_ts_ny.date()
            return StrategyDecision(
                target_exposures=targets, reason="kill_switch", debug=debug
            )

        if float(state.day_return) <= -float(self.daily_loss_limit):
            self._risk_disabled_day = decision_ts_ny.date()
            return StrategyDecision(
                target_exposures=targets, reason="daily_loss_limit", debug=debug
            )

        if self._risk_disabled_day == decision_ts_ny.date():
            return StrategyDecision(
                target_exposures=targets, reason="risk_disabled", debug=debug
            )

        held_symbols = [
            s for s in universe if abs(float(state.positions.get(s, 0.0))) > 1e-8
        ]
        if len(held_symbols) > 1:
            debug["held_symbols"] = held_symbols
            return StrategyDecision(
                target_exposures=targets,
                reason="multi_position_protect_flat",
                debug=debug,
            )

        held_symbol = held_symbols[0] if held_symbols else None
        held_qty = float(state.positions.get(held_symbol, 0.0)) if held_symbol else 0.0
        held_dir = _sign(held_qty) if held_symbol else 0
        debug["held_symbol"] = held_symbol
        debug["held_dir"] = int(held_dir)
        debug["holding_bars"] = {s: int(state.holding_bars.get(s, 0)) for s in universe}

        # If holding, manage exit only (no switching to reduce churn).
        if held_symbol is not None and held_dir != 0:
            hb = int(state.holding_bars.get(held_symbol, 0))
            should_exit, exit_dbg = self._exit_signal(
                symbol=held_symbol,
                df=bars_by_symbol.get(held_symbol),
                decision_ts_ny=decision_ts_ny,
                held_dir=held_dir,
                holding_bars=hb,
            )
            debug["exit"] = exit_dbg
            if should_exit:
                return StrategyDecision(
                    target_exposures=targets,
                    reason="exit_breakout_fail",
                    debug=debug,
                )

            targets[held_symbol] = float(held_dir)
            return StrategyDecision(target_exposures=targets, reason="hold", debug=debug)

        if not crypto_mode:
            # Flat: respect "no new entries after 15:30" hard constraint.
            if decision_ts_ny.time() > time(15, 30):
                return StrategyDecision(
                    target_exposures=targets, reason="entry_cutoff", debug=debug
                )

        candidates: list[dict[str, Any]] = []
        for sym in universe:
            ok, info = self._entry_candidate(
                symbol=sym,
                df=bars_by_symbol.get(sym),
                decision_ts_ny=decision_ts_ny,
                allow_short=bool(state.allow_short),
            )
            debug[f"cand_{sym}"] = info
            if ok:
                candidates.append(info)

        if not candidates:
            return StrategyDecision(target_exposures=targets, reason="no_trade", debug=debug)

        best = max(candidates, key=lambda d: float(d.get("net_edge_bps", -1e9)))
        chosen = str(best["symbol"])
        dir_ = int(best["dir"])

        if (not state.allow_short) and dir_ < 0:
            return StrategyDecision(
                target_exposures=targets, reason="long_only_abstain", debug=debug
            )

        targets[chosen] = float(dir_)

        debug["chosen"] = chosen
        debug["chosen_dir"] = int(dir_)
        debug["gross_exposure"] = float(sum(abs(targets[s]) for s in universe))

        gross = float(sum(abs(targets[s]) for s in universe))
        if gross > 1.0 + 1e-9:
            scale = 1.0 / gross
            targets = {s: float(targets[s]) * scale for s in universe}
            debug["gross_clamped"] = True
            debug["gross_exposure"] = float(sum(abs(targets[s]) for s in universe))

        return StrategyDecision(target_exposures=targets, reason="enter", debug=debug)
