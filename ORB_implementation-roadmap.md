## A) Audit + diagnosis summary (educational research, not financial advice)

### Step 1 — Repo audit (mechanics that matter)

* **Decision timing / “what is time t?”** In `run_backtest()`, the engine iterates bar index `idx[i]` and builds `StrategyState.timestamp = idx[i+1]` (the *next bar open*). The strategy is called with `bars_by_symbol[s].iloc[: i+1]`, meaning your signal can use **completed bars up through `idx[i]`**, and any position change you request is eligible to fill at **`idx[i+1]` open** (1-bar latency). 
* **Fill + cost model (baseline realism):** Orders are executed at the next bar open with a per-side slippage model:

  * BUY fill = `open * (1 + slippage_bps/10000)`
  * SELL fill = `open * (1 - slippage_bps/10000)`
    Sells are processed before buys; buys are cash-limited; no bid/ask is needed. This is exactly the model your cost-aware gating should reference. 
* **Constraint enforcement:** If `cfg.allow_short == False`, the engine clips any negative target exposures to 0 before sizing orders. **The engine does not enforce total gross exposure ≤ 1.0 across symbols** (it sizes per symbol), so if a strategy ever emits exposures in both SPY and QQQ, it must enforce `abs(SPY)+abs(QQQ) ≤ 1.0` internally. 
* **Day boundaries / intraday risk:** The engine tracks `day_start_equity` and computes `state.day_return`. “Kill switches” are strategy-side decisions; the engine doesn’t auto-stop. Time-based flattening must be implemented in the strategy using `state.timestamp` (next bar open time). 

**Implication:** To be “flat by 15:55”, you must request flat when `state.timestamp >= 15:55` (since that target fills at that 15:55 open). Likewise, “no entries after 15:30” should be enforced on `state.timestamp` for new positions.

---

### Step 2 — Reproduce + verify the NEC-X failure mode from outputs

Using your provided `metrics.json`, `equity_curve.csv`, and `trades.csv`:

* **Run window + bar size:** Equity curve runs **2025-07-17 → 2026-01-12**, covering **124 sessions**. Median bar delta is **5 minutes** (regular session gaps overnight). First trade appears on **2025-07-21**, consistent with warmup + gating. 
* **Turnover / churn:** There are **244 fills**, i.e. **122 round trips** (no scaling-in; almost all are full in/out). Symbol split: **QQQ 105 trips**, **SPY 17 trips**. This is high for an intraday strategy with a next-open slippage model.
* **Net vs “gross” (slippage impact):** The trade log shows BUY notionals clustered at **$10,001.25**, which strongly implies **max notional ≈ $10,000** and **slippage_bps ≈ 1.25 per side** (since `10000 * (1 + 0.000125) = 10001.25`). Reversing the slippage model on fills yields:

  * **Gross PnL (no slippage)** ≈ **+$27**
  * **Slippage paid** ≈ **-$305**
  * **Net PnL** ≈ **-$278** (matches `total_return = -0.2776%` on 100k). 
    Put differently: average gross edge was about **+0.22 bps per trip**, while friction was about **2.5 bps per round trip** (2 sides × 1.25 bps), so net expectancy turns negative.
* **Where losses come from (exit taxonomy):** Grouping round trips by exit reason (using the SELL reason):

  * `signal_decay_exit`: **60 trips, -$438**, win rate ~37%, mean hold ~17 min
  * `long_only_exit`: **27 trips, -$288**, win rate ~15%, mean hold ~15 min
  * `max_holding_exit`: **26 trips, +$382**, win rate ~88%, mean hold ~30 min
  * `switch`: **9 trips, +$66**, small count
    This supports the claim that **early “decay” style exits dominate losses**, while the smaller set of **held-to-horizon** trades are where the edge concentrates. 
* **Holding-time buckets:** Trades held **25–30 minutes** were strongly positive (high win rate), while the bulk of shorter holds were negative. That’s consistent with **whipsaw churn**: the strategy pays friction repeatedly without capturing a big enough move.
* **Time-of-day:** Entries **before 12:00 ET** were mildly positive in aggregate, while **after 12:00** were materially negative (midday/afternoon churn). This aligns with the post-mortem’s “low edge, costs dominate” diagnosis and suggests a regime gate to avoid chop-prone windows is justified. 

**Conclusion:** The post-mortem’s main claims are supported by the provided files: the run’s gross edge is near-flat, and the strategy is “death by a thousand cuts” under conservative slippage. The clue you shared (raising `k_cost`, `strength_entry`, `rho_min` helped in another run) is directionally consistent with “more selectivity reduces churn,” but we should not overfit to that single unattached result. 

---

## B) New strategy module (drop-in)

Path: `src/atlas/strategies/orb_trend.py`

```python
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import time
from typing import Any, Optional

import pandas as pd

from atlas.strategies.base import Strategy, StrategyDecision, StrategyState
from atlas.utils.time import NY_TZ


def _sign(x: float) -> int:
    if x > 0:
        return 1
    if x < 0:
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
    m = float(diffs.median())
    return m if m > 0 else 1.0


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
    Intraday ORB + VWAP trend strategy for SPY/QQQ (1–5m bars, RTH only).

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

    # Fixed universe
    spy: str = "SPY"
    qqq: str = "QQQ"

    # ---- Tunable parameters (<= 12 total) ----
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
        # The engine does not enforce this; this is informational.
        return int(max(self.atr_window + 2, self.er_window + 2, self.confirm_bars + 10))

    def _required_symbols(self) -> tuple[str, str]:
        return (self.spy.upper(), self.qqq.upper())

    def _session_start(self, ts_ny: pd.Timestamp) -> pd.Timestamp:
        return ts_ny.normalize() + pd.Timedelta(hours=9, minutes=30)

    def _compute_intraday_vwap(self, df_today: pd.DataFrame) -> float:
        if len(df_today) == 0:
            return float("nan")
        tp = (df_today["high"].astype(float) + df_today["low"].astype(float) + df_today["close"].astype(float)) / 3.0
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
        need = int(math.ceil(float(self.orb_minutes) / max(float(bar_minutes), 1e-9)))
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

        orb_ready, orb_high, orb_low, orb_end = self._compute_orb(df_today, bar_minutes=bar_minutes, session_start=session_start)
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

        # Direction and breakout magnitude (bps beyond range).
        if long_ok:
            dir_ = 1
            breakout_bps = float((last_close - orb_high) / last_close * 10_000.0) if last_close > 0 else 0.0
            info["side"] = "LONG"
        else:
            dir_ = -1
            breakout_bps = float((orb_low - last_close) / last_close * 10_000.0) if last_close > 0 else 0.0
            info["side"] = "SHORT"

        # Expected move proxy: breakout distance + trend-scaled typical move.
        trend_edge_bps = float(er) * float(atr_bps) * math.sqrt(max(float(self.expected_hold_bars), 1.0))
        edge_bps = float(max(breakout_bps, 0.0) + max(trend_edge_bps, 0.0))

        # Engine fill model: next-open slippage_bps per side => RT friction per unit exposure:
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
        info: dict[str, Any] = {"symbol": symbol, "held_dir": int(held_dir), "holding_bars": int(holding_bars)}
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

        orb_ready, orb_high, orb_low, _orb_end = self._compute_orb(df_today, bar_minutes=bar_minutes, session_start=session_start)
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

    def target_exposures(self, bars_by_symbol: dict[str, pd.DataFrame], state: StrategyState) -> StrategyDecision:
        spy, qqq = self._required_symbols()
        decision_ts_ny = _to_ny(pd.Timestamp(state.timestamp))

        # Reset risk disable at day boundary.
        if self._risk_disabled_day is not None and decision_ts_ny.date() != self._risk_disabled_day:
            self._risk_disabled_day = None

        targets = {spy: 0.0, qqq: 0.0}
        debug: dict[str, Any] = {
            "ts": decision_ts_ny.isoformat(),
            "day_return": float(state.day_return),
            "allow_short": bool(state.allow_short),
        }

        # Session/time constraints.
        if decision_ts_ny.time() < time(9, 30) or decision_ts_ny.time() > time(16, 0):
            return StrategyDecision(target_exposures=targets, reason="outside_rth", debug=debug)

        # Hard exit: must be flat by/after 15:55 ET.
        if decision_ts_ny.time() >= time(15, 55):
            return StrategyDecision(target_exposures=targets, reason="forced_flat", debug=debug)

        # Risk controls (daily).
        if float(state.day_return) <= -float(self.kill_switch):
            self._risk_disabled_day = decision_ts_ny.date()
            return StrategyDecision(target_exposures=targets, reason="kill_switch", debug=debug)

        if float(state.day_return) <= -float(self.daily_loss_limit):
            self._risk_disabled_day = decision_ts_ny.date()
            return StrategyDecision(target_exposures=targets, reason="daily_loss_limit", debug=debug)

        if self._risk_disabled_day == decision_ts_ny.date():
            return StrategyDecision(target_exposures=targets, reason="risk_disabled", debug=debug)

        # Determine current holdings.
        held_symbols = [s for s in (spy, qqq) if abs(float(state.positions.get(s, 0.0))) > 1e-8]
        if len(held_symbols) > 1:
            # Safety: never allow >1.0 gross if something external created multiple positions.
            debug["held_symbols"] = held_symbols
            return StrategyDecision(target_exposures=targets, reason="multi_position_protect_flat", debug=debug)

        held_symbol = held_symbols[0] if held_symbols else None
        held_qty = float(state.positions.get(held_symbol, 0.0)) if held_symbol else 0.0
        held_dir = _sign(held_qty) if held_symbol else 0
        debug["held_symbol"] = held_symbol
        debug["held_dir"] = int(held_dir)
        debug["holding_bars"] = {spy: int(state.holding_bars.get(spy, 0)), qqq: int(state.holding_bars.get(qqq, 0))}

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
                return StrategyDecision(target_exposures=targets, reason="exit_breakout_fail", debug=debug)

            targets[held_symbol] = float(held_dir)
            return StrategyDecision(target_exposures=targets, reason="hold", debug=debug)

        # Flat: respect "no new entries after 15:30" hard constraint.
        if decision_ts_ny.time() > time(15, 30):
            return StrategyDecision(target_exposures=targets, reason="entry_cutoff", debug=debug)

        # Compute candidates per symbol; choose the best net edge.
        candidates: list[dict[str, Any]] = []
        for sym in (spy, qqq):
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

        # Long-only fallback: if shorts not allowed, never emit negative exposure.
        if (not state.allow_short) and dir_ < 0:
            return StrategyDecision(target_exposures=targets, reason="long_only_abstain", debug=debug)

        targets[chosen] = float(dir_)

        debug["chosen"] = chosen
        debug["chosen_dir"] = int(dir_)
        debug["gross_exposure"] = float(abs(targets[spy]) + abs(targets[qqq]))

        # Hard clamp, though by construction gross <= 1.0.
        gross = abs(targets[spy]) + abs(targets[qqq])
        if gross > 1.0 + 1e-9:
            scale = 1.0 / gross
            targets = {spy: float(targets[spy]) * scale, qqq: float(targets[qqq]) * scale}
            debug["gross_clamped"] = True
            debug["gross_exposure"] = float(abs(targets[spy]) + abs(targets[qqq]))

        return StrategyDecision(target_exposures=targets, reason="enter", debug=debug)
```

---

## C) Minimal integration patch (registry + TUI + CLI slippage default)

```diff
diff --git a/src/atlas/strategies/registry.py b/src/atlas/strategies/registry.py
--- a/src/atlas/strategies/registry.py
+++ b/src/atlas/strategies/registry.py
@@
 from atlas.strategies.base import Strategy
 from atlas.strategies.ma_crossover import MovingAverageCrossover
 from atlas.strategies.nec_x import NecX
+from atlas.strategies.orb_trend import OrbTrend
@@
     if name in {"nec_x", "nec-x"}:
@@
         return NecX(
@@
             tick_size=_get_float("tick_size", 0.01),
         )
+
+    if name in {"orb_trend", "orb-trend"}:
+        required = {"SPY", "QQQ"}
+        if not required.issubset({s.upper() for s in symbols}):
+            raise ValueError("orb_trend requires --symbols SPY,QQQ")
+
+        def _get_int(key: str, default: int) -> int:
+            raw = params.get(key, params.get(key.lower(), default))
+            return int(raw)
+
+        def _get_float(key: str, default: float) -> float:
+            raw = params.get(key, params.get(key.lower(), default))
+            return float(raw)
+
+        return OrbTrend(
+            orb_minutes=_get_int("orb_minutes", 30),
+            orb_breakout_bps=_get_float("orb_breakout_bps", 4.0),
+            confirm_bars=_get_int("confirm_bars", 2),
+            atr_window=_get_int("atr_window", 20),
+            er_window=_get_int("er_window", 12),
+            er_min=_get_float("er_min", 0.35),
+            expected_hold_bars=_get_int("expected_hold_bars", 12),
+            k_cost=_get_float("k_cost", 2.0),
+            slippage_bps=_get_float("slippage_bps", 1.25),
+            min_hold_bars=_get_int("min_hold_bars", 3),
+            daily_loss_limit=_get_float("daily_loss_limit", 0.010),
+            kill_switch=_get_float("kill_switch", 0.025),
+        )
@@
 def list_strategy_names() -> list[str]:
-    return ["ma_crossover", "nec_x"]
+    return ["ma_crossover", "nec_x", "orb_trend"]

diff --git a/src/atlas/tui/app.py b/src/atlas/tui/app.py
--- a/src/atlas/tui/app.py
+++ b/src/atlas/tui/app.py
@@
 STRATEGY_PARAM_SPECS: dict[str, dict[str, type]] = {
     "ma_crossover": {
         "fast_window": int,
         "slow_window": int,
     },
     "nec_x": {
         "M": int,
         "V": int,
         "Wcorr": int,
         "rho_min": float,
         "strength_entry": float,
         "strength_exit": float,
         "H_max": int,
         "k_cost": float,
         "spread_floor_bps": float,
         "slip_bps": float,
         "daily_loss_limit": float,
         "kill_switch": float,
     },
+    "orb_trend": {
+        "orb_minutes": int,
+        "orb_breakout_bps": float,
+        "confirm_bars": int,
+        "atr_window": int,
+        "er_window": int,
+        "er_min": float,
+        "expected_hold_bars": int,
+        "k_cost": float,
+        "slippage_bps": float,
+        "min_hold_bars": int,
+        "daily_loss_limit": float,
+        "kill_switch": float,
+    },
 }
@@
 STRATEGY_DEFAULT_PARAMS: dict[str, dict[str, Any]] = {
     "ma_crossover": {
         "fast_window": 10,
         "slow_window": 30,
     },
     "nec_x": {
         "M": 6,
         "V": 12,
         "Wcorr": 12,
         "rho_min": 0.60,
         "strength_entry": 0.80,
         "strength_exit": 0.20,
         "H_max": 6,
         "k_cost": 1.25,
         "spread_floor_bps": 0.50,
         "slip_bps": 0.75,
         "daily_loss_limit": 0.010,
         "kill_switch": 0.025,
     },
+    "orb_trend": {
+        "orb_minutes": 30,
+        "orb_breakout_bps": 4.0,
+        "confirm_bars": 2,
+        "atr_window": 20,
+        "er_window": 12,
+        "er_min": 0.35,
+        "expected_hold_bars": 12,
+        "k_cost": 2.0,
+        "slippage_bps": 1.25,
+        "min_hold_bars": 3,
+        "daily_loss_limit": 0.010,
+        "kill_switch": 0.025,
+    },
 }
@@
         if cmd in {"/algorithm", "/strategy", "/aglorithm", "/algorithim"} and args:
             strategy = self._canonicalize_strategy_name(args[0])
             self.state.strategy = strategy
             self._ensure_strategy_params(strategy)
-            if strategy == "nec_x":
+            if strategy in {"nec_x", "orb_trend"}:
                 self.state.symbols = "SPY,QQQ"
                 self.state.bar_timeframe = "5Min"
                 self.state.slippage_bps = 1.25
             self._render_settings()
             return

diff --git a/src/atlas/cli.py b/src/atlas/cli.py
--- a/src/atlas/cli.py
+++ b/src/atlas/cli.py
@@
         slippage_bps=float(
-            (1.25 if strategy in {"nec_x", "nec-x"} else 0.0)
+            (1.25 if strategy in {"nec_x", "nec-x", "orb_trend", "orb-trend"} else 0.0)
             if slippage_bps is None
             else slippage_bps
         ),
```

---

## D) Optional default params JSON snippet

```json
{
  "orb_trend": {
    "orb_minutes": 30,
    "orb_breakout_bps": 4.0,
    "confirm_bars": 2,
    "atr_window": 20,
    "er_window": 12,
    "er_min": 0.35,
    "expected_hold_bars": 12,
    "k_cost": 2.0,
    "slippage_bps": 1.25,
    "min_hold_bars": 3,
    "daily_loss_limit": 0.01,
    "kill_switch": 0.025
  }
}
```
