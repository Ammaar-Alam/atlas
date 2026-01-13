from __future__ import annotations

import math
from collections import deque
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


@dataclass
class NecPDT(Strategy):
    name: str = "nec_pdt"

    # universe (fixed)
    spy: str = "SPY"
    qqq: str = "QQQ"

    # NEC-X core (kept stable; not tuned here)
    M: int = 6
    V: int = 12
    eps: float = 1e-8

    # V2 tunables (see NEC-2_implementation-roadmap.md)
    H: int = 12
    base_thr_bps: float = 10.0
    budget_step_bps: float = 4.0
    atr_lookback_bars: int = 12
    stop_atr_mult: float = 2.0
    trail_atr_mult: float = 2.5
    min_hold_bars: int = 4
    flip_confirm_bars: int = 3
    max_day_trades_per_rolling_5_days: int = 3

    # Conservative standardized costs (round-trip, bps)
    half_spread_bps: float = 1.5
    slippage_bps: float = 2.0
    fee_bps: float = 0.3

    # internal running state (indicators)
    _last_processed: Optional[pd.Timestamp] = field(
        default=None, init=False, repr=False
    )
    _bars_seen: int = field(default=0, init=False, repr=False)
    _prev_close: dict[str, Optional[float]] = field(
        default_factory=dict, init=False, repr=False
    )
    _ema_m: dict[str, Optional[float]] = field(
        default_factory=dict, init=False, repr=False
    )
    _ema_v: dict[str, Optional[float]] = field(
        default_factory=dict, init=False, repr=False
    )
    _trs: dict[str, deque[float]] = field(default_factory=dict, init=False, repr=False)
    _last_features: dict[str, dict[str, float]] = field(
        default_factory=dict, init=False, repr=False
    )

    # PDT budget state
    _session_date: Optional[object] = field(default=None, init=False, repr=False)
    _day_trade_log: deque[dict[str, object]] = field(
        default_factory=deque, init=False, repr=False
    )
    _used_in_window: int = field(default=0, init=False, repr=False)
    _reserved_today: bool = field(default=False, init=False, repr=False)
    _done_today: bool = field(default=False, init=False, repr=False)

    # Position tracking for stops/trailing exits
    _held_symbol: Optional[str] = field(default=None, init=False, repr=False)
    _held_dir: int = field(default=0, init=False, repr=False)
    _entry_price: Optional[float] = field(default=None, init=False, repr=False)
    _stop_price: Optional[float] = field(default=None, init=False, repr=False)
    _best_close: Optional[float] = field(default=None, init=False, repr=False)
    _flip_count: int = field(default=0, init=False, repr=False)

    def warmup_bars(self) -> int:
        return int(max(20, self.M + 2, self.V + 2, self.atr_lookback_bars + 1))

    def _required_symbols(self) -> tuple[str, str]:
        return (self.spy.upper(), self.qqq.upper())

    def _reset_indicators(self) -> None:
        self._last_processed = None
        self._bars_seen = 0
        self._prev_close = {}
        self._ema_m = {}
        self._ema_v = {}
        self._trs = {}
        self._last_features = {}

    def _ensure_session(self, session_date: object) -> None:
        if self._session_date == session_date:
            return
        self._session_date = session_date
        if not self._day_trade_log or self._day_trade_log[-1].get("date") != session_date:
            self._day_trade_log.append({"date": session_date, "day_trades": 0})
        while len(self._day_trade_log) > 5:
            self._day_trade_log.popleft()

        used = 0
        for row in self._day_trade_log:
            used += int(row.get("day_trades", 0) or 0)
        self._used_in_window = int(used)
        self._reserved_today = False
        self._done_today = False
        self._clear_position_state()

    def _clear_position_state(self) -> None:
        self._held_symbol = None
        self._held_dir = 0
        self._entry_price = None
        self._stop_price = None
        self._best_close = None
        self._flip_count = 0

    def _reserve_day_trade(self, session_date: object) -> None:
        self._reserved_today = True
        if self._day_trade_log and self._day_trade_log[-1].get("date") == session_date:
            self._day_trade_log[-1]["day_trades"] = 1
        else:
            self._day_trade_log.append({"date": session_date, "day_trades": 1})
            while len(self._day_trade_log) > 5:
                self._day_trade_log.popleft()

        used = 0
        for row in self._day_trade_log:
            used += int(row.get("day_trades", 0) or 0)
        self._used_in_window = int(used)

    def _cost_roundtrip_bps(self) -> float:
        return float(2.0 * (self.half_spread_bps + self.slippage_bps + self.fee_bps))

    def _update_from_bars(self, bars_by_symbol: dict[str, pd.DataFrame]) -> None:
        spy, qqq = self._required_symbols()
        if spy not in bars_by_symbol or qqq not in bars_by_symbol:
            raise ValueError(f"nec_pdt requires bars for {spy} and {qqq}")

        df_spy = bars_by_symbol[spy]
        df_qqq = bars_by_symbol[qqq]
        common = df_spy.index.intersection(df_qqq.index).sort_values()
        if len(common) < 3:
            return

        if self._last_processed is not None:
            if self._last_processed not in common and self._last_processed < common[0]:
                self._reset_indicators()

        if self._last_processed is None:
            new_index = common
        else:
            new_index = common[common > self._last_processed]
        if len(new_index) == 0:
            return

        alpha_m = 2.0 / (float(self.M) + 1.0)
        alpha_v = 2.0 / (float(self.V) + 1.0)

        for ts in new_index:
            for symbol in (spy, qqq):
                df = bars_by_symbol[symbol]
                row = df.loc[ts]
                open_px = float(row["open"])
                high = float(row["high"])
                low = float(row["low"])
                close = float(row["close"])

                prev_close = self._prev_close.get(symbol)
                if prev_close is None or prev_close <= 0 or close <= 0:
                    r = 0.0
                else:
                    r = math.log(close / prev_close)

                m_prev = self._ema_m.get(symbol)
                v_prev = self._ema_v.get(symbol)
                m = (
                    r
                    if m_prev is None
                    else (alpha_m * r + (1.0 - alpha_m) * float(m_prev))
                )
                v = (
                    abs(r)
                    if v_prev is None
                    else (alpha_v * abs(r) + (1.0 - alpha_v) * float(v_prev))
                )
                score = float(m) / (float(v) + float(self.eps))

                if prev_close is None:
                    tr = float(high - low)
                else:
                    tr = float(
                        max(
                            high - low,
                            abs(high - float(prev_close)),
                            abs(low - float(prev_close)),
                        )
                    )

                dq = self._trs.get(symbol)
                if dq is None:
                    dq = deque(maxlen=int(self.atr_lookback_bars))
                    self._trs[symbol] = dq
                dq.append(float(tr))

                atr = float(sum(dq) / len(dq)) if len(dq) else 0.0
                atr_pct = float(atr / close) if close > 0 else 0.0

                self._ema_m[symbol] = float(m)
                self._ema_v[symbol] = float(v)
                self._prev_close[symbol] = float(close)
                self._last_features[symbol] = {
                    "open": float(open_px),
                    "high": float(high),
                    "low": float(low),
                    "close": float(close),
                    "m": float(m),
                    "v": float(v),
                    "score": float(score),
                    "tr": float(tr),
                    "atr": float(atr),
                    "atr_pct": float(atr_pct),
                    "atr_bps": float(atr_pct * 10_000.0),
                }

            self._last_processed = pd.Timestamp(ts)
            self._bars_seen += 1

    def _ready(self) -> bool:
        spy, qqq = self._required_symbols()
        if self._bars_seen < self.warmup_bars():
            return False
        if spy not in self._last_features or qqq not in self._last_features:
            return False
        if len(self._trs.get(spy, [])) < int(self.atr_lookback_bars) or len(
            self._trs.get(qqq, [])
        ) < int(self.atr_lookback_bars):
            return False
        return True

    def _confirmed_dir(self) -> tuple[int, int, int]:
        spy, qqq = self._required_symbols()
        spy_score = float(self._last_features[spy]["score"])
        qqq_score = float(self._last_features[qqq]["score"])
        dir_spy = _sign(spy_score)
        dir_qqq = _sign(qqq_score)
        confirmed = dir_spy if (dir_spy != 0 and dir_spy == dir_qqq) else 0
        return int(confirmed), int(dir_spy), int(dir_qqq)

    def target_exposures(
        self, bars_by_symbol: dict[str, pd.DataFrame], state: StrategyState
    ) -> StrategyDecision:
        spy, qqq = self._required_symbols()
        decision_ts = pd.Timestamp(state.timestamp)
        if decision_ts.tz is None:
            decision_ts = decision_ts.tz_localize(NY_TZ)
        else:
            decision_ts = decision_ts.tz_convert(NY_TZ)

        self._ensure_session(decision_ts.date())

        targets = {spy: 0.0, qqq: 0.0}
        debug: dict[str, Any] = {
            "ts": decision_ts.isoformat(),
            "bars_seen": int(self._bars_seen),
            "pdt": {
                "log": [
                    {"date": str(row.get("date")), "day_trades": int(row.get("day_trades", 0) or 0)}
                    for row in self._day_trade_log
                ],
                "used_in_window": int(self._used_in_window),
                "budget_remaining": int(
                    max(0, int(self.max_day_trades_per_rolling_5_days) - int(self._used_in_window))
                ),
                "reserved_today": bool(self._reserved_today),
                "done_today": bool(self._done_today),
            },
        }

        held_symbols = [
            s for s in (spy, qqq) if abs(float(state.positions.get(s, 0.0))) > 1e-8
        ]
        if len(held_symbols) > 1:
            return StrategyDecision(
                target_exposures=targets,
                reason="multi_position_not_supported",
                debug=debug,
            )
        held_symbol = held_symbols[0] if held_symbols else None
        held_qty = float(state.positions.get(held_symbol, 0.0)) if held_symbol else 0.0
        held_dir = _sign(held_qty) if held_symbol else 0
        holding_bars = int(state.holding_bars.get(held_symbol, 0)) if held_symbol else 0

        debug["held_symbol"] = held_symbol
        debug["held_dir"] = int(held_dir)
        debug["holding_bars"] = int(holding_bars)

        if decision_ts.time() < time(9, 30) or decision_ts.time() > time(16, 0):
            return StrategyDecision(
                target_exposures=targets, reason="outside_rth", debug=debug
            )

        if decision_ts.time() >= time(15, 55):
            self._done_today = True
            self._clear_position_state()
            return StrategyDecision(
                target_exposures=targets, reason="forced_flat", debug=debug
            )

        self._update_from_bars(bars_by_symbol)
        if spy in self._last_features:
            debug["spy"] = self._last_features[spy]
        if qqq in self._last_features:
            debug["qqq"] = self._last_features[qqq]

        if held_symbol is not None and not self._reserved_today:
            self._reserve_day_trade(decision_ts.date())

        if not self._ready():
            if held_symbol is not None:
                targets[held_symbol] = float(held_dir)
                return StrategyDecision(
                    target_exposures=targets, reason="warmup_hold", debug=debug
                )
            return StrategyDecision(
                target_exposures=targets, reason="warmup", debug=debug
            )

        confirmed_dir, dir_spy, dir_qqq = self._confirmed_dir()
        debug["dir"] = {"confirmed": int(confirmed_dir), spy: int(dir_spy), qqq: int(dir_qqq)}

        if held_symbol is not None:
            last_close = float(self._last_features[held_symbol]["close"])
            atr_pct_now = float(self._last_features[held_symbol]["atr_pct"])

            if self._held_symbol != held_symbol or int(self._held_dir) != int(held_dir):
                self._held_symbol = held_symbol
                self._held_dir = int(held_dir)
                self._flip_count = 0

                entry_price = None
                if holding_bars <= 1:
                    entry_price = float(self._last_features[held_symbol]["open"])
                    debug["entry_bar_open"] = float(entry_price)
                else:
                    entry_price = float(last_close)
                    debug["entry_px_assumed_from_close"] = True

                self._entry_price = float(entry_price)
                atr_pct_entry = float(atr_pct_now)
                if held_dir > 0:
                    self._stop_price = float(
                        float(entry_price) * (1.0 - float(self.stop_atr_mult) * atr_pct_entry)
                    )
                    self._best_close = float(last_close)
                else:
                    self._stop_price = float(
                        float(entry_price) * (1.0 + float(self.stop_atr_mult) * atr_pct_entry)
                    )
                    self._best_close = float(last_close)

            if self._best_close is None:
                self._best_close = float(last_close)
            if held_dir > 0:
                self._best_close = float(max(float(self._best_close), float(last_close)))
            else:
                self._best_close = float(min(float(self._best_close), float(last_close)))

            trail = None
            if atr_pct_now > 0:
                if held_dir > 0:
                    trail = float(
                        float(self._best_close)
                        * (1.0 - float(self.trail_atr_mult) * float(atr_pct_now))
                    )
                else:
                    trail = float(
                        float(self._best_close)
                        * (1.0 + float(self.trail_atr_mult) * float(atr_pct_now))
                    )

            debug.update(
                {
                    "entry_price": float(self._entry_price or 0.0),
                    "stop_price": float(self._stop_price or 0.0),
                    "best_close": float(self._best_close or 0.0),
                    "trail_price": float(trail or 0.0),
                    "atr_pct": float(atr_pct_now),
                }
            )

            if self._stop_price is not None:
                if held_dir > 0 and last_close <= float(self._stop_price):
                    self._done_today = True
                    self._clear_position_state()
                    return StrategyDecision(
                        target_exposures=targets, reason="stop_loss", debug=debug
                    )
                if held_dir < 0 and last_close >= float(self._stop_price):
                    self._done_today = True
                    self._clear_position_state()
                    return StrategyDecision(
                        target_exposures=targets, reason="stop_loss", debug=debug
                    )

            if (
                trail is not None
                and int(holding_bars) >= int(self.min_hold_bars)
                and held_dir > 0
                and last_close <= float(trail)
            ):
                self._done_today = True
                self._clear_position_state()
                return StrategyDecision(
                    target_exposures=targets, reason="trail_stop", debug=debug
                )
            if (
                trail is not None
                and int(holding_bars) >= int(self.min_hold_bars)
                and held_dir < 0
                and last_close >= float(trail)
            ):
                self._done_today = True
                self._clear_position_state()
                return StrategyDecision(
                    target_exposures=targets, reason="trail_stop", debug=debug
                )

            if int(confirmed_dir) == int(held_dir):
                self._flip_count = 0
            elif int(confirmed_dir) == 0 or int(confirmed_dir) == -int(held_dir):
                self._flip_count += 1

            debug["flip_count"] = int(self._flip_count)
            if int(self._flip_count) >= int(self.flip_confirm_bars):
                self._done_today = True
                self._clear_position_state()
                return StrategyDecision(
                    target_exposures=targets, reason="signal_deterioration", debug=debug
                )

            targets[held_symbol] = float(held_dir)
            return StrategyDecision(
                target_exposures=targets, reason="hold", debug=debug
            )

        if self._done_today or self._reserved_today:
            return StrategyDecision(
                target_exposures=targets, reason="one_and_done", debug=debug
            )

        budget_remaining = int(self.max_day_trades_per_rolling_5_days) - int(self._used_in_window)
        if budget_remaining <= 0:
            return StrategyDecision(
                target_exposures=targets, reason="pdt_locked", debug=debug
            )

        if decision_ts.time() > time(15, 30):
            return StrategyDecision(
                target_exposures=targets, reason="entry_time_gate", debug=debug
            )

        if int(confirmed_dir) == 0:
            return StrategyDecision(
                target_exposures=targets, reason="no_confirmation", debug=debug
            )

        if int(confirmed_dir) < 0 and not state.allow_short:
            return StrategyDecision(
                target_exposures=targets, reason="long_only_abstain", debug=debug
            )

        cost_rt = float(self._cost_roundtrip_bps())
        edge_spy = abs(float(self._last_features[spy]["m"])) * float(self.H) * 10_000.0
        edge_qqq = abs(float(self._last_features[qqq]["m"])) * float(self.H) * 10_000.0
        net_spy = float(edge_spy) - float(cost_rt)
        net_qqq = float(edge_qqq) - float(cost_rt)

        chosen = spy if float(net_spy) >= float(net_qqq) else qqq
        chosen_net = float(net_spy) if chosen == spy else float(net_qqq)

        thr = float(self.base_thr_bps) + float(self._used_in_window) * float(self.budget_step_bps)

        debug.update(
            {
                "cost_roundtrip_bps": float(cost_rt),
                "edge_bps": {spy: float(edge_spy), qqq: float(edge_qqq)},
                "net_edge_bps": {spy: float(net_spy), qqq: float(net_qqq)},
                "thr_bps": float(thr),
                "chosen": chosen,
                "chosen_net_edge_bps": float(chosen_net),
            }
        )

        if float(chosen_net) < float(thr):
            return StrategyDecision(
                target_exposures=targets, reason="below_threshold", debug=debug
            )

        self._reserve_day_trade(decision_ts.date())

        targets[chosen] = float(confirmed_dir)
        return StrategyDecision(target_exposures=targets, reason="enter", debug=debug)
