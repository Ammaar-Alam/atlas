from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from datetime import time
from typing import Any, Optional

import numpy as np
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
class NecX(Strategy):
    name: str = "nec_x"

    # universe (v1 fixed)
    spy: str = "SPY"
    qqq: str = "QQQ"

    # v1 knobs (exactly 12)
    M: int = 6
    V: int = 12
    Wcorr: int = 12
    rho_min: float = 0.60
    strength_entry: float = 0.80
    strength_exit: float = 0.20
    H_max: int = 6
    k_cost: float = 1.25
    spread_floor_bps: float = 0.50
    slip_bps: float = 0.75
    daily_loss_limit: float = 0.010
    kill_switch: float = 0.025

    # design constants (not tuned)
    tick_size: float = 0.01
    vol_ratio_min: float = 0.6
    eps: float = 1e-8

    # internal running state (for speed + clean daily resets)
    _last_processed: Optional[pd.Timestamp] = field(default=None, init=False, repr=False)
    _bars_seen: int = field(default=0, init=False, repr=False)
    _risk_disabled_day: Optional[object] = field(default=None, init=False, repr=False)

    _prev_close: dict[str, Optional[float]] = field(default_factory=dict, init=False, repr=False)
    _ema_m: dict[str, Optional[float]] = field(default_factory=dict, init=False, repr=False)
    _ema_v: dict[str, Optional[float]] = field(default_factory=dict, init=False, repr=False)
    _ema_vol_s: dict[str, Optional[float]] = field(default_factory=dict, init=False, repr=False)
    _ema_vol_l: dict[str, Optional[float]] = field(default_factory=dict, init=False, repr=False)
    _vwap_day: dict[str, Optional[object]] = field(default_factory=dict, init=False, repr=False)
    _vwap_num: dict[str, float] = field(default_factory=dict, init=False, repr=False)
    _vwap_den: dict[str, float] = field(default_factory=dict, init=False, repr=False)
    _last_features: dict[str, dict[str, float]] = field(default_factory=dict, init=False, repr=False)

    _rets: dict[str, deque[float]] = field(default_factory=dict, init=False, repr=False)

    def warmup_bars(self) -> int:
        # Minimum to compute Wcorr log-return correlation plus a little stability for EMAs.
        return int(max(self.Wcorr + 1, self.M + 2, self.V + 2, 20))

    def _required_symbols(self) -> tuple[str, str]:
        return (self.spy.upper(), self.qqq.upper())

    def _reset_state(self) -> None:
        self._last_processed = None
        self._bars_seen = 0
        self._prev_close = {}
        self._ema_m = {}
        self._ema_v = {}
        self._ema_vol_s = {}
        self._ema_vol_l = {}
        self._vwap_day = {}
        self._vwap_num = {}
        self._vwap_den = {}
        self._last_features = {}
        self._rets = {}

    def _update_from_bars(self, bars_by_symbol: dict[str, pd.DataFrame]) -> None:
        spy, qqq = self._required_symbols()
        if spy not in bars_by_symbol or qqq not in bars_by_symbol:
            raise ValueError(f"nec_x requires bars for {spy} and {qqq}")

        df_spy = bars_by_symbol[spy]
        df_qqq = bars_by_symbol[qqq]
        common = df_spy.index.intersection(df_qqq.index).sort_values()
        if len(common) < 3:
            return

        if self._last_processed is not None:
            if self._last_processed not in common and self._last_processed < common[0]:
                self._reset_state()

        if self._last_processed is None:
            new_index = common
        else:
            new_index = common[common > self._last_processed]
        if len(new_index) == 0:
            return

        alpha_m = 2.0 / (float(self.M) + 1.0)
        alpha_v = 2.0 / (float(self.V) + 1.0)
        alpha_vs = 2.0 / (4.0 + 1.0)
        alpha_vl = 2.0 / (78.0 + 1.0)

        for ts in new_index:
            for symbol in (spy, qqq):
                df = bars_by_symbol[symbol]
                row = df.loc[ts]
                close = float(row["close"])
                high = float(row["high"])
                low = float(row["low"])
                vol = float(row["volume"])

                prev_close = self._prev_close.get(symbol)
                if prev_close is None or prev_close <= 0 or close <= 0:
                    r = 0.0
                else:
                    r = math.log(close / prev_close)

                m_prev = self._ema_m.get(symbol)
                v_prev = self._ema_v.get(symbol)
                m = r if m_prev is None else (alpha_m * r + (1.0 - alpha_m) * float(m_prev))
                v = abs(r) if v_prev is None else (alpha_v * abs(r) + (1.0 - alpha_v) * float(v_prev))
                score = float(m) / (float(v) + self.eps)

                vs_prev = self._ema_vol_s.get(symbol)
                vl_prev = self._ema_vol_l.get(symbol)
                vs = vol if vs_prev is None else (alpha_vs * vol + (1.0 - alpha_vs) * float(vs_prev))
                vl = vol if vl_prev is None else (alpha_vl * vol + (1.0 - alpha_vl) * float(vl_prev))
                vol_ratio = float(vs) / float(vl) if float(vl) > 0 else 0.0

                d = pd.Timestamp(ts).tz_convert(NY_TZ).date()
                if self._vwap_day.get(symbol) != d:
                    self._vwap_day[symbol] = d
                    self._vwap_num[symbol] = 0.0
                    self._vwap_den[symbol] = 0.0

                tp = (high + low + close) / 3.0
                self._vwap_num[symbol] = float(self._vwap_num.get(symbol, 0.0)) + tp * vol
                self._vwap_den[symbol] = float(self._vwap_den.get(symbol, 0.0)) + vol
                vwap = (
                    float(self._vwap_num[symbol]) / float(self._vwap_den[symbol])
                    if float(self._vwap_den[symbol]) > 0
                    else close
                )
                vwap_dev = (close - vwap) / vwap if vwap != 0 else 0.0

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
                range_frac = float(high - low) / close if close > 0 else 0.0

                dq = self._rets.get(symbol)
                if dq is None:
                    dq = deque(maxlen=int(self.Wcorr))
                    self._rets[symbol] = dq
                dq.append(float(r))

                self._ema_m[symbol] = float(m)
                self._ema_v[symbol] = float(v)
                self._ema_vol_s[symbol] = float(vs)
                self._ema_vol_l[symbol] = float(vl)
                self._prev_close[symbol] = close
                self._last_features[symbol] = {
                    "close": close,
                    "r": float(r),
                    "tr": float(tr),
                    "range_frac": float(range_frac),
                    "vwap": float(vwap),
                    "vwap_dev": float(vwap_dev),
                    "m": float(m),
                    "v": float(v),
                    "score": float(score),
                    "vol_ratio": float(vol_ratio),
                }

            self._last_processed = pd.Timestamp(ts)
            self._bars_seen += 1

    def _ready(self) -> bool:
        spy, qqq = self._required_symbols()
        if self._bars_seen < self.warmup_bars():
            return False
        if spy not in self._last_features or qqq not in self._last_features:
            return False
        if len(self._rets.get(spy, [])) < int(self.Wcorr) or len(self._rets.get(qqq, [])) < int(self.Wcorr):
            return False
        return True

    def _corr(self) -> float:
        spy, qqq = self._required_symbols()
        a = np.array(list(self._rets.get(spy, [])), dtype=float)
        b = np.array(list(self._rets.get(qqq, [])), dtype=float)
        if len(a) < 2 or len(b) < 2:
            return 0.0
        if float(a.std()) == 0.0 or float(b.std()) == 0.0:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])

    def _cost_rt_bps(self, symbol: str) -> float:
        close = float(self._last_features[symbol]["close"])
        tick_bps = 10_000.0 * (float(self.tick_size) / close) if close > 0 else 0.0
        spread_side = max(float(tick_bps), float(self.spread_floor_bps))
        slip_side = float(self.slip_bps)
        fees_rt = 0.0
        return float(2.0 * (spread_side + slip_side) + fees_rt)

    def target_exposures(
        self, bars_by_symbol: dict[str, pd.DataFrame], state: StrategyState
    ) -> StrategyDecision:
        spy, qqq = self._required_symbols()
        decision_ts = pd.Timestamp(state.timestamp)
        if decision_ts.tz is None:
            decision_ts = decision_ts.tz_localize(NY_TZ)
        else:
            decision_ts = decision_ts.tz_convert(NY_TZ)

        if self._risk_disabled_day is not None and decision_ts.date() != self._risk_disabled_day:
            self._risk_disabled_day = None

        self._update_from_bars(bars_by_symbol)

        targets = {spy: 0.0, qqq: 0.0}
        debug: dict[str, Any] = {
            "ts": decision_ts.isoformat(),
            "bars_seen": int(self._bars_seen),
        }

        held_symbols = [s for s in (spy, qqq) if abs(float(state.positions.get(s, 0.0))) > 1e-8]
        held_symbol = held_symbols[0] if len(held_symbols) == 1 else None
        held_qty = float(state.positions.get(held_symbol, 0.0)) if held_symbol else 0.0
        held_dir = _sign(held_qty) if held_symbol else 0

        debug["held_symbol"] = held_symbol
        debug["held_dir"] = held_dir
        debug["holding_bars"] = {s: int(state.holding_bars.get(s, 0)) for s in (spy, qqq)}
        debug["day_return"] = float(state.day_return)

        if decision_ts.time() < time(9, 30) or decision_ts.time() > time(16, 0):
            return StrategyDecision(target_exposures=targets, reason="outside_rth", debug=debug)

        if decision_ts.time() >= time(15, 55):
            return StrategyDecision(target_exposures=targets, reason="forced_flat", debug=debug)

        if float(state.day_return) <= -float(self.kill_switch):
            self._risk_disabled_day = decision_ts.date()
            return StrategyDecision(target_exposures=targets, reason="kill_switch", debug=debug)

        if float(state.day_return) <= -float(self.daily_loss_limit):
            self._risk_disabled_day = decision_ts.date()
            return StrategyDecision(target_exposures=targets, reason="daily_loss_limit", debug=debug)

        if not self._ready():
            return StrategyDecision(target_exposures=targets, reason="warmup", debug=debug)

        spy_score = float(self._last_features[spy]["score"])
        qqq_score = float(self._last_features[qqq]["score"])
        dir_ = _sign(spy_score)
        agree = dir_ != 0 and dir_ == _sign(qqq_score)
        strength = float(min(abs(spy_score), abs(qqq_score)))
        rho = float(self._corr())
        vol_ratio_gate = float(max(self._last_features[spy]["vol_ratio"], self._last_features[qqq]["vol_ratio"]))

        debug.update(
            {
                "spy": self._last_features[spy],
                "qqq": self._last_features[qqq],
                "dir": int(dir_),
                "agree": bool(agree),
                "strength": float(strength),
                "rho": float(rho),
                "vol_ratio_max": float(vol_ratio_gate),
            }
        )

        if not state.allow_short and dir_ < 0:
            if held_symbol is not None:
                return StrategyDecision(target_exposures=targets, reason="long_only_exit", debug=debug)
            return StrategyDecision(target_exposures=targets, reason="long_only_abstain", debug=debug)

        if held_symbol is not None:
            if (not agree) or float(strength) < float(self.strength_exit):
                return StrategyDecision(target_exposures=targets, reason="signal_decay_exit", debug=debug)
            if int(state.holding_bars.get(held_symbol, 0)) >= int(self.H_max):
                return StrategyDecision(target_exposures=targets, reason="max_holding_exit", debug=debug)

            entry_window = time(9, 40) <= decision_ts.time() <= time(15, 30)
            entry_gates_pass = (
                float(rho) >= float(self.rho_min)
                and bool(agree)
                and float(strength) >= float(self.strength_entry)
                and float(vol_ratio_gate) >= float(self.vol_ratio_min)
            )
            if entry_window and entry_gates_pass and dir_ != 0:
                cost_spy = float(self._cost_rt_bps(spy))
                cost_qqq = float(self._cost_rt_bps(qqq))
                exp_spy = abs(float(self._last_features[spy]["m"])) * float(self.H_max) * 10_000.0
                exp_qqq = abs(float(self._last_features[qqq]["m"])) * float(self.H_max) * 10_000.0
                net_spy = float(exp_spy) - float(self.k_cost) * float(cost_spy)
                net_qqq = float(exp_qqq) - float(self.k_cost) * float(cost_qqq)

                debug.update(
                    {
                        "expMove_bps": {spy: float(exp_spy), qqq: float(exp_qqq)},
                        "costRT_bps": {spy: float(cost_spy), qqq: float(cost_qqq)},
                        "netEdge_bps": {spy: float(net_spy), qqq: float(net_qqq)},
                    }
                )

                chosen = spy if net_spy >= net_qqq else qqq
                chosen_net = float(net_spy) if chosen == spy else float(net_qqq)
                if chosen_net > 0.0:
                    desired = {spy: 0.0, qqq: 0.0}
                    desired[chosen] = float(dir_)
                    debug["chosen"] = chosen
                    debug["chosen_netEdge_bps"] = float(chosen_net)

                    if chosen != held_symbol or int(dir_) != int(held_dir):
                        return StrategyDecision(
                            target_exposures=desired, reason="switch", debug=debug
                        )

            targets[held_symbol] = float(held_dir)
            return StrategyDecision(target_exposures=targets, reason="hold", debug=debug)

        if self._risk_disabled_day == decision_ts.date():
            return StrategyDecision(target_exposures=targets, reason="risk_disabled", debug=debug)

        if decision_ts.time() < time(9, 40) or decision_ts.time() > time(15, 30):
            return StrategyDecision(target_exposures=targets, reason="entry_time_gate", debug=debug)

        if float(rho) < float(self.rho_min):
            return StrategyDecision(target_exposures=targets, reason="gate_rho", debug=debug)
        if not agree:
            return StrategyDecision(target_exposures=targets, reason="gate_agree", debug=debug)
        if float(strength) < float(self.strength_entry):
            return StrategyDecision(target_exposures=targets, reason="gate_strength", debug=debug)
        if float(vol_ratio_gate) < float(self.vol_ratio_min):
            return StrategyDecision(target_exposures=targets, reason="gate_liquidity", debug=debug)

        cost_spy = float(self._cost_rt_bps(spy))
        cost_qqq = float(self._cost_rt_bps(qqq))
        exp_spy = abs(float(self._last_features[spy]["m"])) * float(self.H_max) * 10_000.0
        exp_qqq = abs(float(self._last_features[qqq]["m"])) * float(self.H_max) * 10_000.0

        net_spy = float(exp_spy) - float(self.k_cost) * float(cost_spy)
        net_qqq = float(exp_qqq) - float(self.k_cost) * float(cost_qqq)

        debug.update(
            {
                "expMove_bps": {spy: float(exp_spy), qqq: float(exp_qqq)},
                "costRT_bps": {spy: float(cost_spy), qqq: float(cost_qqq)},
                "netEdge_bps": {spy: float(net_spy), qqq: float(net_qqq)},
            }
        )

        chosen = spy if net_spy >= net_qqq else qqq
        chosen_net = float(net_spy) if chosen == spy else float(net_qqq)
        if chosen_net <= 0.0 or dir_ == 0:
            return StrategyDecision(
                target_exposures=targets, reason="net_edge_not_positive", debug=debug
            )

        targets[chosen] = float(dir_)
        debug["chosen"] = chosen
        debug["chosen_netEdge_bps"] = float(chosen_net)
        return StrategyDecision(target_exposures=targets, reason="enter", debug=debug)
