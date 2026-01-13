from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import json

from atlas.strategies.base import Strategy
from atlas.strategies.ema_crossover import EmaCrossover
from atlas.strategies.ma_crossover import MovingAverageCrossover
from atlas.strategies.no_trade import NoTrade
from atlas.strategies.nec_pdt import NecPDT
from atlas.strategies.nec_x import NecX
from atlas.strategies.orb_trend import OrbTrend
from atlas.strategies.orb_trend import OrbTrend
from atlas.strategies.spy_open_close import SpyOpenClose
from atlas.strategies.perp_flare import PerpFlare


@dataclass(frozen=True)
class StrategyBuild:
    name: str
    params: Dict[str, Any]


def _load_params(path: Optional[Path]) -> Dict[str, Any]:
    if path is None:
        return {}
    raw = json.loads(path.read_text())
    if isinstance(raw, dict) and "params" in raw and isinstance(raw["params"], dict):
        return raw["params"]
    if isinstance(raw, dict):
        return raw
    raise ValueError("strategy params json must be an object")


def build_strategy(
    *,
    name: str,
    params_path: Optional[Path],
    symbols: list[str],
    fast_window: int,
    slow_window: int,
    params: Optional[Dict[str, Any]] = None,
) -> Strategy:
    params = params if params is not None else _load_params(params_path)
    if isinstance(params, dict):
        if "params" in params and isinstance(params["params"], dict):
            params = params["params"]
        elif "parameters" in params and isinstance(params["parameters"], dict):
            params = params["parameters"]
        canonical = name.replace("-", "_")
        if canonical in params and isinstance(params[canonical], dict):
            params = params[canonical]
        elif name in params and isinstance(params[name], dict):
            params = params[name]

    if name == "ma_crossover":
        fast = int(params.get("fast_window", fast_window))
        slow = int(params.get("slow_window", slow_window))
        symbol = str(params.get("symbol") or (symbols[0] if symbols else "SPY"))
        return MovingAverageCrossover(fast_window=fast, slow_window=slow, symbol=symbol)

    if name in {"ema_crossover", "ema-crossover"}:
        fast = int(params.get("fast_window", fast_window))
        slow = int(params.get("slow_window", slow_window))
        symbol = str(params.get("symbol") or (symbols[0] if symbols else "SPY"))
        return EmaCrossover(fast_window=fast, slow_window=slow, symbol=symbol)

    if name in {"spy_open_close", "spy-open-close"}:
        symbol = str(params.get("symbol") or (symbols[0] if symbols else "SPY"))
        return SpyOpenClose(symbol=symbol)

    if name in {"no_trade", "no-trade"}:
        return NoTrade()

    if name in {"nec_x", "nec-x"}:
        universe_symbols = [s.strip().upper() for s in symbols if s.strip()]
        if len(universe_symbols) < 2:
            raise ValueError(
                f"nec_x requires at least 2 symbols (got {len(universe_symbols)})"
            )
        spy_symbol, qqq_symbol = universe_symbols[0], universe_symbols[1]

        def _get_int(key: str, default: int) -> int:
            raw = params.get(key, params.get(key.lower(), default))
            return int(raw)

        def _get_float(key: str, default: float) -> float:
            raw = params.get(key, params.get(key.lower(), default))
            return float(raw)

        return NecX(
            spy=spy_symbol,
            qqq=qqq_symbol,
            M=_get_int("M", 6),
            V=_get_int("V", 12),
            Wcorr=_get_int("Wcorr", 12),
            rho_min=_get_float("rho_min", 0.60),
            strength_entry=_get_float("strength_entry", 0.80),
            strength_exit=_get_float("strength_exit", 0.20),
            H_max=_get_int("H_max", 6),
            k_cost=_get_float("k_cost", 1.25),
            spread_floor_bps=_get_float("spread_floor_bps", 0.50),
            slip_bps=_get_float("slip_bps", 0.75),
            daily_loss_limit=_get_float("daily_loss_limit", 0.010),
            kill_switch=_get_float("kill_switch", 0.025),
            tick_size=_get_float("tick_size", 0.01),
        )

    if name in {"nec_pdt", "nec-pdt"}:
        universe_symbols = [s.strip().upper() for s in symbols if s.strip()]
        if len(universe_symbols) < 2:
            raise ValueError(
                f"nec_pdt requires at least 2 symbols (got {len(universe_symbols)})"
            )
        spy_symbol, qqq_symbol = universe_symbols[0], universe_symbols[1]

        def _get_int(key: str, default: int) -> int:
            raw = params.get(key, params.get(key.lower(), default))
            return int(raw)

        def _get_float(key: str, default: float) -> float:
            raw = params.get(key, params.get(key.lower(), default))
            return float(raw)

        return NecPDT(
            spy=spy_symbol,
            qqq=qqq_symbol,
            M=_get_int("M", 6),
            V=_get_int("V", 12),
            eps=_get_float("eps", 1e-8),
            H=_get_int("H", 12),
            base_thr_bps=_get_float("base_thr_bps", 10.0),
            budget_step_bps=_get_float("budget_step_bps", 4.0),
            atr_lookback_bars=_get_int("atr_lookback_bars", 12),
            stop_atr_mult=_get_float("stop_atr_mult", 2.0),
            trail_atr_mult=_get_float("trail_atr_mult", 2.5),
            min_hold_bars=_get_int("min_hold_bars", 4),
            flip_confirm_bars=_get_int("flip_confirm_bars", 3),
            max_day_trades_per_rolling_5_days=_get_int(
                "max_day_trades_per_rolling_5_days", 3
            ),
            half_spread_bps=_get_float("half_spread_bps", 1.5),
            slippage_bps=_get_float("slippage_bps", 2.0),
            fee_bps=float(
                params.get("fee_bps", params.get("fees_bps", _get_float("fee_bps", 0.3)))
            ),
        )

    if name in {"orb_trend", "orb-trend"}:
        universe_symbols = [s.strip().upper() for s in symbols if s.strip()]
        if not universe_symbols:
            raise ValueError("orb_trend requires at least 1 symbol")

        def _get_int(key: str, default: int) -> int:
            raw = params.get(key, params.get(key.lower(), default))
            return int(raw)

        def _get_float(key: str, default: float) -> float:
            raw = params.get(key, params.get(key.lower(), default))
            return float(raw)

        return OrbTrend(
            symbols=tuple(universe_symbols),
            orb_minutes=_get_int("orb_minutes", 30),
            orb_breakout_bps=_get_float("orb_breakout_bps", 4.0),
            confirm_bars=_get_int("confirm_bars", 2),
            atr_window=_get_int("atr_window", 20),
            er_window=_get_int("er_window", 12),
            er_min=_get_float("er_min", 0.35),
            expected_hold_bars=_get_int("expected_hold_bars", 12),
            k_cost=_get_float("k_cost", 2.0),
            slippage_bps=_get_float("slippage_bps", 1.25),
            min_hold_bars=_get_int("min_hold_bars", 3),
            daily_loss_limit=_get_float("daily_loss_limit", 0.010),
            kill_switch=_get_float("kill_switch", 0.025),
        )

    if name in {"perp_flare", "perp-flare"}:
        universe_symbols = [s.strip().upper() for s in symbols if s.strip()]
        if not universe_symbols:
            raise ValueError("perp_flare requires at least 1 symbol")
            
        def _get_int(key: str, default: int) -> int:
            raw = params.get(key, params.get(key.lower(), default))
            return int(raw)

        def _get_float(key: str, default: float) -> float:
            raw = params.get(key, params.get(key.lower(), default))
            return float(raw)

        return PerpFlare(
            symbols=tuple(universe_symbols),
            atr_window=_get_int("atr_window", 14),
            ema_fast=_get_int("ema_fast", 12),
            ema_slow=_get_int("ema_slow", 24),
            er_window=_get_int("er_window", 10),
            breakout_window=_get_int("breakout_window", 20),
            er_min=_get_float("er_min", 0.35),
            taker_fee_bps=_get_float("taker_fee_bps", 3.0),
            half_spread_bps=_get_float("half_spread_bps", 1.0),
            base_slippage_bps=_get_float("base_slippage_bps", 1.5),
            edge_floor_bps=_get_float("edge_floor_bps", 5.0),
            k_cost=_get_float("k_cost", 1.5),
            risk_per_trade=_get_float("risk_per_trade", 0.01),
            stop_atr_mult=_get_float("stop_atr_mult", 2.0),
            trail_atr_mult=_get_float("trail_atr_mult", 3.0),
            max_margin_utilization=_get_float("max_margin_utilization", 0.65),
            max_leverage=_get_float("max_leverage", 10.0),
            maintenance_margin_rate=_get_float("maintenance_margin_rate", 0.05),
            min_liq_buffer_atr=_get_float("min_liq_buffer_atr", 3.0),
        )

    raise ValueError(f"unknown strategy: {name}")


def list_strategy_names() -> list[str]:
    return [
        "spy_open_close",
        "no_trade",
        "ema_crossover",
        "ma_crossover",
        "nec_x",
        "nec_pdt",
        "nec_pdt",
        "orb_trend",
        "perp_flare",
    ]
