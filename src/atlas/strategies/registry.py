from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import json

from atlas.strategies.base import Strategy
from atlas.strategies.ma_crossover import MovingAverageCrossover
from atlas.strategies.nec_x import NecX


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
) -> Strategy:
    params = _load_params(params_path)

    if name == "ma_crossover":
        fast = int(params.get("fast_window", fast_window))
        slow = int(params.get("slow_window", slow_window))
        symbol = str(params.get("symbol") or (symbols[0] if symbols else "SPY"))
        return MovingAverageCrossover(fast_window=fast, slow_window=slow, symbol=symbol)

    if name in {"nec_x", "nec-x"}:
        required = {"SPY", "QQQ"}
        if not required.issubset({s.upper() for s in symbols}):
            raise ValueError("nec_x requires --symbols SPY,QQQ")

        def _get_int(key: str, default: int) -> int:
            raw = params.get(key, params.get(key.lower(), default))
            return int(raw)

        def _get_float(key: str, default: float) -> float:
            raw = params.get(key, params.get(key.lower(), default))
            return float(raw)

        return NecX(
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

    raise ValueError(f"unknown strategy: {name}")


def list_strategy_names() -> list[str]:
    return ["ma_crossover", "nec_x"]
