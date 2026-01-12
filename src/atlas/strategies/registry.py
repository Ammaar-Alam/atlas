from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import json

from atlas.strategies.base import Strategy
from atlas.strategies.ma_crossover import MovingAverageCrossover


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
    fast_window: int,
    slow_window: int,
) -> Strategy:
    params = _load_params(params_path)

    if name == "ma_crossover":
        fast = int(params.get("fast_window", fast_window))
        slow = int(params.get("slow_window", slow_window))
        return MovingAverageCrossover(fast_window=fast, slow_window=slow)

    raise ValueError(f"unknown strategy: {name}")

