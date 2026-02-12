from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from atlas.backtest.derivatives_engine import run_derivatives_backtest
from atlas.backtest.engine import BacktestConfig, run_backtest
from atlas.market import Market, parse_market
from atlas.ml.tune import ObjectiveConfig, WalkForwardConfig, build_walk_forward_segments, score_run
from atlas.strategies.registry import build_strategy

logger = logging.getLogger(__name__)


def _align_bars(
    *, bars_by_symbol: dict[str, pd.DataFrame], symbols: list[str]
) -> tuple[dict[str, pd.DataFrame], pd.DatetimeIndex]:
    common_index: Optional[pd.DatetimeIndex] = None
    for sym in symbols:
        idx = bars_by_symbol[sym].index
        common_index = idx if common_index is None else common_index.intersection(idx)
    if common_index is None or len(common_index) < 3:
        raise ValueError("backtest window has too few aligned bars")
    common_index = common_index.sort_values()
    aligned = {s: bars_by_symbol[s].loc[common_index] for s in symbols}
    return aligned, common_index


def _slice_bars_with_warmup(
    bars_by_symbol: dict[str, pd.DataFrame],
    common_index: pd.DatetimeIndex,
    *,
    score_start: datetime,
    score_end: datetime,
    warmup_bars: int,
) -> dict[str, pd.DataFrame]:
    start_ts = pd.Timestamp(score_start)
    end_ts = pd.Timestamp(score_end)

    start_pos = int(common_index.searchsorted(start_ts, side="left"))
    end_pos = int(common_index.searchsorted(end_ts, side="left"))
    warm_start_pos = max(0, start_pos - int(max(0, warmup_bars)))

    idx_slice = common_index[warm_start_pos:end_pos]
    return {sym: df.loc[idx_slice] for sym, df in bars_by_symbol.items()}


def _run_backtest_for_market(
    *, market: Market, bars_by_symbol: dict[str, pd.DataFrame], strategy, cfg: BacktestConfig, run_dir: Path
) -> None:
    if market == Market.DERIVATIVES:
        run_derivatives_backtest(
            bars_by_symbol=bars_by_symbol,
            strategy=strategy,
            cfg=cfg,
            run_dir=run_dir,
            output_mode="minimal",
        )
        return
    run_backtest(
        bars_by_symbol=bars_by_symbol,
        strategy=strategy,
        cfg=cfg,
        run_dir=run_dir,
        output_mode="minimal",
    )


@dataclass(frozen=True)
class SegmentEval:
    segment: int
    test_start: str
    test_end: str
    score: float
    rejected: bool
    reject_reason: str
    stats: dict[str, Any]
    breakdown: dict[str, float]
    run_dir: str


@dataclass(frozen=True)
class WalkForwardEvalResult:
    run_dir: str
    market: str
    symbols: list[str]
    strategy: str
    params: dict[str, Any]
    backtest: dict[str, Any]
    walk_forward: dict[str, Any]
    objective: dict[str, Any]
    segments: list[dict[str, Any]]
    tests: list[SegmentEval]
    summary: dict[str, Any]


def walk_forward_evaluate(
    *,
    bars_by_symbol: dict[str, pd.DataFrame],
    market: str,
    symbols: list[str],
    strategy: str,
    params: dict[str, Any],
    backtest_cfg: BacktestConfig,
    walk_forward: WalkForwardConfig,
    objective: Optional[ObjectiveConfig],
    run_dir: Path,
    keep_test_runs: bool = True,
) -> WalkForwardEvalResult:
    market_enum = parse_market(market)
    strategy = (strategy or "").strip().lower().replace("-", "_")
    symbols = [s.strip().upper() for s in symbols if s.strip()]
    if not symbols:
        raise ValueError("symbols must be non-empty")

    aligned_bars, common_index = _align_bars(bars_by_symbol=bars_by_symbol, symbols=symbols)
    start_ts = pd.Timestamp(common_index[0]).to_pydatetime()
    end_ts = pd.Timestamp(common_index[-1]).to_pydatetime()

    segments = build_walk_forward_segments(start=start_ts, end=end_ts, cfg=walk_forward)
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "segments.json").write_text(json.dumps([s.to_dict() for s in segments], indent=2))
    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "market": market_enum.value,
                "symbols": symbols,
                "strategy": strategy,
                "params": params,
                "backtest": asdict(backtest_cfg),
                "walk_forward": asdict(walk_forward),
            },
            indent=2,
            default=str,
        )
    )

    objective = objective or ObjectiveConfig()

    # Build once to determine warmup. Param changes would require re-building; we hold params fixed here.
    strat = build_strategy(
        name=strategy,
        params_path=None,
        symbols=symbols,
        fast_window=10,
        slow_window=30,
        params=params,
    )
    warmup = int(strat.warmup_bars())

    tests: list[SegmentEval] = []

    for seg_i, seg in enumerate(segments):
        seg_dir = run_dir / f"segment_{seg_i:03d}"
        test_dir = seg_dir / "test"
        test_dir.mkdir(parents=True, exist_ok=True)

        test_bars = _slice_bars_with_warmup(
            aligned_bars,
            common_index,
            score_start=seg.test.start,
            score_end=seg.test.end,
            warmup_bars=warmup,
        )
        _run_backtest_for_market(
            market=market_enum,
            bars_by_symbol=test_bars,
            strategy=build_strategy(
                name=strategy,
                params_path=None,
                symbols=symbols,
                fast_window=10,
                slow_window=30,
                params=params,
            ),
            cfg=backtest_cfg,
            run_dir=test_dir,
        )

        scored = score_run(
            test_dir,
            objective=objective,
            score_start=seg.test.start,
            score_end=seg.test.end,
        )

        tests.append(
            SegmentEval(
                segment=int(seg_i),
                test_start=pd.Timestamp(seg.test.start).isoformat(),
                test_end=pd.Timestamp(seg.test.end).isoformat(),
                score=float(scored.score),
                rejected=bool(scored.rejected),
                reject_reason=str(scored.reason or ""),
                stats=asdict(scored.stats),
                breakdown=dict(scored.breakdown),
                run_dir=str(test_dir),
            )
        )

        if not keep_test_runs:
            # Keep directory structure stable but delete heavy CSVs to save space.
            for p in (test_dir / "equity_curve.csv", test_dir / "trades.csv"):
                try:
                    if p.exists():
                        p.unlink()
                except Exception:
                    pass

    returns = [float(t.stats.get("total_return", 0.0) or 0.0) for t in tests]
    accepted = [t for t in tests if not bool(t.rejected)]
    accepted_returns = [float(t.stats.get("total_return", 0.0) or 0.0) for t in accepted]
    summary = {
        "segments": int(len(tests)),
        "accepted": int(len(accepted)),
        "rejected": int(len(tests) - len(accepted)),
        "mean_return": float(np.mean(returns)) if returns else 0.0,
        "median_return": float(np.median(returns)) if returns else 0.0,
        "mean_return_accepted": float(np.mean(accepted_returns)) if accepted_returns else 0.0,
        "median_return_accepted": float(np.median(accepted_returns)) if accepted_returns else 0.0,
        "positive_segment_frac": float(sum(1 for r in returns if r > 0) / len(returns)) if returns else 0.0,
        "positive_segment_frac_accepted": float(sum(1 for r in accepted_returns if r > 0) / len(accepted_returns))
        if accepted_returns
        else 0.0,
    }

    objective_dict = asdict(objective)

    result = WalkForwardEvalResult(
        run_dir=str(run_dir),
        market=market_enum.value,
        symbols=symbols,
        strategy=strategy,
        params=dict(params),
        backtest=asdict(backtest_cfg),
        walk_forward=asdict(walk_forward),
        objective=objective_dict,
        segments=[s.to_dict() for s in segments],
        tests=tests,
        summary=summary,
    )
    (run_dir / "walk_forward_eval.json").write_text(
        json.dumps(asdict(result), indent=2, default=str)
    )
    logger.info("walk-forward eval done: %s", run_dir)
    return result
