#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import json
import random
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from atlas.backtest.derivatives_engine import run_derivatives_backtest
from atlas.backtest.engine import BacktestConfig
from atlas.backtest.window_analysis import rolling_window_summary
from atlas.data.bars import parse_bar_timeframe
from atlas.data.universe import load_universe_bars
from atlas.ml.tune import validate_params
from atlas.strategies.registry import build_strategy


@dataclass(frozen=True)
class WindowSpec:
    year: int
    start: datetime
    end: datetime
    length_days: int


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Random local search for perp_weekly_profit_chase over fixed windows. "
            "Optimizes for profitable-window frequency plus weekly hit-rate."
        )
    )
    p.add_argument("--base-params", required=True, help="Path to strategy params JSON")
    p.add_argument("--windows-json", required=True, help="Path to windows.json")
    p.add_argument(
        "--window-indices",
        default="",
        help="Comma-separated 1-based indices from windows.json. Empty = all windows.",
    )
    p.add_argument("--seed", type=int, default=7, help="RNG seed")
    p.add_argument("--trials", type=int, default=20, help="Mutation trials")
    p.add_argument("--keep-top", type=int, default=5, help="Top candidates to persist")
    p.add_argument("--label", default="search", help="Label prefix in outputs")
    p.add_argument(
        "--out-dir",
        default="outputs/evaluations/perp_weekly_profit_chase_search",
        help="Output directory",
    )
    p.add_argument("--symbol", default="BTC/USD")
    p.add_argument("--market", default="derivatives")
    p.add_argument("--data-source", default="coinbase")
    p.add_argument("--bar-timeframe", default="15Min")
    p.add_argument("--prewarm-days", type=int, default=90)
    p.add_argument("--initial-cash", type=float, default=500.0)
    p.add_argument("--max-notional", type=float, default=2500.0)
    p.add_argument("--slippage-bps", type=float, default=1.5)
    p.add_argument("--taker-fee-bps", type=float, default=6.0)
    p.add_argument("--allow-short", action="store_true", default=True)
    return p.parse_args()


def _parse_windows(path: Path, selected_indices: set[int]) -> list[WindowSpec]:
    raw = json.loads(path.read_text())
    out: list[WindowSpec] = []
    for i, item in enumerate(raw, start=1):
        if selected_indices and i not in selected_indices:
            continue
        start = datetime.fromisoformat(str(item["start"]))
        end = datetime.fromisoformat(str(item["end"]))
        if start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)
        if end.tzinfo is None:
            end = end.replace(tzinfo=timezone.utc)
        out.append(
            WindowSpec(
                year=int(item["year"]),
                start=start,
                end=end,
                length_days=int(item.get("length_days", (end - start).days)),
            )
        )
    if not out:
        raise ValueError("No windows selected")
    return out


def _load_strategy_params(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if "perp_weekly_profit_chase" not in payload:
        raise ValueError("Expected key perp_weekly_profit_chase in base params")
    params = dict(payload["perp_weekly_profit_chase"])
    return params


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _mutate(base: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    p = copy.deepcopy(base)
    p.setdefault("daily_loss_hard_stop", 0.0)
    p.setdefault("weekly_loss_hard_stop", 0.0)
    p.setdefault("cooldown_bars_after_exit", 0)
    p.setdefault("trailing_stop_atr_mult", 0.0)
    p.setdefault("break_even_trigger_atr", 0.0)
    p.setdefault("max_hold_bars", 0)

    # Continuous knobs.
    if rng.random() < 0.7:
        p["breakout_buffer_bps"] = round(_clamp(float(p["breakout_buffer_bps"]) + rng.uniform(-2.5, 2.5), 1.0, 12.0), 3)
    if rng.random() < 0.7:
        p["min_atr_bps"] = round(_clamp(float(p["min_atr_bps"]) + rng.uniform(-1.5, 2.0), 1.0, 10.0), 3)
    if rng.random() < 0.7:
        p["base_leverage"] = round(_clamp(float(p["base_leverage"]) + rng.uniform(-0.8, 0.8), 1.2, 5.0), 3)
    if rng.random() < 0.7:
        p["max_leverage"] = round(_clamp(float(p["max_leverage"]) + rng.uniform(-1.0, 1.2), 1.5, 8.0), 3)
    if rng.random() < 0.7:
        p["max_margin_utilization"] = round(_clamp(float(p["max_margin_utilization"]) + rng.uniform(-0.12, 0.12), 0.15, 0.8), 4)
    if rng.random() < 0.7:
        p["stop_atr_mult"] = round(_clamp(float(p["stop_atr_mult"]) + rng.uniform(-0.45, 0.45), 0.7, 3.0), 3)
    if rng.random() < 0.7:
        p["min_liq_buffer_atr"] = round(_clamp(float(p["min_liq_buffer_atr"]) + rng.uniform(-1.0, 1.0), 2.0, 9.0), 3)
    if rng.random() < 0.7:
        p["risk_per_trade"] = round(_clamp(float(p["risk_per_trade"]) + rng.uniform(-0.003, 0.004), 0.003, 0.03), 5)
    if rng.random() < 0.7:
        p["weekly_heartbeat_exposure"] = round(_clamp(float(p["weekly_heartbeat_exposure"]) + rng.uniform(-0.002, 0.003), 0.0, 0.02), 6)
    if rng.random() < 0.7:
        p["weekly_nudge_exposure"] = round(_clamp(float(p["weekly_nudge_exposure"]) + rng.uniform(-0.002, 0.003), 0.0, 0.02), 6)
    if rng.random() < 0.7:
        p["weekly_profit_target"] = round(_clamp(float(p["weekly_profit_target"]) + rng.uniform(-0.0025, 0.003), 0.002, 0.02), 6)
    if rng.random() < 0.7:
        p["weekly_chase_k"] = round(_clamp(float(p["weekly_chase_k"]) + rng.uniform(-0.2, 0.25), 0.0, 1.2), 6)
    if rng.random() < 0.7:
        p["daily_loss_hard_stop"] = round(
            _clamp(float(p.get("daily_loss_hard_stop", 0.0)) + rng.uniform(-0.004, 0.006), 0.0, 0.05), 6
        )
    if rng.random() < 0.7:
        p["weekly_loss_hard_stop"] = round(
            _clamp(float(p.get("weekly_loss_hard_stop", 0.0)) + rng.uniform(-0.008, 0.012), 0.0, 0.12), 6
        )
    if rng.random() < 0.7:
        p["trailing_stop_atr_mult"] = round(
            _clamp(float(p.get("trailing_stop_atr_mult", 0.0)) + rng.uniform(-0.35, 0.40), 0.0, 4.0), 6
        )
    if rng.random() < 0.7:
        p["break_even_trigger_atr"] = round(
            _clamp(float(p.get("break_even_trigger_atr", 0.0)) + rng.uniform(-0.30, 0.35), 0.0, 3.0), 6
        )
    if rng.random() < 0.35:
        p["daily_loss_hard_stop"] = float(rng.choice([0.0, 0.008, 0.010, 0.012, 0.015, 0.020, 0.025]))
    if rng.random() < 0.35:
        p["weekly_loss_hard_stop"] = float(rng.choice([0.0, 0.025, 0.035, 0.050, 0.070, 0.090]))
    if rng.random() < 0.35:
        p["trailing_stop_atr_mult"] = float(rng.choice([0.0, 0.8, 1.0, 1.2, 1.6, 2.0, 2.5]))
    if rng.random() < 0.35:
        p["break_even_trigger_atr"] = float(rng.choice([0.0, 0.5, 0.8, 1.0, 1.4, 2.0]))

    # Discrete knobs.
    if rng.random() < 0.45:
        p["opening_range_minutes"] = int(rng.choice([15, 30, 45, 60, 90]))
    if rng.random() < 0.45:
        p["atr_window"] = int(rng.choice([10, 12, 14, 16, 20, 24]))
    if rng.random() < 0.45:
        p["max_flips_per_day"] = int(rng.choice([1, 2, 3]))
    if rng.random() < 0.35:
        p["weekly_heartbeat_hold_bars"] = int(rng.choice([1, 2, 3]))
    if rng.random() < 0.4:
        p["lookback_short_days"] = round(float(rng.choice([0.5, 0.75, 1.0, 1.5, 2.0])), 3)
    if rng.random() < 0.4:
        p["lookback_long_days"] = round(float(rng.choice([3.0, 5.0, 7.0, 10.0, 14.0])), 3)
    if rng.random() < 0.45:
        p["cooldown_bars_after_exit"] = int(rng.choice([0, 4, 8, 12, 16, 24, 32, 48]))
    if rng.random() < 0.45:
        p["max_hold_bars"] = int(rng.choice([0, 24, 32, 48, 64, 96, 144]))

    # Consistency constraints.
    p["base_leverage"] = float(min(float(p["base_leverage"]), float(p["max_leverage"])))
    p["lookback_long_days"] = float(max(float(p["lookback_long_days"]), float(p["lookback_short_days"]) + 0.5))

    return p


def _score_candidate(
    *,
    candidate_rows: list[dict[str, Any]],
    min_weekly_gate: float = 0.70,
) -> float:
    if not candidate_rows:
        return float("-inf")

    runs = len(candidate_rows)
    profitable_runs = sum(1 for r in candidate_rows if float(r["total_return"]) > 0.0)
    weekly_gate_runs = sum(
        1 for r in candidate_rows if float(r["weekly_positive_frac"]) >= min_weekly_gate
    )

    profitable_frac = profitable_runs / runs
    weekly_gate_frac = weekly_gate_runs / runs
    mean_return = sum(float(r["total_return"]) for r in candidate_rows) / runs
    median_return = sorted(float(r["total_return"]) for r in candidate_rows)[runs // 2]
    worst_return = min(float(r["total_return"]) for r in candidate_rows)
    worst_drawdown = min(float(r["max_drawdown"]) for r in candidate_rows)
    weekly_agg_frac = (
        sum(float(r["weeks_positive"]) for r in candidate_rows)
        / max(1.0, sum(float(r["weeks_total"]) for r in candidate_rows))
    )

    # Heavy emphasis on net-profitable windows, then weekly hit-rate and downside control.
    score = 0.0
    score += 5.0 * profitable_frac
    score += 2.0 * weekly_gate_frac
    score += 1.0 * weekly_agg_frac
    score += 1.5 * mean_return
    score += 0.8 * median_return
    score -= 3.0 * max(0.0, -0.30 - worst_drawdown)  # penalize DD worse than -30%
    score -= 2.2 * max(0.0, -0.35 - worst_return)    # penalize deep tail losses
    return score


def main() -> int:
    args = _parse_args()
    rng = random.Random(int(args.seed))

    out_dir = Path(args.out_dir) / f"{args.label}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = out_dir / "candidate_runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    cands_dir = out_dir / "candidates"
    cands_dir.mkdir(parents=True, exist_ok=True)

    idxs: set[int] = set()
    if str(args.window_indices).strip():
        idxs = {int(x.strip()) for x in str(args.window_indices).split(",") if x.strip()}

    windows = _parse_windows(Path(args.windows_json), idxs)
    base = _load_strategy_params(Path(args.base_params))

    (out_dir / "search_config.json").write_text(
        json.dumps(
            {
                "base_params": str(args.base_params),
                "windows_json": str(args.windows_json),
                "window_indices": sorted(idxs),
                "seed": int(args.seed),
                "trials": int(args.trials),
                "symbol": str(args.symbol),
                "market": str(args.market),
                "data_source": str(args.data_source),
                "bar_timeframe": str(args.bar_timeframe),
                "prewarm_days": int(args.prewarm_days),
                "initial_cash": float(args.initial_cash),
                "max_notional": float(args.max_notional),
                "slippage_bps": float(args.slippage_bps),
                "taker_fee_bps": float(args.taker_fee_bps),
                "allow_short": bool(args.allow_short),
                "windows": [
                    {
                        "year": int(w.year),
                        "start": w.start.isoformat(),
                        "end": w.end.isoformat(),
                        "length_days": int(w.length_days),
                    }
                    for w in windows
                ],
            },
            indent=2,
        )
    )

    # Load bars once per selected window (cached by data layer).
    bars_per_window: dict[int, Any] = {}
    tf = parse_bar_timeframe(str(args.bar_timeframe))
    for i, w in enumerate(windows, start=1):
        load_start = w.start - timedelta(days=int(args.prewarm_days))
        universe = load_universe_bars(
            symbols=[str(args.symbol)],
            data_source=str(args.data_source),
            timeframe=tf,
            start=load_start,
            end=w.end,
            market=str(args.market),
            regular_hours_only=False,
        )
        bars_per_window[i] = universe.bars_by_symbol[str(args.symbol)].copy()

    cfg = BacktestConfig(
        symbols=[str(args.symbol)],
        initial_cash=float(args.initial_cash),
        max_position_notional_usd=float(args.max_notional),
        slippage_bps=float(args.slippage_bps),
        taker_fee_bps=float(args.taker_fee_bps),
        allow_short=bool(args.allow_short),
        maintenance_margin_rate=0.05,
        liquidation_fee_rate=0.005,
    )

    all_rows: list[dict[str, Any]] = []
    candidate_summaries: list[dict[str, Any]] = []

    # Include base candidate as trial 0.
    trials_total = int(args.trials) + 1
    for t in range(trials_total):
        if t == 0:
            params = copy.deepcopy(base)
            origin = "base"
        else:
            params = _mutate(base, rng)
            origin = "mutated"

        if not validate_params("perp_weekly_profit_chase", params):
            continue

        candidate_id = f"cand_{t:03d}"
        rows: list[dict[str, Any]] = []
        for wi, w in enumerate(windows, start=1):
            run_dir = runs_dir / candidate_id / f"w{wi:02d}_{w.year}_{w.start.date()}_{w.end.date()}"
            run_dir.mkdir(parents=True, exist_ok=True)
            strat = build_strategy(
                name="perp_weekly_profit_chase",
                params_path=None,
                symbols=[str(args.symbol)],
                fast_window=10,
                slow_window=30,
                params=params,
            )
            run_derivatives_backtest(
                bars_by_symbol={str(args.symbol): bars_per_window[wi]},
                strategy=strat,
                cfg=cfg,
                run_dir=run_dir,
                debug=False,
                score_start=w.start,
                score_end=w.end,
                no_trade_before=w.start,
            )

            metrics = json.loads((run_dir / "metrics.json").read_text())
            summary, window_rows = rolling_window_summary(
                run_dir=run_dir,
                window=timedelta(days=7),
                step=timedelta(days=7),
                benchmark="spy.us",
            )
            weeks_positive = sum(1 for r in window_rows if float(r.get("return", 0.0)) > 0.0)
            weeks_total = int(summary.windows)

            row = {
                "candidate_id": candidate_id,
                "origin": origin,
                "trial_index": t,
                "window_index": wi,
                "window_year": int(w.year),
                "window_start": w.start.isoformat(),
                "window_end": w.end.isoformat(),
                "total_return": float(metrics.get("total_return", 0.0)),
                "max_drawdown": float(metrics.get("max_drawdown", 0.0)),
                "trades": int(metrics.get("trade_count", metrics.get("trades", 0) or 0)),
                "weeks_total": int(weeks_total),
                "weeks_positive": int(weeks_positive),
                "weekly_positive_frac": float(weeks_positive / max(1, weeks_total)),
                "beat_spy_weekly_frac": float(summary.beat_benchmark_frac or 0.0),
                "mean_weekly_return": float(summary.mean_return),
                "run_dir": str(run_dir),
            }
            rows.append(row)
            all_rows.append(row)

        score = _score_candidate(candidate_rows=rows)
        runs = len(rows)
        profitable_runs = sum(1 for r in rows if float(r["total_return"]) > 0.0)
        weekly_gate_runs = sum(1 for r in rows if float(r["weekly_positive_frac"]) >= 0.70)
        summary_row = {
            "candidate_id": candidate_id,
            "origin": origin,
            "trial_index": int(t),
            "score": float(score),
            "runs": int(runs),
            "profitable_runs": int(profitable_runs),
            "profitable_run_frac": float(profitable_runs / max(1, runs)),
            "weekly_gate_runs": int(weekly_gate_runs),
            "weekly_gate_run_frac": float(weekly_gate_runs / max(1, runs)),
            "mean_total_return": float(sum(float(r["total_return"]) for r in rows) / max(1, runs)),
            "median_total_return": float(sorted(float(r["total_return"]) for r in rows)[runs // 2]),
            "worst_max_drawdown": float(min(float(r["max_drawdown"]) for r in rows)),
            "weekly_positive_frac_agg": float(
                sum(float(r["weeks_positive"]) for r in rows)
                / max(1.0, sum(float(r["weeks_total"]) for r in rows))
            ),
            "params": copy.deepcopy(params),
        }
        candidate_summaries.append(summary_row)
        print(
            f"[{t+1}/{trials_total}] {candidate_id} "
            f"score={summary_row['score']:.4f} "
            f"prof_runs={summary_row['profitable_runs']}/{summary_row['runs']} "
            f"weekly_gate={summary_row['weekly_gate_runs']}/{summary_row['runs']} "
            f"mean_ret={summary_row['mean_total_return']:.4%} "
            f"worst_dd={summary_row['worst_max_drawdown']:.2%}"
        )

    candidate_summaries.sort(key=lambda r: float(r["score"]), reverse=True)

    # Persist detail rows.
    rows_csv = out_dir / "window_rows.csv"
    if all_rows:
        with rows_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)

    # Persist leaderboard and top params.
    leaderboard = out_dir / "leaderboard.json"
    leaderboard.write_text(json.dumps(candidate_summaries, indent=2))

    keep_top = int(max(1, args.keep_top))
    top = candidate_summaries[:keep_top]
    top_rows = []
    for rank, row in enumerate(top, start=1):
        params_payload = {"perp_weekly_profit_chase": row["params"]}
        cand_name = (
            f"{args.label}_seed{int(args.seed)}_rank{rank:02d}_"
            f"prof{int(row['profitable_runs'])}of{int(row['runs'])}_"
            f"wgate{int(row['weekly_gate_runs'])}of{int(row['runs'])}"
        )
        path = cands_dir / f"{cand_name}.json"
        path.write_text(json.dumps(params_payload, indent=2))
        top_rows.append(
            {
                "rank": rank,
                "candidate_id": row["candidate_id"],
                "origin": row["origin"],
                "score": row["score"],
                "profitable_runs": row["profitable_runs"],
                "runs": row["runs"],
                "weekly_gate_runs": row["weekly_gate_runs"],
                "mean_total_return": row["mean_total_return"],
                "worst_max_drawdown": row["worst_max_drawdown"],
                "params_file": str(path),
            }
        )

    top_csv = out_dir / "top_candidates.csv"
    if top_rows:
        with top_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(top_rows[0].keys()))
            writer.writeheader()
            writer.writerows(top_rows)

    result = {
        "out_dir": str(out_dir),
        "leaderboard_json": str(leaderboard),
        "window_rows_csv": str(rows_csv),
        "top_candidates_csv": str(top_csv),
        "top_count": len(top_rows),
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
