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
from atlas.data.benchmarks import spy_total_return
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
            "Strict-gate random search for perp_weekly_profit_chase. "
            "Objective emphasizes: profitable runs that beat SPY + weekly-positive gate "
            "with SPY outperformance on profitable weeks."
        )
    )
    p.add_argument("--base-params", required=True, help="Path to strategy params JSON")
    p.add_argument("--windows-json", required=True, help="Path to windows.json")
    p.add_argument(
        "--window-indices",
        default="",
        help="Comma-separated 1-based indices from windows.json. Empty = all windows.",
    )
    p.add_argument("--seed", type=int, default=11, help="RNG seed")
    p.add_argument("--trials", type=int, default=10, help="Mutation trials")
    p.add_argument("--keep-top", type=int, default=5, help="Top candidates to persist")
    p.add_argument("--label", default="strict_gates", help="Label prefix in outputs")
    p.add_argument(
        "--out-dir",
        default="outputs/evaluations/perp_weekly_profit_chase_strict_gate_search",
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
    p.add_argument("--taker-fee-bps", type=float, default=12.5)
    p.add_argument(
        "--coinbase-fee-model",
        dest="coinbase_fee_model",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply Coinbase fixed per-contract fee model (only active for derivatives+coinbase).",
    )
    p.add_argument(
        "--fixed-fee-per-contract-usd",
        type=float,
        default=0.15,
        help="Fixed fee in USD per contract per side when coinbase fee model is active.",
    )
    p.add_argument(
        "--contract-size-units",
        type=float,
        default=0.01,
        help="Contract size in underlying units (e.g. BTC nano perp = 0.01 BTC).",
    )
    p.add_argument(
        "--allow-short",
        dest="allow_short",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable short exposure (default: true). Use --no-allow-short for long-only.",
    )
    p.add_argument(
        "--weekly-positive-gate",
        type=float,
        default=0.70,
        help="Weekly positive fraction gate per run.",
    )
    p.add_argument(
        "--weekly-beat-spy-gate",
        type=float,
        default=0.70,
        help="Within profitable weeks, min fraction that must beat SPY weekly return.",
    )
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
    if "perp_weekly_profit_chase" in payload and isinstance(payload["perp_weekly_profit_chase"], dict):
        return dict(payload["perp_weekly_profit_chase"])
    if isinstance(payload, dict):
        return dict(payload)
    raise ValueError(f"Unsupported params format: {path}")


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

    p["opening_range_minutes"] = int(rng.choice([15, 20, 30, 45, 60, 90]))
    p["breakout_buffer_bps"] = round(
        _clamp(float(p.get("breakout_buffer_bps", 7.0)) + rng.uniform(-4.0, 4.0), 0.5, 14.0), 6
    )
    p["lookback_short_days"] = round(
        _clamp(float(p.get("lookback_short_days", 1.0)) + rng.uniform(-0.6, 1.2), 0.25, 4.0), 6
    )
    p["lookback_long_days"] = round(
        _clamp(float(p.get("lookback_long_days", 7.0)) + rng.uniform(-3.0, 8.0), 2.0, 20.0), 6
    )
    if float(p["lookback_long_days"]) <= float(p["lookback_short_days"]):
        p["lookback_long_days"] = round(float(p["lookback_short_days"]) + rng.uniform(1.0, 8.0), 6)

    p["momentum_threshold_bps"] = round(
        _clamp(float(p.get("momentum_threshold_bps", 0.0)) + rng.uniform(-8.0, 8.0), -20.0, 40.0), 6
    )
    p["min_atr_bps"] = round(
        _clamp(float(p.get("min_atr_bps", 1.5)) + rng.uniform(-1.0, 3.5), 0.2, 12.0), 6
    )

    p["risk_per_trade"] = round(
        _clamp(float(p.get("risk_per_trade", 0.012)) + rng.uniform(-0.008, 0.02), 0.001, 0.08), 6
    )
    p["base_leverage"] = round(
        _clamp(float(p.get("base_leverage", 2.6)) + rng.uniform(-1.6, 2.5), 0.5, 8.0), 6
    )
    p["max_leverage"] = round(
        _clamp(float(p.get("max_leverage", 3.9)) + rng.uniform(-2.0, 3.0), 1.0, 12.0), 6
    )
    if float(p["max_leverage"]) < float(p["base_leverage"]):
        p["max_leverage"] = float(p["base_leverage"])
    p["max_margin_utilization"] = round(
        _clamp(float(p.get("max_margin_utilization", 0.42)) + rng.uniform(-0.20, 0.20), 0.05, 0.80), 6
    )
    p["stop_atr_mult"] = round(
        _clamp(float(p.get("stop_atr_mult", 2.3)) + rng.uniform(-1.2, 1.8), 0.5, 6.0), 6
    )
    p["min_liq_buffer_atr"] = round(
        _clamp(float(p.get("min_liq_buffer_atr", 4.0)) + rng.uniform(-2.0, 4.0), 1.0, 10.0), 6
    )

    p["weekly_profit_target"] = round(
        _clamp(float(p.get("weekly_profit_target", 0.0068)) + rng.uniform(-0.006, 0.015), 0.001, 0.06), 6
    )
    p["weekly_chase_k"] = round(
        _clamp(float(p.get("weekly_chase_k", 0.32)) + rng.uniform(-0.25, 0.55), 0.0, 2.5), 6
    )
    p["weekly_heartbeat_exposure"] = round(
        _clamp(float(p.get("weekly_heartbeat_exposure", 0.0071)) + rng.uniform(-0.006, 0.05), 0.0, 0.20), 6
    )
    p["weekly_heartbeat_hold_bars"] = int(rng.choice([1, 2, 3, 4, 6, 8, 12]))
    p["weekly_nudge_exposure"] = round(
        _clamp(float(p.get("weekly_nudge_exposure", 0.0)) + rng.uniform(-0.01, 0.06), 0.0, 0.20), 6
    )

    p["max_flips_per_day"] = int(rng.choice([1, 2, 3, 4]))
    p["daily_loss_hard_stop"] = round(
        _clamp(float(p.get("daily_loss_hard_stop", 0.0)) + rng.uniform(-0.01, 0.04), 0.0, 0.12), 6
    )
    p["weekly_loss_hard_stop"] = round(
        _clamp(float(p.get("weekly_loss_hard_stop", 0.0)) + rng.uniform(-0.02, 0.12), 0.0, 0.25), 6
    )
    p["cooldown_bars_after_exit"] = int(rng.choice([0, 2, 4, 8, 12, 16, 24, 32, 48]))
    p["trailing_stop_atr_mult"] = round(
        _clamp(float(p.get("trailing_stop_atr_mult", 0.0)) + rng.uniform(-0.6, 2.4), 0.0, 6.0), 6
    )
    p["break_even_trigger_atr"] = round(
        _clamp(float(p.get("break_even_trigger_atr", 0.0)) + rng.uniform(-0.5, 2.5), 0.0, 5.0), 6
    )
    p["max_hold_bars"] = int(rng.choice([0, 8, 16, 24, 32, 48, 64, 96, 144]))

    return p


def _score_candidate(
    *,
    rows: list[dict[str, Any]],
    weekly_positive_gate: float,
    weekly_beat_spy_gate: float,
) -> float:
    if not rows:
        return float("-inf")

    runs = len(rows)
    run_pb = sum(1 for r in rows if bool(r["run_profitable_and_beat_spy"]))
    weekly_pb = sum(1 for r in rows if bool(r["weekly_gate_and_beat"]))
    mean_return = sum(float(r["total_return"]) for r in rows) / runs
    mean_alpha = sum(float(r["alpha_vs_spy"]) for r in rows) / runs
    worst_return = min(float(r["total_return"]) for r in rows)
    worst_drawdown = min(float(r["max_drawdown"]) for r in rows)

    score = 0.0
    score += 10.0 * (run_pb / runs)
    score += 8.0 * (weekly_pb / runs)
    score += 1.2 * mean_return
    score += 1.2 * mean_alpha
    score -= 8.0 * max(0.0, -0.25 - worst_return)
    score -= 5.0 * max(0.0, -0.25 - worst_drawdown)
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
                "coinbase_fee_model": bool(args.coinbase_fee_model),
                "fixed_fee_per_contract_usd": float(args.fixed_fee_per_contract_usd),
                "contract_size_units": float(args.contract_size_units),
                "allow_short": bool(args.allow_short),
                "weekly_positive_gate": float(args.weekly_positive_gate),
                "weekly_beat_spy_gate": float(args.weekly_beat_spy_gate),
            },
            indent=2,
        )
    )

    coinbase_fee_active = bool(
        args.coinbase_fee_model
        and str(args.market).strip().lower() == "derivatives"
        and str(args.data_source).strip().lower() == "coinbase"
    )
    fixed_fee_per_contract_usd = float(args.fixed_fee_per_contract_usd) if coinbase_fee_active else 0.0
    contract_size_units = float(args.contract_size_units) if coinbase_fee_active else 1.0
    if contract_size_units <= 0.0:
        contract_size_units = 1.0

    tf = parse_bar_timeframe(str(args.bar_timeframe))
    bars_per_window: dict[int, tuple[WindowSpec, Any]] = {}
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
        bars_per_window[i] = (w, universe.bars_by_symbol[str(args.symbol)].copy())

    cfg = BacktestConfig(
        symbols=[str(args.symbol)],
        initial_cash=float(args.initial_cash),
        max_position_notional_usd=float(args.max_notional),
        slippage_bps=float(args.slippage_bps),
        taker_fee_bps=float(args.taker_fee_bps),
        fixed_fee_per_contract_usd=float(fixed_fee_per_contract_usd),
        contract_size_units=float(contract_size_units),
        allow_short=bool(args.allow_short),
        maintenance_margin_rate=0.05,
        liquidation_fee_rate=0.005,
    )

    all_window_rows: list[dict[str, Any]] = []
    candidate_summaries: list[dict[str, Any]] = []

    trials_total = int(args.trials) + 1
    for t in range(trials_total):
        if t == 0:
            params = copy.deepcopy(base)
            origin = "base"
        else:
            params = _mutate(base, rng)
            origin = "mutated"

        if not validate_params("perp_weekly_profit_chase", params):
            print(f"skip invalid params trial={t}")
            continue

        candidate_id = f"cand_{t:03d}"
        run_rows: list[dict[str, Any]] = []
        for wi, (w, bars) in bars_per_window.items():
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
                bars_by_symbol={str(args.symbol): bars},
                strategy=strat,
                cfg=cfg,
                run_dir=run_dir,
                debug=False,
                score_start=w.start,
                score_end=w.end,
                no_trade_before=w.start,
            )

            metrics = json.loads((run_dir / "metrics.json").read_text())
            total_return = float(metrics.get("total_return", 0.0))
            max_drawdown = float(metrics.get("max_drawdown", 0.0))
            trades = int(metrics.get("trade_count", metrics.get("trades", 0) or 0))

            spy = spy_total_return(start=w.start, end=w.end)
            spy_return = float(spy.total_return if spy is not None else 0.0)
            alpha = float(total_return - spy_return)

            summary, window_rows = rolling_window_summary(
                run_dir=run_dir,
                window=timedelta(days=7),
                step=timedelta(days=7),
                benchmark="spy.us",
            )
            positive_weeks = [r for r in window_rows if float(r.get("return", 0.0)) > 0.0]
            positive_weeks_count = int(len(positive_weeks))
            weeks_total = int(summary.windows)
            weekly_positive_frac = float(positive_weeks_count / max(1, weeks_total))

            positive_weeks_beat_spy = int(
                sum(
                    1
                    for r in positive_weeks
                    if float(r.get("return", 0.0)) > float(r.get("benchmark_return", 0.0))
                )
            )
            weekly_positive_beat_spy_frac = float(
                positive_weeks_beat_spy / max(1, positive_weeks_count)
            )

            row = {
                "candidate_id": candidate_id,
                "origin": origin,
                "trial_index": int(t),
                "window_index": wi,
                "window_year": int(w.year),
                "window_start": w.start.isoformat(),
                "window_end": w.end.isoformat(),
                "total_return": float(total_return),
                "max_drawdown": float(max_drawdown),
                "trades": int(trades),
                "spy_return": float(spy_return),
                "alpha_vs_spy": float(alpha),
                "run_profitable_and_beat_spy": bool(total_return > 0.0 and alpha > 0.0),
                "weeks_total": int(weeks_total),
                "weeks_positive": int(positive_weeks_count),
                "weekly_positive_frac": float(weekly_positive_frac),
                "positive_weeks_beat_spy": int(positive_weeks_beat_spy),
                "weekly_positive_beat_spy_frac": float(weekly_positive_beat_spy_frac),
                "weekly_gate_and_beat": bool(
                    weekly_positive_frac >= float(args.weekly_positive_gate)
                    and weekly_positive_beat_spy_frac >= float(args.weekly_beat_spy_gate)
                ),
                "run_dir": str(run_dir),
            }
            run_rows.append(row)
            all_window_rows.append(row)

        score = _score_candidate(
            rows=run_rows,
            weekly_positive_gate=float(args.weekly_positive_gate),
            weekly_beat_spy_gate=float(args.weekly_beat_spy_gate),
        )
        runs = len(run_rows)
        run_pb = int(sum(1 for r in run_rows if bool(r["run_profitable_and_beat_spy"])))
        weekly_pb = int(sum(1 for r in run_rows if bool(r["weekly_gate_and_beat"])))
        mean_return = float(sum(float(r["total_return"]) for r in run_rows) / max(1, runs))
        mean_alpha = float(sum(float(r["alpha_vs_spy"]) for r in run_rows) / max(1, runs))
        worst_return = float(min(float(r["total_return"]) for r in run_rows))
        worst_drawdown = float(min(float(r["max_drawdown"]) for r in run_rows))
        mean_weekly_positive_frac = float(
            sum(float(r["weekly_positive_frac"]) for r in run_rows) / max(1, runs)
        )
        mean_weekly_positive_beat_spy_frac = float(
            sum(float(r["weekly_positive_beat_spy_frac"]) for r in run_rows) / max(1, runs)
        )

        candidate_summary = {
            "candidate_id": candidate_id,
            "origin": origin,
            "trial_index": int(t),
            "score": float(score),
            "runs": int(runs),
            "run_profitable_and_beat_spy_count": int(run_pb),
            "weekly_gate_and_beat_count": int(weekly_pb),
            "mean_total_return": float(mean_return),
            "mean_alpha_vs_spy": float(mean_alpha),
            "worst_total_return": float(worst_return),
            "worst_max_drawdown": float(worst_drawdown),
            "mean_weekly_positive_frac": float(mean_weekly_positive_frac),
            "mean_weekly_positive_beat_spy_frac": float(mean_weekly_positive_beat_spy_frac),
            "params": copy.deepcopy(params),
        }
        candidate_summaries.append(candidate_summary)

        # Checkpoint after each candidate for resumability.
        (out_dir / "leaderboard.partial.json").write_text(json.dumps(candidate_summaries, indent=2))
        print(
            f"[{t+1}/{trials_total}] {candidate_id} "
            f"score={score:.4f} "
            f"run_pb={run_pb}/{runs} weekly_pb={weekly_pb}/{runs} "
            f"mean_ret={mean_return:.4%} mean_alpha={mean_alpha:.4%} "
            f"worst_ret={worst_return:.2%} worst_dd={worst_drawdown:.2%}"
        )

    candidate_summaries.sort(
        key=lambda r: (
            int(r["run_profitable_and_beat_spy_count"]),
            int(r["weekly_gate_and_beat_count"]),
            float(r["score"]),
        ),
        reverse=True,
    )

    rows_csv = out_dir / "window_rows.csv"
    if all_window_rows:
        with rows_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_window_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_window_rows)

    leaderboard = out_dir / "leaderboard.json"
    leaderboard.write_text(json.dumps(candidate_summaries, indent=2))

    keep_top = int(max(1, args.keep_top))
    top = candidate_summaries[:keep_top]
    top_rows: list[dict[str, Any]] = []
    for rank, row in enumerate(top, start=1):
        params_payload = {"perp_weekly_profit_chase": row["params"]}
        cand_name = (
            f"{args.label}_seed{int(args.seed)}_rank{rank:02d}_"
            f"runpb{int(row['run_profitable_and_beat_spy_count'])}of{int(row['runs'])}_"
            f"wpb{int(row['weekly_gate_and_beat_count'])}of{int(row['runs'])}"
        )
        path = cands_dir / f"{cand_name}.json"
        path.write_text(json.dumps(params_payload, indent=2))
        top_rows.append(
            {
                "rank": rank,
                "candidate_id": row["candidate_id"],
                "origin": row["origin"],
                "score": row["score"],
                "run_profitable_and_beat_spy_count": row["run_profitable_and_beat_spy_count"],
                "weekly_gate_and_beat_count": row["weekly_gate_and_beat_count"],
                "runs": row["runs"],
                "mean_total_return": row["mean_total_return"],
                "mean_alpha_vs_spy": row["mean_alpha_vs_spy"],
                "worst_total_return": row["worst_total_return"],
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
