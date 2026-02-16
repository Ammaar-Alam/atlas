#!/usr/bin/env python3
from __future__ import annotations

import argparse
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
from atlas.ml.tune import get_search_space, sample_params, validate_params
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
            "Strict-gate random search for any strategy with tuning space. "
            "Objective emphasizes profitable runs, SPY outperformance, and weekly-positive consistency."
        )
    )
    p.add_argument("--strategy", required=True, help="Strategy name in registry and tuning space")
    p.add_argument("--base-params", required=True, help="Path to strategy params JSON")
    p.add_argument("--windows-json", required=True, help="Path to windows.json")
    p.add_argument(
        "--window-indices",
        default="",
        help="Comma-separated 1-based indices from windows.json. Empty = all windows.",
    )
    p.add_argument("--seed", type=int, default=11, help="RNG seed")
    p.add_argument("--trials", type=int, default=40, help="Random samples (base candidate is always included)")
    p.add_argument("--keep-top", type=int, default=5, help="Top candidates to persist")
    p.add_argument("--label", default="strict_strategy", help="Label prefix in outputs")
    p.add_argument(
        "--out-dir",
        default="outputs/evaluations/strategy_strict_gate_search",
        help="Output directory",
    )
    p.add_argument("--symbol", default="BTC-PERP")
    p.add_argument(
        "--symbols",
        default="",
        help="Comma-separated symbols. Overrides --symbol when provided.",
    )
    p.add_argument("--market", default="derivatives")
    p.add_argument("--data-source", default="coinbase")
    p.add_argument(
        "--csv-path",
        default="",
        help="CSV file path when --data-source csv and using single symbol.",
    )
    p.add_argument(
        "--csv-dir",
        default="",
        help="CSV directory when --data-source csv and using multiple symbols.",
    )
    p.add_argument("--bar-timeframe", default="1H")
    p.add_argument("--prewarm-days", type=int, default=30)
    p.add_argument("--initial-cash", type=float, default=500.0)
    p.add_argument("--max-notional", type=float, default=5000.0)
    p.add_argument("--slippage-bps", type=float, default=1.5)
    p.add_argument("--taker-fee-bps", type=float, default=10.0)
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
        "--fixed-fee-map",
        default="",
        help="Optional symbol->fixed fee map, e.g. BTC-PERP:0.15,ETH-PERP:0.15",
    )
    p.add_argument(
        "--contract-size-map",
        default="",
        help="Optional symbol->contract size map, e.g. BTC-PERP:0.01,ETH-PERP:0.10",
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
    p.add_argument(
        "--drift-frac",
        type=float,
        default=0.50,
        help="Clamp random samples to +/- drift_frac around base parameters.",
    )
    return p.parse_args()


def _parse_symbols(symbols_arg: str, symbol_arg: str) -> list[str]:
    if str(symbols_arg or "").strip():
        parts = [s.strip() for s in str(symbols_arg).split(",") if s.strip()]
        if parts:
            return parts
    return [str(symbol_arg).strip()]


def _parse_symbol_float_map(raw: str) -> dict[str, float]:
    out: dict[str, float] = {}
    text = str(raw or "").strip()
    if not text:
        return out
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"bad symbol map entry {part!r}; expected SYMBOL:value")
        sym, val = part.split(":", 1)
        sym = sym.strip()
        if not sym:
            raise ValueError(f"empty symbol in map entry {part!r}")
        out[sym] = float(val.strip())
    return out


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
                year=int(item.get("year", start.year)),
                start=start,
                end=end,
                length_days=int(item.get("length_days", (end - start).days)),
            )
        )
    if not out:
        raise ValueError("No windows selected")
    return out


def _load_strategy_params(path: Path, strategy: str) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if isinstance(payload, dict):
        if strategy in payload and isinstance(payload[strategy], dict):
            return dict(payload[strategy])
        if "atlas_profile" in payload:
            other = {k: v for k, v in payload.items() if k != "atlas_profile"}
            if strategy in other and isinstance(other[strategy], dict):
                return dict(other[strategy])
        if all(not isinstance(v, dict) for v in payload.values()):
            return dict(payload)
    raise ValueError(f"Unsupported params format for strategy={strategy}: {path}")


def _score_candidate(
    *,
    rows: list[dict[str, Any]],
    weekly_positive_gate: float,
    weekly_beat_spy_gate: float,
) -> float:
    if not rows:
        return float("-inf")
    runs = len(rows)
    returns = [float(r["total_return"]) for r in rows]
    alphas = [float(r["alpha_vs_spy"]) for r in rows]
    run_profitable = sum(1 for r in rows if float(r["total_return"]) > 0.0)
    run_pb = sum(1 for r in rows if bool(r["run_profitable_and_beat_spy"]))
    weekly_gate = sum(1 for r in rows if float(r["weekly_positive_frac"]) >= float(weekly_positive_gate))
    weekly_pb = sum(1 for r in rows if bool(r["weekly_gate_and_beat"]))
    mean_return = sum(returns) / runs
    mean_alpha = sum(alphas) / runs
    median_return = float(sorted(returns)[runs // 2]) if runs % 2 else float(
        0.5 * (sorted(returns)[runs // 2 - 1] + sorted(returns)[runs // 2])
    )
    median_alpha = float(sorted(alphas)[runs // 2]) if runs % 2 else float(
        0.5 * (sorted(alphas)[runs // 2 - 1] + sorted(alphas)[runs // 2])
    )
    # Winsorize means so single pathological windows cannot dominate ranking.
    mean_return_w = max(-1.0, min(1.0, float(mean_return)))
    mean_alpha_w = max(-1.0, min(1.0, float(mean_alpha)))
    mean_weekly_pos = sum(float(r["weekly_positive_frac"]) for r in rows) / runs
    mean_weekly_beat = sum(float(r["weekly_positive_beat_spy_frac"]) for r in rows) / runs
    worst_return = min(returns)
    best_return = max(returns)
    worst_drawdown = min(float(r["max_drawdown"]) for r in rows)

    score = 0.0
    score += 4.0 * (run_profitable / runs)
    score += 6.0 * (run_pb / runs)
    score += 18.0 * (weekly_gate / runs)
    score += 18.0 * (weekly_pb / runs)
    score += 100.0 * mean_weekly_pos
    score += 20.0 * mean_weekly_beat
    score += 2.0 * median_return
    score += 2.0 * median_alpha
    score += 0.75 * mean_return_w
    score += 0.75 * mean_alpha_w
    # Harsh downside penalties (pessimistic scoring).
    score -= 15.0 * max(0.0, -0.10 - worst_return)
    score -= 10.0 * max(0.0, -0.10 - worst_drawdown)
    # Reject lottery-like profiles that rely on huge outlier runs.
    score -= 10.0 * max(0.0, best_return - 1.0)
    score -= 8.0 * max(0.0, mean_return - 0.50)
    score -= 6.0 * max(0.0, mean_alpha - 0.50)
    score -= 7.0 * max(0.0, -0.20 - worst_return)
    score -= 4.0 * max(0.0, -0.20 - worst_drawdown)
    return score


def main() -> int:
    args = _parse_args()
    strategy = str(args.strategy).strip()
    symbols = _parse_symbols(str(args.symbols), str(args.symbol))
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
    base = _load_strategy_params(Path(args.base_params), strategy)
    if not validate_params(strategy, base):
        raise ValueError(f"Base params invalid for {strategy}: {args.base_params}")
    space = get_search_space(strategy)

    (out_dir / "search_config.json").write_text(
        json.dumps(
            {
                "strategy": strategy,
                "base_params": str(args.base_params),
                "windows_json": str(args.windows_json),
                "window_indices": sorted(idxs),
                "seed": int(args.seed),
                "trials": int(args.trials),
                "symbol": str(args.symbol),
                "symbols": list(symbols),
                "market": str(args.market),
                "data_source": str(args.data_source),
                "csv_path": str(args.csv_path or ""),
                "csv_dir": str(args.csv_dir or ""),
                "bar_timeframe": str(args.bar_timeframe),
                "prewarm_days": int(args.prewarm_days),
                "initial_cash": float(args.initial_cash),
                "max_notional": float(args.max_notional),
                "slippage_bps": float(args.slippage_bps),
                "taker_fee_bps": float(args.taker_fee_bps),
                "coinbase_fee_model": bool(args.coinbase_fee_model),
                "fixed_fee_per_contract_usd": float(args.fixed_fee_per_contract_usd),
                "contract_size_units": float(args.contract_size_units),
                "fixed_fee_map": _parse_symbol_float_map(str(args.fixed_fee_map)),
                "contract_size_map": _parse_symbol_float_map(str(args.contract_size_map)),
                "allow_short": bool(args.allow_short),
                "weekly_positive_gate": float(args.weekly_positive_gate),
                "weekly_beat_spy_gate": float(args.weekly_beat_spy_gate),
                "drift_frac": float(args.drift_frac),
            },
            indent=2,
        )
    )

    coinbase_fee_active = bool(
        args.coinbase_fee_model
        and str(args.market).strip().lower() == "derivatives"
    )
    fixed_fee_map = _parse_symbol_float_map(str(args.fixed_fee_map))
    contract_size_map = _parse_symbol_float_map(str(args.contract_size_map))
    fixed_fee_per_contract_usd = float(args.fixed_fee_per_contract_usd) if coinbase_fee_active else 0.0
    contract_size_units = float(args.contract_size_units) if coinbase_fee_active else 1.0
    if contract_size_units <= 0.0:
        contract_size_units = 1.0

    tf = parse_bar_timeframe(str(args.bar_timeframe))
    csv_path = Path(str(args.csv_path)) if str(args.csv_path).strip() else None
    csv_dir = Path(str(args.csv_dir)) if str(args.csv_dir).strip() else None
    bars_per_window: dict[int, tuple[WindowSpec, Any]] = {}
    for i, w in enumerate(windows, start=1):
        load_start = w.start - timedelta(days=int(args.prewarm_days))
        universe = load_universe_bars(
            symbols=list(symbols),
            data_source=str(args.data_source),
            timeframe=tf,
            start=load_start,
            end=w.end,
            csv_path=csv_path,
            csv_dir=csv_dir,
            market=str(args.market),
            regular_hours_only=False,
        )
        bars = {
            s: universe.bars_by_symbol[s].copy()
            for s in symbols
            if s in universe.bars_by_symbol
        }
        if len(bars) != len(symbols):
            missing = [s for s in symbols if s not in bars]
            raise ValueError(f"missing bars for symbols={missing} in window={i}")
        bars_per_window[i] = (w, bars)

    cfg = BacktestConfig(
        symbols=list(symbols),
        initial_cash=float(args.initial_cash),
        max_position_notional_usd=float(args.max_notional),
        slippage_bps=float(args.slippage_bps),
        taker_fee_bps=float(args.taker_fee_bps),
        fixed_fee_per_contract_usd=float(fixed_fee_per_contract_usd),
        contract_size_units=float(contract_size_units),
        fixed_fee_per_contract_usd_by_symbol=dict(fixed_fee_map),
        contract_size_units_by_symbol=dict(contract_size_map),
        allow_short=bool(args.allow_short),
        maintenance_margin_rate=0.05,
        liquidation_fee_rate=0.005,
    )

    all_window_rows: list[dict[str, Any]] = []
    candidate_summaries: list[dict[str, Any]] = []

    trials_total = int(args.trials) + 1
    for ti in range(trials_total):
        if ti == 0:
            params = dict(base)
            origin = "base"
        else:
            params = sample_params(
                strategy=strategy,
                rng=rng,
                space=space,
                incumbent=base,
                drift_frac=float(args.drift_frac),
                max_attempts=1000,
            )
            origin = "mutated"

        cid = f"cand_{ti:03d}"
        rows: list[dict[str, Any]] = []
        for wi, (w, bars) in bars_per_window.items():
            run_dir = runs_dir / cid / f"w{wi:02d}_{w.year}_{w.start.date()}_{w.end.date()}"
            run_dir.mkdir(parents=True, exist_ok=True)
            strat = build_strategy(
                name=strategy,
                params_path=None,
                symbols=list(symbols),
                fast_window=10,
                slow_window=30,
                params=params,
            )
            run_derivatives_backtest(
                bars_by_symbol=bars,
                strategy=strat,
                cfg=cfg,
                run_dir=run_dir,
                debug=False,
                score_start=w.start,
                score_end=w.end,
                no_trade_before=w.start,
            )
            metrics = json.loads((run_dir / "metrics.json").read_text())
            weekly_summary, weekly_rows = rolling_window_summary(
                run_dir=run_dir,
                window=timedelta(days=7),
                step=timedelta(days=7),
                benchmark="spy.us",
            )
            spy = spy_total_return(start=w.start, end=w.end)
            spy_ret = float(spy.total_return if spy is not None else 0.0)
            profitable_weeks = [r for r in weekly_rows if float(r.get("return", 0.0)) > 0.0]
            profitable_weeks_total = len(profitable_weeks)
            profitable_weeks_beat = sum(
                1
                for r in profitable_weeks
                if float(r.get("return", 0.0)) > float(r.get("benchmark_return", 0.0) or 0.0)
            )
            weekly_pos_frac = float(
                sum(1 for r in weekly_rows if float(r.get("return", 0.0)) > 0.0)
                / max(1, len(weekly_rows))
            )
            weekly_pos_beat_spy_frac = float(
                profitable_weeks_beat / max(1, profitable_weeks_total)
            )
            row = {
                "candidate_id": cid,
                "origin": origin,
                "trial_index": int(ti),
                "window_index": int(wi),
                "window_year": int(w.year),
                "window_start": w.start.isoformat(),
                "window_end": w.end.isoformat(),
                "total_return": float(metrics.get("total_return", 0.0)),
                "alpha_vs_spy": float(metrics.get("total_return", 0.0) - spy_ret),
                "max_drawdown": float(metrics.get("max_drawdown", 0.0)),
                "trades": int(metrics.get("trade_count", metrics.get("trades", 0) or 0)),
                "weekly_positive_frac": weekly_pos_frac,
                "weekly_positive_beat_spy_frac": weekly_pos_beat_spy_frac,
                "run_profitable_and_beat_spy": bool(
                    float(metrics.get("total_return", 0.0)) > 0.0
                    and float(metrics.get("total_return", 0.0)) > float(spy_ret)
                ),
                "weekly_gate_and_beat": bool(
                    weekly_pos_frac >= float(args.weekly_positive_gate)
                    and weekly_pos_beat_spy_frac >= float(args.weekly_beat_spy_gate)
                ),
                "run_dir": str(run_dir),
            }
            rows.append(row)
            all_window_rows.append(row)

        score = _score_candidate(
            rows=rows,
            weekly_positive_gate=float(args.weekly_positive_gate),
            weekly_beat_spy_gate=float(args.weekly_beat_spy_gate),
        )
        summary = {
            "candidate_id": cid,
            "origin": origin,
            "trial_index": int(ti),
            "score": float(score),
            "runs": len(rows),
            "run_profitable_count": int(sum(1 for r in rows if float(r["total_return"]) > 0.0)),
            "run_profitable_and_beat_spy_count": int(sum(1 for r in rows if bool(r["run_profitable_and_beat_spy"]))),
            "weekly_gate_count": int(sum(1 for r in rows if float(r["weekly_positive_frac"]) >= float(args.weekly_positive_gate))),
            "weekly_gate_and_beat_count": int(sum(1 for r in rows if bool(r["weekly_gate_and_beat"]))),
            "mean_total_return": float(sum(float(r["total_return"]) for r in rows) / max(1, len(rows))),
            "mean_alpha_vs_spy": float(sum(float(r["alpha_vs_spy"]) for r in rows) / max(1, len(rows))),
            "worst_total_return": float(min(float(r["total_return"]) for r in rows)),
            "worst_max_drawdown": float(min(float(r["max_drawdown"]) for r in rows)),
            "mean_weekly_positive_frac": float(sum(float(r["weekly_positive_frac"]) for r in rows) / max(1, len(rows))),
            "mean_weekly_positive_beat_spy_frac": float(
                sum(float(r["weekly_positive_beat_spy_frac"]) for r in rows) / max(1, len(rows))
            ),
            "params": params,
        }
        candidate_summaries.append(summary)
        candidate_summaries.sort(key=lambda r: float(r["score"]), reverse=True)
        keep = max(1, int(args.keep_top))
        candidate_summaries = candidate_summaries[: max(keep, len(candidate_summaries))]

        print(
            f"[{ti+1}/{trials_total}] {cid} score={float(summary['score']):.4f} "
            f"run+={int(summary['run_profitable_count'])}/{len(rows)} "
            f"run+&beat={int(summary['run_profitable_and_beat_spy_count'])}/{len(rows)} "
            f"weekly_gate={int(summary['weekly_gate_count'])}/{len(rows)} "
            f"weekly_gate&beat={int(summary['weekly_gate_and_beat_count'])}/{len(rows)} "
            f"mean_ret={100.0*float(summary['mean_total_return']):.4f}% "
            f"mean_alpha={100.0*float(summary['mean_alpha_vs_spy']):.4f}% "
            f"worst_ret={100.0*float(summary['worst_total_return']):.2f}% "
            f"worst_dd={100.0*float(summary['worst_max_drawdown']):.2f}%"
        )

        (out_dir / "leaderboard.partial.json").write_text(json.dumps(candidate_summaries, indent=2))

    keep = max(1, int(args.keep_top))
    top = sorted(candidate_summaries, key=lambda r: float(r["score"]), reverse=True)[:keep]
    for item in top:
        cid = str(item["candidate_id"])
        (cands_dir / f"{cid}.json").write_text(json.dumps({strategy: item["params"]}, indent=2))

    leaderboard_json = out_dir / "leaderboard.json"
    leaderboard_csv = out_dir / "leaderboard.csv"
    window_rows_csv = out_dir / "window_rows.csv"
    top_csv = out_dir / "top_candidates.csv"

    leaderboard = sorted(candidate_summaries, key=lambda r: float(r["score"]), reverse=True)
    leaderboard_json.write_text(json.dumps(leaderboard, indent=2))

    with leaderboard_csv.open("w", newline="") as f:
        fieldnames = [
            "candidate_id",
            "origin",
            "trial_index",
            "score",
            "runs",
            "run_profitable_count",
            "run_profitable_and_beat_spy_count",
            "weekly_gate_count",
            "weekly_gate_and_beat_count",
            "mean_total_return",
            "mean_alpha_vs_spy",
            "worst_total_return",
            "worst_max_drawdown",
            "mean_weekly_positive_frac",
            "mean_weekly_positive_beat_spy_frac",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in leaderboard:
            w.writerow({k: r.get(k) for k in fieldnames})

    with window_rows_csv.open("w", newline="") as f:
        fieldnames = [
            "candidate_id",
            "origin",
            "trial_index",
            "window_index",
            "window_year",
            "window_start",
            "window_end",
            "total_return",
            "alpha_vs_spy",
            "max_drawdown",
            "trades",
            "weekly_positive_frac",
            "weekly_positive_beat_spy_frac",
            "run_profitable_and_beat_spy",
            "weekly_gate_and_beat",
            "run_dir",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in all_window_rows:
            w.writerow({k: r.get(k) for k in fieldnames})

    with top_csv.open("w", newline="") as f:
        fieldnames = [
            "rank",
            "candidate_id",
            "score",
            "run_profitable_count",
            "run_profitable_and_beat_spy_count",
            "weekly_gate_count",
            "weekly_gate_and_beat_count",
            "mean_total_return",
            "mean_alpha_vs_spy",
            "mean_weekly_positive_frac",
            "mean_weekly_positive_beat_spy_frac",
            "params_file",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for i, r in enumerate(top, start=1):
            w.writerow(
                {
                    "rank": i,
                    "candidate_id": r["candidate_id"],
                    "score": r["score"],
                    "run_profitable_count": r["run_profitable_count"],
                    "run_profitable_and_beat_spy_count": r["run_profitable_and_beat_spy_count"],
                    "weekly_gate_count": r["weekly_gate_count"],
                    "weekly_gate_and_beat_count": r["weekly_gate_and_beat_count"],
                    "mean_total_return": r["mean_total_return"],
                    "mean_alpha_vs_spy": r["mean_alpha_vs_spy"],
                    "mean_weekly_positive_frac": r["mean_weekly_positive_frac"],
                    "mean_weekly_positive_beat_spy_frac": r["mean_weekly_positive_beat_spy_frac"],
                    "params_file": str(cands_dir / f"{r['candidate_id']}.json"),
                }
            )

    result = {
        "out_dir": str(out_dir),
        "strategy": strategy,
        "leaderboard_json": str(leaderboard_json),
        "leaderboard_csv": str(leaderboard_csv),
        "window_rows_csv": str(window_rows_csv),
        "top_candidates_csv": str(top_csv),
        "top_count": len(top),
    }
    (out_dir / "run_summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
