#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import glob
import json
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
            "Evaluate perp_weekly_profit_chase parameter files across fixed windows "
            "and aggregate run-level + weekly-window metrics."
        )
    )
    p.add_argument("--windows-json", required=True, help="Path to windows.json")
    p.add_argument(
        "--window-indices",
        default="",
        help="Comma-separated 1-based indices from windows.json. Empty = all windows.",
    )
    p.add_argument(
        "--params-file",
        action="append",
        default=[],
        help="Path to a params JSON file. Can be repeated.",
    )
    p.add_argument(
        "--params-glob",
        action="append",
        default=[],
        help="Glob pattern for params files. Can be repeated.",
    )
    p.add_argument(
        "--label",
        default="candidate_eval",
        help="Label prefix in outputs/evaluations/perp_weekly_profit_chase_candidate_eval",
    )
    p.add_argument(
        "--out-root",
        default="outputs/evaluations/perp_weekly_profit_chase_candidate_eval",
        help="Output root directory",
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
        "--min-weekly-gate",
        type=float,
        default=0.70,
        help="Threshold for counting a run as weekly-gate-passing.",
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


def _resolve_param_files(args: argparse.Namespace) -> list[Path]:
    files: list[Path] = []
    for p in args.params_file:
        files.append(Path(p))
    for pat in args.params_glob:
        for p in sorted(glob.glob(pat)):
            files.append(Path(p))
    dedup: list[Path] = []
    seen: set[str] = set()
    for p in files:
        key = str(p)
        if key in seen:
            continue
        seen.add(key)
        dedup.append(p)
    if not dedup:
        raise ValueError("Provide at least one --params-file or --params-glob")
    missing = [str(p) for p in dedup if not p.exists()]
    if missing:
        raise ValueError(f"Missing params files: {missing}")
    return dedup


def _candidate_id(path: Path) -> str:
    parent = path.parent.parent.name if path.parent.name == "candidates" else path.parent.name
    return f"{parent}__{path.stem}"


def _score_candidate(rows: list[dict[str, Any]], *, min_weekly_gate: float) -> float:
    if not rows:
        return float("-inf")
    runs = len(rows)
    profitable_runs = sum(1 for r in rows if float(r["total_return"]) > 0.0)
    weekly_gate_runs = sum(1 for r in rows if float(r["weekly_positive_frac"]) >= float(min_weekly_gate))
    profitable_frac = profitable_runs / runs
    weekly_gate_frac = weekly_gate_runs / runs
    mean_return = sum(float(r["total_return"]) for r in rows) / runs
    median_return = sorted(float(r["total_return"]) for r in rows)[runs // 2]
    worst_return = min(float(r["total_return"]) for r in rows)
    worst_drawdown = min(float(r["max_drawdown"]) for r in rows)
    weekly_agg_frac = (
        sum(float(r["weeks_positive"]) for r in rows)
        / max(1.0, sum(float(r["weeks_total"]) for r in rows))
    )
    score = 0.0
    score += 5.0 * profitable_frac
    score += 2.0 * weekly_gate_frac
    score += 1.0 * weekly_agg_frac
    score += 1.5 * mean_return
    score += 0.8 * median_return
    score -= 3.0 * max(0.0, -0.30 - worst_drawdown)
    score -= 2.2 * max(0.0, -0.35 - worst_return)
    return score


def main() -> int:
    args = _parse_args()
    idxs: set[int] = set()
    if str(args.window_indices).strip():
        idxs = {int(x.strip()) for x in str(args.window_indices).split(",") if x.strip()}
    windows = _parse_windows(Path(args.windows_json), idxs)
    param_files = _resolve_param_files(args)

    out_dir = (
        Path(args.out_root)
        / f"{args.label}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = out_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    tf = parse_bar_timeframe(str(args.bar_timeframe))
    bars_per_window: dict[int, Any] = {}
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

    coinbase_fee_active = bool(
        args.coinbase_fee_model
        and str(args.market).strip().lower() == "derivatives"
        and str(args.data_source).strip().lower() == "coinbase"
    )
    fixed_fee_per_contract_usd = float(args.fixed_fee_per_contract_usd) if coinbase_fee_active else 0.0
    contract_size_units = float(args.contract_size_units) if coinbase_fee_active else 1.0
    if contract_size_units <= 0.0:
        contract_size_units = 1.0

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

    window_rows: list[dict[str, Any]] = []
    candidate_rows_map: dict[str, list[dict[str, Any]]] = {}
    candidate_meta: dict[str, dict[str, Any]] = {}

    for pf in param_files:
        cid = _candidate_id(pf)
        params = _load_strategy_params(pf)
        if not validate_params("perp_weekly_profit_chase", params):
            print(f"skip invalid params: {pf}")
            continue
        candidate_meta[cid] = {
            "candidate_id": cid,
            "params_file": str(pf),
            "params": params,
        }
        candidate_rows_map[cid] = []
        for wi, w in enumerate(windows, start=1):
            run_dir = runs_dir / cid / f"w{wi:02d}_{w.year}_{w.start.date()}_{w.end.date()}"
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
            summary, window_segments = rolling_window_summary(
                run_dir=run_dir,
                window=timedelta(days=7),
                step=timedelta(days=7),
                benchmark="spy.us",
            )
            weeks_positive = sum(1 for r in window_segments if float(r.get("return", 0.0)) > 0.0)
            weeks_total = int(summary.windows)
            row = {
                "candidate_id": cid,
                "params_file": str(pf),
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
                "mean_weekly_return": float(summary.mean_return),
                "beat_spy_weekly_frac": float(summary.beat_benchmark_frac or 0.0),
                "run_dir": str(run_dir),
            }
            window_rows.append(row)
            candidate_rows_map[cid].append(row)

    leaderboard_rows: list[dict[str, Any]] = []
    for cid, rows in candidate_rows_map.items():
        if not rows:
            continue
        runs = len(rows)
        rets = [float(r["total_return"]) for r in rows]
        dds = [float(r["max_drawdown"]) for r in rows]
        weeks_total = int(sum(int(r["weeks_total"]) for r in rows))
        weeks_positive = int(sum(int(r["weeks_positive"]) for r in rows))
        profitable_runs = sum(1 for r in rows if float(r["total_return"]) > 0.0)
        weekly_gate_runs = sum(
            1 for r in rows if float(r["weekly_positive_frac"]) >= float(args.min_weekly_gate)
        )
        leaderboard_rows.append(
            {
                "candidate_id": cid,
                "params_file": str(candidate_meta[cid]["params_file"]),
                "runs": runs,
                "profitable_runs": int(profitable_runs),
                "profitable_run_frac": float(profitable_runs / max(1, runs)),
                "weekly_gate_runs": int(weekly_gate_runs),
                "weekly_gate_run_frac": float(weekly_gate_runs / max(1, runs)),
                "weeks_total": int(weeks_total),
                "weeks_positive": int(weeks_positive),
                "aggregate_weekly_positive_frac": float(weeks_positive / max(1, weeks_total)),
                "mean_total_return": float(sum(rets) / max(1, runs)),
                "median_total_return": float(sorted(rets)[runs // 2]),
                "worst_total_return": float(min(rets)),
                "best_total_return": float(max(rets)),
                "worst_max_drawdown": float(min(dds)),
                "mean_max_drawdown": float(sum(dds) / max(1, runs)),
                "score": float(_score_candidate(rows, min_weekly_gate=float(args.min_weekly_gate))),
            }
        )

    leaderboard_rows.sort(key=lambda r: float(r["score"]), reverse=True)

    rows_csv = out_dir / "window_rows.csv"
    if window_rows:
        with rows_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(window_rows[0].keys()))
            writer.writeheader()
            writer.writerows(window_rows)

    leaderboard_csv = out_dir / "leaderboard.csv"
    if leaderboard_rows:
        with leaderboard_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(leaderboard_rows[0].keys()))
            writer.writeheader()
            writer.writerows(leaderboard_rows)

    summary = {
        "out_dir": str(out_dir),
        "windows_json": str(args.windows_json),
        "window_indices": sorted(idxs),
        "params_files": [str(p) for p in param_files],
        "slippage_bps": float(args.slippage_bps),
        "taker_fee_bps": float(args.taker_fee_bps),
        "coinbase_fee_model": bool(args.coinbase_fee_model),
        "fixed_fee_per_contract_usd": float(args.fixed_fee_per_contract_usd),
        "contract_size_units": float(args.contract_size_units),
        "leaderboard_csv": str(leaderboard_csv),
        "window_rows_csv": str(rows_csv),
        "winner": leaderboard_rows[0] if leaderboard_rows else None,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
