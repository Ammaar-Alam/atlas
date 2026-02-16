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
            "Evaluate strategy parameter files across fixed windows and aggregate "
            "run-level + weekly-window metrics."
        )
    )
    p.add_argument("--strategy", required=True, help="Strategy name in registry")
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
        default="strategy_eval",
        help="Label prefix in outputs/evaluations/strategy_eval",
    )
    p.add_argument(
        "--out-root",
        default="outputs/evaluations/strategy_eval",
        help="Output root directory",
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
        "--min-weekly-gate",
        type=float,
        default=0.70,
        help="Threshold for counting a run as weekly-gate-passing.",
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
    symbols = _parse_symbols(str(args.symbols), str(args.symbol))
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
    csv_path = Path(str(args.csv_path)) if str(args.csv_path).strip() else None
    csv_dir = Path(str(args.csv_dir)) if str(args.csv_dir).strip() else None
    bars_per_window: dict[int, Any] = {}
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
        bars_per_window[i] = bars

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

    window_rows: list[dict[str, Any]] = []
    candidate_rows_map: dict[str, list[dict[str, Any]]] = {}
    candidate_meta: dict[str, dict[str, Any]] = {}

    for pf in param_files:
        cid = _candidate_id(pf)
        params = _load_strategy_params(pf, str(args.strategy))
        if not validate_params(str(args.strategy), params):
            print(f"skip invalid params for strategy={args.strategy}: {pf}")
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
                name=str(args.strategy),
                params_path=None,
                symbols=list(symbols),
                fast_window=10,
                slow_window=30,
                params=params,
            )
            run_derivatives_backtest(
                bars_by_symbol=bars_per_window[wi],
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
                "strategy": str(args.strategy),
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
                "strategy": str(args.strategy),
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
                "mean_total_return": float(sum(rets) / max(1, len(rets))),
                "median_total_return": float(sorted(rets)[len(rets) // 2]),
                "worst_total_return": float(min(rets)),
                "best_total_return": float(max(rets)),
                "worst_max_drawdown": float(min(dds)),
                "mean_max_drawdown": float(sum(dds) / max(1, len(dds))),
                "score": float(_score_candidate(rows, min_weekly_gate=float(args.min_weekly_gate))),
            }
        )
    leaderboard_rows = sorted(leaderboard_rows, key=lambda r: float(r["score"]), reverse=True)

    window_rows_csv = out_dir / "window_rows.csv"
    leaderboard_csv = out_dir / "leaderboard.csv"
    leaderboard_json = out_dir / "leaderboard.json"
    evaluation_result_json = out_dir / "evaluation_result.json"

    with window_rows_csv.open("w", newline="") as f:
        fieldnames = [
            "strategy",
            "candidate_id",
            "params_file",
            "window_index",
            "window_year",
            "window_start",
            "window_end",
            "total_return",
            "max_drawdown",
            "trades",
            "weeks_total",
            "weeks_positive",
            "weekly_positive_frac",
            "mean_weekly_return",
            "beat_spy_weekly_frac",
            "run_dir",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in window_rows:
            w.writerow(r)

    with leaderboard_csv.open("w", newline="") as f:
        fieldnames = [
            "strategy",
            "candidate_id",
            "params_file",
            "runs",
            "profitable_runs",
            "profitable_run_frac",
            "weekly_gate_runs",
            "weekly_gate_run_frac",
            "weeks_total",
            "weeks_positive",
            "aggregate_weekly_positive_frac",
            "mean_total_return",
            "median_total_return",
            "worst_total_return",
            "best_total_return",
            "worst_max_drawdown",
            "mean_max_drawdown",
            "score",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in leaderboard_rows:
            w.writerow(r)

    leaderboard_json.write_text(json.dumps(leaderboard_rows, indent=2))
    winner = leaderboard_rows[0] if leaderboard_rows else None
    result = {
        "out_dir": str(out_dir),
        "strategy": str(args.strategy),
        "windows_json": str(args.windows_json),
        "window_indices": sorted(list(idxs)),
        "params_files": [str(p) for p in param_files],
        "symbol": str(args.symbol),
        "symbols": list(symbols),
        "market": str(args.market),
        "data_source": str(args.data_source),
        "bar_timeframe": str(args.bar_timeframe),
        "initial_cash": float(args.initial_cash),
        "max_notional": float(args.max_notional),
        "slippage_bps": float(args.slippage_bps),
        "taker_fee_bps": float(args.taker_fee_bps),
        "coinbase_fee_model": bool(args.coinbase_fee_model),
        "fixed_fee_per_contract_usd": float(fixed_fee_per_contract_usd),
        "contract_size_units": float(contract_size_units),
        "fixed_fee_map": dict(fixed_fee_map),
        "contract_size_map": dict(contract_size_map),
        "allow_short": bool(args.allow_short),
        "leaderboard_csv": str(leaderboard_csv),
        "window_rows_csv": str(window_rows_csv),
        "winner": winner,
    }
    evaluation_result_json.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
