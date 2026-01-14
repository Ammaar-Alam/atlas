"""Time-split + stress-test evaluation for the PerpHawk strategy.

This script is intentionally lightweight and deterministic. It can run on any
bars source supported by the repo (sample/csv/coinbase) as long as the market is
DERIVATIVES and the symbols exist in that data source.

Examples
--------
python scripts/eval_perp_hawk.py --data-source sample --symbols BTC-PERP ETH-PERP --timeframe 1Min
python scripts/eval_perp_hawk.py --data-source csv --csv-dir data/my_perp_bars --symbols BTC-PERP --timeframe 5Min
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from atlas.backtest.derivatives_engine import run_derivatives_backtest
from atlas.backtest.engine import BacktestConfig
from atlas.backtest.metrics import compute_metrics
from atlas.data.bars import parse_bar_timeframe
from atlas.data.universe import load_universe_bars
from atlas.market import Market
from atlas.strategies.registry import build_strategy


def _slice_bars(
    bars_by_symbol: dict[str, pd.DataFrame],
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for s, df in bars_by_symbol.items():
        out[s] = df.loc[(df.index >= start) & (df.index <= end)].copy()
    return out


def _add_funding_shock(
    bars_by_symbol: dict[str, pd.DataFrame],
    *,
    seed: int = 7,
    base_rate_per_hour: float = 0.0,
    spike_rate_per_hour: float = 0.0002,
    spike_prob: float = 0.02,
) -> dict[str, pd.DataFrame]:
    """Inject a synthetic funding series (hourly rate) for stress-testing.

    This does NOT attempt to be a faithful funding model; it is purely for
    robustness testing of strategies that use the funding channel.
    """

    rng = np.random.default_rng(seed)
    out: dict[str, pd.DataFrame] = {}
    for s, df in bars_by_symbol.items():
        z = df.copy()
        spikes = rng.random(len(z)) < spike_prob
        shock = np.where(spikes, spike_rate_per_hour, base_rate_per_hour)
        # Alternate sign across symbols so the portfolio isn't trivially biased.
        if hash(s) % 2 == 0:
            shock = -shock
        z["funding_rate"] = shock.astype(float)
        out[s] = z
    return out


def _pnl_decomp(eq_df: pd.DataFrame) -> dict[str, float]:
    """Summarize the PnL decomposition columns produced by derivatives_engine."""

    def _sum(col: str) -> float:
        return float(eq_df[col].fillna(0.0).sum()) if col in eq_df.columns else 0.0

    return {
        "price_pnl": _sum("price_pnl"),
        "funding_pnl": _sum("funding_pnl"),
        "fees_paid": -_sum("fees_paid"),
        "liquidation_fee": -_sum("liquidation_fee"),
        "slippage_cost_est": -_sum("slippage_cost_est"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-source", choices=["sample", "csv", "coinbase"], default="sample")
    ap.add_argument("--csv-dir", type=str, default=None, help="If data-source=csv, directory containing <SYMBOL>_<tf>_sample.csv-style files")
    ap.add_argument("--cache-dir", type=str, default="data/cache", help="Cache directory (for non-CSV sources)")
    ap.add_argument("--symbols", nargs="+", default=["BTC-PERP"], help="Derivative symbols, e.g. BTC-PERP")
    ap.add_argument("--timeframe", default="1Min", help="Bar timeframe (e.g. 1Min, 5Min, 15Min, 1H)")
    ap.add_argument("--out-root", default="outputs/evals", help="Root directory for evaluation runs")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    tf = parse_bar_timeframe(args.timeframe)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # Load bars
    if args.data_source == "csv":
        if not args.csv_dir:
            raise SystemExit("--csv-dir is required when --data-source=csv")
        from atlas.data.csv_loader import load_bars_csv

        bars_by_symbol: dict[str, pd.DataFrame] = {}
        for s in args.symbols:
            p = Path(args.csv_dir) / f"{s}_{tf.name.lower()}_sample.csv"
            if not p.exists():
                # fall back to <symbol>_<tf>.csv
                p2 = Path(args.csv_dir) / f"{s}_{tf.name.lower()}.csv"
                p = p2
            bars_by_symbol[s] = load_bars_csv(p)
    else:
        uni = load_universe_bars(
            market=Market.DERIVATIVES,
            symbols=args.symbols,
            data_source=args.data_source,
            timeframe=tf,
            start=None,
            end=None,
        )
        bars_by_symbol = uni.bars_by_symbol

    # Ensure common index
    common_idx = None
    for df in bars_by_symbol.values():
        common_idx = df.index if common_idx is None else common_idx.intersection(df.index)
    if common_idx is None or len(common_idx) < 50:
        raise SystemExit("Not enough overlapping bars to run eval")

    for s in list(bars_by_symbol.keys()):
        bars_by_symbol[s] = bars_by_symbol[s].loc[common_idx].copy()

    # Time splits (50/25/25)
    n = len(common_idx)
    i_train_end = int(n * 0.50)
    i_val_end = int(n * 0.75)
    splits = {
        "train": (common_idx[0], common_idx[i_train_end - 1]),
        "val": (common_idx[i_train_end], common_idx[i_val_end - 1]),
        "test": (common_idx[i_val_end], common_idx[-1]),
    }

    scenarios: dict[str, dict[str, Any]] = {
        "base": {"slippage_bps": 1.25, "taker_fee_bps": 3.0},
        "cost_x2": {"slippage_bps": 2.5, "taker_fee_bps": 6.0},
        "wide": {"slippage_bps": 8.0, "taker_fee_bps": 8.0},
        "funding_shock": {"slippage_bps": 2.5, "taker_fee_bps": 6.0, "funding_shock": True},
    }

    print("\n=== PerpHawk evaluation ===")
    print(f"symbols={args.symbols} timeframe={tf.name} bars={n}")
    print("splits:")
    for k, (a, b) in splits.items():
        print(f"  {k}: {a.isoformat()} -> {b.isoformat()} ({(common_idx <= b).sum() - (common_idx < a).sum()} bars)")

    # Strategy instantiation via registry for parity with CLI
    strat = build_strategy(
        name="perp_hawk",
        params_path=None,
        symbols=args.symbols,
        fast_window=20,
        slow_window=50,
        params={},
    )
    print("strategy:")
    try:
        print(asdict(strat))
    except Exception:
        print(strat)

    for split_name, (start, end) in splits.items():
        split_bars = _slice_bars(bars_by_symbol, start=start, end=end)
        for scen_name, scen in scenarios.items():
            scen_bars = split_bars
            if scen.get("funding_shock"):
                scen_bars = _add_funding_shock(
                    scen_bars,
                    seed=args.seed,
                )

            run_id = f"perp_hawk_{split_name}_{scen_name}"
            cfg = BacktestConfig(
                symbols=tuple(args.symbols),
                initial_cash=10_000.0,
                max_position_notional_usd=10_000.0,
                slippage_bps=float(scen["slippage_bps"]),
                allow_short=True,
                taker_fee_bps=float(scen["taker_fee_bps"]),
                maintenance_margin_rate=0.05,
                liquidation_fee_rate=0.005,
            )

            run_dir = out_root / run_id
            run_derivatives_backtest(
                bars_by_symbol=scen_bars,
                strategy=strat,
                cfg=cfg,
                run_dir=run_dir,
                progress=None,
            )

            eq_path = run_dir / "equity_curve.csv"
            trades_path = run_dir / "trades.csv"
            eq_df = pd.read_csv(eq_path)
            trades_df = pd.read_csv(trades_path)
            eq_df["timestamp"] = pd.to_datetime(eq_df["timestamp"], utc=True)
            eq_df = eq_df.set_index("timestamp").sort_index()
            metrics = compute_metrics(eq_df, trades_df)
            decomp = _pnl_decomp(eq_df)

            print(f"\n[{split_name} | {scen_name}]  run_dir={run_dir}")
            print(
                f"  total_return={metrics.total_return:.4%}  max_dd={metrics.max_drawdown:.4%}  sharpe={metrics.sharpe:.2f}  trades={metrics.trades}"
            )
            print(
                "  pnl_decomp (USD): "
                + ", ".join([f"{k}={v:,.2f}" for k, v in decomp.items()])
            )


if __name__ == "__main__":
    main()
