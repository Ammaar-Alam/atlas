from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from atlas.backtest.metrics import compute_metrics
from atlas.strategies.base import Strategy

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BacktestConfig:
    symbol: str
    initial_cash: float
    max_position_notional_usd: float
    slippage_bps: float
    allow_short: bool


@dataclass(frozen=True)
class BacktestOutputs:
    run_dir: Path
    trades_csv: Path
    trades_json: Path
    equity_curve_csv: Path
    metrics_json: Path


def run_backtest(
    *,
    bars: pd.DataFrame,
    strategy: Strategy,
    cfg: BacktestConfig,
    run_dir: Path,
) -> BacktestOutputs:
    run_dir.mkdir(parents=True, exist_ok=True)

    cash = float(cfg.initial_cash)
    position_qty = 0.0
    current_exposure = 0.0
    pending_target_exposure: Optional[float] = None
    pending_reason: Optional[str] = None

    slippage = float(cfg.slippage_bps) / 10_000.0

    trades_rows: list[dict] = []
    equity_rows: list[dict] = []

    idx = bars.index
    if len(idx) < 3:
        raise ValueError("need at least 3 bars to run a backtest")

    for i in range(len(bars)):
        ts = idx[i]
        open_px = float(bars["open"].iloc[i])
        close_px = float(bars["close"].iloc[i])

        if i > 0 and pending_target_exposure is not None:
            target_exposure = pending_target_exposure
            if not cfg.allow_short and target_exposure < 0:
                target_exposure = 0.0

            target_notional = target_exposure * float(cfg.max_position_notional_usd)
            target_qty = target_notional / open_px if open_px > 0 else 0.0
            delta_qty = target_qty - position_qty

            if abs(delta_qty) > 1e-8:
                side = "BUY" if delta_qty > 0 else "SELL"
                fill_px = open_px * (1.0 + slippage) if delta_qty > 0 else open_px * (1.0 - slippage)

                if delta_qty > 0:
                    max_affordable = cash / fill_px if fill_px > 0 else 0.0
                    if delta_qty > max_affordable:
                        delta_qty = max_affordable
                        target_qty = position_qty + delta_qty

                cash -= delta_qty * fill_px
                position_qty = target_qty
                current_exposure = (
                    (position_qty * open_px) / float(cfg.max_position_notional_usd)
                    if float(cfg.max_position_notional_usd) > 0
                    else 0.0
                )

                trades_rows.append(
                    {
                        "timestamp": ts.isoformat(),
                        "symbol": cfg.symbol,
                        "side": side,
                        "qty": float(abs(delta_qty)),
                        "fill_price": float(fill_px),
                        "notional": float(abs(delta_qty) * fill_px),
                        "cash_after": float(cash),
                        "position_qty_after": float(position_qty),
                        "strategy_reason": pending_reason,
                    }
                )
            else:
                current_exposure = target_exposure

            pending_target_exposure = None
            pending_reason = None

        equity = cash + position_qty * close_px
        equity_rows.append(
            {
                "timestamp": ts.isoformat(),
                "symbol": cfg.symbol,
                "close": close_px,
                "cash": float(cash),
                "position_qty": float(position_qty),
                "equity": float(equity),
            }
        )

        if i == len(bars) - 1:
            break

        decision = strategy.target_exposure(bars.iloc[: i + 1])
        next_exposure = float(decision.target_exposure)
        if not cfg.allow_short and next_exposure < 0:
            next_exposure = 0.0

        if abs(next_exposure - current_exposure) > 1e-8:
            pending_target_exposure = next_exposure
            pending_reason = decision.reason

    trades = pd.DataFrame(trades_rows)
    equity_curve = pd.DataFrame(equity_rows)
    equity_curve["timestamp"] = pd.to_datetime(equity_curve["timestamp"], errors="raise")
    equity_curve = equity_curve.set_index("timestamp").sort_index()

    metrics = compute_metrics(equity_curve, trades)

    trades_csv = run_dir / "trades.csv"
    trades_json = run_dir / "trades.json"
    equity_curve_csv = run_dir / "equity_curve.csv"
    metrics_json = run_dir / "metrics.json"

    trades.to_csv(trades_csv, index=False)
    trades.to_json(trades_json, orient="records", indent=2)
    equity_curve.to_csv(equity_curve_csv, index=True)
    metrics_json.write_text(json.dumps(metrics.to_dict(), indent=2))

    logger.info("backtest done: final equity %.2f", float(equity_curve['equity'].iloc[-1]))
    logger.info("outputs: %s", run_dir)

    return BacktestOutputs(
        run_dir=run_dir,
        trades_csv=trades_csv,
        trades_json=trades_json,
        equity_curve_csv=equity_curve_csv,
        metrics_json=metrics_json,
    )
