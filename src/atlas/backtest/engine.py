from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import pandas as pd

from atlas.backtest.metrics import compute_metrics
from atlas.strategies.base import Strategy, StrategyState

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BacktestConfig:
    symbols: list[str]
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
    decisions_jsonl: Path


@dataclass(frozen=True)
class BacktestProgress:
    i: int
    n: int
    timestamp: str
    cash: float
    equity: float
    day_pnl: float
    day_return: float
    total_return: float
    drawdown: float
    positions: dict[str, float]
    targets: dict[str, float]
    closes: dict[str, float]
    fills: int
    last_trade: Optional[dict[str, Any]] = None


def run_backtest(
    *,
    bars_by_symbol: dict[str, pd.DataFrame],
    strategy: Strategy,
    cfg: BacktestConfig,
    run_dir: Path,
    progress: Optional[Callable[[BacktestProgress], None]] = None,
    progress_interval_s: float = 0.25,
) -> BacktestOutputs:
    run_dir.mkdir(parents=True, exist_ok=True)
    decisions_jsonl = run_dir / "decisions.jsonl"

    cash = float(cfg.initial_cash)
    symbols = [s.strip().upper() for s in cfg.symbols if s.strip()]
    if not symbols:
        raise ValueError("cfg.symbols must be non-empty")

    missing = [s for s in symbols if s not in bars_by_symbol]
    if missing:
        raise ValueError(f"bars missing symbols: {missing}")

    common_index: Optional[pd.DatetimeIndex] = None
    for symbol in symbols:
        idx = bars_by_symbol[symbol].index
        if common_index is None:
            common_index = idx
        else:
            common_index = common_index.intersection(idx)
    if common_index is None:
        raise ValueError("no bars provided")
    common_index = common_index.sort_values()
    if len(common_index) < 3:
        raise ValueError("need at least 3 aligned bars to run a backtest")

    bars_by_symbol = {
        s: bars_by_symbol[s].loc[common_index].copy() for s in symbols
    }

    position_qty: dict[str, float] = {s: 0.0 for s in symbols}
    holding_bars: dict[str, int] = {s: 0 for s in symbols}
    current_targets: dict[str, float] = {s: 0.0 for s in symbols}
    pending_target_exposures: Optional[dict[str, float]] = None
    pending_reason: Optional[str] = None

    max_notional = float(cfg.max_position_notional_usd)
    slippage = float(cfg.slippage_bps) / 10_000.0

    trades_rows: list[dict] = []
    equity_rows: list[dict] = []

    idx = common_index
    current_day: Optional[object] = None
    day_start_equity = float(cfg.initial_cash)
    peak_equity = float(cfg.initial_cash)
    last_progress_t = 0.0

    with decisions_jsonl.open("w") as f_decisions:
        for i in range(len(idx)):
            ts = pd.Timestamp(idx[i])
            opens = {s: float(bars_by_symbol[s]["open"].iloc[i]) for s in symbols}
            closes = {s: float(bars_by_symbol[s]["close"].iloc[i]) for s in symbols}

            if current_day != ts.date():
                current_day = ts.date()
                day_start_equity = cash + sum(
                    position_qty[s] * opens[s] for s in symbols
                )

            if i > 0 and pending_target_exposures is not None:
                targets: dict[str, float] = {
                    s: float(pending_target_exposures.get(s, 0.0)) for s in symbols
                }
                if not cfg.allow_short:
                    targets = {s: max(0.0, v) for s, v in targets.items()}

                orders: list[tuple[str, float, float]] = []
                for s in symbols:
                    open_px = opens[s]
                    target_notional = targets[s] * max_notional
                    target_qty = target_notional / open_px if open_px > 0 else 0.0
                    delta_qty = target_qty - position_qty[s]
                    if abs(delta_qty) > 1e-8:
                        orders.append((s, delta_qty, target_qty))
                    else:
                        current_targets[s] = float(targets[s])

                sells = [o for o in orders if o[1] < 0]
                buys = [o for o in orders if o[1] > 0]

                def _exec(symbol: str, delta_qty: float, target_qty: float) -> None:
                    nonlocal cash
                    open_px = opens[symbol]
                    side = "BUY" if delta_qty > 0 else "SELL"
                    fill_px = (
                        open_px * (1.0 + slippage)
                        if delta_qty > 0
                        else open_px * (1.0 - slippage)
                    )

                    if delta_qty > 0:
                        max_affordable = cash / fill_px if fill_px > 0 else 0.0
                        if delta_qty > max_affordable:
                            delta_qty = max_affordable
                            target_qty = position_qty[symbol] + delta_qty

                    cash -= delta_qty * fill_px
                    position_qty[symbol] = target_qty
                    current_targets[symbol] = (
                        (position_qty[symbol] * open_px) / max_notional
                        if max_notional > 0
                        else 0.0
                    )

                    trades_rows.append(
                        {
                            "timestamp": ts.isoformat(),
                            "symbol": symbol,
                            "side": side,
                            "qty": float(abs(delta_qty)),
                            "fill_price": float(fill_px),
                            "notional": float(abs(delta_qty) * fill_px),
                            "cash_after": float(cash),
                            "position_qty_after": float(position_qty[symbol]),
                            "strategy_reason": pending_reason,
                        }
                    )

                for s, delta, tgt in sells:
                    _exec(s, delta, tgt)
                for s, delta, tgt in buys:
                    _exec(s, delta, tgt)

                pending_target_exposures = None
                pending_reason = None

            equity = cash + sum(position_qty[s] * closes[s] for s in symbols)
            day_pnl = float(equity - day_start_equity)
            day_return = float(day_pnl / day_start_equity) if day_start_equity > 0 else 0.0
            peak_equity = max(peak_equity, float(equity))
            drawdown = float((float(equity) / peak_equity) - 1.0) if peak_equity > 0 else 0.0

            for s in symbols:
                if abs(position_qty[s]) > 1e-8:
                    holding_bars[s] = holding_bars.get(s, 0) + 1
                else:
                    holding_bars[s] = 0

            row: dict[str, object] = {
                "timestamp": ts.isoformat(),
                "cash": float(cash),
                "equity": float(equity),
                "day_start_equity": float(day_start_equity),
                "day_pnl": float(day_pnl),
                "day_return": float(day_return),
            }
            for s in symbols:
                row[f"{s}_close"] = float(closes[s])
                row[f"{s}_position_qty"] = float(position_qty[s])
                row[f"{s}_holding_bars"] = int(holding_bars[s])
            equity_rows.append(row)

            if progress is not None:
                now = time.monotonic()
                if i == 0 or (now - last_progress_t) >= float(progress_interval_s):
                    last_progress_t = now
                    total_return = (
                        float(equity) / float(cfg.initial_cash) - 1.0
                        if float(cfg.initial_cash) > 0
                        else 0.0
                    )
                    last_trade = trades_rows[-1] if trades_rows else None
                    progress(
                        BacktestProgress(
                            i=int(i + 1),
                            n=int(len(idx)),
                            timestamp=ts.isoformat(),
                            cash=float(cash),
                            equity=float(equity),
                            day_pnl=float(day_pnl),
                            day_return=float(day_return),
                            total_return=float(total_return),
                            drawdown=float(drawdown),
                            positions={s: float(position_qty[s]) for s in symbols},
                            targets={s: float(current_targets.get(s, 0.0)) for s in symbols},
                            closes={s: float(closes[s]) for s in symbols},
                            fills=int(len(trades_rows)),
                            last_trade=last_trade,
                        )
                    )

            if i == len(idx) - 1:
                break

            next_open_ts = pd.Timestamp(idx[i + 1])
            state = StrategyState(
                timestamp=next_open_ts,
                allow_short=cfg.allow_short,
                cash=float(cash),
                positions={s: float(position_qty[s]) for s in symbols},
                equity=float(equity),
                day_start_equity=float(day_start_equity),
                day_pnl=float(day_pnl),
                day_return=float(day_return),
                holding_bars={s: int(holding_bars[s]) for s in symbols},
            )
            history = {s: bars_by_symbol[s].iloc[: i + 1] for s in symbols}
            decision = strategy.target_exposures(history, state)

            next_targets = {s: float(decision.target_exposures.get(s, 0.0)) for s in symbols}
            if not cfg.allow_short:
                next_targets = {s: max(0.0, v) for s, v in next_targets.items()}

            decision_row = {
                "timestamp": pd.Timestamp(state.timestamp).isoformat(),
                "targets": next_targets,
                "reason": decision.reason,
                "debug": decision.debug,
            }
            f_decisions.write(json.dumps(decision_row) + "\n")

            if any(abs(next_targets[s] - current_targets.get(s, 0.0)) > 1e-8 for s in symbols):
                pending_target_exposures = next_targets
                pending_reason = decision.reason

    trade_cols = [
        "timestamp",
        "symbol",
        "side",
        "qty",
        "fill_price",
        "notional",
        "cash_after",
        "position_qty_after",
        "strategy_reason",
    ]
    trades = pd.DataFrame(trades_rows, columns=trade_cols)
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
        decisions_jsonl=decisions_jsonl,
    )
