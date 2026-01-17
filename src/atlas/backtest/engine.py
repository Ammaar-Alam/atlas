from __future__ import annotations

import json
import logging
import time
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import pandas as pd

from atlas.backtest.metrics import compute_metrics
from atlas.strategies.base import Strategy, StrategyState
from atlas.utils.time import NY_TZ

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BacktestConfig:
    symbols: list[str]
    initial_cash: float
    max_position_notional_usd: float
    slippage_bps: float
    allow_short: bool
    taker_fee_bps: float = 3.0
    maintenance_margin_rate: float = 0.05
    liquidation_fee_rate: float = 0.01


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
    debug: bool = False,
    output_mode: str = "full",
) -> BacktestOutputs:
    mode = (output_mode or "full").strip().lower()
    if mode not in {"full", "minimal"}:
        raise ValueError(f"unsupported output_mode: {output_mode!r} (expected 'full' or 'minimal')")

    run_dir.mkdir(parents=True, exist_ok=True)
    decisions_jsonl = run_dir / "decisions.jsonl"
    trade_debug_jsonl = run_dir / "trade_debug.jsonl"
    write_decisions = (mode == "full")
    write_trades_json = (mode == "full")
    write_metrics_json = (mode == "full")

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

    open_by_symbol = {s: bars_by_symbol[s]["open"].to_numpy(copy=False) for s in symbols}
    close_by_symbol = {s: bars_by_symbol[s]["close"].to_numpy(copy=False) for s in symbols}

    position_qty: dict[str, float] = {s: 0.0 for s in symbols}
    holding_bars: dict[str, int] = {s: 0 for s in symbols}
    current_targets: dict[str, float] = {s: 0.0 for s in symbols}
    pending_target_exposures: Optional[dict[str, float]] = None
    pending_reason: Optional[str] = None
    pending_decision_context: Optional[dict[str, Any]] = None

    max_notional = float(cfg.max_position_notional_usd)
    slippage = float(cfg.slippage_bps) / 10_000.0

    trades_rows: list[dict] = []
    equity_rows: list[dict] = []

    idx = common_index
    current_day: Optional[object] = None
    day_start_equity = float(cfg.initial_cash)
    peak_equity = float(cfg.initial_cash)
    last_progress_t = 0.0

    def _bar_snapshot(symbol: str, *, row_i: int) -> dict[str, float]:
        df = bars_by_symbol[symbol]
        cols = ["open", "high", "low", "close", "volume", "funding_rate"]
        out: dict[str, float] = {}
        for col in cols:
            if col not in df.columns:
                continue
            try:
                out[col] = float(df[col].iloc[row_i])
            except Exception:
                continue
        return out

    with ExitStack() as stack:
        f_decisions = (
            stack.enter_context(decisions_jsonl.open("w")) if write_decisions else None
        )
        f_trade_debug = (
            stack.enter_context(trade_debug_jsonl.open("w")) if debug else None
        )

        for i in range(len(idx)):
            ts = pd.Timestamp(idx[i])
            opens = {s: float(open_by_symbol[s][i]) for s in symbols}
            closes = {s: float(close_by_symbol[s][i]) for s in symbols}

            execution_bars = (
                {s: _bar_snapshot(s, row_i=i) for s in symbols} if debug else {}
            )

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

                    trade_row = {
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
                    trades_rows.append(trade_row)

                    if f_trade_debug is not None:
                        f_trade_debug.write(
                            json.dumps(
                                {
                                    "timestamp": ts.isoformat(),
                                    "trade": trade_row,
                                    "decision": pending_decision_context,
                                    "execution": {
                                        "timestamp": ts.isoformat(),
                                        "bars": execution_bars,
                                    },
                                }
                            )
                            + "\n"
                        )

                for s, delta, tgt in sells:
                    _exec(s, delta, tgt)
                for s, delta, tgt in buys:
                    _exec(s, delta, tgt)

                pending_target_exposures = None
                pending_reason = None
                pending_decision_context = None

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
                extra={
                    "max_position_notional_usd": float(cfg.max_position_notional_usd),
                    "slippage_bps": float(cfg.slippage_bps),
                    "taker_fee_bps": float(cfg.taker_fee_bps),
                },
            )
            history = {s: bars_by_symbol[s].iloc[: i + 1] for s in symbols}
            decision = strategy.target_exposures(history, state)

            next_targets = {s: float(decision.target_exposures.get(s, 0.0)) for s in symbols}
            if not cfg.allow_short:
                next_targets = {s: max(0.0, v) for s, v in next_targets.items()}

            decision_row = None
            if write_decisions or debug:
                decision_row = {
                    "timestamp": pd.Timestamp(state.timestamp).isoformat(),
                    "targets": next_targets,
                    "reason": decision.reason,
                    "debug": decision.debug,
                }
                if debug:
                    decision_row["snapshot"] = {
                        "signal_bar_end_timestamp": ts.isoformat(),
                        "signal_bars": {s: _bar_snapshot(s, row_i=i) for s in symbols},
                        "state": {
                            "cash": float(cash),
                            "equity": float(equity),
                            "day_start_equity": float(day_start_equity),
                            "day_pnl": float(day_pnl),
                            "day_return": float(day_return),
                            "positions": {s: float(position_qty[s]) for s in symbols},
                            "holding_bars": {s: int(holding_bars[s]) for s in symbols},
                            "current_targets": {
                                s: float(current_targets.get(s, 0.0)) for s in symbols
                            },
                            "config": {
                                "max_position_notional_usd": float(cfg.max_position_notional_usd),
                                "slippage_bps": float(cfg.slippage_bps),
                                "allow_short": bool(cfg.allow_short),
                            },
                        },
                    }
                if f_decisions is not None:
                    f_decisions.write(json.dumps(decision_row) + "\n")

            if any(abs(next_targets[s] - current_targets.get(s, 0.0)) > 1e-8 for s in symbols):
                pending_target_exposures = next_targets
                pending_reason = decision.reason
                if debug:
                    pending_decision_context = {
                        "timestamp": (decision_row or {}).get("timestamp"),
                        "reason": decision.reason,
                        "targets": next_targets,
                        "debug": decision.debug,
                        "snapshot": (decision_row or {}).get("snapshot"),
                    }

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
    ts = pd.to_datetime(equity_curve["timestamp"], errors="raise", utc=True).dt.tz_convert(NY_TZ)
    equity_curve = equity_curve.drop(columns=["timestamp"])
    equity_curve.index = ts
    equity_curve.index.name = "timestamp"
    equity_curve = equity_curve.sort_index()

    trades_csv = run_dir / "trades.csv"
    trades_json = run_dir / "trades.json"
    equity_curve_csv = run_dir / "equity_curve.csv"
    metrics_json = run_dir / "metrics.json"

    trades.to_csv(trades_csv, index=False)
    equity_curve.to_csv(equity_curve_csv, index=True)
    if write_trades_json:
        trades.to_json(trades_json, orient="records", indent=2)
    if write_metrics_json:
        metrics = compute_metrics(equity_curve, trades)
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
