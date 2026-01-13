from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import pandas as pd
import numpy as np

from atlas.backtest.metrics import compute_metrics
from atlas.backtest.engine import BacktestConfig, BacktestOutputs, BacktestProgress
from atlas.strategies.base import Strategy, StrategyState
from atlas.utils.time import NY_TZ
from atlas.logging_utils import get_logger

logger = get_logger(__name__)

# Constants
TAKER_FEE_BPS = 3.0  # Default 3bps taker
FUNDING_INTERVAL_HOURS = 1  # Approximate accumulation. Real CEX settle every 4h or 8h. 
# We will model funding as continuous accrual or per-bar check if needed.
# For simplicity in this engine: We won't simulate exact 8h timestamps unless critical.
# We will just pass the current funding rate to strategy. 

def run_derivatives_backtest(
    *,
    bars_by_symbol: dict[str, pd.DataFrame],
    strategy: Strategy,
    cfg: BacktestConfig,
    run_dir: Path,
    progress: Optional[Callable[[BacktestProgress], None]] = None,
    progress_interval_s: float = 0.25,
) -> BacktestOutputs:
    """
    Derivatives-specific backtest engine.
    Handles:
    - Margin & Leverage checks
    - Liquidation logic
    - Funding Rate storage/passing (simulated as 0 if not provided, or loaded)
    - Shorting allowed by default
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    decisions_jsonl = run_dir / "decisions.jsonl"

    cash = float(cfg.initial_cash)
    symbols = [s.strip().upper() for s in cfg.symbols if s.strip()]
    if not symbols:
        raise ValueError("cfg.symbols must be non-empty")

    common_index: Optional[pd.DatetimeIndex] = None
    for symbol in symbols:
        if symbol not in bars_by_symbol:
            raise ValueError(f"Missing bars for {symbol}")
        idx = bars_by_symbol[symbol].index
        if common_index is None:
            common_index = idx
        else:
            common_index = common_index.intersection(idx)
    
    if common_index is None or len(common_index) < 3:
        raise ValueError("Insufficient aligned bars")
        
    common_index = common_index.sort_values()
    
    # Reindex bars
    bars_by_symbol = {s: bars_by_symbol[s].loc[common_index].copy() for s in symbols}
    
    # State initialization
    position_qty: dict[str, float] = {s: 0.0 for s in symbols}
    entry_prices: dict[str, float] = {s: 0.0 for s in symbols} # Avg entry
    holding_bars: dict[str, int] = {s: 0 for s in symbols}
    current_targets: dict[str, float] = {s: 0.0 for s in symbols}
    
    # Funding Rate Mock (For now 0.0, or random, or loaded if we had it)
    # Ideally bars_by_symbol should contain funding? Or separate dict?
    # For now, we assume 0.0 funding 
    current_funding_rates = {s: 0.0 for s in symbols} 

    pending_target_exposures: Optional[dict[str, float]] = None
    pending_reason: Optional[str] = None
    
    # Config
    max_notional_cap = float(cfg.max_position_notional_usd)
    slippage_rate = float(cfg.slippage_bps) / 10_000.0
    fee_rate = TAKER_FEE_BPS / 10_000.0

    trades_rows: list[dict] = []
    equity_rows: list[dict] = []
    
    idx = common_index
    current_day: Optional[object] = None
    day_start_equity = float(cfg.initial_cash)
    peak_equity = float(cfg.initial_cash)
    last_progress_t = 0.0

    # Liquidation Params
    MAINTENANCE_MARGIN = 0.05 # 5%
    LIQ_FEE = 0.01 # 1% penalty

    with decisions_jsonl.open("w") as f_decisions:
        for i in range(len(idx)):
            ts = pd.Timestamp(idx[i])
            opens = {s: float(bars_by_symbol[s]["open"].iloc[i]) for s in symbols}
            closes = {s: float(bars_by_symbol[s]["close"].iloc[i]) for s in symbols}

            # Mark to Market Equity
            # Equity = Cash + Unrealized PnL
            unrealized_pnl = 0.0
            total_margin_used = 0.0
            
            for s in symbols:
                qty = position_qty[s]
                if abs(qty) > 1e-9:
                    px = opens[s] # Use open for 'start of bar' check? Or close? 
                    # Usually mark-to-market at close of previous bar ~ open of this bar
                    # For liquidation check, we check against OPEN of this bar before processing orders
                    entry = entry_prices[s]
                    pnl = (px - entry) * qty
                    unrealized_pnl += pnl
                    total_margin_used += abs(qty * px) * MAINTENANCE_MARGIN

            equity = cash + unrealized_pnl
            
            # Liquidation Check
            if equity < total_margin_used: # Simple check: Equity < Maintenance Margin
                 # Liquidate ALL
                 logger.warning(f"LIQUIDATION at {ts}: Equity {equity} < Maint {total_margin_used}")
                 for s in symbols:
                     qty = position_qty[s]
                     if abs(qty) > 1e-9:
                         px = opens[s]
                         # Liquidate at open
                         fill_val = abs(qty) * px
                         trans_cost = fill_val * (slippage_rate + fee_rate + LIQ_FEE)
                         realized_pnl = (px - entry_prices[s]) * qty
                         cash += realized_pnl - trans_cost
                         
                         trades_rows.append({
                             "timestamp": ts.isoformat(),
                             "symbol": s,
                             "side": "SELL" if qty > 0 else "BUY",
                             "qty": abs(qty),
                             "fill_price": px,
                             "notional": fill_val,
                             "cash_after": cash,
                             "position_qty_after": 0.0,
                             "strategy_reason": "LIQUIDATION",
                         })
                         position_qty[s] = 0.0
                         entry_prices[s] = 0.0
                         current_targets[s] = 0.0
                 
                 # Reset equity
                 equity = cash # Cash is now the remaining equity
            
            # Daily stats update
            if current_day != ts.date():
                current_day = ts.date()
                day_start_equity = equity

            # Process Pending Orders (from previous bar's decision)
            if i > 0 and pending_target_exposures is not None:
                targets = pending_target_exposures
                
                # Execution
                for s in symbols:
                    target_pct = targets.get(s, 0.0)
                    
                    # Notional Target based on Equity (Dynamic sizing)
                    # Or Config Max? Usually strategies target % of Equity.
                    # But engine.py uses `target_qty = target_notional / open_px` where target_notional = target * max_notional
                    # And max_notional is cfg.max_position_notional_usd.
                    # This implies fixed max size sizing.
                    # Let's stick to engine.py convention for consistency.
                    
                    target_notional = target_pct * max_notional_cap
                    px = opens[s]
                    if px <= 0: continue
                    
                    desired_qty = target_notional / px
                    current_q = position_qty[s]
                    delta_q = desired_qty - current_q
                    
                    if abs(delta_q) * px < 10.0: # Min trade size $10
                        if abs(desired_qty) < 1e-9 and abs(current_q) > 1e-9:
                             # Closing dust?
                             pass
                        else:
                             # Skip tiny adjustments
                             current_targets[s] = target_pct
                             continue

                    # Execute
                    # Price with slippage
                    fill_px = px * (1.0 + slippage_rate) if delta_q > 0 else px * (1.0 - slippage_rate)
                    
                    # Transaction Cost
                    notional_val = abs(delta_q) * fill_px
                    cost = notional_val * fee_rate # Taker fee
                    
                    # Update Cash:
                    # Logic: Cash isn't used to "buy" the asset in perps. 
                    # Cash is collateral. PnL is settled.
                    # But valid accounting:
                    # Open Long 1 BTC @ 100k. Cash doesn't change (only fee).
                    # Close Long. Cash changes by (Exit - Entry) * Qty - Fee.
                    # To keep compatible with engine.py "Cash" model (which assumes authorized spot or margin):
                    # We will use "Collateral" model.
                    # But engine.py does `cash -= delta * fill_px`. This is Spot accounting.
                    # FOR DERIVATIVES: Cash only changes on Realized PnL and Fees.
                    
                    avg_entry = entry_prices[s]
                    
                    if (current_q > 0 and delta_q < 0) or (current_q < 0 and delta_q > 0):
                        # Closing or Flipping
                        # Amount closed
                        closed_qty = min(abs(current_q), abs(delta_q)) * np.sign(delta_q) # Signed quantity being closed
                        
                        # PnL on closed portion
                        # Long (+q), Sell (-d): PnL = (Fill - Entry) * Abs(closed)
                        # Short (-q), Buy (+d): PnL = (Entry - Fill) * Abs(closed)
                        # Generic: (Fill - Entry) * (-closed) ?? No.
                        # Realized PnL = closed_qty * (Fill - Entry) ??
                        # If Long (pos > 0), closed_qty is negative. 
                        # fill > entry -> Profit. 
                        # -1 * (110 - 100) = -10. Wrong.
                        # We are SELLING. Revenue is +Fill. Cost was Entry.
                        # Standard formulation: 
                        # Realized = (Fill_Price - Entry_Price) * (Qty_Closed for Longs) 
                        #            (Entry_Price - Fill_Price) * (Qty_Closed for Shorts)
                        
                        # Let's use simpler PnL accumulation:
                        # old_val = qty * entry
                        # new_val = qty * current_px
                        # It's path dependent.
                        
                        # Correct Perp Accounting:
                        # Cash balance update = Realized PnL - Fees
                        
                        # Determine realized pnl
                        direction = 1 if current_q > 0 else -1
                        close_amt = min(abs(current_q), abs(delta_q))
                        
                        pnl = 0.0
                        if direction == 1: # Long closing
                             pnl = (fill_px - avg_entry) * close_amt
                        else: # Short closing
                             pnl = (avg_entry - fill_px) * close_amt
                        
                        cash += pnl
                        
                    cash -= cost
                    
                    # Update Position
                    new_qty = current_q + delta_q
                    
                    # Update Avg Entry
                    if new_qty == 0:
                        entry_prices[s] = 0.0
                    elif (current_q >= 0 and delta_q > 0) or (current_q <= 0 and delta_q < 0):
                        # Increasing position -> Weighted Average
                        # total_cost = current_q * current_entry + delta_q * fill_px
                        # new_entry = total_cost / new_qty
                        # NOTE: careful with signs.
                        curr_val = abs(current_q) * entry_prices[s]
                        add_val = abs(delta_q) * fill_px
                        entry_prices[s] = (curr_val + add_val) / abs(new_qty)
                    elif (current_q > 0 and new_qty < 0) or (current_q < 0 and new_qty > 0):
                        # Flipped. Entry is now fill_px for the remainder
                        entry_prices[s] = fill_px
                    else:
                        # Reducing but not flipping -> Entry price stays same!
                        pass
                        
                    position_qty[s] = new_qty
                    current_targets[s] = target_pct
                    
                    trades_rows.append({
                        "timestamp": ts.isoformat(),
                        "symbol": s,
                        "side": "BUY" if delta_q > 0 else "SELL",
                        "qty": abs(delta_q),
                        "fill_price": fill_px,
                        "notional": notional_val,
                        "cash_after": cash,
                        "position_qty_after": new_qty,
                        "strategy_reason": pending_reason,
                    })

                pending_target_exposures = None
                pending_reason = None

            # Calculate Equity for Tracking (Mark to Market at CLOSE)
            unrealized_pnl = 0.0
            for s in symbols:
                qty = position_qty[s]
                if abs(qty) > 1e-9:
                    px = closes[s]
                    entry = entry_prices[s]
                    if qty > 0:
                        unrealized_pnl += (px - entry) * qty
                    else:
                         unrealized_pnl += (entry - px) * abs(qty)
            
            equity = cash + unrealized_pnl
            
            # Record Equity Row
            row = {
                "timestamp": ts.isoformat(),
                "cash": cash,
                "equity": equity,
                "day_start_equity": day_start_equity,
                "day_pnl": equity - day_start_equity,
                "day_return": (equity - day_start_equity)/day_start_equity if day_start_equity else 0.0,
            }
            for s in symbols:
                row[f"{s}_close"] = closes[s]
                row[f"{s}_position_qty"] = position_qty[s]
                row[f"{s}_holding_bars"] = holding_bars[s]
            equity_rows.append(row)

            # Progress
            if progress is not None:
                now = time.monotonic()
                if i == 0 or (now - last_progress_t) >= float(progress_interval_s):
                    last_progress_t = now
                    total_ret = (equity / cfg.initial_cash - 1.0) if cfg.initial_cash else 0.0
                    peak_equity = max(peak_equity, equity)
                    dd = (equity / peak_equity - 1.0) if peak_equity > 0 else 0.0
                    
                    progress(BacktestProgress(
                        i=i+1, n=len(idx), timestamp=ts.isoformat(),
                        cash=cash, equity=equity, day_pnl=row["day_pnl"],
                        day_return=row["day_return"], total_return=total_ret, drawdown=dd,
                        positions={s: position_qty[s] for s in symbols},
                        targets=current_targets,
                        closes=closes,
                        fills=len(trades_rows),
                        last_trade=trades_rows[-1] if trades_rows else None
                    ))

            # Strategy Step
            if i < len(idx) - 1:
                next_ts = pd.Timestamp(idx[i+1])
                
                # Update holding bars
                for s in symbols:
                    if abs(position_qty[s]) > 1e-9:
                        holding_bars[s] += 1
                    else:
                        holding_bars[s] = 0

                state = StrategyState(
                    timestamp=next_ts,
                    allow_short=True, # Always allowed in perps
                    cash=cash,
                    positions={s: position_qty[s] for s in symbols},
                    equity=equity,
                    day_start_equity=day_start_equity,
                    day_pnl=row["day_pnl"],
                    day_return=row["day_return"],
                    holding_bars={s: holding_bars[s] for s in symbols},
                    extra={"funding_rates": current_funding_rates}
                )
                
                # Slice history and run
                history = {s: bars_by_symbol[s].iloc[: i + 1] for s in symbols}
                decision = strategy.target_exposures(history, state)
                
                pending_target_exposures = decision.target_exposures
                pending_reason = decision.reason
                
                # Log decision
                decision_row = {
                    "timestamp": next_ts.isoformat(),
                    "targets": pending_target_exposures,
                    "reason": pending_reason,
                    "debug": decision.debug
                }
                f_decisions.write(json.dumps(decision_row) + "\n")

    # Output generation
    trade_cols = [
        "timestamp", "symbol", "side", "qty", "fill_price", "notional",
        "cash_after", "position_qty_after", "strategy_reason"
    ]
    trades = pd.DataFrame(trades_rows, columns=trade_cols)
    equity_curve = pd.DataFrame(equity_rows)
    if not equity_curve.empty:
        ts = pd.to_datetime(equity_curve["timestamp"], utc=True).dt.tz_convert(NY_TZ)
        equity_curve.index = ts
        equity_curve["timestamp"] = ts # Keep column for metrics compatibility if needed? No, usually index.
        equity_curve = equity_curve.drop(columns=["timestamp"])
        equity_curve.index.name = "timestamp"

    metrics = compute_metrics(equity_curve, trades)
    
    trades.to_csv(run_dir / "trades.csv", index=False)
    trades.to_json(run_dir / "trades.json", orient="records", indent=2)
    equity_curve.to_csv(run_dir / "equity_curve.csv", index=True)
    (run_dir / "metrics.json").write_text(json.dumps(metrics.to_dict(), indent=2))
    
    logger.info(f"Derivatives backtest done. Final Eq: {equity:.2f}")

    return BacktestOutputs(
        run_dir=run_dir,
        trades_csv=run_dir / "trades.csv",
        trades_json=run_dir / "trades.json",
        equity_curve_csv=run_dir / "equity_curve.csv",
        metrics_json=run_dir / "metrics.json",
        decisions_jsonl=run_dir / "decisions.jsonl",
    )
