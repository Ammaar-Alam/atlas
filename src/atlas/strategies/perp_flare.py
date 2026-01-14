from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from atlas.strategies.base import Strategy, StrategyDecision, StrategyState


@dataclass
class PerpFlare(Strategy):
    """
    ER-Gated Donchian Breakout with strict admission control and liquidation buffers.
    Target: BTC-PERP, ETH-PERP.
    """

    symbols: tuple[str, ...]

    # Feature params
    atr_window: int = 14
    ema_fast: int = 12
    ema_slow: int = 24
    er_window: int = 10
    breakout_window: int = 20

    # Gates
    er_min: float = 0.35

    # Environment assumptions (bps). In backtests/paper, these may be overridden
    # via StrategyState.extra to match the engine's simulated costs.
    taker_fee_bps: float = 3.0
    half_spread_bps: float = 1.0
    base_slippage_bps: float = 1.5

    # Admission
    edge_floor_bps: float = 5.0
    k_cost: float = 1.5

    # Risk / Sizing
    risk_per_trade: float = 0.01
    stop_atr_mult: float = 2.0
    trail_atr_mult: float = 3.0
    max_margin_utilization: float = 0.65
    max_leverage: float = 10.0

    # Liquidation buffer (environment assumption; may be overridden via StrategyState.extra)
    maintenance_margin_rate: float = 0.05
    min_liq_buffer_atr: float = 3.0

    def warmup_bars(self) -> int:
        return max(
            self.atr_window,
            self.ema_slow,
            self.er_window,
            self.breakout_window,
        ) + 10

    def target_exposures(
        self,
        bars_by_symbol: dict[str, pd.DataFrame],
        state: StrategyState,
    ) -> StrategyDecision:
        # 0. Prep
        equity = state.equity
        positions = state.positions
        extra = dict(state.extra or {})
        current_funding = dict(extra.get("funding_rates") or {})

        max_position_notional_usd = float(extra.get("max_position_notional_usd") or 0.0)
        slippage_bps = float(
            extra.get("slippage_bps")
            if extra.get("slippage_bps") is not None
            else (self.half_spread_bps + self.base_slippage_bps)
        )
        taker_fee_bps = float(extra.get("taker_fee_bps") or self.taker_fee_bps)
        maintenance_margin_rate = float(extra.get("maintenance_margin_rate") or self.maintenance_margin_rate)

        target_exposures = {s: 0.0 for s in self.symbols}
        decision_meta: dict[str, object] = {}

        # 1. Compute Features per symbol
        candidates = []
        
        for sym in self.symbols:
            df = bars_by_symbol.get(sym)
            if df is None or len(df) < self.warmup_bars():
                continue
                
            close = df["close"].values
            high = df["high"].values
            low = df["low"].values
            
            # --- Features ---
            
            # ATR
            tr1 = high - low
            tr2 = np.abs(high - np.roll(close, 1))
            tr3 = np.abs(low - np.roll(close, 1))
            tr = np.maximum(tr1, np.maximum(tr2, tr3))
            atr = pd.Series(tr).rolling(self.atr_window).mean().values
            atr_val = atr[-1]
            atr_bps = (atr_val / close[-1]) * 10000
            
            # EMAs
            ema_f = pd.Series(close).ewm(span=self.ema_fast, adjust=False).mean().values[-1]
            ema_s = pd.Series(close).ewm(span=self.ema_slow, adjust=False).mean().values[-1]
            
            # Trend Score
            trend_score = (ema_f - ema_s) / (atr_val if atr_val > 0 else 1.0)
            
            # Efficiency Ratio (ER)
            change = np.abs(close[-1] - close[-1 - self.er_window])
            volatility = np.sum(np.abs(np.diff(close[-self.er_window - 1:])))
            er = change / volatility if volatility > 0 else 0.0
            
            # Breakout Bands (strictly strictly prior to current bar? 
            # Reference says "exclude current bar" for signal, but for live trading bar -1 is the just-closed bar)
            hh = np.max(high[-self.breakout_window-1:-1]) 
            ll = np.min(low[-self.breakout_window-1:-1])
            
            # 2. Gate & Signal
            
            c_t = close[-1]
            
            # Directional Setup
            long_signal = (c_t > hh) and (trend_score > 0) and (er >= self.er_min)
            short_signal = (c_t < ll) and (trend_score < 0) and (er >= self.er_min)
            
            if not long_signal and not short_signal:
                continue
                
            # Breakout Strength
            b_long = 10000 * max(0, (c_t - hh) / c_t)
            b_short = 10000 * max(0, (ll - c_t) / c_t)
            
            # Edge Proxy
            edge = (b_long if long_signal else b_short) + 0.5 * abs(trend_score) * atr_bps
            
            # Friction Estimate
            funding_rate = current_funding.get(sym, 0.0) # Assume e.g. 0.0001 for 0.01%
            # Lambda_f approx 0.0?? Let's assume passed in or default. 
            # Reference: lambda_f * |f_t| * 10^4
            # We'll just assume a penalty multiplier of 100 for now (conservative)
            friction = 2 * (slippage_bps + taker_fee_bps) + 100 * abs(funding_rate) * 10000
            
            # Admission
            required_edge = self.edge_floor_bps + self.k_cost * friction
            
            decision_meta[sym] = {
                "edge": edge,
                "friction": friction,
                "er": er,
                "trend": trend_score,
                "atr": atr_val,
                "close": c_t
            }
            
            if edge < required_edge:
                continue
                
            # Relative score
            score = edge - self.k_cost * friction
            direction = 1 if long_signal else -1
            
            candidates.append({
                "symbol": sym,
                "score": score,
                "dir": direction,
                "atr": atr_val,
                "price": c_t,
                "funding": funding_rate
            })

        # 3. Selection & Sizing
        # Simplify: Pick Top 1 for now (or top N fits)
        # Reference: "Pick highest Score; otherwise abstain" implies single selection

        # Actually, existing positions might need to be managed (trailing stops). 
        # Strategy interface implies "target state". 
        
        # If we have an existing position:
        for sym, qty in positions.items():
            if abs(qty) > 0:
                # Check hold/exit logic
                # Trailing stop?
                # For now, let's keep it simple: If we are not entering, do we exit?
                # The reference has: "Daily loss limit... Cooldown... Time stop"
                # For this MVP, let's assume we hold unless stopped out (handled by engine) or signal flips?
                # Or better: regenerate target every bar.
                if sym not in [c["symbol"] for c in candidates]:
                     # No new entry signal. Do we hold?
                     # A real system would track state. 
                     # For MVP: If no signal, we exit. (Simple breakout logic usually implies "in or out")
                     # Note: Reference "Time stop... Trailing stop".
                     # This implies stateful management.
                     # We will rely on re-entry signal persistence for now 
                     # OR allow holding if no counter-signal?
                     pass
        
        if candidates:
            # Sort by score desc
            candidates.sort(key=lambda x: x["score"], reverse=True)
            best = candidates[0]
            
            sym = best["symbol"]
            price = best["price"]
            atr = best["atr"]
            direction = best["dir"]
            
            # Risk limits
            risk_usd = equity * self.risk_per_trade
            stop_dist = self.stop_atr_mult * atr

            if stop_dist <= 0 or price <= 0 or risk_usd <= 0:
                return StrategyDecision(target_exposures)

            desired_qty_base = risk_usd / stop_dist
            desired_notional = desired_qty_base * price

            # Risk cap from equity/leverage rules.
            risk_cap_notional = equity * self.max_leverage * self.max_margin_utilization
            desired_notional = min(desired_notional, risk_cap_notional)

            # Engine-level cap (StrategyDecision exposures are interpreted as a
            # multiplier on cfg.max_position_notional_usd).
            if max_position_notional_usd <= 0:
                return StrategyDecision(target_exposures)
            desired_notional = min(desired_notional, max_position_notional_usd)
            desired_qty_base = desired_notional / price if price > 0 else 0.0

            # Liquidation Buffer Check
            # Long Liq: P_liq = (qP - E) / (q(1-mmr)) ??
            # Simplified: Ensure we are far from potential liquidation
            mmr = maintenance_margin_rate
            
            # Approx Liq Price (Isolated logic roughly)
            # Long: Entry - (Equity / Qty) ... relative to margin
            # Let's use the formula from spec:
            # Long: P_liq ~= (q*P - E_allocated_roughly?) 
            # The spec formula P_liq = (qP - E)/q(1-mmr) assumes cross logic where E is total equity?
            # Yes, standard cross margin.
            
            if desired_qty_base <= 0:
                return StrategyDecision(target_exposures)

            if direction == 1:
                liq_price = (desired_qty_base * price - equity) / (desired_qty_base * (1 - mmr))
                # If negative (fully solvent), liq is 0 (or lower)
                if liq_price < 0: liq_price = 0
                dist_to_liq = price - liq_price
            else:
                # Short
                liq_price = (equity + desired_qty_base * price) / (desired_qty_base * (1 + mmr))
                dist_to_liq = liq_price - price
            
            if dist_to_liq < (self.min_liq_buffer_atr * atr):
                # Too risky, reduce size or abort
                # Simple abort according to spec "Gate"
                target_exposures[sym] = 0.0
            else:
                # Convert notional into target exposure for the engines.
                exposure = desired_notional / max_position_notional_usd if max_position_notional_usd > 0 else 0.0
                target_exposures[sym] = float(exposure) * float(direction)

        return StrategyDecision(target_exposures)
