from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd

from atlas.strategies.base import Strategy, StrategyDecision, StrategyState


def _estimate_dt_hours(index: pd.DatetimeIndex) -> float:
    if len(index) < 2:
        return 1.0 / 60.0
    dt = (pd.Timestamp(index[-1]) - pd.Timestamp(index[-2])).total_seconds() / 3600.0
    if dt <= 0:
        return 1.0 / 60.0
    return float(dt)


def _to_float_series(values: pd.Series) -> np.ndarray:
    arr = values.astype(float).to_numpy(copy=False)
    return np.asarray(arr, dtype=float)


def _rolling_log_return_sigma(prices: np.ndarray, *, window: int) -> float:
    if window <= 1:
        return 0.0
    if prices.size < window + 1:
        return 0.0
    tail = prices[-(window + 1) :]
    if np.any(tail <= 0):
        return 0.0
    rets = np.diff(np.log(tail))
    if rets.size < 2:
        return 0.0
    return float(np.std(rets, ddof=0))


def _rolling_std(values: np.ndarray, *, window: int) -> float:
    if window <= 1:
        return 0.0
    if values.size < window:
        return 0.0
    tail = values[-window:]
    if tail.size < 2:
        return 0.0
    return float(np.std(tail, ddof=0))


def _safe_div(n: float, d: float) -> float:
    return float(n / d) if d else 0.0


@dataclass
class BasisCarry(Strategy):
    """
    Market-neutral Basis & Carry (Cash-and-Carry) hedge:

    - Long "spot" leg (spot_symbol)
    - Short perp leg (perp_symbol)

    Optional: when allow_reverse=True, can flip to reverse cash-and-carry
    (short spot / long perp) in negative funding regimes.

    Assumptions:
    - Bars for both legs are aligned (engine intersects timestamps).
    - Perp funding is provided either as a `funding_rate` column (preferred) or
      via StrategyState.extra["funding_rates"][perp_symbol]. Funding is treated
      as an hourly rate (fraction per hour), consistent with derivatives_engine.
    """

    spot_symbol: str
    perp_symbol: str

    # Signal / edge
    funding_ema_alpha: float = 0.20
    funding_entry_bps_per_day: float = 10.0
    funding_exit_bps_per_day: float = 0.0
    edge_horizon_hours: float = 8.0
    min_basis_bps: float = 5.0
    min_basis_exit_bps: float = 0.0
    basis_mean_bps: float = 0.0
    basis_halflife_hours: float = 24.0
    basis_momentum_window_bars: int = 30
    max_basis_widening_bps_per_hour: float = 10.0
    basis_vol_window_bars: int = 120
    lambda_basis_vol: float = 1.0
    edge_saturation_bps: float = 50.0

    # Risk / sizing
    collateral_buffer_frac: float = 0.10
    z_sigma_daily: float = 3.0
    spot_vol_window_bars: int = 120
    max_leverage: float = 3.0
    max_margin_utilization: float = 0.50
    maintenance_margin_rate: float = 0.05

    # Turnover control
    rebalance_drift_frac: float = 0.02
    rebalance_min_notional_usd: float = 100.0
    min_trade_notional_usd: float = 200.0

    # Regime
    allow_reverse: bool = False
    require_funding_rate: bool = False

    def warmup_bars(self) -> int:
        return (
            max(
                int(self.basis_momentum_window_bars) + 2,
                int(self.basis_vol_window_bars) + 2,
                int(self.spot_vol_window_bars) + 2,
                30,
            )
            + 5
        )

    def target_exposures(
        self, bars_by_symbol: dict[str, pd.DataFrame], state: StrategyState
    ) -> StrategyDecision:
        symbols = (self.spot_symbol, self.perp_symbol)
        target = {self.spot_symbol: 0.0, self.perp_symbol: 0.0}

        spot_df = bars_by_symbol.get(self.spot_symbol)
        perp_df = bars_by_symbol.get(self.perp_symbol)
        if spot_df is None or perp_df is None:
            return StrategyDecision(target, reason="missing bars")
        if len(spot_df) < self.warmup_bars() or len(perp_df) < self.warmup_bars():
            return StrategyDecision(target, reason="warmup")

        # Strategy requires the ability to hold a short perp (or reverse short spot).
        if not bool(state.allow_short):
            return StrategyDecision(target, reason="short disabled")

        extra = dict(state.extra or {})
        max_position_notional_usd = float(extra.get("max_position_notional_usd") or 0.0)
        if max_position_notional_usd <= 0:
            return StrategyDecision(target, reason="max_notional=0")

        # Align close series to a common time index (defensive; engine typically already aligned).
        spot_close_s = spot_df["close"]
        perp_close_s = perp_df["close"]
        spot_close_s, perp_close_s = spot_close_s.align(perp_close_s, join="inner")
        if len(spot_close_s) < self.warmup_bars():
            return StrategyDecision(target, reason="insufficient overlap")

        idx = spot_close_s.index
        dt_hours = _estimate_dt_hours(idx)
        horizon_hours = max(float(self.edge_horizon_hours), dt_hours)
        horizon_bars = max(1, int(round(horizon_hours / dt_hours))) if dt_hours > 0 else 1

        spot_close = _to_float_series(spot_close_s)
        perp_close = _to_float_series(perp_close_s)
        if spot_close[-1] <= 0 or perp_close[-1] <= 0:
            return StrategyDecision(target, reason="bad prices")

        basis = (perp_close - spot_close) / spot_close
        b_t = float(basis[-1])
        basis_bps = float(b_t * 10000.0)

        # Funding series (preferred) or a last-known current rate from engine context.
        current_funding_rates = dict(extra.get("funding_rates") or {})
        fallback_funding = float(current_funding_rates.get(self.perp_symbol, 0.0) or 0.0)
        has_funding_history = "funding_rate" in perp_df.columns
        if has_funding_history:
            funding_s = perp_df["funding_rate"]
            funding_s = funding_s.reindex(idx).fillna(0.0)
            funding = _to_float_series(funding_s)
        else:
            funding = np.full_like(spot_close, fallback_funding, dtype=float)

        has_any_funding = has_funding_history or (self.perp_symbol in current_funding_rates)
        if self.require_funding_rate and not has_any_funding:
            return StrategyDecision(target, reason="funding missing")

        alpha = float(self.funding_ema_alpha)
        if not (0.0 < alpha <= 1.0):
            alpha = 0.20
        funding_ema = float(pd.Series(funding).ewm(alpha=alpha, adjust=False).mean().iloc[-1])
        funding_ema_bps_per_day = funding_ema * 24.0 * 10000.0

        # Basis momentum (bps/hour): positive means basis widening (perp richer vs spot).
        mom_w = int(self.basis_momentum_window_bars)
        if mom_w > 0 and basis.size > mom_w:
            basis_mom_per_hour = (float(basis[-1]) - float(basis[-1 - mom_w])) / (
                float(mom_w) * float(dt_hours)
            )
        else:
            basis_mom_per_hour = 0.0
        basis_mom_bps_per_hour = basis_mom_per_hour * 10000.0

        # Basis volatility over horizon (bps).
        vol_w = int(self.basis_vol_window_bars)
        if vol_w > 1 and basis.size > vol_w:
            basis_deltas = np.diff(basis[-(vol_w + 1) :])
        else:
            basis_deltas = np.diff(basis)
        sigma_basis_bar = _rolling_std(basis_deltas, window=max(2, min(vol_w, basis_deltas.size)))
        sigma_basis_horizon = sigma_basis_bar * math.sqrt(float(horizon_bars))
        sigma_basis_bps = sigma_basis_horizon * 10000.0

        # Frictions: modeled in bps.
        slippage_bps = float(extra.get("slippage_bps") or 0.0)
        taker_fee_bps = float(extra.get("taker_fee_bps") or 0.0)
        round_trip_cost_bps = 2.0 * (slippage_bps + taker_fee_bps)
        required_edge_bps = round_trip_cost_bps + float(self.lambda_basis_vol) * float(sigma_basis_bps)

        # Mean-reverting basis expectation over horizon (continuous-time exponential decay).
        basis_mean = float(self.basis_mean_bps) / 10000.0
        halflife = float(self.basis_halflife_hours)
        if halflife <= 0:
            expected_basis_end = basis_mean
        else:
            k = math.log(2.0) / halflife
            expected_basis_end = basis_mean + (b_t - basis_mean) * math.exp(-k * horizon_hours)
        delta_b_expected = expected_basis_end - b_t

        # Compute candidate edge for both regimes.
        def _expected_edge_bps(side: int) -> tuple[float, float, float]:
            # side=+1 -> long spot / short perp (cash-and-carry)
            # side=-1 -> short spot / long perp (reverse cash-and-carry)
            expected_funding_frac = float(side) * float(funding_ema) * float(horizon_hours)
            expected_basis_pnl_frac = -float(side) * float(delta_b_expected)
            expected_edge = expected_funding_frac + expected_basis_pnl_frac
            return (
                expected_edge * 10000.0,
                expected_funding_frac * 10000.0,
                expected_basis_pnl_frac * 10000.0,
            )

        carry_edge_bps, carry_funding_bps, carry_basis_bps = _expected_edge_bps(+1)
        rev_edge_bps, rev_funding_bps, rev_basis_bps = _expected_edge_bps(-1)

        # Determine current side from existing notionals (if any).
        positions = dict(state.positions or {})
        spot_qty = float(positions.get(self.spot_symbol, 0.0) or 0.0)
        perp_qty = float(positions.get(self.perp_symbol, 0.0) or 0.0)
        spot_notional = spot_qty * float(spot_close[-1])
        perp_notional = perp_qty * float(perp_close[-1])
        gross_notional = abs(spot_notional) + abs(perp_notional)
        net_notional_imbalance = spot_notional + perp_notional

        current_side: Optional[int] = None
        if abs(spot_notional) > 1e-6 and abs(perp_notional) > 1e-6:
            if spot_notional > 0 and perp_notional < 0:
                current_side = +1
            elif spot_notional < 0 and perp_notional > 0:
                current_side = -1

        # Regime gates.
        max_widen_bps_per_hour = float(self.max_basis_widening_bps_per_hour)
        carry_ok = (
            (basis_bps >= float(self.min_basis_bps))
            and (funding_ema_bps_per_day >= float(self.funding_entry_bps_per_day))
            and (basis_mom_bps_per_hour <= max_widen_bps_per_hour)
            and (carry_edge_bps > required_edge_bps)
        )
        rev_ok = (
            bool(self.allow_reverse)
            and (basis_bps <= -float(self.min_basis_bps))
            and (funding_ema_bps_per_day <= -float(self.funding_entry_bps_per_day))
            and (basis_mom_bps_per_hour >= -max_widen_bps_per_hour)
            and (rev_edge_bps > required_edge_bps)
        )

        chosen_side: Optional[int] = None
        chosen_edge_bps = 0.0
        chosen_funding_bps = 0.0
        chosen_basis_bps = 0.0
        if carry_ok and rev_ok:
            if rev_edge_bps > carry_edge_bps:
                chosen_side, chosen_edge_bps, chosen_funding_bps, chosen_basis_bps = (
                    -1,
                    rev_edge_bps,
                    rev_funding_bps,
                    rev_basis_bps,
                )
            else:
                chosen_side, chosen_edge_bps, chosen_funding_bps, chosen_basis_bps = (
                    +1,
                    carry_edge_bps,
                    carry_funding_bps,
                    carry_basis_bps,
                )
        elif carry_ok:
            chosen_side, chosen_edge_bps, chosen_funding_bps, chosen_basis_bps = (
                +1,
                carry_edge_bps,
                carry_funding_bps,
                carry_basis_bps,
            )
        elif rev_ok:
            chosen_side, chosen_edge_bps, chosen_funding_bps, chosen_basis_bps = (
                -1,
                rev_edge_bps,
                rev_funding_bps,
                rev_basis_bps,
            )

        # Exit logic: if we can't justify a position, target flat.
        in_position = gross_notional >= float(self.min_trade_notional_usd)
        if chosen_side is None:
            # If already in a position, apply a weaker exit gate before forcing flat.
            if in_position:
                exit_side = current_side
                if exit_side is None:
                    return StrategyDecision(target, reason="exit (unknown side)")

                funding_exit_ok = (
                    (funding_ema_bps_per_day >= float(self.funding_exit_bps_per_day))
                    if exit_side == +1
                    else (funding_ema_bps_per_day <= -float(self.funding_exit_bps_per_day))
                )
                basis_exit_ok = (
                    (basis_bps >= float(self.min_basis_exit_bps))
                    if exit_side == +1
                    else (basis_bps <= -float(self.min_basis_exit_bps))
                )
                exit_edge_bps, _, _ = _expected_edge_bps(exit_side)
                keep = funding_exit_ok and basis_exit_ok and (exit_edge_bps > required_edge_bps)
                if keep:
                    # Keep current exposures to avoid churn.
                    cur_spot_exposure = _safe_div(spot_notional, max_position_notional_usd)
                    cur_perp_exposure = _safe_div(perp_notional, max_position_notional_usd)
                    target[self.spot_symbol] = float(cur_spot_exposure)
                    target[self.perp_symbol] = float(cur_perp_exposure)
                    return StrategyDecision(
                        target,
                        reason=f"hold ({'carry' if exit_side == 1 else 'reverse'})",
                        debug={
                            "symbols": symbols,
                            "basis_bps": basis_bps,
                            "funding_ema_bps_per_day": funding_ema_bps_per_day,
                            "required_edge_bps": required_edge_bps,
                            "expected_edge_bps": exit_edge_bps,
                        },
                    )
            return StrategyDecision(
                target,
                reason="flat (no edge)",
                debug={
                    "symbols": symbols,
                    "basis_bps": basis_bps,
                    "basis_mom_bps_per_hour": basis_mom_bps_per_hour,
                    "sigma_basis_bps": sigma_basis_bps,
                    "funding_ema_bps_per_day": funding_ema_bps_per_day,
                    "required_edge_bps": required_edge_bps,
                    "carry_edge_bps": carry_edge_bps,
                    "rev_edge_bps": rev_edge_bps,
                },
            )

        # Risk / sizing
        equity = float(state.equity)
        mmr = float(extra.get("maintenance_margin_rate") or self.maintenance_margin_rate)
        margin_used = abs(float(mmr)) * float(gross_notional)
        free_collateral = max(0.0, equity - margin_used)

        # Spot sigma_daily estimate from rolling log-return volatility.
        sigma_bar = _rolling_log_return_sigma(spot_close, window=int(self.spot_vol_window_bars))
        bars_per_day = int(round(24.0 / dt_hours)) if dt_hours > 0 else 1440
        sigma_daily = sigma_bar * math.sqrt(float(max(bars_per_day, 1)))

        denom = float(abs(mmr)) + float(self.z_sigma_daily) * float(sigma_daily)
        if denom <= 0:
            n_max = 0.0
        else:
            n_max = (free_collateral * (1.0 - float(self.collateral_buffer_frac))) / denom

        # Additional hard caps.
        # `max_leverage` is interpreted as a gross leverage cap across both legs.
        leverage_cap = max(0.0, equity * float(self.max_leverage) / 2.0)
        n_max = min(n_max, leverage_cap) if leverage_cap > 0 else n_max
        if abs(mmr) > 0 and float(self.max_margin_utilization) > 0:
            gross_cap = equity * float(self.max_margin_utilization) / abs(mmr)
            n_max = min(n_max, gross_cap / 2.0)

        net_edge_bps = float(chosen_edge_bps - required_edge_bps)
        if net_edge_bps <= 0:
            return StrategyDecision(target, reason="flat (net edge<=0)")

        sat = float(self.edge_saturation_bps)
        if sat <= 0:
            size_frac = 1.0
        else:
            size_frac = max(0.0, min(1.0, net_edge_bps / sat))

        n_target = (equity * float(self.max_leverage) / 2.0) * size_frac if equity > 0 else 0.0
        n_target = min(n_target, max_position_notional_usd)
        n_final = min(n_max, n_target, max_position_notional_usd)

        if n_final < float(self.min_trade_notional_usd):
            return StrategyDecision(target, reason="flat (min_notional)")

        # Desired exposures: equal and opposite notionals.
        desired_spot_exposure = (float(chosen_side) * n_final) / max_position_notional_usd
        desired_perp_exposure = (-float(chosen_side) * n_final) / max_position_notional_usd

        # Turnover control: only rebalance if drift is meaningful.
        drift_thresh = float(self.rebalance_drift_frac) * equity if equity > 0 else 0.0
        size_thresh = float(self.rebalance_min_notional_usd)

        spot_size_ok = abs(abs(spot_notional) - n_final) <= size_thresh if in_position else False
        perp_size_ok = abs(abs(perp_notional) - n_final) <= size_thresh if in_position else False
        drift_ok = abs(net_notional_imbalance) <= drift_thresh if drift_thresh > 0 else False
        side_ok = (current_side == chosen_side) if current_side is not None else False

        if in_position and side_ok and drift_ok and spot_size_ok and perp_size_ok:
            cur_spot_exposure = _safe_div(spot_notional, max_position_notional_usd)
            cur_perp_exposure = _safe_div(perp_notional, max_position_notional_usd)
            target[self.spot_symbol] = float(cur_spot_exposure)
            target[self.perp_symbol] = float(cur_perp_exposure)
            reason = f"hold ({'carry' if chosen_side == 1 else 'reverse'})"
        else:
            target[self.spot_symbol] = float(desired_spot_exposure)
            target[self.perp_symbol] = float(desired_perp_exposure)
            if in_position and current_side is not None and current_side != chosen_side:
                reason = f"flip ({'carry' if chosen_side == 1 else 'reverse'})"
            else:
                reason = f"enter ({'carry' if chosen_side == 1 else 'reverse'})"

        debug: dict[str, Any] = {
            "symbols": symbols,
            "spot_close": float(spot_close[-1]),
            "perp_close": float(perp_close[-1]),
            "basis_bps": float(basis_bps),
            "basis_mom_bps_per_hour": float(basis_mom_bps_per_hour),
            "funding_rate": float(funding[-1]),
            "funding_ema_bps_per_day": float(funding_ema_bps_per_day),
            "edge_horizon_hours": float(horizon_hours),
            "sigma_basis_bps": float(sigma_basis_bps),
            "required_edge_bps": float(required_edge_bps),
            "carry_edge_bps": float(carry_edge_bps),
            "rev_edge_bps": float(rev_edge_bps),
            "chosen_side": int(chosen_side),
            "chosen_edge_bps": float(chosen_edge_bps),
            "chosen_funding_bps": float(chosen_funding_bps),
            "chosen_basis_bps": float(chosen_basis_bps),
            "net_edge_bps": float(net_edge_bps),
            "equity": float(equity),
            "gross_notional": float(gross_notional),
            "net_notional_imbalance": float(net_notional_imbalance),
            "margin_used": float(margin_used),
            "free_collateral": float(free_collateral),
            "sigma_daily": float(sigma_daily),
            "n_max": float(n_max),
            "n_target": float(n_target),
            "n_final": float(n_final),
            "desired_exposures": {
                self.spot_symbol: float(desired_spot_exposure),
                self.perp_symbol: float(desired_perp_exposure),
            },
            "current_positions": {
                self.spot_symbol: float(spot_qty),
                self.perp_symbol: float(perp_qty),
            },
        }

        return StrategyDecision(target, reason=reason, debug=debug)
