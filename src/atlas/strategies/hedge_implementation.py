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


def _safe_div(n: float, d: float) -> float:
    return float(n / d) if d else 0.0


def _rolling_cov_2x2(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    if x.size < 2 or y.size < 2:
        return np.zeros((2, 2), dtype=float)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size != y.size:
        n = min(x.size, y.size)
        x = x[-n:]
        y = y[-n:]
    mat = np.vstack([x, y])
    try:
        cov = np.cov(mat, ddof=0)
    except Exception:
        cov = np.zeros((2, 2), dtype=float)
    if cov.shape != (2, 2):
        return np.zeros((2, 2), dtype=float)
    return np.asarray(cov, dtype=float)


def _rolling_std(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    if x.size < 2:
        return 0.0
    return float(np.std(x, ddof=0))


@dataclass
class HedgeImplementation(Strategy):
    """
    Hedge-Implementation.md: market-neutral spot/perp hedge with explicit edge, costs, risk, and sizing.

    Model:
    - Spot mid ~ spot close (S_t)
    - Perp mark ~ perp close (P_t)
    - Basis b_t = (P_t - S_t) / S_t
    - Funding rate uses `funding_rate` column when present (hourly rate, fraction per hour)

    Signs:
    - s=+1 => long perp / short spot (reverse carry)
    - s=-1 => short perp / long spot (cash-and-carry)
    """

    spot_symbol: str
    perp_symbol: str

    # Horizon & estimators (Section 3 / 7)
    edge_horizon_hours: float = 8.0
    funding_ema_alpha: float = 0.20
    basis_halflife_hours: float = 24.0
    theta_intercept_bps: float = 0.0
    theta_funding_beta: float = 0.25  # basis bps per (funding bps/day)

    # Costs / financing (Section 2 / 7)
    include_expected_rebalance_costs: bool = True
    cov_window_bars: int = 240
    rebalance_delta_max: float = 0.02
    rebalance_turnover_frac_per_unit_delta: float = 0.50
    spot_financing_rate_per_hour: float = 0.0  # applies only when short spot (s=+1)

    # Risk & sizing (Section 4 / 7)
    z_risk: float = 1.0
    lambda_risk: float = 8.0  # mean-variance penalty on f=N/equity
    z_liq: float = 2.33  # ~ N(0,1) 99th percentile
    collateral_buffer_frac: float = 0.10
    max_leverage: float = 3.0  # gross cap across both legs
    max_margin_utilization: float = 0.50
    maintenance_margin_rate: float = 0.05
    min_trade_notional_usd: float = 200.0
    rebalance_min_notional_usd: float = 100.0
    flip_hysteresis_bps: float = 2.0
    require_funding_rate: bool = False

    def warmup_bars(self) -> int:
        return max(int(self.cov_window_bars) + 5, 60)

    def target_exposures(
        self, bars_by_symbol: dict[str, pd.DataFrame], state: StrategyState
    ) -> StrategyDecision:
        target = {self.spot_symbol: 0.0, self.perp_symbol: 0.0}

        spot_df = bars_by_symbol.get(self.spot_symbol)
        perp_df = bars_by_symbol.get(self.perp_symbol)
        if spot_df is None or perp_df is None:
            return StrategyDecision(target, reason="missing bars")
        if len(spot_df) < self.warmup_bars() or len(perp_df) < self.warmup_bars():
            return StrategyDecision(target, reason="warmup")

        # Hedge requires the ability to short *one* leg.
        if not bool(state.allow_short):
            return StrategyDecision(target, reason="short disabled")

        extra = dict(state.extra or {})
        max_position_notional_usd = float(extra.get("max_position_notional_usd") or 0.0)
        if max_position_notional_usd <= 0:
            return StrategyDecision(target, reason="max_notional=0")

        # Align close series to a common index (defensive; engine typically already aligned).
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

        # Compute basis b_t.
        basis = (perp_close - spot_close) / spot_close
        b_t = float(basis[-1])
        one_plus_b = 1.0 + b_t
        if one_plus_b <= 0:
            return StrategyDecision(target, reason="invalid basis (1+b<=0)")

        # Funding (preferred: column; fallback: current funding rate passed by engine).
        current_funding_rates = dict(extra.get("funding_rates") or {})
        fallback_funding = float(current_funding_rates.get(self.perp_symbol, 0.0) or 0.0)
        has_funding_history = "funding_rate" in perp_df.columns
        if has_funding_history:
            funding_s = perp_df["funding_rate"].reindex(idx).fillna(0.0)
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

        # Baseline estimator A: OU/ARX mean for basis.
        theta_bps = float(self.theta_intercept_bps) + float(self.theta_funding_beta) * float(
            funding_ema_bps_per_day
        )
        theta = theta_bps / 10000.0
        halflife = float(self.basis_halflife_hours)
        if halflife <= 0:
            delta_b_hat = theta - b_t
        else:
            kappa_b = math.log(2.0) / halflife  # per hour
            delta_b_hat = (theta - b_t) * (1.0 - math.exp(-kappa_b * horizon_hours))

        # Funding forecast over horizon (continuous approximation).
        r_hat = funding_ema * float(horizon_hours)

        # Core edge statistic X_hat (per notional, fraction).
        x_hat = (delta_b_hat / one_plus_b) - r_hat
        s = 1 if x_hat > 0 else (-1 if x_hat < 0 else 0)

        # Current positions + hedge mismatch ratio δ_t.
        positions = dict(state.positions or {})
        spot_qty = float(positions.get(self.spot_symbol, 0.0) or 0.0)
        perp_qty = float(positions.get(self.perp_symbol, 0.0) or 0.0)
        spot_notional = spot_qty * float(spot_close[-1])
        perp_notional = perp_qty * float(perp_close[-1])
        gross_notional = abs(spot_notional) + abs(perp_notional)
        epsilon_notional = spot_notional + perp_notional
        n_scale = max(abs(spot_notional), abs(perp_notional), 1e-9)
        delta_curr = float(epsilon_notional / n_scale)

        current_side: Optional[int] = None
        if abs(spot_notional) > 1e-6 and abs(perp_notional) > 1e-6:
            if spot_notional < 0 and perp_notional > 0:
                current_side = +1
            elif spot_notional > 0 and perp_notional < 0:
                current_side = -1

        # Risk estimates: build Σ_{t,h} from cov of (spot returns, basis changes).
        log_spot = np.log(np.clip(spot_close, 1e-12, None))
        log_perp = np.log(np.clip(perp_close, 1e-12, None))
        r_spot = np.diff(log_spot)
        r_perp = np.diff(log_perp)
        db = np.diff(basis)

        # Align lengths: r_* and db are based on bar-to-bar changes.
        n_changes = min(r_spot.size, db.size, r_perp.size)
        if n_changes < 2:
            return StrategyDecision(target, reason="insufficient returns")
        r_spot = r_spot[-n_changes:]
        r_perp = r_perp[-n_changes:]
        db = db[-n_changes:]

        w_cov = int(self.cov_window_bars)
        tail = max(2, min(w_cov, n_changes))
        r_tail = r_spot[-tail:]
        db_tail = db[-tail:]
        cov_bar = _rolling_cov_2x2(r_tail, db_tail)
        cov_h = cov_bar * float(max(horizon_bars, 1))

        # sigma_spot_h: diffusive scale.
        sigma_spot_bar = _rolling_std(r_tail)
        sigma_spot_h = sigma_spot_bar * math.sqrt(float(max(horizon_bars, 1)))

        # sigma_edge per notional from w' Σ w.
        w0 = float(delta_curr) if gross_notional >= float(self.min_trade_notional_usd) else 0.0
        w1 = 1.0 / one_plus_b
        var_edge = float(w0 * (cov_h[0, 0] * w0 + cov_h[0, 1] * w1) + w1 * (cov_h[1, 0] * w0 + cov_h[1, 1] * w1))
        sigma_edge = math.sqrt(max(0.0, var_edge))

        # Costs: kappa_hat (per notional, fraction).
        slippage_bps = float(extra.get("slippage_bps") or 0.0)
        taker_fee_bps = float(extra.get("taker_fee_bps") or 0.0)
        cost_side_bps = float(slippage_bps + taker_fee_bps)
        entry_exit_cost = 4.0 * cost_side_bps / 10000.0  # enter+exit across both legs

        rebalance_cost = 0.0
        delta_max = float(self.rebalance_delta_max)
        if self.include_expected_rebalance_costs and delta_max > 0:
            # Approximate hedge-mismatch diffusion from return differences (perp - spot).
            delta_step = (r_perp - r_spot)[-tail:]
            sigma_delta_bar = _rolling_std(delta_step)
            expected_rebalances = float(max(horizon_bars, 1)) * (sigma_delta_bar**2) / max(
                delta_max**2, 1e-12
            )
            turnover_frac = float(self.rebalance_turnover_frac_per_unit_delta) * delta_max
            turnover_frac = max(0.0, min(2.0, turnover_frac))
            # Each rebalance trades ~ turnover_frac * N per leg (both legs), paying one-side costs.
            rebalance_cost = expected_rebalances * (2.0 * turnover_frac) * (cost_side_bps / 10000.0)

        kappa_hat = float(entry_exit_cost + rebalance_cost)

        # Financing drag (per notional, fraction) for the spot leg when it's short.
        phi_hat = 0.0
        if float(self.spot_financing_rate_per_hour) > 0 and s == +1:
            phi_hat = float(self.spot_financing_rate_per_hour) * float(horizon_hours)

        # Net edge magnitude E_hat and decision inequality.
        e_hat = abs(float(x_hat)) - float(kappa_hat) - float(phi_hat)
        risk_gate = float(self.z_risk) * float(sigma_edge)

        # If we have no directional preference or no net edge, exit/avoid.
        if s == 0 or not (e_hat > risk_gate):
            if gross_notional >= float(self.min_trade_notional_usd):
                return StrategyDecision(
                    target,
                    reason="exit (no edge)",
                    debug={
                        "x_hat_bps": float(x_hat) * 10000.0,
                        "e_hat_bps": float(e_hat) * 10000.0,
                        "risk_gate_bps": float(risk_gate) * 10000.0,
                        "sigma_edge_bps": float(sigma_edge) * 10000.0,
                    },
                )
            return StrategyDecision(
                target,
                reason="flat (no edge)",
                debug={
                    "x_hat_bps": float(x_hat) * 10000.0,
                    "e_hat_bps": float(e_hat) * 10000.0,
                    "risk_gate_bps": float(risk_gate) * 10000.0,
                    "sigma_edge_bps": float(sigma_edge) * 10000.0,
                },
            )

        # Sizing: mean-variance optimum N* = E_hat / (lambda * sigma^2), then caps.
        sigma2 = float(sigma_edge) ** 2
        lam = float(self.lambda_risk)
        n_star = 0.0
        if e_hat > 0 and lam > 0 and sigma2 > 0:
            # Interpret lambda_risk as a dimensionless mean-variance penalty on *fraction of equity*.
            # f* = E_hat / (lambda * sigma^2), N* = f* * equity
            n_star = (float(e_hat) / (lam * sigma2)) * max(0.0, float(state.equity))

        equity = float(state.equity)
        mmr = float(extra.get("maintenance_margin_rate") or self.maintenance_margin_rate)
        denom = float(abs(mmr)) + float(self.z_liq) * float(sigma_spot_h)
        if denom <= 0 or equity <= 0:
            n_lev = 0.0
        else:
            n_lev = (equity * (1.0 - float(self.collateral_buffer_frac))) / denom

        # Gross leverage + margin-utilization caps for a 2-leg hedge.
        n_cap_leverage = max(0.0, equity * float(self.max_leverage) / 2.0)
        n_cap_margin = float("inf")
        if abs(mmr) > 0 and float(self.max_margin_utilization) > 0:
            n_cap_margin = (equity * float(self.max_margin_utilization)) / (2.0 * abs(mmr))

        n_final = min(
            float(max_position_notional_usd),
            float(n_star) if n_star > 0 else float(max_position_notional_usd),
            float(n_lev) if n_lev > 0 else float(max_position_notional_usd),
            float(n_cap_leverage) if n_cap_leverage > 0 else float(max_position_notional_usd),
            float(n_cap_margin) if n_cap_margin > 0 else float(max_position_notional_usd),
        )
        n_final = float(max(0.0, n_final))

        if n_final < float(self.min_trade_notional_usd):
            return StrategyDecision(target, reason="flat (min_notional)")

        desired_spot_exposure = (-float(s) * n_final) / max_position_notional_usd
        desired_perp_exposure = (float(s) * n_final) / max_position_notional_usd

        # Turnover control: rebalance only if mismatch exceeds δ_max or size drift is material.
        in_position = gross_notional >= float(self.min_trade_notional_usd)
        size_ok = (
            abs(abs(spot_notional) - n_final) <= float(self.rebalance_min_notional_usd)
            and abs(abs(perp_notional) - n_final) <= float(self.rebalance_min_notional_usd)
            if in_position
            else False
        )
        drift_ok = abs(delta_curr) <= float(delta_max) if delta_max > 0 else False
        side_ok = (current_side == s) if current_side is not None else False

        # Flip hysteresis: require extra edge to flip sides.
        hysteresis = float(self.flip_hysteresis_bps) / 10000.0
        wants_flip = bool(in_position and current_side is not None and current_side != s)
        if wants_flip and (e_hat - risk_gate) <= hysteresis:
            # Keep current exposures rather than churn.
            cur_spot = _safe_div(spot_notional, max_position_notional_usd)
            cur_perp = _safe_div(perp_notional, max_position_notional_usd)
            target[self.spot_symbol] = float(cur_spot)
            target[self.perp_symbol] = float(cur_perp)
            reason = "hold (flip hysteresis)"
        elif in_position and side_ok and drift_ok and size_ok:
            cur_spot = _safe_div(spot_notional, max_position_notional_usd)
            cur_perp = _safe_div(perp_notional, max_position_notional_usd)
            target[self.spot_symbol] = float(cur_spot)
            target[self.perp_symbol] = float(cur_perp)
            reason = "hold"
        else:
            target[self.spot_symbol] = float(desired_spot_exposure)
            target[self.perp_symbol] = float(desired_perp_exposure)
            if wants_flip:
                reason = "flip"
            elif in_position:
                reason = "rebalance"
            else:
                reason = "enter"

        debug: dict[str, Any] = {
            "symbols": (self.spot_symbol, self.perp_symbol),
            "spot_close": float(spot_close[-1]),
            "perp_close": float(perp_close[-1]),
            "basis_bps": float(b_t) * 10000.0,
            "funding_rate": float(funding[-1]),
            "funding_ema_bps_per_day": float(funding_ema_bps_per_day),
            "theta_bps": float(theta_bps),
            "delta_b_hat_bps": float(delta_b_hat) * 10000.0,
            "r_hat_bps": float(r_hat) * 10000.0,
            "x_hat_bps": float(x_hat) * 10000.0,
            "kappa_hat_bps": float(kappa_hat) * 10000.0,
            "phi_hat_bps": float(phi_hat) * 10000.0,
            "e_hat_bps": float(e_hat) * 10000.0,
            "sigma_edge_bps": float(sigma_edge) * 10000.0,
            "risk_gate_bps": float(risk_gate) * 10000.0,
            "dt_hours": float(dt_hours),
            "horizon_hours": float(horizon_hours),
            "horizon_bars": int(horizon_bars),
            "delta_curr": float(delta_curr),
            "delta_max": float(delta_max),
            "current_side": int(current_side) if current_side is not None else None,
            "s_target": int(s),
            "equity": float(equity),
            "mmr": float(mmr),
            "sigma_spot_h": float(sigma_spot_h),
            "n_star": float(n_star),
            "n_lev": float(n_lev),
            "n_cap_leverage": float(n_cap_leverage),
            "n_cap_margin": float(n_cap_margin) if math.isfinite(n_cap_margin) else None,
            "n_final": float(n_final),
            "positions_notional": {
                self.spot_symbol: float(spot_notional),
                self.perp_symbol: float(perp_notional),
            },
            "targets": {
                self.spot_symbol: float(target[self.spot_symbol]),
                self.perp_symbol: float(target[self.perp_symbol]),
            },
            "cost_side_bps": float(cost_side_bps),
            "entry_exit_cost_bps": float(entry_exit_cost) * 10000.0,
            "rebalance_cost_bps": float(rebalance_cost) * 10000.0,
        }

        return StrategyDecision(target, reason=reason, debug=debug)
