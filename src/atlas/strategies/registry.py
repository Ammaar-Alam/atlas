from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import json

from atlas.strategies.base import Strategy
from atlas.strategies.ema_crossover import EmaCrossover
from atlas.strategies.ma_crossover import MovingAverageCrossover
from atlas.strategies.no_trade import NoTrade
from atlas.strategies.nec_pdt import NecPDT
from atlas.strategies.nec_x import NecX
from atlas.strategies.orb_trend import OrbTrend
from atlas.strategies.spy_open_close import SpyOpenClose
from atlas.strategies.perp_flare import PerpFlare
from atlas.strategies.perp_hawk import PerpHawk
from atlas.strategies.perp_scalp import PerpScalp
from atlas.strategies.basis_carry import BasisCarry
from atlas.strategies.hedge_implementation import HedgeImplementation


@dataclass(frozen=True)
class StrategyBuild:
    name: str
    params: Dict[str, Any]


def _load_params(path: Optional[Path]) -> Dict[str, Any]:
    if path is None:
        return {}
    raw = json.loads(path.read_text())
    if isinstance(raw, dict) and "params" in raw and isinstance(raw["params"], dict):
        return raw["params"]
    if isinstance(raw, dict):
        return raw
    raise ValueError("strategy params json must be an object")


def build_strategy(
    *,
    name: str,
    params_path: Optional[Path],
    symbols: list[str],
    fast_window: int,
    slow_window: int,
    params: Optional[Dict[str, Any]] = None,
) -> Strategy:
    params = params if params is not None else _load_params(params_path)
    if isinstance(params, dict):
        if "params" in params and isinstance(params["params"], dict):
            params = params["params"]
        elif "parameters" in params and isinstance(params["parameters"], dict):
            params = params["parameters"]
        canonical = name.replace("-", "_")
        if canonical in params and isinstance(params[canonical], dict):
            params = params[canonical]
        elif name in params and isinstance(params[name], dict):
            params = params[name]

    if name == "ma_crossover":
        fast = int(params.get("fast_window", fast_window))
        slow = int(params.get("slow_window", slow_window))
        symbol = str(params.get("symbol") or (symbols[0] if symbols else "SPY"))
        return MovingAverageCrossover(fast_window=fast, slow_window=slow, symbol=symbol)

    if name in {"ema_crossover", "ema-crossover"}:
        fast = int(params.get("fast_window", fast_window))
        slow = int(params.get("slow_window", slow_window))
        symbol = str(params.get("symbol") or (symbols[0] if symbols else "SPY"))
        return EmaCrossover(fast_window=fast, slow_window=slow, symbol=symbol)

    if name in {"spy_open_close", "spy-open-close"}:
        symbol = str(params.get("symbol") or (symbols[0] if symbols else "SPY"))
        return SpyOpenClose(symbol=symbol)

    if name in {"no_trade", "no-trade"}:
        return NoTrade()

    if name in {"nec_x", "nec-x"}:
        universe_symbols = [s.strip().upper() for s in symbols if s.strip()]
        if len(universe_symbols) < 2:
            raise ValueError(
                f"nec_x requires at least 2 symbols (got {len(universe_symbols)})"
            )
        spy_symbol, qqq_symbol = universe_symbols[0], universe_symbols[1]

        def _get_int(key: str, default: int) -> int:
            raw = params.get(key, params.get(key.lower(), default))
            return int(raw)

        def _get_float(key: str, default: float) -> float:
            raw = params.get(key, params.get(key.lower(), default))
            return float(raw)

        return NecX(
            spy=spy_symbol,
            qqq=qqq_symbol,
            M=_get_int("M", 6),
            V=_get_int("V", 12),
            Wcorr=_get_int("Wcorr", 12),
            rho_min=_get_float("rho_min", 0.60),
            strength_entry=_get_float("strength_entry", 0.80),
            strength_exit=_get_float("strength_exit", 0.20),
            H_max=_get_int("H_max", 6),
            k_cost=_get_float("k_cost", 1.25),
            spread_floor_bps=_get_float("spread_floor_bps", 0.50),
            slip_bps=_get_float("slip_bps", 0.75),
            daily_loss_limit=_get_float("daily_loss_limit", 0.010),
            kill_switch=_get_float("kill_switch", 0.025),
            tick_size=_get_float("tick_size", 0.01),
        )

    if name in {"nec_pdt", "nec-pdt"}:
        universe_symbols = [s.strip().upper() for s in symbols if s.strip()]
        if len(universe_symbols) < 2:
            raise ValueError(
                f"nec_pdt requires at least 2 symbols (got {len(universe_symbols)})"
            )
        spy_symbol, qqq_symbol = universe_symbols[0], universe_symbols[1]

        def _get_int(key: str, default: int) -> int:
            raw = params.get(key, params.get(key.lower(), default))
            return int(raw)

        def _get_float(key: str, default: float) -> float:
            raw = params.get(key, params.get(key.lower(), default))
            return float(raw)

        return NecPDT(
            spy=spy_symbol,
            qqq=qqq_symbol,
            M=_get_int("M", 6),
            V=_get_int("V", 12),
            eps=_get_float("eps", 1e-8),
            H=_get_int("H", 12),
            base_thr_bps=_get_float("base_thr_bps", 10.0),
            budget_step_bps=_get_float("budget_step_bps", 4.0),
            atr_lookback_bars=_get_int("atr_lookback_bars", 12),
            stop_atr_mult=_get_float("stop_atr_mult", 2.0),
            trail_atr_mult=_get_float("trail_atr_mult", 2.5),
            min_hold_bars=_get_int("min_hold_bars", 4),
            flip_confirm_bars=_get_int("flip_confirm_bars", 3),
            max_day_trades_per_rolling_5_days=_get_int(
                "max_day_trades_per_rolling_5_days", 3
            ),
            half_spread_bps=_get_float("half_spread_bps", 1.5),
            slippage_bps=_get_float("slippage_bps", 2.0),
            fee_bps=float(
                params.get("fee_bps", params.get("fees_bps", _get_float("fee_bps", 0.3)))
            ),
        )

    if name in {"orb_trend", "orb-trend"}:
        universe_symbols = [s.strip().upper() for s in symbols if s.strip()]
        if not universe_symbols:
            raise ValueError("orb_trend requires at least 1 symbol")

        def _get_int(key: str, default: int) -> int:
            raw = params.get(key, params.get(key.lower(), default))
            return int(raw)

        def _get_float(key: str, default: float) -> float:
            raw = params.get(key, params.get(key.lower(), default))
            return float(raw)

        return OrbTrend(
            symbols=tuple(universe_symbols),
            orb_minutes=_get_int("orb_minutes", 30),
            orb_breakout_bps=_get_float("orb_breakout_bps", 4.0),
            confirm_bars=_get_int("confirm_bars", 2),
            atr_window=_get_int("atr_window", 20),
            er_window=_get_int("er_window", 12),
            er_min=_get_float("er_min", 0.35),
            expected_hold_bars=_get_int("expected_hold_bars", 12),
            k_cost=_get_float("k_cost", 2.0),
            slippage_bps=_get_float("slippage_bps", 1.25),
            min_hold_bars=_get_int("min_hold_bars", 3),
            daily_loss_limit=_get_float("daily_loss_limit", 0.010),
            kill_switch=_get_float("kill_switch", 0.025),
        )

    if name in {"perp_flare", "perp-flare"}:
        universe_symbols = [s.strip().upper() for s in symbols if s.strip()]
        if not universe_symbols:
            raise ValueError("perp_flare requires at least 1 symbol")
            
        def _get_int(key: str, default: int) -> int:
            raw = params.get(key, params.get(key.lower(), default))
            return int(raw)

        def _get_float(key: str, default: float) -> float:
            raw = params.get(key, params.get(key.lower(), default))
            return float(raw)

        def _get_opt_float(key: str, default: Optional[float]) -> Optional[float]:
            raw = params.get(key, params.get(key.lower(), default))
            if raw is None:
                return default
            return float(raw)

        def _get_str(key: str, default: str) -> str:
            raw = params.get(key, params.get(key.lower(), default))
            return str(raw)

        return PerpFlare(
            symbols=tuple(universe_symbols),
            atr_window=_get_int("atr_window", 14),
            ema_fast=_get_int("ema_fast", 12),
            ema_slow=_get_int("ema_slow", 24),
            er_window=_get_int("er_window", 10),
            breakout_window=_get_int("breakout_window", 20),
            er_min=_get_float("er_min", 0.35),
            taker_fee_bps=_get_float("taker_fee_bps", 3.0),
            half_spread_bps=_get_float("half_spread_bps", 1.0),
            base_slippage_bps=_get_float("base_slippage_bps", 1.5),
            edge_floor_bps=_get_float("edge_floor_bps", 5.0),
            k_cost=_get_float("k_cost", 1.5),
            risk_per_trade=_get_float("risk_per_trade", 0.01),
            stop_atr_mult=_get_float("stop_atr_mult", 2.0),
            trail_atr_mult=_get_float("trail_atr_mult", 3.0),
            max_margin_utilization=_get_float("max_margin_utilization", 0.65),
            max_leverage=_get_float("max_leverage", 10.0),
            sizing_mode=_get_str("sizing_mode", "risk"),
            target_leverage=_get_opt_float("target_leverage", None),
            maintenance_margin_rate=_get_float("maintenance_margin_rate", 0.05),
            min_liq_buffer_atr=_get_float("min_liq_buffer_atr", 3.0),
        )

    if name in {"perp_hawk", "perp-hawk"}:
        universe_symbols = [s.strip().upper() for s in symbols if s.strip()]
        if not universe_symbols:
            raise ValueError("perp_hawk requires at least 1 symbol")

        def _get_int(key: str, default: int) -> int:
            raw = params.get(key, params.get(key.lower(), default))
            return int(raw)

        def _get_float(key: str, default: float) -> float:
            raw = params.get(key, params.get(key.lower(), default))
            return float(raw)

        def _get_bool(key: str, default: bool) -> bool:
            raw = params.get(key, params.get(key.lower(), default))
            if isinstance(raw, bool):
                return raw
            if isinstance(raw, (int, float)):
                return bool(int(raw))
            if isinstance(raw, str):
                return raw.strip().lower() in {"1", "true", "t", "yes", "y", "on"}
            return bool(default)

        return PerpHawk(
            atr_window=_get_int("atr_window", 14),
            ema_fast=_get_int("ema_fast", 20),
            ema_slow=_get_int("ema_slow", 60),
            er_window=_get_int("er_window", 20),
            breakout_window=_get_int("breakout_window", 20),
            breakout_buffer_bps=_get_float("breakout_buffer_bps", 2.0),
            er_min=_get_float("er_min", 0.30),
            trend_z_min=_get_float("trend_z_min", 0.25),
            min_atr_bps=_get_float("min_atr_bps", 5.0),
            allow_trend_entry_without_breakout=_get_bool(
                "allow_trend_entry_without_breakout", True
            ),
            risk_budget=_get_float("risk_budget", 0.010),
            stop_atr_mult=_get_float("stop_atr_mult", 2.2),
            trail_atr_mult=_get_float("trail_atr_mult", 3.2),
            max_positions=_get_int("max_positions", 2),
            rebalance_exposure_threshold=_get_float(
                "rebalance_exposure_threshold", 0.05
            ),
            max_leverage=_get_float("max_leverage", 3.0),
            max_margin_utilization=_get_float("max_margin_utilization", 0.35),
            funding_entry_bps_per_day=_get_float("funding_entry_bps_per_day", 25.0),
            funding_exit_bps_per_day=_get_float("funding_exit_bps_per_day", 60.0),
            daily_loss_limit=_get_float("daily_loss_limit", 0.02),
            kill_switch=_get_float("kill_switch", 0.10),
            min_hold_bars=_get_int("min_hold_bars", 3),
            flip_confirm_bars=_get_int("flip_confirm_bars", 3),
            cooldown_bars=_get_int("cooldown_bars", 5),
        )

    if name in {"perp_scalp", "perp-scalp"}:
        universe_symbols = [s.strip().upper() for s in symbols if s.strip()]
        if not universe_symbols:
            raise ValueError("perp_scalp requires at least 1 symbol")

        def _get_int(key: str, default: int) -> int:
            raw = params.get(key, params.get(key.lower(), default))
            return int(raw)

        def _get_float(key: str, default: float) -> float:
            raw = params.get(key, params.get(key.lower(), default))
            return float(raw)

        def _get_opt_float(key: str, default: Optional[float]) -> Optional[float]:
            raw = params.get(key, params.get(key.lower(), default))
            if raw is None:
                return default
            return float(raw)

        def _get_str(key: str, default: str) -> str:
            raw = params.get(key, params.get(key.lower(), default))
            return str(raw)

        return PerpScalp(
            symbols=tuple(universe_symbols),
            atr_window=_get_int("atr_window", 14),
            ema_fast=_get_int("ema_fast", 8),
            ema_slow=_get_int("ema_slow", 21),
            er_window=_get_int("er_window", 10),
            breakout_window=_get_int("breakout_window", 8),
            breakout_buffer_bps=_get_float("breakout_buffer_bps", 1.0),
            er_min=_get_float("er_min", 0.25),
            trend_z_min=_get_float("trend_z_min", 0.15),
            min_atr_bps=_get_float("min_atr_bps", 8.0),
            edge_floor_bps=_get_float("edge_floor_bps", 3.0),
            k_cost=_get_float("k_cost", 1.5),
            taker_fee_bps=_get_float("taker_fee_bps", 3.0),
            slippage_bps=_get_float("slippage_bps", 1.5),
            funding_entry_bps_per_day=_get_float("funding_entry_bps_per_day", 40.0),
            funding_exit_bps_per_day=_get_float("funding_exit_bps_per_day", 80.0),
            risk_per_trade=_get_float("risk_per_trade", 0.005),
            stop_atr_mult=_get_float("stop_atr_mult", 1.2),
            trail_atr_mult=_get_float("trail_atr_mult", 1.8),
            take_profit_atr_mult=_get_float("take_profit_atr_mult", 1.5),
            max_hold_bars=_get_int("max_hold_bars", 12),
            min_hold_bars=_get_int("min_hold_bars", 2),
            flip_confirm_bars=_get_int("flip_confirm_bars", 2),
            cooldown_bars=_get_int("cooldown_bars", 4),
            sizing_mode=_get_str("sizing_mode", "risk"),
            target_leverage=_get_opt_float("target_leverage", None),
            max_leverage=_get_float("max_leverage", 5.0),
            max_margin_utilization=_get_float("max_margin_utilization", 0.40),
            maintenance_margin_rate=_get_float("maintenance_margin_rate", 0.05),
            min_liq_buffer_atr=_get_float("min_liq_buffer_atr", 2.5),
            daily_loss_limit=_get_float("daily_loss_limit", 0.02),
            kill_switch=_get_float("kill_switch", 0.10),
        )

    if name in {"basis_carry", "basis-carry", "cash_and_carry", "cash-and-carry"}:
        universe_symbols = [s.strip().upper() for s in symbols if s.strip()]
        if len(universe_symbols) < 2:
            raise ValueError("basis_carry requires 2 symbols (spot, perp)")

        pair = universe_symbols[:2]

        def _is_perp(sym: str) -> bool:
            s = (sym or "").strip().upper()
            return s.endswith("-PERP") or s.endswith("-CDE")

        spot_symbol = pair[0]
        perp_symbol = pair[1]
        if _is_perp(pair[0]) and not _is_perp(pair[1]):
            spot_symbol = pair[1]
            perp_symbol = pair[0]
        elif _is_perp(pair[1]) and not _is_perp(pair[0]):
            spot_symbol = pair[0]
            perp_symbol = pair[1]

        def _get_int(key: str, default: int) -> int:
            raw = params.get(key, params.get(key.lower(), default))
            return int(raw)

        def _get_float(key: str, default: float) -> float:
            raw = params.get(key, params.get(key.lower(), default))
            return float(raw)

        def _get_bool(key: str, default: bool) -> bool:
            raw = params.get(key, params.get(key.lower(), default))
            if isinstance(raw, bool):
                return raw
            if isinstance(raw, (int, float)):
                return bool(int(raw))
            if isinstance(raw, str):
                return raw.strip().lower() in {"1", "true", "t", "yes", "y", "on"}
            return bool(default)

        return BasisCarry(
            spot_symbol=spot_symbol,
            perp_symbol=perp_symbol,
            funding_ema_alpha=_get_float("funding_ema_alpha", 0.20),
            funding_entry_bps_per_day=_get_float("funding_entry_bps_per_day", 10.0),
            funding_exit_bps_per_day=_get_float("funding_exit_bps_per_day", 0.0),
            edge_horizon_hours=_get_float("edge_horizon_hours", 8.0),
            min_basis_bps=_get_float("min_basis_bps", 5.0),
            min_basis_exit_bps=_get_float("min_basis_exit_bps", 0.0),
            basis_mean_bps=_get_float("basis_mean_bps", 0.0),
            basis_halflife_hours=_get_float("basis_halflife_hours", 24.0),
            basis_momentum_window_bars=_get_int("basis_momentum_window_bars", 30),
            max_basis_widening_bps_per_hour=_get_float("max_basis_widening_bps_per_hour", 10.0),
            basis_vol_window_bars=_get_int("basis_vol_window_bars", 120),
            lambda_basis_vol=_get_float("lambda_basis_vol", 1.0),
            edge_saturation_bps=_get_float("edge_saturation_bps", 50.0),
            collateral_buffer_frac=_get_float("collateral_buffer_frac", 0.10),
            z_sigma_daily=_get_float("z_sigma_daily", 3.0),
            spot_vol_window_bars=_get_int("spot_vol_window_bars", 120),
            max_leverage=_get_float("max_leverage", 3.0),
            max_margin_utilization=_get_float("max_margin_utilization", 0.50),
            maintenance_margin_rate=_get_float("maintenance_margin_rate", 0.05),
            rebalance_drift_frac=_get_float("rebalance_drift_frac", 0.02),
            rebalance_min_notional_usd=_get_float("rebalance_min_notional_usd", 100.0),
            min_trade_notional_usd=_get_float("min_trade_notional_usd", 200.0),
            allow_reverse=_get_bool("allow_reverse", False),
            require_funding_rate=_get_bool("require_funding_rate", False),
        )

    if name in {"hedge_implementation", "hedge-implementation", "hedge_impl", "hedge-impl", "hedge"}:
        universe_symbols = [s.strip().upper() for s in symbols if s.strip()]
        if len(universe_symbols) < 2:
            raise ValueError("hedge requires 2 symbols (spot, perp)")

        pair = universe_symbols[:2]

        def _is_perp(sym: str) -> bool:
            s = (sym or "").strip().upper()
            return s.endswith("-PERP") or s.endswith("-CDE")

        spot_symbol = pair[0]
        perp_symbol = pair[1]
        if _is_perp(pair[0]) and not _is_perp(pair[1]):
            spot_symbol = pair[1]
            perp_symbol = pair[0]
        elif _is_perp(pair[1]) and not _is_perp(pair[0]):
            spot_symbol = pair[0]
            perp_symbol = pair[1]

        def _get_int(key: str, default: int) -> int:
            raw = params.get(key, params.get(key.lower(), default))
            return int(raw)

        def _get_float(key: str, default: float) -> float:
            raw = params.get(key, params.get(key.lower(), default))
            return float(raw)

        def _get_bool(key: str, default: bool) -> bool:
            raw = params.get(key, params.get(key.lower(), default))
            if isinstance(raw, bool):
                return raw
            if isinstance(raw, (int, float)):
                return bool(int(raw))
            if isinstance(raw, str):
                return raw.strip().lower() in {"1", "true", "t", "yes", "y", "on"}
            return bool(default)

        return HedgeImplementation(
            spot_symbol=spot_symbol,
            perp_symbol=perp_symbol,
            edge_horizon_hours=_get_float("edge_horizon_hours", 8.0),
            funding_ema_alpha=_get_float("funding_ema_alpha", 0.20),
            basis_halflife_hours=_get_float("basis_halflife_hours", 24.0),
            theta_intercept_bps=_get_float("theta_intercept_bps", 0.0),
            theta_funding_beta=_get_float("theta_funding_beta", 0.25),
            include_expected_rebalance_costs=_get_bool("include_expected_rebalance_costs", True),
            cov_window_bars=_get_int("cov_window_bars", 240),
            rebalance_delta_max=_get_float("rebalance_delta_max", 0.02),
            rebalance_turnover_frac_per_unit_delta=_get_float(
                "rebalance_turnover_frac_per_unit_delta", 0.50
            ),
            spot_financing_rate_per_hour=_get_float("spot_financing_rate_per_hour", 0.0),
            z_risk=_get_float("z_risk", 1.0),
            lambda_risk=_get_float("lambda_risk", 8.0),
            z_liq=_get_float("z_liq", 2.33),
            collateral_buffer_frac=_get_float("collateral_buffer_frac", 0.10),
            max_leverage=_get_float("max_leverage", 3.0),
            max_margin_utilization=_get_float("max_margin_utilization", 0.50),
            maintenance_margin_rate=_get_float("maintenance_margin_rate", 0.05),
            min_trade_notional_usd=_get_float("min_trade_notional_usd", 200.0),
            rebalance_min_notional_usd=_get_float("rebalance_min_notional_usd", 100.0),
            flip_hysteresis_bps=_get_float("flip_hysteresis_bps", 2.0),
            require_funding_rate=_get_bool("require_funding_rate", False),
        )

    raise ValueError(f"unknown strategy: {name}")


def list_strategy_names() -> list[str]:
    return [
        "spy_open_close",
        "no_trade",
        "ema_crossover",
        "ma_crossover",
        "nec_x",
        "nec_pdt",
        "orb_trend",
        "perp_flare",
        "perp_hawk",
        "perp_scalp",
        "hedge",
        "basis_carry",
    ]
