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
from atlas.strategies.perp_weekly_trend_reset import PerpWeeklyTrendReset
from atlas.strategies.perp_weekly_profit_chase import PerpWeeklyProfitChase
from atlas.strategies.basis_carry import BasisCarry
from atlas.strategies.hedge_implementation import HedgeImplementation
from atlas.strategies.crypto_ensemble import CryptoEnsemble
from atlas.strategies.crypto_meta import CryptoMeta
from atlas.strategies.crypto_momentum import CryptoMomentum
from atlas.strategies.crypto_weekly_lock_momentum import CryptoWeeklyLockMomentum
from atlas.strategies.crypto_rotation import CryptoRotation
from atlas.strategies.crypto_tsm import CryptoTSM
from atlas.strategies.crypto_regime_fusion import CryptoRegimeFusion
from atlas.strategies.crypto_regime_vol_target import CryptoRegimeVolTarget
from atlas.strategies.crypto_vol_squeeze import CryptoVolSqueeze
from atlas.strategies.crypto_7d_positive_gate import Crypto7DPositiveGate
from atlas.strategies.perp_quant_fusion import PerpQuantFusion
from atlas.strategies.perp_research_vol_momentum import PerpResearchVolMomentum
from atlas.strategies.perp_regime_adaptive_trend_capture import (
    PerpRegimeAdaptiveTrendCapture,
)
from atlas.strategies.perp_trend_vol_guard import PerpTrendVolGuard
from atlas.strategies.perp_weekly_carry_shield import PerpWeeklyCarryShield


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

        def _get_bool(key: str, default: bool) -> bool:
            raw = params.get(key, params.get(key.lower(), default))
            if isinstance(raw, bool):
                return raw
            if isinstance(raw, (int, float)):
                return bool(int(raw))
            if isinstance(raw, str):
                return raw.strip().lower() in {"1", "true", "t", "yes", "y", "on"}
            return bool(default)

        def _get_str(key: str, default: str) -> str:
            raw = params.get(key, params.get(key.lower(), default))
            return str(raw)

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

    if name in {"perp_weekly_trend_reset", "perp-weekly-trend-reset"}:
        universe_symbols = [s.strip().upper() for s in symbols if s.strip()]
        if not universe_symbols:
            raise ValueError("perp_weekly_trend_reset requires at least 1 symbol")

        def _get_int(key: str, default: int) -> int:
            raw = params.get(key, params.get(key.lower(), default))
            return int(raw)

        def _get_float(key: str, default: float) -> float:
            raw = params.get(key, params.get(key.lower(), default))
            return float(raw)

        def _get_opt_int(key: str, default: Optional[int]) -> Optional[int]:
            raw = params.get(key, params.get(key.lower(), default))
            if raw is None:
                return default
            return int(raw)

        def _get_bool(key: str, default: bool) -> bool:
            raw = params.get(key, params.get(key.lower(), default))
            if isinstance(raw, bool):
                return raw
            if isinstance(raw, (int, float)):
                return bool(int(raw))
            if isinstance(raw, str):
                return raw.strip().lower() in {"1", "true", "t", "yes", "y", "on"}
            return bool(default)

        return PerpWeeklyTrendReset(
            symbols=tuple(universe_symbols),
            lookback_days=_get_int("lookback_days", 14),
            momentum_threshold_bps=_get_float("momentum_threshold_bps", 0.0),
            ema_fast=_get_int("ema_fast", 12),
            ema_slow=_get_int("ema_slow", 48),
            require_ema_confirmation=_get_bool("require_ema_confirmation", False),
            target_leverage=_get_float("target_leverage", 8.0),
            max_margin_utilization=_get_float("max_margin_utilization", 0.80),
            maintenance_margin_rate=_get_float("maintenance_margin_rate", 0.05),
            stop_atr_mult=_get_float("stop_atr_mult", 2.5),
            trail_atr_mult=_get_float("trail_atr_mult", 4.0),
            atr_window=_get_int("atr_window", 14),
            min_liq_buffer_atr=_get_float("min_liq_buffer_atr", 4.0),
            use_stops=_get_bool("use_stops", False),
            rebalance_weekday_utc=_get_int("rebalance_weekday_utc", 0),
            rebalance_hour_utc=_get_int("rebalance_hour_utc", 0),
            rebalance_minute_utc=_get_int("rebalance_minute_utc", 0),
            weekly_nudge_exposure=_get_float("weekly_nudge_exposure", 0.002),
            min_trade_notional_usd=_get_float("min_trade_notional_usd", 10.0),
            heartbeat_exposure=_get_float("heartbeat_exposure", 0.03),
            heartbeat_hold_bars=_get_int("heartbeat_hold_bars", 12),
            max_hold_bars=_get_opt_int("max_hold_bars", None),
        )

    if name in {"perp_weekly_profit_chase", "perp-weekly-profit-chase"}:
        universe_symbols = [s.strip().upper() for s in symbols if s.strip()]
        if not universe_symbols:
            raise ValueError("perp_weekly_profit_chase requires at least 1 symbol")

        def _get_int(key: str, default: int) -> int:
            raw = params.get(key, params.get(key.lower(), default))
            return int(raw)

        def _get_float(key: str, default: float) -> float:
            raw = params.get(key, params.get(key.lower(), default))
            return float(raw)

        return PerpWeeklyProfitChase(
            symbols=tuple(universe_symbols),
            rebalance_weekday_utc=_get_int("rebalance_weekday_utc", 0),
            rebalance_hour_utc=_get_int("rebalance_hour_utc", 0),
            rebalance_minute_utc=_get_int("rebalance_minute_utc", 5),
            weekly_profit_target=_get_float("weekly_profit_target", 0.01),
            weekly_chase_k=_get_float("weekly_chase_k", 2.0),
            atr_window=_get_int("atr_window", 14),
            opening_range_minutes=_get_int("opening_range_minutes", 60),
            breakout_buffer_bps=_get_float("breakout_buffer_bps", 8.0),
            lookback_short_days=_get_float("lookback_short_days", 1.0),
            lookback_long_days=_get_float("lookback_long_days", 7.0),
            momentum_threshold_bps=_get_float("momentum_threshold_bps", 0.0),
            min_atr_bps=_get_float("min_atr_bps", 5.0),
            sizing_mode=str(params.get("sizing_mode", params.get("sizing_mode".lower(), "leverage"))),
            risk_per_trade=_get_float("risk_per_trade", 0.03),
            base_leverage=_get_float("base_leverage", 8.0),
            max_leverage=_get_float("max_leverage", 25.0),
            max_margin_utilization=_get_float("max_margin_utilization", 0.95),
            maintenance_margin_rate=_get_float("maintenance_margin_rate", 0.05),
            stop_atr_mult=_get_float("stop_atr_mult", 2.0),
            min_liq_buffer_atr=_get_float("min_liq_buffer_atr", 3.0),
            min_trade_notional_usd=_get_float("min_trade_notional_usd", 10.0),
            weekly_heartbeat_exposure=_get_float("weekly_heartbeat_exposure", 0.01),
            weekly_heartbeat_hold_bars=_get_int("weekly_heartbeat_hold_bars", 1),
            weekly_nudge_exposure=_get_float("weekly_nudge_exposure", 0.002),
            max_flips_per_day=_get_int("max_flips_per_day", 3),
            daily_loss_hard_stop=_get_float("daily_loss_hard_stop", 0.0),
            weekly_loss_hard_stop=_get_float("weekly_loss_hard_stop", 0.0),
            cooldown_bars_after_exit=_get_int("cooldown_bars_after_exit", 0),
            trailing_stop_atr_mult=_get_float("trailing_stop_atr_mult", 0.0),
            break_even_trigger_atr=_get_float("break_even_trigger_atr", 0.0),
            max_hold_bars=_get_int("max_hold_bars", 0),
        )

    if name in {"perp_trend_vol_guard", "perp-trend-vol-guard", "perp_tvg", "perp-tvg"}:
        universe_symbols = [s.strip().upper() for s in symbols if s.strip()]
        if not universe_symbols:
            raise ValueError("perp_trend_vol_guard requires at least 1 symbol")

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

        return PerpTrendVolGuard(
            symbols=tuple(universe_symbols),
            ema_fast=_get_int("ema_fast", 18),
            ema_slow=_get_int("ema_slow", 72),
            momentum_window_bars=_get_int("momentum_window_bars", 24),
            breakout_window=_get_int("breakout_window", 20),
            breakout_buffer_bps=_get_float("breakout_buffer_bps", 2.0),
            atr_window=_get_int("atr_window", 20),
            trend_strength_min=_get_float("trend_strength_min", 0.18),
            min_atr_bps=_get_float("min_atr_bps", 4.0),
            edge_floor_bps=_get_float("edge_floor_bps", 6.0),
            k_cost=_get_float("k_cost", 1.8),
            slippage_bps=_get_float("slippage_bps", 1.25),
            taker_fee_bps=_get_float("taker_fee_bps", 3.0),
            risk_budget=_get_float("risk_budget", 0.010),
            stop_atr_mult=_get_float("stop_atr_mult", 2.2),
            target_vol_bps_per_bar=_get_float("target_vol_bps_per_bar", 80.0),
            max_positions=_get_int("max_positions", 2),
            max_gross_exposure=_get_float("max_gross_exposure", 0.80),
            max_per_symbol_exposure=_get_float("max_per_symbol_exposure", 0.45),
            rebalance_interval_bars=_get_int("rebalance_interval_bars", 2),
            rebalance_exposure_threshold=_get_float(
                "rebalance_exposure_threshold", 0.03
            ),
            min_trade_notional_usd=_get_float("min_trade_notional_usd", 20.0),
            min_hold_bars=_get_int("min_hold_bars", 6),
            flip_confirm_bars=_get_int("flip_confirm_bars", 2),
            market_vol_reduce_bps=_get_float("market_vol_reduce_bps", 100.0),
            market_vol_off_bps=_get_float("market_vol_off_bps", 160.0),
            weekly_loss_limit=_get_float("weekly_loss_limit", 0.03),
            enable_weekly_profit_lock=_get_bool("enable_weekly_profit_lock", True),
            weekly_profit_target=_get_float("weekly_profit_target", 0.02),
            weekly_lock_risk_scale=_get_float("weekly_lock_risk_scale", 0.35),
            daily_loss_limit=_get_float("daily_loss_limit", 0.02),
            kill_switch=_get_float("kill_switch", 0.12),
        )

    if name in {"perp_quant_fusion", "perp-quant-fusion", "perp_qf", "perp-qf"}:
        universe_symbols = [s.strip().upper() for s in symbols if s.strip()]
        if not universe_symbols:
            raise ValueError("perp_quant_fusion requires at least 1 symbol")

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

        return PerpQuantFusion(
            symbols=tuple(universe_symbols),
            atr_window=_get_int("atr_window", 14),
            ema_fast=_get_int("ema_fast", 20),
            ema_slow=_get_int("ema_slow", 60),
            er_window=_get_int("er_window", 20),
            choppiness_window=_get_int("choppiness_window", 14),
            breakout_window=_get_int("breakout_window", 20),
            breakout_buffer_bps=_get_float("breakout_buffer_bps", 2.0),
            trend_z_min=_get_float("trend_z_min", 0.20),
            er_min=_get_float("er_min", 0.30),
            er_exit_min=_get_float("er_exit_min", 0.18),
            choppiness_max=_get_float("choppiness_max", 62.0),
            choppiness_exit_max=_get_float("choppiness_exit_max", 68.0),
            min_atr_bps=_get_float("min_atr_bps", 5.0),
            edge_floor_bps=_get_float("edge_floor_bps", 6.0),
            k_cost=_get_float("k_cost", 1.8),
            slippage_bps=_get_float("slippage_bps", 1.25),
            taker_fee_bps=_get_float("taker_fee_bps", 3.0),
            risk_budget=_get_float("risk_budget", 0.02),
            stop_atr_mult=_get_float("stop_atr_mult", 2.2),
            max_positions=_get_int("max_positions", 3),
            max_gross_exposure=_get_float("max_gross_exposure", 1.0),
            max_per_symbol_exposure=_get_float("max_per_symbol_exposure", 0.50),
            rebalance_exposure_threshold=_get_float(
                "rebalance_exposure_threshold", 0.03
            ),
            min_trade_notional_usd=_get_float("min_trade_notional_usd", 20.0),
            min_hold_bars=_get_int("min_hold_bars", 3),
            flip_confirm_bars=_get_int("flip_confirm_bars", 2),
            daily_loss_limit=_get_float("daily_loss_limit", 0.025),
            kill_switch=_get_float("kill_switch", 0.12),
            enable_weekly_profit_lock=_get_bool("enable_weekly_profit_lock", False),
            weekly_profit_target=_get_float("weekly_profit_target", 0.02),
            weekly_lock_risk_scale=_get_float("weekly_lock_risk_scale", 0.25),
            enable_weekly_heartbeat=_get_bool("enable_weekly_heartbeat", False),
            heartbeat_weekday_utc=_get_int("heartbeat_weekday_utc", 0),
            heartbeat_hour_utc=_get_int("heartbeat_hour_utc", 0),
            heartbeat_minute_utc=_get_int("heartbeat_minute_utc", 5),
            heartbeat_exposure=_get_float("heartbeat_exposure", 0.01),
            heartbeat_hold_bars=_get_int("heartbeat_hold_bars", 1),
        )

    if name in {
        "perp_research_vol_momentum",
        "perp-research-vol-momentum",
        "perp_research_vm",
        "perp-research-vm",
    }:
        universe_symbols = [s.strip().upper() for s in symbols if s.strip()]
        if not universe_symbols:
            raise ValueError("perp_research_vol_momentum requires at least 1 symbol")

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

        def _get_str(key: str, default: str) -> str:
            raw = params.get(key, params.get(key.lower(), default))
            return str(raw)

        def _get_int_list(key: str, default: tuple[int, ...]) -> tuple[int, ...]:
            raw = params.get(key, params.get(key.lower(), default))
            if raw is None:
                return tuple(int(v) for v in default)
            if isinstance(raw, (list, tuple)):
                values = [int(v) for v in raw]
                return tuple(values) if values else tuple(int(v) for v in default)
            if isinstance(raw, str):
                parts = [p.strip() for p in raw.replace(";", ",").split(",") if p.strip()]
                if parts:
                    return tuple(int(p) for p in parts)
            return tuple(int(v) for v in default)

        return PerpResearchVolMomentum(
            symbols=tuple(universe_symbols),
            rebalance_weekday_utc=_get_int("rebalance_weekday_utc", 0),
            rebalance_days_utc=_get_int_list("rebalance_days_utc", (0,)),
            rebalance_hour_utc=_get_int("rebalance_hour_utc", 0),
            rebalance_minute_utc=_get_int("rebalance_minute_utc", 0),
            long_momentum_bars=_get_int("long_momentum_bars", 24 * 14),
            short_momentum_bars=_get_int("short_momentum_bars", 24 * 2),
            ema_fast=_get_int("ema_fast", 24),
            ema_slow=_get_int("ema_slow", 24 * 7),
            atr_window=_get_int("atr_window", 24 * 2),
            vol_lookback_bars=_get_int("vol_lookback_bars", 24 * 5),
            vol_regime_window=_get_int("vol_regime_window", 24 * 30),
            min_abs_long_momentum_bps=_get_float("min_abs_long_momentum_bps", 45.0),
            min_atr_bps=_get_float("min_atr_bps", 8.0),
            trend_strength_min=_get_float("trend_strength_min", 0.10),
            edge_floor_bps=_get_float("edge_floor_bps", 8.0),
            k_cost=_get_float("k_cost", 2.6),
            expected_hold_bars=_get_int("expected_hold_bars", 120),
            signal_decay_factor=_get_float("signal_decay_factor", 0.55),
            min_net_edge_bps=_get_float("min_net_edge_bps", 18.0),
            trend_consistency_min=_get_float("trend_consistency_min", 0.75),
            trend_consistency_subwindows=_get_int("trend_consistency_subwindows", 4),
            target_vol_per_bar=_get_float("target_vol_per_bar", 0.0065),
            vol_floor=_get_float("vol_floor", 0.0020),
            max_leverage=_get_float("max_leverage", 4.0),
            max_margin_utilization=_get_float("max_margin_utilization", 0.40),
            max_gross_exposure=_get_float("max_gross_exposure", 0.95),
            max_per_symbol_exposure=_get_float("max_per_symbol_exposure", 0.95),
            max_positions=_get_int("max_positions", 1),
            min_trade_notional_usd=_get_float("min_trade_notional_usd", 25.0),
            rebalance_exposure_threshold=_get_float("rebalance_exposure_threshold", 0.04),
            vol_pctl_low=_get_float("vol_pctl_low", 0.15),
            vol_pctl_high=_get_float("vol_pctl_high", 0.82),
            crash_vol_z=_get_float("crash_vol_z", 1.25),
            crash_reversal_bps=_get_float("crash_reversal_bps", 55.0),
            crash_risk_scale=_get_float("crash_risk_scale", 0.30),
            vol_off_z=_get_float("vol_off_z", 2.4),
            stop_atr_mult=_get_float("stop_atr_mult", 3.2),
            trail_atr_mult=_get_float("trail_atr_mult", 4.2),
            min_hold_bars=_get_int("min_hold_bars", 24),
            max_hold_bars=_get_int("max_hold_bars", 24 * 10),
            max_loss_per_trade_pct=_get_float("max_loss_per_trade_pct", 0.015),
            weekly_loss_limit=_get_float("weekly_loss_limit", 0.03),
            daily_loss_limit=_get_float("daily_loss_limit", 0.02),
            kill_switch=_get_float("kill_switch", 0.20),
            mom_h1_bars=_get_int("mom_h1_bars", 48),
            mom_h2_bars=_get_int("mom_h2_bars", 168),
            mom_h3_bars=_get_int("mom_h3_bars", 504),
            mom_h4_bars=_get_int("mom_h4_bars", 1512),
            mom_w1=_get_float("mom_w1", 0.15),
            mom_w2=_get_float("mom_w2", 0.25),
            mom_w3=_get_float("mom_w3", 0.30),
            mom_w4=_get_float("mom_w4", 0.30),
            mom_z_scale=_get_float("mom_z_scale", 2.0),
            mom_score_min=_get_float("mom_score_min", 0.20),
            trend_regression_bars=_get_int("trend_regression_bars", 504),
            trend_tstat_entry=_get_float("trend_tstat_entry", 2.2),
            trend_tstat_full=_get_float("trend_tstat_full", 4.0),
            trend_tstat_exit=_get_float("trend_tstat_exit", 1.0),
            er_window_bars=_get_int("er_window_bars", 168),
            er_min=_get_float("er_min", 0.28),
            er_full=_get_float("er_full", 0.45),
            vol_short_span=_get_int("vol_short_span", 48),
            vol_long_span=_get_int("vol_long_span", 336),
            vol_ratio_delever=_get_float("vol_ratio_delever", 1.25),
            vol_ratio_off=_get_float("vol_ratio_off", 1.80),
            vol_ratio_power=_get_float("vol_ratio_power", 1.5),
            min_contracts=_get_int("min_contracts", 1),
            qty_rounding=_get_str("qty_rounding", "floor"),
            include_fixed_fee_in_cost=_get_bool("include_fixed_fee_in_cost", True),
            mom_exit_score=_get_float("mom_exit_score", 0.12),
            flip_exit_mom_score=_get_float("flip_exit_mom_score", 0.22),
            cooldown_bars=_get_int("cooldown_bars", 24),
            use_daily_loss_lockout=_get_bool("use_daily_loss_lockout", False),
            use_weekly_loss_lockout=_get_bool("use_weekly_loss_lockout", False),
        )

    if name in {
        "perp_regime_adaptive_trend_capture",
        "perp-regime-adaptive-trend-capture",
        "perp_ratc",
        "perp-ratc",
    }:
        universe_symbols = [s.strip().upper() for s in symbols if s.strip()]
        if not universe_symbols:
            raise ValueError("perp_regime_adaptive_trend_capture requires at least 1 symbol")

        def _get_int(key: str, default: int) -> int:
            raw = params.get(key, params.get(key.lower(), default))
            return int(raw)

        def _get_float(key: str, default: float) -> float:
            raw = params.get(key, params.get(key.lower(), default))
            return float(raw)

        return PerpRegimeAdaptiveTrendCapture(
            symbols=tuple(universe_symbols),
            mom_horizon_a=_get_int("mom_horizon_a", 168),
            mom_horizon_b=_get_int("mom_horizon_b", 504),
            mom_horizon_c=_get_int("mom_horizon_c", 1008),
            ema_fast_regime=_get_int("ema_fast_regime", 72),
            ema_slow_regime=_get_int("ema_slow_regime", 504),
            bear_exit_bps=_get_float("bear_exit_bps", 120.0),
            short_entry_bps=_get_float("short_entry_bps", 300.0),
            cooldown_bars=_get_int("cooldown_bars", 168),
            long_base_exposure=_get_float("long_base_exposure", 0.55),
            short_base_exposure=_get_float("short_base_exposure", 0.35),
            extreme_vol_scale=_get_float("extreme_vol_scale", 0.40),
            high_vol_scale=_get_float("high_vol_scale", 0.70),
            extreme_vol_rank=_get_float("extreme_vol_rank", 0.85),
            high_vol_rank=_get_float("high_vol_rank", 0.75),
            vol_lookback_bars=_get_int("vol_lookback_bars", 120),
            vol_regime_window=_get_int("vol_regime_window", 720),
            crash_threshold_bps=_get_float("crash_threshold_bps", 350.0),
            max_hold_bars=_get_int("max_hold_bars", 2016),
            rebalance_exposure_threshold=_get_float("rebalance_exposure_threshold", 0.02),
            daily_loss_limit=_get_float("daily_loss_limit", 0.05),
            weekly_loss_limit=_get_float("weekly_loss_limit", 0.07),
            kill_switch=_get_float("kill_switch", 0.25),
        )

    if name in {
        "perp_weekly_carry_shield",
        "perp-weekly-carry-shield",
        "perp_carry_shield",
        "perp-carry-shield",
    }:
        universe_symbols = [s.strip().upper() for s in symbols if s.strip()]
        if not universe_symbols:
            raise ValueError("perp_weekly_carry_shield requires at least 1 symbol")

        def _get_int(key: str, default: int) -> int:
            raw = params.get(key, params.get(key.lower(), default))
            return int(raw)

        def _get_float(key: str, default: float) -> float:
            raw = params.get(key, params.get(key.lower(), default))
            return float(raw)

        return PerpWeeklyCarryShield(
            symbols=tuple(universe_symbols),
            rebalance_weekday_utc=_get_int("rebalance_weekday_utc", 0),
            rebalance_hour_utc=_get_int("rebalance_hour_utc", 0),
            rebalance_minute_utc=_get_int("rebalance_minute_utc", 0),
            atr_window=_get_int("atr_window", 20),
            ema_fast=_get_int("ema_fast", 16),
            ema_slow=_get_int("ema_slow", 48),
            er_window=_get_int("er_window", 20),
            choppiness_window=_get_int("choppiness_window", 20),
            momentum_bars=_get_int("momentum_bars", 24),
            trend_z_min=_get_float("trend_z_min", 0.20),
            er_min=_get_float("er_min", 0.28),
            choppiness_max=_get_float("choppiness_max", 62.0),
            momentum_threshold_bps=_get_float("momentum_threshold_bps", 10.0),
            min_atr_bps=_get_float("min_atr_bps", 5.0),
            edge_floor_bps=_get_float("edge_floor_bps", 8.0),
            k_cost=_get_float("k_cost", 2.2),
            expected_move_atr_mult=_get_float("expected_move_atr_mult", 2.5),
            slippage_bps=_get_float("slippage_bps", 1.25),
            taker_fee_bps=_get_float("taker_fee_bps", 3.0),
            risk_budget=_get_float("risk_budget", 0.008),
            stop_atr_mult=_get_float("stop_atr_mult", 2.8),
            max_margin_utilization=_get_float("max_margin_utilization", 0.35),
            max_leverage=_get_float("max_leverage", 3.0),
            max_positions=_get_int("max_positions", 2),
            max_gross_exposure=_get_float("max_gross_exposure", 1.0),
            max_per_symbol_exposure=_get_float("max_per_symbol_exposure", 0.50),
            min_trade_notional_usd=_get_float("min_trade_notional_usd", 20.0),
            min_hold_bars=_get_int("min_hold_bars", 6),
            rebalance_exposure_threshold=_get_float(
                "rebalance_exposure_threshold", 0.04
            ),
            daily_loss_limit=_get_float("daily_loss_limit", 0.02),
            kill_switch=_get_float("kill_switch", 0.12),
            weekly_profit_target=_get_float("weekly_profit_target", 0.006),
            weekly_loss_limit=_get_float("weekly_loss_limit", 0.006),
            weekly_heartbeat_exposure=_get_float("weekly_heartbeat_exposure", 0.01),
            weekly_heartbeat_hold_bars=_get_int("weekly_heartbeat_hold_bars", 1),
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

    if name in {"crypto_ensemble", "crypto-ensemble", "crypto_ens", "crypto-ens"}:
        universe_symbols = [s.strip().upper() for s in symbols if s.strip()]
        if not universe_symbols:
            raise ValueError("crypto_ensemble requires at least 1 symbol")

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

        def _get_opt_str(key: str, default: Optional[str]) -> Optional[str]:
            raw = params.get(key, params.get(key.lower(), default))
            if raw is None:
                return default
            s = str(raw).strip()
            return s if s else default

        return CryptoEnsemble(
            symbols=tuple(universe_symbols),
            market_symbol=_get_opt_str("market_symbol", "BTC/USD"),
            ema_fast=_get_int("ema_fast", 20),
            ema_slow=_get_int("ema_slow", 80),
            atr_window=_get_int("atr_window", 20),
            er_window=_get_int("er_window", 40),
            breakout_window=_get_int("breakout_window", 60),
            momentum_window=_get_int("momentum_window", 240),
            er_trend_min=_get_float("er_trend_min", 0.35),
            er_range_max=_get_float("er_range_max", 0.20),
            trend_z_min=_get_float("trend_z_min", 0.20),
            min_atr_bps=_get_float("min_atr_bps", 6.0),
            meanrev_ewm_span=_get_int("meanrev_ewm_span", 120),
            meanrev_entry_z=_get_float("meanrev_entry_z", 1.75),
            meanrev_exit_z=_get_float("meanrev_exit_z", 0.50),
            rsi_window=_get_int("rsi_window", 14),
            rsi_oversold=_get_float("rsi_oversold", 35.0),
            rsi_overbought=_get_float("rsi_overbought", 65.0),
            require_vwap_alignment_for_trend=_get_bool(
                "require_vwap_alignment_for_trend", True
            ),
            meanrev_disable_cost_rt_bps=_get_float("meanrev_disable_cost_rt_bps", 30.0),
            meanrev_allow_bear_trend_long_only=_get_bool(
                "meanrev_allow_bear_trend_long_only", False
            ),
            meanrev_setup_max_bars=_get_int("meanrev_setup_max_bars", 8),
            meanrev_reversal_min_bps=_get_float("meanrev_reversal_min_bps", 0.0),
            meanrev_size_mult=_get_float("meanrev_size_mult", 0.35),
            meanrev_stop_atr_mult=_get_float("meanrev_stop_atr_mult", 0.0),
            meanrev_trail_atr_mult=_get_float("meanrev_trail_atr_mult", 0.0),
            meanrev_max_hold_bars=_get_int("meanrev_max_hold_bars", 0),
            breakout_buffer_bps=_get_float("breakout_buffer_bps", 2.0),
            confirm_bars=_get_int("confirm_bars", 2),
            max_positions=_get_int("max_positions", 3),
            max_gross_exposure=_get_float("max_gross_exposure", 1.0),
            max_exposure_per_symbol=_get_float("max_exposure_per_symbol", 1.0),
            risk_budget=_get_float("risk_budget", 0.02),
            stop_atr_mult=_get_float("stop_atr_mult", 2.0),
            trail_atr_mult=_get_float("trail_atr_mult", 3.0),
            take_profit_atr_mult=_get_float("take_profit_atr_mult", 0.0),
            max_hold_bars=_get_int("max_hold_bars", 0),
            min_hold_bars=_get_int("min_hold_bars", 3),
            cooldown_bars=_get_int("cooldown_bars", 6),
            flip_confirm_bars=_get_int("flip_confirm_bars", 3),
            min_dollar_volume_ewma=_get_float("min_dollar_volume_ewma", 50_000.0),
            dv_ewm_span=_get_int("dv_ewm_span", 60),
            rebalance_exposure_threshold=_get_float(
                "rebalance_exposure_threshold", 0.05
            ),
            min_trade_notional_usd=_get_float("min_trade_notional_usd", 25.0),
            slippage_bps=_get_float("slippage_bps", 3.0),
            taker_fee_bps=_get_float("taker_fee_bps", 25.0),
            edge_floor_bps=_get_float("edge_floor_bps", 4.0),
            k_cost=_get_float("k_cost", 2.0),
            daily_loss_limit=_get_float("daily_loss_limit", 0.03),
            kill_switch=_get_float("kill_switch", 0.12),
            kill_switch_cooldown_days=_get_int("kill_switch_cooldown_days", 7),
            market_drawdown_off=_get_float("market_drawdown_off", 0.15),
            market_drawdown_reduce=_get_float("market_drawdown_reduce", 0.08),
            market_vol_off_bps=_get_float("market_vol_off_bps", 250.0),
            market_vol_reduce_bps=_get_float("market_vol_reduce_bps", 150.0),
            market_peak_halflife_bars=_get_int("market_peak_halflife_bars", 240),
            heartbeat_every_bars=_get_int("heartbeat_every_bars", 0),
            heartbeat_notional_usd=_get_float("heartbeat_notional_usd", 25.0),
        )

    if name in {"crypto_tsm", "crypto-tsm", "crypto_trend", "crypto-trend"}:
        universe_symbols = [s.strip().upper() for s in symbols if s.strip()]
        if not universe_symbols:
            raise ValueError("crypto_tsm requires at least 1 symbol")

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

        def _get_opt_str(key: str, default: Optional[str]) -> Optional[str]:
            raw = params.get(key, params.get(key.lower(), default))
            if raw is None:
                return default
            s = str(raw).strip()
            return s if s else default

        return CryptoTSM(
            symbols=tuple(universe_symbols),
            market_symbol=_get_opt_str("market_symbol", "BTC/USD"),
            ema_fast=_get_int("ema_fast", 24),
            ema_slow=_get_int("ema_slow", 120),
            atr_window=_get_int("atr_window", 24),
            momentum_window=_get_int("momentum_window", 240),
            confirm_bars=_get_int("confirm_bars", 3),
            exit_confirm_bars=_get_int("exit_confirm_bars", 3),
            max_positions=_get_int("max_positions", 2),
            max_gross_exposure=_get_float("max_gross_exposure", 1.0),
            max_exposure_per_symbol=_get_float("max_exposure_per_symbol", 1.0),
            risk_budget=_get_float("risk_budget", 0.05),
            stop_atr_mult=_get_float("stop_atr_mult", 3.0),
            trail_atr_mult=_get_float("trail_atr_mult", 5.0),
            take_profit_atr_mult=_get_float("take_profit_atr_mult", 0.0),
            max_hold_bars=_get_int("max_hold_bars", 0),
            min_hold_bars=_get_int("min_hold_bars", 6),
            cooldown_bars=_get_int("cooldown_bars", 12),
            rebalance_interval_bars=_get_int("rebalance_interval_bars", 4),
            rebalance_exposure_threshold=_get_float("rebalance_exposure_threshold", 0.05),
            min_dollar_volume_ewma=_get_float("min_dollar_volume_ewma", 100_000.0),
            dv_ewm_span=_get_int("dv_ewm_span", 60),
            min_trade_notional_usd=_get_float("min_trade_notional_usd", 25.0),
            slippage_bps=_get_float("slippage_bps", 3.0),
            taker_fee_bps=_get_float("taker_fee_bps", 25.0),
            edge_floor_bps=_get_float("edge_floor_bps", 8.0),
            k_cost=_get_float("k_cost", 2.0),
            daily_loss_limit=_get_float("daily_loss_limit", 0.05),
            kill_switch=_get_float("kill_switch", 0.20),
            kill_switch_cooldown_days=_get_int("kill_switch_cooldown_days", 7),
            market_drawdown_off=_get_float("market_drawdown_off", 0.20),
            market_drawdown_reduce=_get_float("market_drawdown_reduce", 0.10),
            market_vol_off_bps=_get_float("market_vol_off_bps", 300.0),
            market_vol_reduce_bps=_get_float("market_vol_reduce_bps", 180.0),
            market_peak_halflife_bars=_get_int("market_peak_halflife_bars", 240),
        )

    if name in {"crypto_rotation", "crypto-rotation", "crypto_rot", "crypto-rot"}:
        universe_symbols = [s.strip().upper() for s in symbols if s.strip()]
        if not universe_symbols:
            raise ValueError("crypto_rotation requires at least 1 symbol")

        def _get_int(key: str, default: int) -> int:
            raw = params.get(key, params.get(key.lower(), default))
            return int(raw)

        def _get_float(key: str, default: float) -> float:
            raw = params.get(key, params.get(key.lower(), default))
            return float(raw)

        def _get_opt_str(key: str, default: Optional[str]) -> Optional[str]:
            raw = params.get(key, params.get(key.lower(), default))
            if raw is None:
                return default
            s = str(raw).strip()
            return s if s else default

        return CryptoRotation(
            symbols=tuple(universe_symbols),
            market_symbol=_get_opt_str("market_symbol", "BTC/USD"),
            rebalance_interval_bars=_get_int("rebalance_interval_bars", 28),
            min_trade_notional_usd=_get_float("min_trade_notional_usd", 25.0),
            rebalance_exposure_threshold=_get_float("rebalance_exposure_threshold", 0.02),
            mom_short_bars=_get_int("mom_short_bars", 28),
            mom_med_bars=_get_int("mom_med_bars", 120),
            mom_long_bars=_get_int("mom_long_bars", 360),
            w_mom_short=_get_float("w_mom_short", 0.20),
            w_mom_med=_get_float("w_mom_med", 0.30),
            w_mom_long=_get_float("w_mom_long", 0.50),
            vol_window_bars=_get_int("vol_window_bars", 120),
            vol_target_bps_per_bar=_get_float("vol_target_bps_per_bar", 80.0),
            max_total_exposure=_get_float("max_total_exposure", 1.0),
            max_exposure_per_symbol=_get_float("max_exposure_per_symbol", 0.60),
            top_k=_get_int("top_k", 2),
            score_floor=_get_float("score_floor", 0.0),
            dv_ewm_span=_get_int("dv_ewm_span", 60),
            min_dollar_volume_ewma=_get_float("min_dollar_volume_ewma", 50_000.0),
            slippage_bps=_get_float("slippage_bps", 3.0),
            taker_fee_bps=_get_float("taker_fee_bps", 25.0),
            k_cost=_get_float("k_cost", 1.0),
            edge_floor_bps=_get_float("edge_floor_bps", 0.0),
            daily_loss_limit=_get_float("daily_loss_limit", 0.05),
            kill_switch=_get_float("kill_switch", 0.25),
            kill_switch_cooldown_days=_get_int("kill_switch_cooldown_days", 7),
            market_drawdown_off=_get_float("market_drawdown_off", 0.25),
            market_drawdown_reduce=_get_float("market_drawdown_reduce", 0.12),
            market_vol_off_bps=_get_float("market_vol_off_bps", 300.0),
            market_vol_reduce_bps=_get_float("market_vol_reduce_bps", 180.0),
            market_peak_halflife_bars=_get_int("market_peak_halflife_bars", 240),
            market_mom_bars=_get_int("market_mom_bars", 0),
            market_mom_off=_get_float("market_mom_off", 0.0),
            market_mom_reduce=_get_float("market_mom_reduce", 0.0),
            heartbeat_every_bars=_get_int("heartbeat_every_bars", 0),
            heartbeat_notional_usd=_get_float("heartbeat_notional_usd", 25.0),
        )

    if name in {"crypto_momentum", "crypto-momentum"}:
        universe_symbols = [s.strip().upper() for s in symbols if s.strip()]
        if not universe_symbols:
            raise ValueError("crypto_momentum requires at least 1 symbol")

        def _get_int(key: str, default: int) -> int:
            raw = params.get(key, params.get(key.lower(), default))
            return int(raw)

        def _get_float(key: str, default: float) -> float:
            raw = params.get(key, params.get(key.lower(), default))
            return float(raw)

        def _get_opt_str(key: str, default: Optional[str]) -> Optional[str]:
            raw = params.get(key, params.get(key.lower(), default))
            if raw is None:
                return default
            s = str(raw).strip()
            return s if s else default

        return CryptoMomentum(
            symbols=tuple(universe_symbols),
            market_symbol=_get_opt_str("market_symbol", "BTC/USD"),
            momentum_window_bars=_get_int("momentum_window_bars", 240),
            max_total_exposure=_get_float("max_total_exposure", 1.0),
            max_exposure_per_symbol=_get_float("max_exposure_per_symbol", 1.0),
            rebalance_interval_bars=_get_int("rebalance_interval_bars", 28),
            rebalance_exposure_threshold=_get_float("rebalance_exposure_threshold", 0.10),
            min_trade_notional_usd=_get_float("min_trade_notional_usd", 25.0),
            heartbeat_every_bars=_get_int("heartbeat_every_bars", 0),
            heartbeat_notional_usd=_get_float("heartbeat_notional_usd", 1.0),
        )

    if name in {
        "crypto_weekly_lock_momentum",
        "crypto-weekly-lock-momentum",
        "crypto_weekly_lock",
        "crypto-weekly-lock",
    }:
        universe_symbols = [s.strip().upper() for s in symbols if s.strip()]
        if not universe_symbols:
            raise ValueError("crypto_weekly_lock_momentum requires at least 1 symbol")

        def _get_int(key: str, default: int) -> int:
            raw = params.get(key, params.get(key.lower(), default))
            return int(raw)

        def _get_float(key: str, default: float) -> float:
            raw = params.get(key, params.get(key.lower(), default))
            return float(raw)

        def _get_opt_str(key: str, default: Optional[str]) -> Optional[str]:
            raw = params.get(key, params.get(key.lower(), default))
            if raw is None:
                return default
            s = str(raw).strip()
            return s if s else default

        return CryptoWeeklyLockMomentum(
            symbols=tuple(universe_symbols),
            market_symbol=_get_opt_str("market_symbol", "BTC/USD"),
            rebalance_interval_bars=_get_int("rebalance_interval_bars", 8),
            rebalance_exposure_threshold=_get_float("rebalance_exposure_threshold", 0.05),
            mom_short_bars=_get_int("mom_short_bars", 28),
            mom_med_bars=_get_int("mom_med_bars", 84),
            mom_long_bars=_get_int("mom_long_bars", 252),
            w_mom_short=_get_float("w_mom_short", 0.20),
            w_mom_med=_get_float("w_mom_med", 0.35),
            w_mom_long=_get_float("w_mom_long", 0.45),
            vol_window_bars=_get_int("vol_window_bars", 84),
            top_k=_get_int("top_k", 2),
            score_floor=_get_float("score_floor", 0.0),
            max_total_exposure=_get_float("max_total_exposure", 1.0),
            max_exposure_per_symbol=_get_float("max_exposure_per_symbol", 0.70),
            vol_target_bps_per_bar=_get_float("vol_target_bps_per_bar", 90.0),
            min_trade_notional_usd=_get_float("min_trade_notional_usd", 25.0),
            regime_ema_bars=_get_int("regime_ema_bars", 168),
            regime_mom_bars=_get_int("regime_mom_bars", 84),
            regime_mom_off=_get_float("regime_mom_off", 0.0),
            regime_dd_off=_get_float("regime_dd_off", 0.15),
            regime_dd_reduce=_get_float("regime_dd_reduce", 0.08),
            regime_peak_lookback_bars=_get_int("regime_peak_lookback_bars", 252),
            weekly_profit_target=_get_float("weekly_profit_target", 0.010),
            weekly_loss_limit=_get_float("weekly_loss_limit", 0.012),
            daily_loss_limit=_get_float("daily_loss_limit", 0.03),
            kill_switch=_get_float("kill_switch", 0.15),
            kill_switch_cooldown_days=_get_int("kill_switch_cooldown_days", 5),
        )

    if name in {"crypto_vol_squeeze", "crypto-vol-squeeze", "crypto_squeeze", "crypto-squeeze"}:
        universe_symbols = [s.strip().upper() for s in symbols if s.strip()]
        if not universe_symbols:
            raise ValueError("crypto_vol_squeeze requires at least 1 symbol")

        def _get_int(key: str, default: int) -> int:
            raw = params.get(key, params.get(key.lower(), default))
            return int(raw)

        def _get_float(key: str, default: float) -> float:
            raw = params.get(key, params.get(key.lower(), default))
            return float(raw)

        def _get_opt_str(key: str, default: Optional[str]) -> Optional[str]:
            raw = params.get(key, params.get(key.lower(), default))
            if raw is None:
                return default
            s = str(raw).strip()
            return s if s else default

        return CryptoVolSqueeze(
            symbols=tuple(universe_symbols),
            market_symbol=_get_opt_str("market_symbol", "BTC/USD"),
            rebalance_interval_bars=_get_int("rebalance_interval_bars", 28),
            rebalance_exposure_threshold=_get_float(
                "rebalance_exposure_threshold", 0.05
            ),
            min_trade_notional_usd=_get_float("min_trade_notional_usd", 25.0),
            bb_window=_get_int("bb_window", 20),
            bb_k=_get_float("bb_k", 2.0),
            squeeze_lookback=_get_int("squeeze_lookback", 120),
            squeeze_percentile=_get_float("squeeze_percentile", 0.20),
            donchian_window=_get_int("donchian_window", 40),
            atr_window=_get_int("atr_window", 20),
            min_atr_bps=_get_float("min_atr_bps", 12.0),
            entry_breakout_buffer_bps=_get_float("entry_breakout_buffer_bps", 8.0),
            expected_move_atr_mult=_get_float("expected_move_atr_mult", 2.0),
            cost_k=_get_float("cost_k", 2.5),
            edge_floor_bps=_get_float("edge_floor_bps", 4.0),
            slippage_bps=_get_float("slippage_bps", 3.0),
            taker_fee_bps=_get_float("taker_fee_bps", 25.0),
            max_total_exposure=_get_float("max_total_exposure", 1.0),
            max_exposure_per_symbol=_get_float("max_exposure_per_symbol", 0.55),
            vol_target_bps_per_bar=_get_float("vol_target_bps_per_bar", 70.0),
            exposure_scale_on_squeeze=_get_float("exposure_scale_on_squeeze", 1.0),
            min_hold_bars=_get_int("min_hold_bars", 12),
            max_hold_bars=_get_int("max_hold_bars", 56),
            exit_mom_bars=_get_int("exit_mom_bars", 24),
            exit_mom_threshold=_get_float("exit_mom_threshold", 0.0),
            daily_loss_limit=_get_float("daily_loss_limit", 0.03),
            kill_switch=_get_float("kill_switch", 0.12),
            kill_switch_cooldown_days=_get_int("kill_switch_cooldown_days", 4),
            market_drawdown_off=_get_float("market_drawdown_off", 0.20),
            market_drawdown_reduce=_get_float("market_drawdown_reduce", 0.10),
            market_vol_off_bps=_get_float("market_vol_off_bps", 260.0),
            market_vol_reduce_bps=_get_float("market_vol_reduce_bps", 160.0),
        )

    if name in {
        "crypto_7d_positive_gate",
        "crypto-7d-positive-gate",
        "crypto_7d_gate",
        "crypto-7d-gate",
    }:
        universe_symbols = [s.strip().upper() for s in symbols if s.strip()]
        if not universe_symbols:
            raise ValueError("crypto_7d_positive_gate requires at least 1 symbol")

        def _get_int(key: str, default: int) -> int:
            raw = params.get(key, params.get(key.lower(), default))
            return int(raw)

        def _get_float(key: str, default: float) -> float:
            raw = params.get(key, params.get(key.lower(), default))
            return float(raw)

        def _get_opt_str(key: str, default: Optional[str]) -> Optional[str]:
            raw = params.get(key, params.get(key.lower(), default))
            if raw is None:
                return default
            s = str(raw).strip()
            return s if s else default

        return Crypto7DPositiveGate(
            symbols=tuple(universe_symbols),
            market_symbol=_get_opt_str("market_symbol", "BTC/USD"),
            pos7_lookback_windows=_get_int("pos7_lookback_windows", 120),
            pos7_on=_get_float("pos7_on", 0.60),
            pos7_off=_get_float("pos7_off", 0.52),
            pos7_reset_floor=_get_float("pos7_reset_floor", 0.50),
            trend_ema_bars=_get_int("trend_ema_bars", 96),
            trend_on=_get_float("trend_on", 0.005),
            trend_off=_get_float("trend_off", -0.002),
            donchian_bars=_get_int("donchian_bars", 36),
            entry_buffer_bps=_get_float("entry_buffer_bps", 4.0),
            atr_bars=_get_int("atr_bars", 20),
            min_atr_bps=_get_float("min_atr_bps", 6.0),
            expected_move_atr_mult=_get_float("expected_move_atr_mult", 2.2),
            edge_floor_bps=_get_float("edge_floor_bps", 8.0),
            taker_fee_bps=_get_float("taker_fee_bps", 25.0),
            slippage_bps=_get_float("slippage_bps", 3.0),
            min_reentry_bars=_get_int("min_reentry_bars", 28),
            max_hold_bars=_get_int("max_hold_bars", 280),
            cooldown_bars=_get_int("cooldown_bars", 32),
            max_total_exposure=_get_float("max_total_exposure", 0.70),
            max_exposure_per_symbol=_get_float("max_exposure_per_symbol", 0.40),
            vol_target_bps_per_bar=_get_float("vol_target_bps_per_bar", 75.0),
            edge_scale_bps=_get_float("edge_scale_bps", 40.0),
            min_trade_notional_usd=_get_float("min_trade_notional_usd", 20.0),
            stop_atr_mult=_get_float("stop_atr_mult", 2.2),
            take_profit_atr_mult=_get_float("take_profit_atr_mult", 4.0),
            rebalance_exposure_threshold=_get_float(
                "rebalance_exposure_threshold", 0.06
            ),
            daily_loss_limit=_get_float("daily_loss_limit", 0.03),
            kill_switch=_get_float("kill_switch", 0.20),
            mkt_peak_lookback=_get_int("mkt_peak_lookback", 120),
            mkt_vol_bars=_get_int("mkt_vol_bars", 40),
            mkt_dd_off=_get_float("mkt_dd_off", 0.20),
            mkt_vol_off_bps=_get_float("mkt_vol_off_bps", 200.0),
        )

    if name in {"crypto_meta", "crypto-meta"}:
        universe_symbols = [s.strip().upper() for s in symbols if s.strip()]
        if not universe_symbols:
            raise ValueError("crypto_meta requires at least 1 symbol")

        def _get_int(key: str, default: int) -> int:
            raw = params.get(key, params.get(key.lower(), default))
            return int(raw)

        def _get_float(key: str, default: float) -> float:
            raw = params.get(key, params.get(key.lower(), default))
            return float(raw)

        def _get_str(key: str, default: str) -> str:
            raw = params.get(key, params.get(key.lower(), default))
            return str(raw)

        def _get_opt_str(key: str, default: Optional[str]) -> Optional[str]:
            raw = params.get(key, params.get(key.lower(), default))
            if raw is None:
                return default
            s = str(raw).strip()
            return s if s else default

        def _parse_symbols_csv(raw: str) -> list[str]:
            parts = [p.strip().upper() for p in str(raw or "").split(",") if p.strip()]
            out: list[str] = []
            seen: set[str] = set()
            for p in parts:
                if p and p not in seen:
                    seen.add(p)
                    out.append(p)
            return out

        def _load_child_params_file(raw_path: str) -> dict[str, Any]:
            rel = str(raw_path or "").strip()
            if not rel:
                return {}

            repo_root = Path.cwd().resolve()
            preset_root = (repo_root / "strategy_params").resolve()
            child_path = Path(rel)
            resolved = child_path.resolve() if child_path.is_absolute() else (repo_root / child_path).resolve()

            if preset_root not in resolved.parents:
                raise ValueError("crypto_meta child params file must be under strategy_params/")
            if resolved.suffix.lower() != ".json":
                raise ValueError("crypto_meta child params file must be a .json")
            return _load_params(resolved)

        market_symbol = _get_opt_str("market_symbol", "BTC/USD")
        regime_mom_bars = _get_int("regime_mom_bars", 168)
        regime_mom_threshold = _get_float("regime_mom_threshold", 0.0)
        regime_er_bars = _get_int("regime_er_bars", 84)
        regime_er_min = _get_float("regime_er_min", 0.25)
        rotation_weight_trending = _get_float("rotation_weight_trending", 1.0)
        rotation_weight_ranging = _get_float("rotation_weight_ranging", 0.0)

        ensemble_params_file = _get_str(
            "ensemble_params_file",
            "strategy_params/crypto_ensemble_ultra_6h_coinbase_heartbeat.json",
        )
        rotation_params_file = _get_str(
            "rotation_params_file",
            "strategy_params/crypto_rotation_2022_candidate_r2_momfilter_v11_6h_coinbase_nohb.json",
        )

        ensemble_symbols_csv = _get_str("ensemble_symbols", "BTC/USD,ETH/USD")
        rotation_symbols_csv = _get_str("rotation_symbols", "")

        ensemble_symbols = [
            s for s in _parse_symbols_csv(ensemble_symbols_csv) if s in universe_symbols
        ]
        if not ensemble_symbols:
            ensemble_symbols = universe_symbols[: min(2, len(universe_symbols))]

        rotation_symbols = [
            s for s in _parse_symbols_csv(rotation_symbols_csv) if s in universe_symbols
        ]
        if not rotation_symbols:
            rotation_symbols = list(universe_symbols)

        ensemble_params = _load_child_params_file(ensemble_params_file)
        rotation_params = _load_child_params_file(rotation_params_file)

        ensemble = build_strategy(
            name="crypto_ensemble",
            params_path=None,
            params=ensemble_params,
            symbols=ensemble_symbols,
            fast_window=fast_window,
            slow_window=slow_window,
        )
        rotation = build_strategy(
            name="crypto_rotation",
            params_path=None,
            params=rotation_params,
            symbols=rotation_symbols,
            fast_window=fast_window,
            slow_window=slow_window,
        )

        return CryptoMeta(
            symbols=tuple(universe_symbols),
            market_symbol=str(market_symbol or "BTC/USD"),
            regime_mom_bars=regime_mom_bars,
            regime_mom_threshold=regime_mom_threshold,
            regime_er_bars=regime_er_bars,
            regime_er_min=regime_er_min,
            rotation_weight_trending=rotation_weight_trending,
            rotation_weight_ranging=rotation_weight_ranging,
            ensemble_params_file=str(ensemble_params_file),
            rotation_params_file=str(rotation_params_file),
            ensemble_symbols=str(ensemble_symbols_csv),
            rotation_symbols=str(rotation_symbols_csv),
            ensemble=ensemble,
            rotation=rotation,
        )

    if name in {
        "crypto_regime_vol_target",
        "crypto-regime-vol-target",
        "crypto_rvt",
        "crypto-rvt",
    }:
        universe_symbols = [s.strip().upper() for s in symbols if s.strip()]
        if not universe_symbols:
            raise ValueError("crypto_regime_vol_target requires at least 1 symbol")

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

        def _get_opt_str(key: str, default: Optional[str]) -> Optional[str]:
            raw = params.get(key, params.get(key.lower(), default))
            if raw is None:
                return default
            s = str(raw).strip()
            return s if s else default

        return CryptoRegimeVolTarget(
            symbols=tuple(universe_symbols),
            market_symbol=_get_opt_str("market_symbol", "BTC/USD"),
            fast_window=_get_int("fast_window", 20),
            slow_window=_get_int("slow_window", 80),
            regime_window=_get_int("regime_window", 200),
            regime_slope_bars=_get_int("regime_slope_bars", 10),
            momentum_window_bars=_get_int("momentum_window_bars", 120),
            atr_window=_get_int("atr_window", 20),
            top_k=_get_int("top_k", 2),
            target_vol_bps_per_bar=_get_float("target_vol_bps_per_bar", 70.0),
            max_total_exposure=_get_float("max_total_exposure", 1.0),
            max_exposure_per_symbol=_get_float("max_exposure_per_symbol", 0.70),
            rebalance_interval_bars=_get_int("rebalance_interval_bars", 8),
            rebalance_exposure_threshold=_get_float(
                "rebalance_exposure_threshold", 0.04
            ),
            min_trade_notional_usd=_get_float("min_trade_notional_usd", 25.0),
            market_drawdown_reduce=_get_float("market_drawdown_reduce", 0.08),
            market_drawdown_off=_get_float("market_drawdown_off", 0.16),
            market_peak_lookback_bars=_get_int("market_peak_lookback_bars", 240),
            weekly_loss_limit=_get_float("weekly_loss_limit", 0.04),
            enable_weekly_profit_lock=_get_bool("enable_weekly_profit_lock", True),
            weekly_profit_target=_get_float("weekly_profit_target", 0.03),
            daily_loss_limit=_get_float("daily_loss_limit", 0.03),
            kill_switch=_get_float("kill_switch", 0.15),
            kill_switch_cooldown_days=_get_int("kill_switch_cooldown_days", 5),
            trailing_stop_pct=_get_float("trailing_stop_pct", 0.10),
            min_hold_bars=_get_int("min_hold_bars", 6),
        )

    if name in {
        "crypto_regime_fusion",
        "crypto-regime-fusion",
        "crypto_regime",
        "crypto-regime",
    }:
        universe_symbols = [s.strip().upper() for s in symbols if s.strip()]
        if not universe_symbols:
            raise ValueError("crypto_regime_fusion requires at least 1 symbol")

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

        def _get_opt_str(key: str, default: Optional[str]) -> Optional[str]:
            raw = params.get(key, params.get(key.lower(), default))
            if raw is None:
                return default
            s = str(raw).strip()
            return s if s else default

        return CryptoRegimeFusion(
            symbols=tuple(universe_symbols),
            market_symbol=_get_opt_str("market_symbol", "BTC/USD"),
            regime_momentum_bars=_get_int("regime_momentum_bars", 120),
            regime_er_bars=_get_int("regime_er_bars", 80),
            regime_atr_bars=_get_int("regime_atr_bars", 40),
            regime_ema_fast=_get_int("regime_ema_fast", 24),
            regime_ema_slow=_get_int("regime_ema_slow", 96),
            regime_trend_mom=_get_float("regime_trend_mom", 0.02),
            regime_trend_er_min=_get_float("regime_trend_er_min", 0.30),
            regime_trend_strength_min=_get_float("regime_trend_strength_min", 0.20),
            regime_range_abs_mom_max=_get_float("regime_range_abs_mom_max", 0.015),
            regime_range_er_max=_get_float("regime_range_er_max", 0.18),
            momentum_window_bars=_get_int("momentum_window_bars", 120),
            vol_window_bars=_get_int("vol_window_bars", 80),
            trend_top_k=_get_int("trend_top_k", 3),
            trend_score_floor=_get_float("trend_score_floor", 0.0),
            max_total_exposure=_get_float("max_total_exposure", 1.0),
            max_exposure_per_symbol=_get_float("max_exposure_per_symbol", 0.55),
            neutral_exposure_scale=_get_float("neutral_exposure_scale", 0.25),
            meanrev_window_bars=_get_int("meanrev_window_bars", 72),
            meanrev_entry_z=_get_float("meanrev_entry_z", 1.5),
            meanrev_exit_z=_get_float("meanrev_exit_z", 0.5),
            meanrev_max_z=_get_float("meanrev_max_z", 4.0),
            range_min_exposure=_get_float("range_min_exposure", 0.15),
            range_max_exposure=_get_float("range_max_exposure", 0.45),
            rebalance_interval_bars=_get_int("rebalance_interval_bars", 8),
            rebalance_exposure_threshold=_get_float(
                "rebalance_exposure_threshold", 0.03
            ),
            min_trade_notional_usd=_get_float("min_trade_notional_usd", 25.0),
            daily_loss_limit=_get_float("daily_loss_limit", 0.04),
            kill_switch=_get_float("kill_switch", 0.12),
            kill_switch_cooldown_days=_get_int("kill_switch_cooldown_days", 2),
            heartbeat_every_bars=_get_int("heartbeat_every_bars", 0),
            heartbeat_notional_usd=_get_float("heartbeat_notional_usd", 1.0),
            heartbeat_max_exposure_delta=_get_float(
                "heartbeat_max_exposure_delta", 0.02
            ),
            heartbeat_respect_min_trade_notional=_get_bool(
                "heartbeat_respect_min_trade_notional", False
            ),
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
        "crypto_ensemble",
        "crypto_meta",
        "crypto_tsm",
        "crypto_rotation",
        "crypto_momentum",
        "crypto_weekly_lock_momentum",
        "crypto_regime_vol_target",
        "crypto_regime_fusion",
        "crypto_vol_squeeze",
        "crypto_7d_positive_gate",
        "perp_flare",
        "perp_hawk",
        "perp_scalp",
        "perp_trend_vol_guard",
        "perp_quant_fusion",
        "perp_research_vol_momentum",
        "perp_regime_adaptive_trend_capture",
        "perp_weekly_carry_shield",
        "perp_weekly_trend_reset",
        "perp_weekly_profit_chase",
        "hedge",
        "basis_carry",
    ]
