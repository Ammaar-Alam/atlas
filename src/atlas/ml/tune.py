from __future__ import annotations

import concurrent.futures
import json
import logging
import math
import os
import random
import shutil
import statistics
import tempfile
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from threading import Event
from typing import Any, Callable, Optional, Union

import pandas as pd

from atlas.backtest.derivatives_engine import run_derivatives_backtest
from atlas.backtest.engine import BacktestConfig, BacktestOutputs, run_backtest
from atlas.backtest.metrics import compute_metrics
from atlas.market import Market, parse_market
from atlas.strategies.base import Strategy
from atlas.strategies.registry import build_strategy

logger = logging.getLogger(__name__)


def parse_duration_spec(spec: str) -> timedelta:
    """
    Parse compact duration strings used across Atlas config.

    Supported suffixes:
    - min (minutes)
    - h   (hours)
    - d   (days)
    - w   (weeks)
    - m   (months ~= 30d)
    - y   (years  ~= 365d)
    """
    raw = (spec or "").strip().lower()
    if not raw:
        raise ValueError("duration spec is required")
    if raw.endswith("min"):
        return timedelta(minutes=int(raw[:-3]))
    if raw.endswith("h"):
        return timedelta(hours=int(raw[:-1]))
    if raw.endswith("d"):
        return timedelta(days=int(raw[:-1]))
    if raw.endswith("w"):
        return timedelta(weeks=int(raw[:-1]))
    if raw.endswith("m"):
        return timedelta(days=30 * int(raw[:-1]))
    if raw.endswith("y"):
        return timedelta(days=365 * int(raw[:-1]))
    raise ValueError(f"unsupported duration spec: {spec!r}")


@dataclass(frozen=True)
class IntRange:
    name: str
    lo: int
    hi: int
    step: int = 1
    log: bool = False

    def sample(self, rng: random.Random) -> int:
        lo = int(self.lo)
        hi = int(self.hi)
        step = max(1, int(self.step))
        if lo > hi:
            lo, hi = hi, lo
        if lo == hi:
            return lo
        if self.log:
            # Sample uniformly in log-space across [lo, hi] inclusive.
            lo_f = math.log(float(max(lo, 1)))
            hi_f = math.log(float(max(hi, 1)))
            draw = math.exp(rng.uniform(lo_f, hi_f))
            value = int(round(draw))
        else:
            value = rng.randint(lo, hi)
        # Snap to step grid.
        value = int(round(value / step) * step)
        return int(max(lo, min(hi, value)))


@dataclass(frozen=True)
class FloatRange:
    name: str
    lo: float
    hi: float
    log: bool = False
    decimals: Optional[int] = None

    def sample(self, rng: random.Random) -> float:
        lo = float(self.lo)
        hi = float(self.hi)
        if lo > hi:
            lo, hi = hi, lo
        if lo == hi:
            value = lo
        elif self.log:
            lo_f = math.log(float(max(lo, 1e-12)))
            hi_f = math.log(float(max(hi, 1e-12)))
            value = float(math.exp(rng.uniform(lo_f, hi_f)))
        else:
            value = float(rng.uniform(lo, hi))
        if self.decimals is not None:
            value = round(float(value), int(self.decimals))
        return float(max(lo, min(hi, value)))


Param = Union[IntRange, FloatRange]


def _perp_flare_space() -> list[Param]:
    # These bounds are intentionally "wide but plausible".
    return [
        IntRange("atr_window", 8, 80, log=True),
        IntRange("ema_fast", 4, 120, log=True),
        IntRange("ema_slow", 10, 260, log=True),
        IntRange("er_window", 5, 80, log=True),
        IntRange("breakout_window", 8, 160, log=True),
        FloatRange("er_min", 0.15, 0.85, decimals=3),
        FloatRange("edge_floor_bps", 0.0, 30.0, decimals=3),
        FloatRange("k_cost", 0.25, 8.0, decimals=3),
        FloatRange("risk_per_trade", 0.002, 0.25, log=True, decimals=6),
        FloatRange("stop_atr_mult", 0.75, 8.0, decimals=3),
        FloatRange("trail_atr_mult", 0.75, 14.0, decimals=3),
        FloatRange("max_margin_utilization", 0.10, 0.95, decimals=4),
        FloatRange("max_leverage", 1.0, 25.0, decimals=4),
        FloatRange("min_liq_buffer_atr", 0.5, 12.0, decimals=3),
    ]


def _orb_trend_space() -> list[Param]:
    # Excludes environment params like slippage_bps (handled by engine/backtest config).
    return [
        IntRange("orb_minutes", 5, 90, log=True),
        FloatRange("orb_breakout_bps", 1.0, 25.0, decimals=3),
        IntRange("confirm_bars", 1, 6),
        IntRange("atr_window", 8, 60, log=True),
        IntRange("er_window", 5, 60, log=True),
        FloatRange("er_min", 0.10, 0.85, decimals=3),
        IntRange("expected_hold_bars", 4, 60, log=True),
        FloatRange("k_cost", 0.5, 6.0, decimals=3),
        IntRange("min_hold_bars", 1, 12),
        FloatRange("daily_loss_limit", 0.003, 0.05, decimals=4),
        FloatRange("kill_switch", 0.01, 0.10, decimals=4),
    ]


def _hedge_space() -> list[Param]:
    # Pair hedge (spot + perp). We tune the forecasting horizon, mean reversion, risk gating,
    # and turnover controls. Fees/slippage are handled by the backtest config.
    return [
        FloatRange("edge_horizon_hours", 2.0, 24.0, log=True, decimals=3),
        FloatRange("basis_halflife_hours", 6.0, 96.0, log=True, decimals=3),
        FloatRange("theta_intercept_bps", -50.0, 50.0, decimals=3),
        IntRange("cov_window_bars", 60, 720, log=True),
        FloatRange("rebalance_delta_max", 0.005, 0.05, log=True, decimals=6),
        FloatRange("rebalance_turnover_frac_per_unit_delta", 0.20, 1.00, decimals=4),
        FloatRange("z_risk", 0.50, 2.50, decimals=4),
        FloatRange("lambda_risk", 2.0, 40.0, log=True, decimals=6),
        FloatRange("z_liq", 1.5, 3.5, decimals=4),
        FloatRange("collateral_buffer_frac", 0.05, 0.30, decimals=4),
        FloatRange("flip_hysteresis_bps", 0.0, 10.0, decimals=4),
    ]

def _crypto_ensemble_space() -> list[Param]:
    # Keep the search space deliberately constrained: focus on signal/regime + primary risk knobs.
    return [
        IntRange("ema_fast", 8, 80, log=True),
        IntRange("ema_slow", 24, 260, log=True),
        IntRange("atr_window", 8, 80, log=True),
        IntRange("er_window", 10, 160, log=True),
        IntRange("breakout_window", 12, 260, log=True),
        IntRange("momentum_window", 24, 720, log=True),
        FloatRange("er_trend_min", 0.20, 0.90, decimals=3),
        FloatRange("er_range_max", 0.05, 0.50, decimals=3),
        FloatRange("trend_z_min", 0.05, 0.80, decimals=3),
        FloatRange("min_atr_bps", 1.0, 25.0, decimals=3),
        IntRange("meanrev_ewm_span", 20, 360, log=True),
        FloatRange("meanrev_entry_z", 0.75, 3.50, decimals=3),
        FloatRange("breakout_buffer_bps", 0.0, 12.0, decimals=3),
        IntRange("max_positions", 1, 6),
        FloatRange("max_gross_exposure", 0.25, 1.25, decimals=4),
        FloatRange("risk_budget", 0.002, 0.10, log=True, decimals=6),
        FloatRange("stop_atr_mult", 1.0, 6.0, decimals=3),
        FloatRange("trail_atr_mult", 1.2, 12.0, decimals=3),
        IntRange("cooldown_bars", 0, 24),
        IntRange("flip_confirm_bars", 1, 6),
        FloatRange("k_cost", 0.25, 8.0, decimals=3),
        FloatRange("edge_floor_bps", 0.0, 30.0, decimals=3),
    ]

def _crypto_tsm_space() -> list[Param]:
    # Time-series momentum / trend. Keep the space modest to avoid overfitting.
    return [
        IntRange("ema_fast", 6, 96, log=True),
        IntRange("ema_slow", 24, 360, log=True),
        IntRange("atr_window", 8, 120, log=True),
        FloatRange("min_atr_bps", 1.0, 25.0, decimals=3),
        IntRange("momentum_window", 12, 960, log=True),
        IntRange("confirm_bars", 1, 8),
        IntRange("exit_confirm_bars", 1, 8),
        IntRange("max_positions", 1, 6),
        FloatRange("max_gross_exposure", 0.25, 1.50, decimals=4),
        FloatRange("risk_budget", 0.002, 0.25, log=True, decimals=6),
        FloatRange("stop_atr_mult", 1.0, 8.0, decimals=3),
        FloatRange("trail_atr_mult", 1.2, 16.0, decimals=3),
        IntRange("cooldown_bars", 0, 48),
        IntRange("rebalance_interval_bars", 1, 24),
        FloatRange("k_cost", 0.25, 8.0, decimals=3),
        FloatRange("edge_floor_bps", 0.0, 40.0, decimals=3),
    ]

def _crypto_rotation_space() -> list[Param]:
    # Cross-sectional rotation. Keep the space modest and validate key ordering constraints.
    return [
        IntRange("rebalance_interval_bars", 4, 84, log=True),
        IntRange("mom_short_bars", 6, 84, log=True),
        IntRange("mom_med_bars", 24, 360, log=True),
        IntRange("mom_long_bars", 72, 960, log=True),
        IntRange("vol_window_bars", 24, 480, log=True),
        FloatRange("vol_target_bps_per_bar", 20.0, 180.0, decimals=3),
        IntRange("top_k", 1, 5),
        FloatRange("max_total_exposure", 0.25, 1.50, decimals=4),
        FloatRange("max_exposure_per_symbol", 0.10, 1.00, decimals=4),
        FloatRange("rebalance_exposure_threshold", 0.0, 0.12, decimals=4),
        FloatRange("score_floor", -0.25, 0.50, decimals=4),
        FloatRange("k_cost", 0.25, 6.0, decimals=3),
        FloatRange("edge_floor_bps", 0.0, 40.0, decimals=3),
    ]


def _crypto_momentum_space() -> list[Param]:
    return [
        IntRange("momentum_window_bars", 24, 960, log=True),
        FloatRange("max_total_exposure", 0.20, 1.50, decimals=4),
        FloatRange("max_exposure_per_symbol", 0.20, 1.25, decimals=4),
        IntRange("rebalance_interval_bars", 1, 56, log=True),
        FloatRange("rebalance_exposure_threshold", 0.0, 0.25, decimals=4),
        FloatRange("min_trade_notional_usd", 1.0, 50.0, decimals=3),
        IntRange("heartbeat_every_bars", 0, 84),
        FloatRange("heartbeat_notional_usd", 0.0, 10.0, decimals=3),
    ]


def _crypto_regime_fusion_space() -> list[Param]:
    return [
        IntRange("regime_momentum_bars", 48, 240, log=True),
        IntRange("regime_er_bars", 32, 160, log=True),
        IntRange("regime_ema_fast", 8, 48, log=True),
        IntRange("regime_ema_slow", 24, 192, log=True),
        FloatRange("regime_trend_mom", 0.005, 0.040, decimals=4),
        FloatRange("regime_trend_er_min", 0.15, 0.60, decimals=3),
        FloatRange("regime_trend_strength_min", 0.05, 0.60, decimals=3),
        FloatRange("regime_range_abs_mom_max", 0.001, 0.030, decimals=4),
        FloatRange("regime_range_er_max", 0.05, 0.40, decimals=3),
        IntRange("momentum_window_bars", 48, 240, log=True),
        IntRange("vol_window_bars", 24, 160, log=True),
        IntRange("trend_top_k", 1, 4),
        FloatRange("max_total_exposure", 0.30, 1.20, decimals=4),
        FloatRange("max_exposure_per_symbol", 0.15, 1.00, decimals=4),
        FloatRange("neutral_exposure_scale", 0.0, 0.60, decimals=4),
        IntRange("meanrev_window_bars", 24, 168, log=True),
        FloatRange("meanrev_entry_z", 0.8, 3.0, decimals=3),
        FloatRange("meanrev_exit_z", 0.2, 1.5, decimals=3),
        FloatRange("range_min_exposure", 0.0, 0.40, decimals=4),
        FloatRange("range_max_exposure", 0.10, 0.80, decimals=4),
        IntRange("rebalance_interval_bars", 2, 32, log=True),
        FloatRange("rebalance_exposure_threshold", 0.0, 0.15, decimals=4),
        FloatRange("daily_loss_limit", 0.01, 0.08, decimals=4),
        FloatRange("kill_switch", 0.05, 0.30, decimals=4),
        IntRange("kill_switch_cooldown_days", 1, 7),
        IntRange("heartbeat_every_bars", 0, 84),
        FloatRange("heartbeat_notional_usd", 0.0, 10.0, decimals=3),
        FloatRange("heartbeat_max_exposure_delta", 0.0, 0.10, decimals=4),
    ]


def _crypto_regime_vol_target_space() -> list[Param]:
    return [
        IntRange("fast_window", 8, 64, log=True),
        IntRange("slow_window", 24, 220, log=True),
        IntRange("regime_window", 80, 360, log=True),
        IntRange("regime_slope_bars", 2, 48, log=True),
        IntRange("momentum_window_bars", 24, 360, log=True),
        IntRange("atr_window", 8, 80, log=True),
        IntRange("top_k", 1, 4),
        FloatRange("target_vol_bps_per_bar", 10.0, 160.0, decimals=3),
        FloatRange("max_total_exposure", 0.20, 1.50, decimals=4),
        FloatRange("max_exposure_per_symbol", 0.10, 1.00, decimals=4),
        IntRange("rebalance_interval_bars", 1, 32, log=True),
        FloatRange("rebalance_exposure_threshold", 0.0, 0.20, decimals=4),
        FloatRange("market_drawdown_reduce", 0.02, 0.25, decimals=4),
        FloatRange("market_drawdown_off", 0.05, 0.40, decimals=4),
        IntRange("market_peak_lookback_bars", 48, 480, log=True),
        FloatRange("weekly_loss_limit", 0.01, 0.08, decimals=4),
        FloatRange("weekly_profit_target", 0.005, 0.08, decimals=4),
        FloatRange("daily_loss_limit", 0.005, 0.08, decimals=4),
        FloatRange("kill_switch", 0.02, 0.30, decimals=4),
        IntRange("kill_switch_cooldown_days", 1, 14),
        FloatRange("trailing_stop_pct", 0.02, 0.25, decimals=4),
        IntRange("min_hold_bars", 1, 32),
    ]


def _crypto_vol_squeeze_space() -> list[Param]:
    return [
        IntRange("rebalance_interval_bars", 8, 56, log=True),
        FloatRange("rebalance_exposure_threshold", 0.0, 0.20, decimals=4),
        IntRange("bb_window", 10, 80, log=True),
        FloatRange("bb_k", 1.2, 3.5, decimals=3),
        IntRange("squeeze_lookback", 40, 320, log=True),
        FloatRange("squeeze_percentile", 0.05, 0.45, decimals=4),
        IntRange("donchian_window", 10, 120, log=True),
        IntRange("atr_window", 8, 80, log=True),
        FloatRange("min_atr_bps", 1.0, 40.0, decimals=3),
        FloatRange("entry_breakout_buffer_bps", 0.0, 25.0, decimals=3),
        FloatRange("expected_move_atr_mult", 0.8, 5.0, decimals=3),
        FloatRange("cost_k", 0.25, 8.0, decimals=3),
        FloatRange("edge_floor_bps", 0.0, 30.0, decimals=3),
        FloatRange("max_total_exposure", 0.20, 1.50, decimals=4),
        FloatRange("max_exposure_per_symbol", 0.10, 1.00, decimals=4),
        FloatRange("vol_target_bps_per_bar", 10.0, 180.0, decimals=3),
        FloatRange("exposure_scale_on_squeeze", 0.0, 1.5, decimals=4),
        IntRange("min_hold_bars", 1, 40),
        IntRange("max_hold_bars", 4, 168, log=True),
        IntRange("exit_mom_bars", 4, 120, log=True),
        FloatRange("exit_mom_threshold", -0.02, 0.02, decimals=4),
        FloatRange("daily_loss_limit", 0.005, 0.08, decimals=4),
        FloatRange("kill_switch", 0.02, 0.30, decimals=4),
        IntRange("kill_switch_cooldown_days", 1, 14),
        FloatRange("market_drawdown_reduce", 0.02, 0.25, decimals=4),
        FloatRange("market_drawdown_off", 0.05, 0.40, decimals=4),
        FloatRange("market_vol_reduce_bps", 40.0, 260.0, decimals=3),
        FloatRange("market_vol_off_bps", 80.0, 400.0, decimals=3),
    ]


def _crypto_weekly_lock_momentum_space() -> list[Param]:
    return [
        IntRange("rebalance_interval_bars", 2, 32, log=True),
        FloatRange("rebalance_exposure_threshold", 0.0, 0.20, decimals=4),
        IntRange("mom_short_bars", 8, 84, log=True),
        IntRange("mom_med_bars", 24, 168, log=True),
        IntRange("mom_long_bars", 72, 720, log=True),
        FloatRange("w_mom_short", 0.0, 1.0, decimals=4),
        FloatRange("w_mom_med", 0.0, 1.0, decimals=4),
        FloatRange("w_mom_long", 0.0, 1.0, decimals=4),
        IntRange("vol_window_bars", 24, 240, log=True),
        IntRange("top_k", 1, 4),
        FloatRange("score_floor", -0.25, 0.50, decimals=4),
        FloatRange("max_total_exposure", 0.20, 1.50, decimals=4),
        FloatRange("max_exposure_per_symbol", 0.10, 1.00, decimals=4),
        FloatRange("vol_target_bps_per_bar", 10.0, 220.0, decimals=3),
        IntRange("regime_ema_bars", 24, 360, log=True),
        IntRange("regime_mom_bars", 12, 240, log=True),
        FloatRange("regime_mom_off", -0.05, 0.05, decimals=4),
        FloatRange("regime_dd_reduce", 0.02, 0.25, decimals=4),
        FloatRange("regime_dd_off", 0.05, 0.40, decimals=4),
        IntRange("regime_peak_lookback_bars", 24, 480, log=True),
        FloatRange("weekly_profit_target", 0.002, 0.04, decimals=4),
        FloatRange("weekly_loss_limit", 0.002, 0.06, decimals=4),
        FloatRange("daily_loss_limit", 0.005, 0.08, decimals=4),
        FloatRange("kill_switch", 0.02, 0.30, decimals=4),
        IntRange("kill_switch_cooldown_days", 1, 14),
    ]


def _perp_quant_fusion_space() -> list[Param]:
    return [
        IntRange("atr_window", 8, 48, log=True),
        IntRange("ema_fast", 6, 48, log=True),
        IntRange("ema_slow", 20, 200, log=True),
        IntRange("er_window", 8, 64, log=True),
        IntRange("choppiness_window", 8, 48, log=True),
        IntRange("breakout_window", 8, 96, log=True),
        FloatRange("breakout_buffer_bps", 0.0, 12.0, decimals=3),
        FloatRange("trend_z_min", 0.05, 1.0, decimals=3),
        FloatRange("er_min", 0.10, 0.80, decimals=3),
        FloatRange("er_exit_min", 0.05, 0.70, decimals=3),
        FloatRange("choppiness_max", 45.0, 75.0, decimals=3),
        FloatRange("choppiness_exit_max", 50.0, 85.0, decimals=3),
        FloatRange("min_atr_bps", 0.5, 25.0, decimals=3),
        FloatRange("edge_floor_bps", 0.0, 30.0, decimals=3),
        FloatRange("k_cost", 0.25, 6.0, decimals=3),
        FloatRange("risk_budget", 0.002, 0.10, log=True, decimals=6),
        FloatRange("stop_atr_mult", 0.8, 6.0, decimals=3),
        IntRange("max_positions", 1, 4),
        FloatRange("max_gross_exposure", 0.20, 2.0, decimals=4),
        FloatRange("max_per_symbol_exposure", 0.10, 1.0, decimals=4),
        FloatRange("rebalance_exposure_threshold", 0.0, 0.20, decimals=4),
        IntRange("min_hold_bars", 1, 16),
        IntRange("flip_confirm_bars", 1, 8),
        FloatRange("daily_loss_limit", 0.005, 0.08, decimals=4),
        FloatRange("kill_switch", 0.02, 0.30, decimals=4),
        FloatRange("weekly_profit_target", 0.005, 0.08, decimals=4),
        FloatRange("weekly_lock_risk_scale", 0.0, 1.0, decimals=4),
        FloatRange("heartbeat_exposure", 0.0, 0.10, decimals=4),
        IntRange("heartbeat_hold_bars", 1, 8),
    ]


def _perp_trend_vol_guard_space() -> list[Param]:
    return [
        IntRange("ema_fast", 4, 48, log=True),
        IntRange("ema_slow", 12, 220, log=True),
        IntRange("momentum_window_bars", 6, 96, log=True),
        IntRange("breakout_window", 8, 96, log=True),
        FloatRange("breakout_buffer_bps", 0.0, 14.0, decimals=3),
        IntRange("atr_window", 8, 80, log=True),
        FloatRange("trend_strength_min", 0.05, 1.20, decimals=3),
        FloatRange("min_atr_bps", 0.5, 30.0, decimals=3),
        FloatRange("edge_floor_bps", 0.0, 30.0, decimals=3),
        FloatRange("k_cost", 0.25, 8.0, decimals=3),
        FloatRange("risk_budget", 0.001, 0.08, log=True, decimals=6),
        FloatRange("stop_atr_mult", 0.8, 8.0, decimals=3),
        FloatRange("target_vol_bps_per_bar", 10.0, 220.0, decimals=3),
        IntRange("max_positions", 1, 4),
        FloatRange("max_gross_exposure", 0.20, 2.0, decimals=4),
        FloatRange("max_per_symbol_exposure", 0.10, 1.0, decimals=4),
        IntRange("rebalance_interval_bars", 1, 16, log=True),
        FloatRange("rebalance_exposure_threshold", 0.0, 0.20, decimals=4),
        IntRange("min_hold_bars", 1, 32),
        IntRange("flip_confirm_bars", 1, 8),
        FloatRange("market_vol_reduce_bps", 10.0, 300.0, decimals=3),
        FloatRange("market_vol_off_bps", 20.0, 450.0, decimals=3),
        FloatRange("weekly_loss_limit", 0.005, 0.08, decimals=4),
        FloatRange("weekly_profit_target", 0.005, 0.08, decimals=4),
        FloatRange("weekly_lock_risk_scale", 0.0, 1.0, decimals=4),
        FloatRange("weekly_chase_target", 0.0, 0.03, decimals=4),
        FloatRange("weekly_chase_k", 0.0, 12.0, decimals=4),
        FloatRange("weekly_chase_max_extra_exposure", 0.0, 0.50, decimals=4),
        IntRange("weekly_chase_start_weekday_utc", 0, 6),
        FloatRange("fallback_floor_exposure", 0.0, 0.35, decimals=4),
        FloatRange("fallback_trend_strength_min", 0.0, 0.80, decimals=4),
        FloatRange("fallback_min_momentum_bps", 0.0, 100.0, decimals=3),
        FloatRange("fallback_min_atr_bps", 0.5, 25.0, decimals=3),
        FloatRange("daily_loss_limit", 0.005, 0.08, decimals=4),
        FloatRange("kill_switch", 0.02, 0.30, decimals=4),
    ]


def _perp_weekly_carry_shield_space() -> list[Param]:
    return [
        IntRange("atr_window", 8, 80, log=True),
        IntRange("ema_fast", 4, 48, log=True),
        IntRange("ema_slow", 12, 200, log=True),
        IntRange("er_window", 6, 80, log=True),
        IntRange("choppiness_window", 6, 80, log=True),
        IntRange("momentum_bars", 4, 120, log=True),
        FloatRange("trend_z_min", 0.05, 1.0, decimals=3),
        FloatRange("er_min", 0.10, 0.90, decimals=3),
        FloatRange("choppiness_max", 40.0, 80.0, decimals=3),
        FloatRange("momentum_threshold_bps", 0.0, 200.0, decimals=3),
        FloatRange("min_atr_bps", 0.5, 25.0, decimals=3),
        FloatRange("edge_floor_bps", 0.0, 30.0, decimals=3),
        FloatRange("k_cost", 0.25, 8.0, decimals=3),
        FloatRange("expected_move_atr_mult", 0.8, 6.0, decimals=3),
        FloatRange("risk_budget", 0.001, 0.05, log=True, decimals=6),
        FloatRange("stop_atr_mult", 0.8, 8.0, decimals=3),
        FloatRange("max_margin_utilization", 0.05, 0.95, decimals=4),
        FloatRange("max_leverage", 1.0, 15.0, decimals=4),
        IntRange("max_positions", 1, 4),
        FloatRange("max_gross_exposure", 0.20, 2.0, decimals=4),
        FloatRange("max_per_symbol_exposure", 0.10, 1.0, decimals=4),
        IntRange("min_hold_bars", 1, 24),
        FloatRange("rebalance_exposure_threshold", 0.0, 0.20, decimals=4),
        FloatRange("daily_loss_limit", 0.003, 0.08, decimals=4),
        FloatRange("kill_switch", 0.02, 0.30, decimals=4),
        FloatRange("weekly_profit_target", 0.001, 0.03, decimals=4),
        FloatRange("weekly_loss_limit", 0.001, 0.03, decimals=4),
        FloatRange("fallback_trend_floor_exposure", 0.0, 0.30, decimals=4),
        FloatRange("fallback_trend_floor_er_min", 0.0, 0.80, decimals=3),
        FloatRange("fallback_trend_floor_choppiness_max", 35.0, 90.0, decimals=3),
        FloatRange("fallback_trend_floor_min_momentum_bps", 0.0, 120.0, decimals=3),
        FloatRange("weekly_heartbeat_exposure", 0.0, 0.10, decimals=4),
        IntRange("weekly_heartbeat_hold_bars", 1, 8),
    ]


def _perp_weekly_profit_chase_space() -> list[Param]:
    return [
        IntRange("atr_window", 8, 60, log=True),
        IntRange("opening_range_minutes", 15, 120, log=True),
        FloatRange("breakout_buffer_bps", 0.0, 12.0, decimals=3),
        FloatRange("min_atr_bps", 1.0, 20.0, decimals=3),
        FloatRange("weekly_profit_target", 0.005, 0.05, decimals=4),
        FloatRange("weekly_chase_k", 0.0, 5.0, decimals=3),
        FloatRange("risk_per_trade", 0.005, 0.10, log=True, decimals=6),
        FloatRange("base_leverage", 1.0, 20.0, decimals=3),
        FloatRange("max_leverage", 2.0, 25.0, decimals=3),
        FloatRange("max_margin_utilization", 0.10, 0.95, decimals=4),
        FloatRange("stop_atr_mult", 1.0, 6.0, decimals=3),
        FloatRange("min_liq_buffer_atr", 1.0, 12.0, decimals=3),
        FloatRange("weekly_heartbeat_exposure", 0.0, 0.10, decimals=4),
        IntRange("weekly_heartbeat_hold_bars", 1, 12),
        IntRange("max_flips_per_day", 1, 6),
        FloatRange("daily_loss_hard_stop", 0.0, 0.06, decimals=4),
        FloatRange("weekly_loss_hard_stop", 0.0, 0.15, decimals=4),
        IntRange("cooldown_bars_after_exit", 0, 192),
        FloatRange("trailing_stop_atr_mult", 0.0, 6.0, decimals=3),
        FloatRange("break_even_trigger_atr", 0.0, 4.0, decimals=3),
        IntRange("max_hold_bars", 0, 192),
    ]


def _perp_weekly_trend_reset_space() -> list[Param]:
    return [
        IntRange("lookback_days", 5, 60, log=True),
        FloatRange("momentum_threshold_bps", 0.0, 250.0, decimals=3),
        IntRange("ema_fast", 4, 48, log=True),
        IntRange("ema_slow", 12, 240, log=True),
        IntRange("atr_window", 8, 60, log=True),
        FloatRange("target_leverage", 1.0, 30.0, decimals=3),
        FloatRange("max_margin_utilization", 0.10, 0.95, decimals=4),
        FloatRange("stop_atr_mult", 1.0, 8.0, decimals=3),
        FloatRange("trail_atr_mult", 1.2, 16.0, decimals=3),
        FloatRange("min_liq_buffer_atr", 1.0, 12.0, decimals=3),
        FloatRange("heartbeat_exposure", 0.0, 0.20, decimals=4),
        IntRange("heartbeat_hold_bars", 2, 96, log=True),
    ]


def _perp_regime_adaptive_trend_capture_space() -> list[Param]:
    return [
        IntRange("mom_horizon_a", 72, 336, log=True),
        IntRange("mom_horizon_b", 168, 1008, log=True),
        IntRange("mom_horizon_c", 336, 2016, log=True),
        IntRange("ema_fast_regime", 24, 168, log=True),
        IntRange("ema_slow_regime", 168, 1512, log=True),
        FloatRange("bear_exit_bps", 40.0, 260.0, decimals=3),
        FloatRange("short_entry_bps", 120.0, 700.0, decimals=3),
        IntRange("cooldown_bars", 12, 336, log=True),
        FloatRange("long_base_exposure", 0.15, 0.95, decimals=4),
        FloatRange("short_base_exposure", 0.0, 0.70, decimals=4),
        FloatRange("extreme_vol_scale", 0.10, 0.80, decimals=4),
        FloatRange("high_vol_scale", 0.25, 0.98, decimals=4),
        FloatRange("extreme_vol_rank", 0.75, 0.98, decimals=4),
        FloatRange("high_vol_rank", 0.50, 0.90, decimals=4),
        IntRange("vol_lookback_bars", 24, 240, log=True),
        IntRange("vol_regime_window", 168, 2160, log=True),
        FloatRange("crash_threshold_bps", 120.0, 800.0, decimals=3),
        IntRange("max_hold_bars", 72, 4032, log=True),
        FloatRange("rebalance_exposure_threshold", 0.0, 0.08, decimals=4),
        FloatRange("daily_loss_limit", 0.01, 0.12, decimals=4),
        FloatRange("weekly_loss_limit", 0.02, 0.25, decimals=4),
        FloatRange("kill_switch", 0.08, 0.55, decimals=4),
    ]


def get_search_space(strategy: str) -> list[Param]:
    strategy = (strategy or "").strip().lower().replace("-", "_")
    if strategy == "perp_flare":
        return _perp_flare_space()
    if strategy == "perp_weekly_trend_reset":
        return _perp_weekly_trend_reset_space()
    if strategy == "perp_weekly_profit_chase":
        return _perp_weekly_profit_chase_space()
    if strategy == "orb_trend":
        return _orb_trend_space()
    if strategy == "hedge":
        return _hedge_space()
    if strategy == "crypto_ensemble":
        return _crypto_ensemble_space()
    if strategy == "crypto_tsm":
        return _crypto_tsm_space()
    if strategy == "crypto_rotation":
        return _crypto_rotation_space()
    if strategy == "crypto_momentum":
        return _crypto_momentum_space()
    if strategy == "crypto_regime_vol_target":
        return _crypto_regime_vol_target_space()
    if strategy == "crypto_regime_fusion":
        return _crypto_regime_fusion_space()
    if strategy == "crypto_vol_squeeze":
        return _crypto_vol_squeeze_space()
    if strategy == "crypto_weekly_lock_momentum":
        return _crypto_weekly_lock_momentum_space()
    if strategy == "perp_quant_fusion":
        return _perp_quant_fusion_space()
    if strategy == "perp_trend_vol_guard":
        return _perp_trend_vol_guard_space()
    if strategy == "perp_weekly_carry_shield":
        return _perp_weekly_carry_shield_space()
    if strategy == "perp_regime_adaptive_trend_capture":
        return _perp_regime_adaptive_trend_capture_space()
    raise ValueError(f"no tuning space defined for strategy: {strategy}")


def _validate_perp_flare_params(params: dict[str, Any]) -> bool:
    try:
        if int(params["ema_fast"]) >= int(params["ema_slow"]):
            return False
        if float(params["trail_atr_mult"]) < float(params["stop_atr_mult"]):
            return False
        if int(params["breakout_window"]) < 2:
            return False
        if float(params["risk_per_trade"]) <= 0:
            return False
        if not (0.0 < float(params["max_margin_utilization"]) <= 1.0):
            return False
        if float(params["max_leverage"]) <= 0:
            return False
        return True
    except Exception:
        return False


def _validate_perp_weekly_profit_chase_params(params: dict[str, Any]) -> bool:
    try:
        if int(params["atr_window"]) < 2:
            return False
        if int(params["opening_range_minutes"]) <= 0:
            return False
        if float(params["weekly_profit_target"]) <= 0:
            return False
        if float(params["risk_per_trade"]) <= 0:
            return False
        if float(params["base_leverage"]) <= 0:
            return False
        if float(params["max_leverage"]) <= 0:
            return False
        if float(params["max_leverage"]) < float(params["base_leverage"]):
            return False
        if not (0.0 < float(params["max_margin_utilization"]) <= 1.0):
            return False
        if float(params["stop_atr_mult"]) <= 0:
            return False
        if float(params["min_liq_buffer_atr"]) < 0:
            return False
        if float(params["weekly_heartbeat_exposure"]) < 0:
            return False
        if int(params["weekly_heartbeat_hold_bars"]) < 1:
            return False
        if int(params["max_flips_per_day"]) < 1:
            return False
        if float(params.get("daily_loss_hard_stop", 0.0)) < 0.0:
            return False
        if float(params.get("weekly_loss_hard_stop", 0.0)) < 0.0:
            return False
        if int(params.get("cooldown_bars_after_exit", 0)) < 0:
            return False
        if float(params.get("trailing_stop_atr_mult", 0.0)) < 0.0:
            return False
        if float(params.get("break_even_trigger_atr", 0.0)) < 0.0:
            return False
        if int(params.get("max_hold_bars", 0)) < 0:
            return False
        return True
    except Exception:
        return False


def _validate_perp_weekly_trend_reset_params(params: dict[str, Any]) -> bool:
    try:
        if int(params["ema_fast"]) >= int(params["ema_slow"]):
            return False
        if int(params["lookback_days"]) <= 0:
            return False
        if int(params["atr_window"]) < 2:
            return False
        if float(params["target_leverage"]) <= 0:
            return False
        if not (0.0 < float(params["max_margin_utilization"]) <= 1.0):
            return False
        if float(params["trail_atr_mult"]) < float(params["stop_atr_mult"]):
            return False
        if float(params["min_liq_buffer_atr"]) < 0:
            return False
        if float(params["heartbeat_exposure"]) < 0:
            return False
        if int(params["heartbeat_hold_bars"]) < 1:
            return False
        return True
    except Exception:
        return False


def _validate_orb_trend_params(params: dict[str, Any]) -> bool:
    try:
        if int(params["orb_minutes"]) <= 0:
            return False
        if int(params["confirm_bars"]) <= 0:
            return False
        if int(params["atr_window"]) < 2:
            return False
        if int(params["er_window"]) < 2:
            return False
        if not (0.0 < float(params["er_min"]) <= 1.0):
            return False
        if int(params["expected_hold_bars"]) <= 0:
            return False
        if float(params["k_cost"]) < 0:
            return False
        if int(params["min_hold_bars"]) < 0:
            return False
        if not (0.0 < float(params["daily_loss_limit"]) < 1.0):
            return False
        if not (0.0 < float(params["kill_switch"]) < 1.0):
            return False
        return True
    except Exception:
        return False


def _validate_hedge_params(params: dict[str, Any]) -> bool:
    try:
        if float(params.get("edge_horizon_hours", 0.0)) <= 0:
            return False
        if float(params.get("basis_halflife_hours", 0.0)) <= 0:
            return False
        if int(params.get("cov_window_bars", 0)) < 20:
            return False
        if float(params.get("rebalance_delta_max", 0.0)) <= 0:
            return False
        if float(params.get("rebalance_turnover_frac_per_unit_delta", 0.0)) <= 0:
            return False
        if float(params.get("z_risk", 0.0)) <= 0:
            return False
        if float(params.get("lambda_risk", 0.0)) <= 0:
            return False
        if float(params.get("z_liq", 0.0)) <= 0:
            return False
        if not (0.0 <= float(params.get("collateral_buffer_frac", 0.0)) < 1.0):
            return False
        # Some hedge params may be fixed externally (not part of the search space). If present,
        # validate them; otherwise allow them to be set via defaults/base params.
        max_leverage = params.get("max_leverage")
        if max_leverage is not None and float(max_leverage) <= 0:
            return False
        max_margin_util = params.get("max_margin_utilization")
        if max_margin_util is not None and not (0.0 < float(max_margin_util) <= 1.0):
            return False
        if float(params.get("flip_hysteresis_bps", 0.0)) < 0:
            return False
        return True
    except Exception:
        return False


def validate_params(strategy: str, params: dict[str, Any]) -> bool:
    strategy = (strategy or "").strip().lower().replace("-", "_")
    if strategy == "perp_flare":
        return _validate_perp_flare_params(params)
    if strategy == "perp_weekly_trend_reset":
        return _validate_perp_weekly_trend_reset_params(params)
    if strategy == "perp_weekly_profit_chase":
        return _validate_perp_weekly_profit_chase_params(params)
    if strategy == "orb_trend":
        return _validate_orb_trend_params(params)
    if strategy == "hedge":
        return _validate_hedge_params(params)
    if strategy == "crypto_ensemble":
        try:
            if int(params.get("ema_fast", 0)) >= int(params.get("ema_slow", 0)):
                return False
            if float(params.get("trail_atr_mult", 0.0)) < float(params.get("stop_atr_mult", 0.0)):
                return False
            er_trend_min = float(params.get("er_trend_min", 0.0))
            er_range_max = float(params.get("er_range_max", 0.0))
            if not (0.0 < er_range_max < er_trend_min <= 1.0):
                return False
            if int(params.get("atr_window", 0)) < 2:
                return False
            if int(params.get("er_window", 0)) < 2:
                return False
            if int(params.get("breakout_window", 0)) < 2:
                return False
            if int(params.get("momentum_window", 0)) < 2:
                return False
            if float(params.get("risk_budget", 0.0)) <= 0:
                return False
            if float(params.get("max_gross_exposure", 0.0)) <= 0:
                return False
            if int(params.get("max_positions", 0)) <= 0:
                return False
            if float(params.get("k_cost", 0.0)) < 0:
                return False
            if float(params.get("edge_floor_bps", 0.0)) < 0:
                return False
            if float(params.get("min_atr_bps", 0.0)) < 0:
                return False
            if float(params.get("breakout_buffer_bps", 0.0)) < 0:
                return False
            if float(params.get("meanrev_entry_z", 0.0)) <= 0:
                return False
            return True
        except Exception:
            return False
    if strategy == "crypto_tsm":
        try:
            if int(params.get("ema_fast", 0)) >= int(params.get("ema_slow", 0)):
                return False
            if float(params.get("trail_atr_mult", 0.0)) < float(params.get("stop_atr_mult", 0.0)):
                return False
            if int(params.get("atr_window", 0)) < 2:
                return False
            if float(params.get("min_atr_bps", 0.0)) < 0:
                return False
            if int(params.get("momentum_window", 0)) < 2:
                return False
            if int(params.get("confirm_bars", 0)) <= 0:
                return False
            if int(params.get("exit_confirm_bars", 0)) <= 0:
                return False
            min_hold_bars = 0
            if "min_hold_bars" in params:
                min_hold_bars = int(params.get("min_hold_bars", 0))
                if min_hold_bars < 0:
                    return False
            if "max_hold_bars" in params:
                max_hold_bars = int(params.get("max_hold_bars", 0))
                if max_hold_bars < 0:
                    return False
                if max_hold_bars > 0 and max_hold_bars < int(min_hold_bars):
                    return False
            if "take_profit_atr_mult" in params and float(params.get("take_profit_atr_mult", 0.0)) < 0:
                return False
            if float(params.get("risk_budget", 0.0)) <= 0:
                return False
            if float(params.get("max_gross_exposure", 0.0)) <= 0:
                return False
            if int(params.get("max_positions", 0)) <= 0:
                return False
            if int(params.get("rebalance_interval_bars", 0)) <= 0:
                return False
            if float(params.get("k_cost", 0.0)) < 0:
                return False
            if float(params.get("edge_floor_bps", 0.0)) < 0:
                return False
            if ("market_drawdown_reduce" in params) or ("market_drawdown_off" in params):
                md_reduce = float(params.get("market_drawdown_reduce", 0.0) or 0.0)
                md_off = float(params.get("market_drawdown_off", 0.0) or 0.0)
                if not (0.0 < md_reduce < md_off < 1.0):
                    return False
            if ("market_vol_reduce_bps" in params) or ("market_vol_off_bps" in params):
                vol_reduce = float(params.get("market_vol_reduce_bps", 0.0) or 0.0)
                vol_off = float(params.get("market_vol_off_bps", 0.0) or 0.0)
                if not (0.0 < vol_reduce < vol_off):
                    return False
            if "market_peak_halflife_bars" in params and int(params.get("market_peak_halflife_bars", 0)) <= 0:
                return False
            return True
        except Exception:
            return False
    if strategy == "crypto_rotation":
        try:
            if int(params.get("rebalance_interval_bars", 0)) <= 0:
                return False
            mom_s = int(params.get("mom_short_bars", 0) or 0)
            mom_m = int(params.get("mom_med_bars", 0) or 0)
            mom_l = int(params.get("mom_long_bars", 0) or 0)
            if not (0 < mom_s < mom_m < mom_l):
                return False
            if int(params.get("vol_window_bars", 0)) < 2:
                return False
            if float(params.get("vol_target_bps_per_bar", 0.0)) < 0:
                return False
            if int(params.get("top_k", 0)) <= 0:
                return False
            if float(params.get("max_total_exposure", 0.0)) <= 0:
                return False
            if float(params.get("max_exposure_per_symbol", 0.0)) <= 0:
                return False
            if float(params.get("rebalance_exposure_threshold", 0.0)) < 0:
                return False
            if float(params.get("k_cost", 0.0)) < 0:
                return False
            if float(params.get("edge_floor_bps", 0.0)) < 0:
                return False
            if "heartbeat_every_bars" in params and int(params.get("heartbeat_every_bars", 0)) < 0:
                return False
            if "heartbeat_notional_usd" in params and float(params.get("heartbeat_notional_usd", 0.0)) < 0:
                return False
            if ("market_drawdown_reduce" in params) or ("market_drawdown_off" in params):
                md_reduce = float(params.get("market_drawdown_reduce", 0.0) or 0.0)
                md_off = float(params.get("market_drawdown_off", 0.0) or 0.0)
                if not (0.0 < md_reduce < md_off < 1.0):
                    return False
            if ("market_vol_reduce_bps" in params) or ("market_vol_off_bps" in params):
                vol_reduce = float(params.get("market_vol_reduce_bps", 0.0) or 0.0)
                vol_off = float(params.get("market_vol_off_bps", 0.0) or 0.0)
                if not (0.0 < vol_reduce < vol_off):
                    return False
            if "market_peak_halflife_bars" in params and int(params.get("market_peak_halflife_bars", 0)) <= 0:
                return False
            if ("w_mom_short" in params) or ("w_mom_med" in params) or ("w_mom_long" in params):
                w_s = float(params.get("w_mom_short", 0.0) or 0.0)
                w_m = float(params.get("w_mom_med", 0.0) or 0.0)
                w_l = float(params.get("w_mom_long", 0.0) or 0.0)
                if w_s < 0 or w_m < 0 or w_l < 0:
                    return False
                if (w_s + w_m + w_l) <= 0:
                    return False
            return True
        except Exception:
            return False
    if strategy == "crypto_momentum":
        try:
            if int(params.get("momentum_window_bars", 0)) < 2:
                return False
            if float(params.get("max_total_exposure", 0.0)) <= 0:
                return False
            if float(params.get("max_exposure_per_symbol", 0.0)) <= 0:
                return False
            if float(params.get("max_exposure_per_symbol", 0.0)) < (
                float(params.get("max_total_exposure", 0.0)) / 8.0
            ):
                return False
            if int(params.get("rebalance_interval_bars", 0)) <= 0:
                return False
            if float(params.get("rebalance_exposure_threshold", 0.0)) < 0:
                return False
            if float(params.get("min_trade_notional_usd", 0.0)) < 0:
                return False
            if int(params.get("heartbeat_every_bars", 0)) < 0:
                return False
            if float(params.get("heartbeat_notional_usd", 0.0)) < 0:
                return False
            return True
        except Exception:
            return False
    if strategy == "crypto_regime_vol_target":
        try:
            if int(params.get("fast_window", 0)) >= int(params.get("slow_window", 0)):
                return False
            if int(params.get("regime_window", 0)) < int(params.get("slow_window", 0)):
                return False
            if int(params.get("regime_slope_bars", 0)) < 1:
                return False
            if int(params.get("momentum_window_bars", 0)) < 2:
                return False
            if int(params.get("atr_window", 0)) < 2:
                return False
            if int(params.get("top_k", 0)) <= 0:
                return False
            if float(params.get("target_vol_bps_per_bar", 0.0)) < 0:
                return False
            if float(params.get("max_total_exposure", 0.0)) <= 0:
                return False
            if float(params.get("max_exposure_per_symbol", 0.0)) <= 0:
                return False
            if float(params.get("max_exposure_per_symbol", 0.0)) > float(
                params.get("max_total_exposure", 0.0)
            ):
                return False
            if int(params.get("rebalance_interval_bars", 0)) <= 0:
                return False
            if float(params.get("rebalance_exposure_threshold", 0.0)) < 0:
                return False
            if float(params.get("market_drawdown_reduce", 0.0)) <= 0:
                return False
            if float(params.get("market_drawdown_off", 0.0)) <= 0:
                return False
            if float(params.get("market_drawdown_reduce", 0.0)) >= float(
                params.get("market_drawdown_off", 0.0)
            ):
                return False
            if int(params.get("market_peak_lookback_bars", 0)) < 2:
                return False
            if float(params.get("weekly_loss_limit", 0.0)) <= 0:
                return False
            if float(params.get("weekly_profit_target", 0.0)) <= 0:
                return False
            if float(params.get("daily_loss_limit", 0.0)) <= 0:
                return False
            if float(params.get("kill_switch", 0.0)) <= float(
                params.get("daily_loss_limit", 0.0)
            ):
                return False
            if int(params.get("kill_switch_cooldown_days", 0)) < 0:
                return False
            if float(params.get("trailing_stop_pct", 0.0)) < 0:
                return False
            if int(params.get("min_hold_bars", 0)) < 0:
                return False
            return True
        except Exception:
            return False
    if strategy == "crypto_regime_fusion":
        try:
            if int(params.get("regime_ema_fast", 0)) >= int(params.get("regime_ema_slow", 0)):
                return False
            if int(params.get("regime_momentum_bars", 0)) <= 1:
                return False
            if int(params.get("regime_er_bars", 0)) <= 1:
                return False
            if not (0.0 < float(params.get("regime_trend_er_min", 0.0)) <= 1.0):
                return False
            if not (0.0 < float(params.get("regime_range_er_max", 0.0)) < 1.0):
                return False
            if float(params.get("regime_range_er_max", 0.0)) >= float(
                params.get("regime_trend_er_min", 0.0)
            ):
                return False
            if float(params.get("max_total_exposure", 0.0)) <= 0:
                return False
            if float(params.get("max_exposure_per_symbol", 0.0)) <= 0:
                return False
            if int(params.get("trend_top_k", 0)) <= 0:
                return False
            if int(params.get("momentum_window_bars", 0)) <= 1:
                return False
            if int(params.get("vol_window_bars", 0)) <= 1:
                return False
            if int(params.get("meanrev_window_bars", 0)) <= 1:
                return False
            if float(params.get("meanrev_entry_z", 0.0)) <= 0:
                return False
            if float(params.get("meanrev_exit_z", 0.0)) < 0:
                return False
            if float(params.get("meanrev_exit_z", 0.0)) >= float(
                params.get("meanrev_entry_z", 0.0)
            ):
                return False
            if float(params.get("range_min_exposure", 0.0)) < 0:
                return False
            if float(params.get("range_max_exposure", 0.0)) < 0:
                return False
            if float(params.get("range_min_exposure", 0.0)) > float(
                params.get("range_max_exposure", 0.0)
            ):
                return False
            if int(params.get("rebalance_interval_bars", 0)) <= 0:
                return False
            if float(params.get("rebalance_exposure_threshold", 0.0)) < 0:
                return False
            if float(params.get("daily_loss_limit", 0.0)) <= 0:
                return False
            if float(params.get("kill_switch", 0.0)) <= 0:
                return False
            if float(params.get("kill_switch", 0.0)) <= float(
                params.get("daily_loss_limit", 0.0)
            ):
                return False
            if int(params.get("kill_switch_cooldown_days", 0)) < 0:
                return False
            if int(params.get("heartbeat_every_bars", 0)) < 0:
                return False
            if float(params.get("heartbeat_notional_usd", 0.0)) < 0:
                return False
            if float(params.get("heartbeat_max_exposure_delta", 0.0)) < 0:
                return False
            return True
        except Exception:
            return False
    if strategy == "crypto_vol_squeeze":
        try:
            if int(params.get("bb_window", 0)) < 2:
                return False
            if float(params.get("bb_k", 0.0)) <= 0:
                return False
            if int(params.get("squeeze_lookback", 0)) < int(params.get("bb_window", 0)):
                return False
            sq = float(params.get("squeeze_percentile", 0.0))
            if not (0.0 < sq < 1.0):
                return False
            if int(params.get("donchian_window", 0)) < 2:
                return False
            if int(params.get("atr_window", 0)) < 2:
                return False
            if float(params.get("min_atr_bps", 0.0)) < 0:
                return False
            if float(params.get("entry_breakout_buffer_bps", 0.0)) < 0:
                return False
            if float(params.get("expected_move_atr_mult", 0.0)) <= 0:
                return False
            if float(params.get("cost_k", 0.0)) < 0:
                return False
            if float(params.get("edge_floor_bps", 0.0)) < 0:
                return False
            if float(params.get("max_total_exposure", 0.0)) <= 0:
                return False
            if float(params.get("max_exposure_per_symbol", 0.0)) <= 0:
                return False
            if float(params.get("max_exposure_per_symbol", 0.0)) > float(
                params.get("max_total_exposure", 0.0)
            ):
                return False
            if float(params.get("vol_target_bps_per_bar", 0.0)) < 0:
                return False
            if int(params.get("rebalance_interval_bars", 0)) <= 0:
                return False
            if float(params.get("rebalance_exposure_threshold", 0.0)) < 0:
                return False
            if int(params.get("min_hold_bars", 0)) < 1:
                return False
            if int(params.get("max_hold_bars", 0)) < int(params.get("min_hold_bars", 0)):
                return False
            if int(params.get("exit_mom_bars", 0)) < 1:
                return False
            if float(params.get("daily_loss_limit", 0.0)) <= 0:
                return False
            if float(params.get("kill_switch", 0.0)) <= 0:
                return False
            if float(params.get("kill_switch", 0.0)) <= float(
                params.get("daily_loss_limit", 0.0)
            ):
                return False
            md_reduce = float(params.get("market_drawdown_reduce", 0.0) or 0.0)
            md_off = float(params.get("market_drawdown_off", 0.0) or 0.0)
            if not (0.0 < md_reduce < md_off < 1.0):
                return False
            vol_reduce = float(params.get("market_vol_reduce_bps", 0.0) or 0.0)
            vol_off = float(params.get("market_vol_off_bps", 0.0) or 0.0)
            if not (0.0 < vol_reduce < vol_off):
                return False
            if int(params.get("kill_switch_cooldown_days", 0)) < 0:
                return False
            return True
        except Exception:
            return False
    if strategy == "crypto_weekly_lock_momentum":
        try:
            m1 = int(params.get("mom_short_bars", 0))
            m2 = int(params.get("mom_med_bars", 0))
            m3 = int(params.get("mom_long_bars", 0))
            if not (0 < m1 < m2 < m3):
                return False
            w1 = float(params.get("w_mom_short", 0.0))
            w2 = float(params.get("w_mom_med", 0.0))
            w3 = float(params.get("w_mom_long", 0.0))
            if w1 < 0 or w2 < 0 or w3 < 0:
                return False
            if (w1 + w2 + w3) <= 0:
                return False
            if int(params.get("vol_window_bars", 0)) < 2:
                return False
            if int(params.get("top_k", 0)) <= 0:
                return False
            if float(params.get("max_total_exposure", 0.0)) <= 0:
                return False
            if float(params.get("max_exposure_per_symbol", 0.0)) <= 0:
                return False
            if float(params.get("max_exposure_per_symbol", 0.0)) > float(
                params.get("max_total_exposure", 0.0)
            ):
                return False
            if float(params.get("vol_target_bps_per_bar", 0.0)) < 0:
                return False
            if int(params.get("rebalance_interval_bars", 0)) <= 0:
                return False
            if float(params.get("rebalance_exposure_threshold", 0.0)) < 0:
                return False
            if int(params.get("regime_ema_bars", 0)) < 2:
                return False
            if int(params.get("regime_mom_bars", 0)) < 2:
                return False
            if int(params.get("regime_peak_lookback_bars", 0)) < 2:
                return False
            dd_reduce = float(params.get("regime_dd_reduce", 0.0))
            dd_off = float(params.get("regime_dd_off", 0.0))
            if not (0.0 < dd_reduce < dd_off < 1.0):
                return False
            if float(params.get("weekly_profit_target", 0.0)) <= 0:
                return False
            if float(params.get("weekly_loss_limit", 0.0)) <= 0:
                return False
            if float(params.get("daily_loss_limit", 0.0)) <= 0:
                return False
            if float(params.get("kill_switch", 0.0)) <= 0:
                return False
            if float(params.get("kill_switch", 0.0)) <= float(
                params.get("daily_loss_limit", 0.0)
            ):
                return False
            if int(params.get("kill_switch_cooldown_days", 0)) < 0:
                return False
            return True
        except Exception:
            return False
    if strategy == "perp_quant_fusion":
        try:
            if int(params.get("ema_fast", 0)) >= int(params.get("ema_slow", 0)):
                return False
            if int(params.get("atr_window", 0)) < 2:
                return False
            if int(params.get("er_window", 0)) < 2:
                return False
            if int(params.get("choppiness_window", 0)) < 2:
                return False
            if int(params.get("breakout_window", 0)) < 2:
                return False
            if float(params.get("breakout_buffer_bps", 0.0)) < 0:
                return False
            if float(params.get("trend_z_min", 0.0)) < 0:
                return False
            if not (0.0 < float(params.get("er_min", 0.0)) <= 1.0):
                return False
            if not (0.0 <= float(params.get("er_exit_min", 0.0)) < 1.0):
                return False
            if float(params.get("er_exit_min", 0.0)) >= float(params.get("er_min", 0.0)):
                return False
            if float(params.get("choppiness_exit_max", 0.0)) < float(
                params.get("choppiness_max", 0.0)
            ):
                return False
            if float(params.get("edge_floor_bps", 0.0)) < 0:
                return False
            if float(params.get("k_cost", 0.0)) < 0:
                return False
            if float(params.get("risk_budget", 0.0)) <= 0:
                return False
            if float(params.get("stop_atr_mult", 0.0)) <= 0:
                return False
            if ("max_margin_utilization" in params) and not (
                0.0 < float(params.get("max_margin_utilization", 0.0)) <= 1.0
            ):
                return False
            if ("max_leverage" in params) and float(params.get("max_leverage", 0.0)) <= 0:
                return False
            if int(params.get("max_positions", 0)) <= 0:
                return False
            if float(params.get("max_gross_exposure", 0.0)) <= 0:
                return False
            if float(params.get("max_per_symbol_exposure", 0.0)) <= 0:
                return False
            if float(params.get("max_per_symbol_exposure", 0.0)) > float(
                params.get("max_gross_exposure", 0.0)
            ):
                return False
            if float(params.get("rebalance_exposure_threshold", 0.0)) < 0:
                return False
            if int(params.get("min_hold_bars", 0)) < 1:
                return False
            if int(params.get("flip_confirm_bars", 0)) < 1:
                return False
            if float(params.get("daily_loss_limit", 0.0)) <= 0:
                return False
            if float(params.get("kill_switch", 0.0)) <= 0:
                return False
            if float(params.get("kill_switch", 0.0)) <= float(
                params.get("daily_loss_limit", 0.0)
            ):
                return False
            if float(params.get("weekly_profit_target", 0.0)) <= 0:
                return False
            if not (0.0 <= float(params.get("weekly_lock_risk_scale", 0.0)) <= 1.0):
                return False
            if float(params.get("heartbeat_exposure", 0.0)) < 0:
                return False
            if int(params.get("heartbeat_hold_bars", 0)) < 1:
                return False
            return True
        except Exception:
            return False
    if strategy == "perp_trend_vol_guard":
        try:
            if int(params.get("ema_fast", 0)) >= int(params.get("ema_slow", 0)):
                return False
            if int(params.get("momentum_window_bars", 0)) < 2:
                return False
            if int(params.get("breakout_window", 0)) < 2:
                return False
            if int(params.get("atr_window", 0)) < 2:
                return False
            if float(params.get("breakout_buffer_bps", 0.0)) < 0:
                return False
            if float(params.get("trend_strength_min", 0.0)) < 0:
                return False
            if float(params.get("min_atr_bps", 0.0)) < 0:
                return False
            if float(params.get("edge_floor_bps", 0.0)) < 0:
                return False
            if float(params.get("k_cost", 0.0)) < 0:
                return False
            if float(params.get("risk_budget", 0.0)) <= 0:
                return False
            if float(params.get("stop_atr_mult", 0.0)) <= 0:
                return False
            if float(params.get("target_vol_bps_per_bar", 0.0)) < 0:
                return False
            if int(params.get("max_positions", 0)) <= 0:
                return False
            if float(params.get("max_gross_exposure", 0.0)) <= 0:
                return False
            if float(params.get("max_per_symbol_exposure", 0.0)) <= 0:
                return False
            if float(params.get("max_per_symbol_exposure", 0.0)) > float(
                params.get("max_gross_exposure", 0.0)
            ):
                return False
            if int(params.get("rebalance_interval_bars", 0)) <= 0:
                return False
            if float(params.get("rebalance_exposure_threshold", 0.0)) < 0:
                return False
            if int(params.get("min_hold_bars", 0)) < 1:
                return False
            if int(params.get("flip_confirm_bars", 0)) < 1:
                return False
            if float(params.get("market_vol_reduce_bps", 0.0)) <= 0:
                return False
            if float(params.get("market_vol_off_bps", 0.0)) <= float(
                params.get("market_vol_reduce_bps", 0.0)
            ):
                return False
            if float(params.get("weekly_loss_limit", 0.0)) <= 0:
                return False
            if float(params.get("weekly_profit_target", 0.0)) <= 0:
                return False
            if not (0.0 <= float(params.get("weekly_lock_risk_scale", 0.0)) <= 1.0):
                return False
            if float(params.get("weekly_chase_target", 0.0)) < 0:
                return False
            if float(params.get("weekly_chase_k", 0.0)) < 0:
                return False
            if float(params.get("weekly_chase_max_extra_exposure", 0.0)) < 0:
                return False
            if float(params.get("weekly_chase_max_extra_exposure", 0.0)) > float(
                params.get("max_per_symbol_exposure", 0.0)
            ):
                return False
            if not (0 <= int(params.get("weekly_chase_start_weekday_utc", 0)) <= 6):
                return False
            if float(params.get("fallback_floor_exposure", 0.0)) < 0:
                return False
            if float(params.get("fallback_floor_exposure", 0.0)) > float(
                params.get("max_per_symbol_exposure", 0.0)
            ):
                return False
            if float(params.get("fallback_trend_strength_min", 0.0)) < 0:
                return False
            if float(params.get("fallback_min_momentum_bps", 0.0)) < 0:
                return False
            if float(params.get("fallback_min_atr_bps", 0.0)) < 0:
                return False
            if float(params.get("daily_loss_limit", 0.0)) <= 0:
                return False
            if float(params.get("kill_switch", 0.0)) <= float(
                params.get("daily_loss_limit", 0.0)
            ):
                return False
            return True
        except Exception:
            return False
    if strategy == "perp_weekly_carry_shield":
        try:
            if int(params.get("ema_fast", 0)) >= int(params.get("ema_slow", 0)):
                return False
            if int(params.get("atr_window", 0)) < 2:
                return False
            if int(params.get("er_window", 0)) < 2:
                return False
            if int(params.get("choppiness_window", 0)) < 2:
                return False
            if int(params.get("momentum_bars", 0)) < 2:
                return False
            if float(params.get("trend_z_min", 0.0)) < 0:
                return False
            if not (0.0 < float(params.get("er_min", 0.0)) <= 1.0):
                return False
            if float(params.get("choppiness_max", 0.0)) <= 0:
                return False
            if float(params.get("momentum_threshold_bps", 0.0)) < 0:
                return False
            if float(params.get("min_atr_bps", 0.0)) < 0:
                return False
            if float(params.get("edge_floor_bps", 0.0)) < 0:
                return False
            if float(params.get("k_cost", 0.0)) < 0:
                return False
            if float(params.get("expected_move_atr_mult", 0.0)) <= 0:
                return False
            if float(params.get("risk_budget", 0.0)) <= 0:
                return False
            if float(params.get("stop_atr_mult", 0.0)) <= 0:
                return False
            if int(params.get("max_positions", 0)) <= 0:
                return False
            if float(params.get("max_gross_exposure", 0.0)) <= 0:
                return False
            if float(params.get("max_per_symbol_exposure", 0.0)) <= 0:
                return False
            if float(params.get("max_per_symbol_exposure", 0.0)) > float(
                params.get("max_gross_exposure", 0.0)
            ):
                return False
            if int(params.get("min_hold_bars", 0)) < 1:
                return False
            if float(params.get("rebalance_exposure_threshold", 0.0)) < 0:
                return False
            if float(params.get("daily_loss_limit", 0.0)) <= 0:
                return False
            if float(params.get("kill_switch", 0.0)) <= 0:
                return False
            if float(params.get("kill_switch", 0.0)) <= float(
                params.get("daily_loss_limit", 0.0)
            ):
                return False
            if float(params.get("weekly_profit_target", 0.0)) <= 0:
                return False
            if float(params.get("weekly_loss_limit", 0.0)) <= 0:
                return False
            if float(params.get("fallback_trend_floor_exposure", 0.0)) < 0:
                return False
            if float(params.get("fallback_trend_floor_exposure", 0.0)) > float(
                params.get("max_per_symbol_exposure", 0.0)
            ):
                return False
            if float(params.get("fallback_trend_floor_er_min", 0.0)) < 0:
                return False
            if float(params.get("fallback_trend_floor_er_min", 0.0)) > 1.0:
                return False
            if float(params.get("fallback_trend_floor_choppiness_max", 100.0)) <= 0:
                return False
            if float(params.get("fallback_trend_floor_min_momentum_bps", 0.0)) < 0:
                return False
            if float(params.get("weekly_heartbeat_exposure", 0.0)) < 0:
                return False
            if int(params.get("weekly_heartbeat_hold_bars", 0)) < 1:
                return False
            return True
        except Exception:
            return False
    if strategy == "perp_regime_adaptive_trend_capture":
        try:
            mom_a = int(params.get("mom_horizon_a", 0))
            mom_b = int(params.get("mom_horizon_b", 0))
            mom_c = int(params.get("mom_horizon_c", 0))
            if not (mom_a > 0 and mom_b > 0 and mom_c > 0):
                return False
            if not (mom_a < mom_b < mom_c):
                return False

            ema_f = int(params.get("ema_fast_regime", 0))
            ema_s = int(params.get("ema_slow_regime", 0))
            if not (ema_f > 0 and ema_s > 0 and ema_f < ema_s):
                return False

            bear_exit = float(params.get("bear_exit_bps", 0.0))
            short_entry = float(params.get("short_entry_bps", 0.0))
            if bear_exit <= 0.0 or short_entry <= 0.0:
                return False
            if short_entry < bear_exit:
                return False

            cooldown = int(params.get("cooldown_bars", -1))
            if cooldown < 0:
                return False

            long_exp = float(params.get("long_base_exposure", 0.0))
            short_exp = float(params.get("short_base_exposure", 0.0))
            if not (0.0 <= short_exp <= 1.0 and 0.0 <= long_exp <= 1.0):
                return False
            if long_exp <= 0.0 and short_exp <= 0.0:
                return False

            ex_scale = float(params.get("extreme_vol_scale", 0.0))
            hi_scale = float(params.get("high_vol_scale", 0.0))
            if not (0.0 <= ex_scale <= 1.0 and 0.0 <= hi_scale <= 1.0):
                return False

            ex_rank = float(params.get("extreme_vol_rank", 0.0))
            hi_rank = float(params.get("high_vol_rank", 0.0))
            if not (0.0 <= hi_rank <= 1.0 and 0.0 <= ex_rank <= 1.0):
                return False
            if not (hi_rank < ex_rank):
                return False

            vol_lb = int(params.get("vol_lookback_bars", 0))
            vol_rw = int(params.get("vol_regime_window", 0))
            if not (vol_lb > 0 and vol_rw > vol_lb):
                return False

            crash = float(params.get("crash_threshold_bps", 0.0))
            if crash <= 0.0:
                return False

            max_hold = int(params.get("max_hold_bars", 0))
            if max_hold <= 0:
                return False

            reb_thr = float(params.get("rebalance_exposure_threshold", 0.0))
            if not (0.0 <= reb_thr <= 1.0):
                return False

            dloss = float(params.get("daily_loss_limit", 0.0))
            wloss = float(params.get("weekly_loss_limit", 0.0))
            kill = float(params.get("kill_switch", 0.0))
            if not (0.0 < dloss < 1.0 and 0.0 < wloss < 1.0 and 0.0 < kill < 1.0):
                return False
            if dloss > wloss:
                return False
            if wloss > kill:
                return False
            return True
        except Exception:
            return False
    return True


def sample_params(
    *,
    strategy: str,
    rng: random.Random,
    space: list[Param],
    incumbent: Optional[dict[str, Any]] = None,
    drift_frac: Optional[float] = None,
    max_attempts: int = 500,
) -> dict[str, Any]:
    """
    Sample a parameter set from the given search space.

    If `incumbent` and `drift_frac` are provided, sampled values are clamped to
    +/- drift_frac around incumbent values (still respecting global bounds).
    """
    drift_frac = None if drift_frac is None else float(max(0.0, drift_frac))
    for _ in range(max(1, int(max_attempts))):
        params: dict[str, Any] = {}
        for spec in space:
            params[spec.name] = spec.sample(rng)

        if incumbent is not None and drift_frac is not None and drift_frac > 0:
            for spec in space:
                if spec.name not in incumbent:
                    continue
                base = incumbent.get(spec.name)
                if base is None:
                    continue
                try:
                    if isinstance(spec, IntRange):
                        base_f = float(int(base))
                        lo = max(spec.lo, int(math.floor(base_f * (1.0 - drift_frac))))
                        hi = min(spec.hi, int(math.ceil(base_f * (1.0 + drift_frac))))
                        params[spec.name] = int(
                            max(lo, min(hi, int(params[spec.name])))
                        )
                    else:
                        base_f = float(base)
                        lo = max(spec.lo, float(base_f * (1.0 - drift_frac)))
                        hi = min(spec.hi, float(base_f * (1.0 + drift_frac)))
                        params[spec.name] = float(
                            max(lo, min(hi, float(params[spec.name])))
                        )
                except Exception:
                    continue

        if validate_params(strategy, params):
            return params

    raise RuntimeError("failed to sample a valid parameter set")


@dataclass(frozen=True)
class ObjectiveConfig:
    # Hard constraints (violations reject the trial)
    max_drawdown_limit: float = 0.40  # reject if max_drawdown < -limit
    worst_day_limit: float = 0.20  # reject if worst_day_return < -limit
    turnover_cap: float = 250.0  # gross_notional / avg_equity
    min_trades: int = 2
    require_no_liquidations: bool = True

    # Soft objective weights (maximize)
    w_total_return: float = 0.75
    # Prefer daily Sharpe for comparability across timeframes.
    w_sharpe_daily: float = 0.25
    w_sharpe: float = 0.05
    w_positive_trading_days: float = 0.10
    w_drawdown: float = 1.00  # penalty on abs(max_drawdown)
    w_turnover: float = 0.003  # penalty on turnover
    w_worst_day: float = 0.50  # penalty on abs(worst_day_return)


@dataclass(frozen=True)
class WalkForwardConfig:
    train: str = "30d"
    validate: str = "7d"
    test: str = "7d"
    step: str = "7d"

    def train_td(self) -> timedelta:
        return parse_duration_spec(self.train)

    def validate_td(self) -> timedelta:
        return parse_duration_spec(self.validate)

    def test_td(self) -> timedelta:
        return parse_duration_spec(self.test)

    def step_td(self) -> timedelta:
        return parse_duration_spec(self.step)


@dataclass(frozen=True)
class TuneConfig:
    trials_per_segment: int = 60
    jobs: int = 1
    seed: int = 7
    drift_frac: Optional[float] = 0.50
    improvement_margin: float = 0.0
    objective: ObjectiveConfig = field(default_factory=ObjectiveConfig)
    walk_forward: WalkForwardConfig = field(default_factory=WalkForwardConfig)
    keep_best_test_runs: bool = True


_TUNE_WORKER_BARS_BY_SYMBOL: Optional[dict[str, pd.DataFrame]] = None
_TUNE_WORKER_COMMON_INDEX: Optional[pd.DatetimeIndex] = None


def _init_tune_worker(
    bars_by_symbol: dict[str, pd.DataFrame], common_index: pd.DatetimeIndex
) -> None:
    global _TUNE_WORKER_BARS_BY_SYMBOL, _TUNE_WORKER_COMMON_INDEX
    _TUNE_WORKER_BARS_BY_SYMBOL = bars_by_symbol
    _TUNE_WORKER_COMMON_INDEX = common_index


def _evaluate_train_validate_trial(
    *,
    market: str,
    symbols: list[str],
    strategy: str,
    backtest_cfg: BacktestConfig,
    objective: ObjectiveConfig,
    segment: int,
    trial: int,
    params: dict[str, Any],
    train_start: datetime,
    train_end: datetime,
    validate_start: datetime,
    validate_end: datetime,
) -> dict[str, Any]:
    bars_by_symbol = _TUNE_WORKER_BARS_BY_SYMBOL
    common_index = _TUNE_WORKER_COMMON_INDEX
    if bars_by_symbol is None or common_index is None:
        raise RuntimeError("tune worker not initialized")

    market_enum = parse_market(market)

    with tempfile.TemporaryDirectory(prefix="atlas_tune_") as tmp:
        root = Path(tmp)
        train_dir = root / "train"
        validate_dir = root / "validate"
        train_dir.mkdir(parents=True, exist_ok=True)
        validate_dir.mkdir(parents=True, exist_ok=True)

        strategy_train = _build_strategy_instance(
            strategy_name=strategy,
            symbols=symbols,
            params=params,
        )
        warmup = strategy_train.warmup_bars()

        train_bars = _slice_bars_with_warmup(
            bars_by_symbol,
            common_index,
            score_start=train_start,
            score_end=train_end,
            warmup_bars=warmup,
        )
        _run_backtest_for_market(
            market=market_enum,
            bars_by_symbol=train_bars,
            strategy=strategy_train,
            cfg=backtest_cfg,
            run_dir=train_dir,
            output_mode="minimal",
            score_start=train_start,
            score_end=train_end,
            no_trade_before=train_start,
        )
        train_score = score_run(
            train_dir,
            objective=objective,
            score_start=train_start,
            score_end=train_end,
        )

        validate_bars = _slice_bars_with_warmup(
            bars_by_symbol,
            common_index,
            score_start=validate_start,
            score_end=validate_end,
            warmup_bars=warmup,
        )
        _run_backtest_for_market(
            market=market_enum,
            bars_by_symbol=validate_bars,
            strategy=_build_strategy_instance(
                strategy_name=strategy,
                symbols=symbols,
                params=params,
            ),
            cfg=backtest_cfg,
            run_dir=validate_dir,
            output_mode="minimal",
            score_start=validate_start,
            score_end=validate_end,
            no_trade_before=validate_start,
        )
        validate_score = score_run(
            validate_dir,
            objective=objective,
            score_start=validate_start,
            score_end=validate_end,
        )

    selection_score = 0.25 * float(train_score.score) + 0.75 * float(validate_score.score)
    rejected = bool(train_score.rejected or validate_score.rejected)
    reject_reason = str(train_score.reason or validate_score.reason or "")

    return {
        "segment": int(segment),
        "trial": int(trial),
        "params": dict(params),
        "selection_score": float(selection_score),
        "rejected": bool(rejected),
        "reject_reason": reject_reason,
        "train": {
            "score": float(train_score.score),
            "rejected": bool(train_score.rejected),
            "reject_reason": str(train_score.reason or ""),
            "stats": asdict(train_score.stats),
            "breakdown": dict(train_score.breakdown),
        },
        "validate": {
            "score": float(validate_score.score),
            "rejected": bool(validate_score.rejected),
            "reject_reason": str(validate_score.reason or ""),
            "stats": asdict(validate_score.stats),
            "breakdown": dict(validate_score.breakdown),
        },
    }


@dataclass(frozen=True)
class Window:
    start: datetime
    end: datetime

    def to_dict(self) -> dict[str, str]:
        return {"start": self.start.isoformat(), "end": self.end.isoformat()}


@dataclass(frozen=True)
class WalkForwardSegment:
    train: Window
    validate: Window
    test: Window

    def to_dict(self) -> dict[str, dict[str, str]]:
        return {
            "train": self.train.to_dict(),
            "validate": self.validate.to_dict(),
            "test": self.test.to_dict(),
        }


@dataclass(frozen=True)
class BacktestStats:
    total_return: float
    max_drawdown: float
    sharpe: float
    sharpe_daily: float
    trades: int
    gross_notional: float
    turnover: float
    worst_day_return: float
    positive_trading_day_frac: float
    liquidation_count: int


@dataclass(frozen=True)
class ScoreResult:
    score: float
    rejected: bool
    reason: str
    stats: BacktestStats
    breakdown: dict[str, float]


def _read_run_stats(
    run_dir: Path,
    *,
    score_start: Optional[datetime] = None,
    score_end: Optional[datetime] = None,
) -> BacktestStats:
    trades_path = run_dir / "trades.csv"
    trades = pd.read_csv(trades_path) if trades_path.exists() else pd.DataFrame()

    trade_ts = None
    if len(trades) and "timestamp" in trades.columns:
        trade_ts = pd.to_datetime(trades["timestamp"], errors="coerce", utc=True)

    equity_path = run_dir / "equity_curve.csv"
    equity = pd.read_csv(equity_path, parse_dates=["timestamp"])
    equity["timestamp"] = pd.to_datetime(equity["timestamp"], errors="coerce", utc=True)
    equity = equity.dropna(subset=["timestamp"])
    equity = equity.set_index("timestamp").sort_index()
    if len(equity) and equity.index.tz is None:
        equity.index = pd.to_datetime(equity.index, utc=True)

    start_ts: Optional[pd.Timestamp] = None
    end_ts: Optional[pd.Timestamp] = None
    if score_start is not None:
        start_ts = pd.Timestamp(score_start)
        start_ts = (
            start_ts.tz_localize("UTC")
            if start_ts.tz is None
            else start_ts.tz_convert("UTC")
        )
    if score_end is not None:
        end_ts = pd.Timestamp(score_end)
        end_ts = (
            end_ts.tz_localize("UTC") if end_ts.tz is None else end_ts.tz_convert("UTC")
        )

    if start_ts is not None:
        equity = equity[equity.index >= start_ts]
        if trade_ts is not None:
            mask = trade_ts.notna() & (trade_ts >= start_ts)
            trades = trades.loc[mask]
            trade_ts = trade_ts.loc[mask]
    if end_ts is not None:
        equity = equity[equity.index < end_ts]
        if trade_ts is not None:
            mask = trade_ts.notna() & (trade_ts < end_ts)
            trades = trades.loc[mask]
            trade_ts = trade_ts.loc[mask]

    gross_notional = (
        float(trades["notional"].sum())
        if len(trades) and "notional" in trades.columns
        else 0.0
    )

    if len(equity):
        metrics = compute_metrics(equity, trades)
        total_return = float(metrics.total_return)
        max_drawdown = float(metrics.max_drawdown)
        sharpe = float(metrics.sharpe)
        sharpe_daily = float(metrics.sharpe_daily)
        trade_count = int(metrics.trades)
    else:
        total_return = 0.0
        max_drawdown = 0.0
        sharpe = 0.0
        sharpe_daily = 0.0
        trade_count = int(len(trades))

    avg_equity = float(equity["equity"].astype(float).mean()) if len(equity) else 0.0
    turnover = (gross_notional / avg_equity) if avg_equity > 0 else 0.0

    if "day_return" in equity.columns and len(equity):
        daily = equity["day_return"].astype(float).groupby(equity.index.date).last()
        worst_day_return = float(daily.min()) if len(daily) else 0.0
    else:
        worst_day_return = 0.0

    if trade_ts is not None and len(trade_ts):
        trading_days = set(trade_ts.dropna().dt.date)
    else:
        trading_days = set()

    positive_trading_day_frac = 0.0
    if trading_days and "day_return" in equity.columns and len(equity):
        daily = equity["day_return"].astype(float).groupby(equity.index.date).last()
        vals = [float(daily.get(day, 0.0)) for day in sorted(trading_days)]
        if vals:
            positive_trading_day_frac = float(sum(1 for v in vals if v > 0) / len(vals))

    liquidation_count = 0
    if len(trades) and "strategy_reason" in trades.columns:
        liquidation_count = int(
            (trades["strategy_reason"].astype(str) == "LIQUIDATION").sum()
        )

    return BacktestStats(
        total_return=float(total_return),
        max_drawdown=float(max_drawdown),
        sharpe=float(sharpe),
        sharpe_daily=float(sharpe_daily),
        trades=int(trade_count),
        gross_notional=float(gross_notional),
        turnover=float(turnover),
        worst_day_return=float(worst_day_return),
        positive_trading_day_frac=float(positive_trading_day_frac),
        liquidation_count=int(liquidation_count),
    )


def score_run(
    run_dir: Path,
    *,
    objective: ObjectiveConfig,
    score_start: Optional[datetime] = None,
    score_end: Optional[datetime] = None,
) -> ScoreResult:
    stats = _read_run_stats(run_dir, score_start=score_start, score_end=score_end)

    if stats.trades < int(objective.min_trades):
        return ScoreResult(
            score=float("-inf"),
            rejected=True,
            reason=f"min_trades (got {stats.trades})",
            stats=stats,
            breakdown={},
        )

    if objective.require_no_liquidations and stats.liquidation_count > 0:
        return ScoreResult(
            score=float("-inf"),
            rejected=True,
            reason=f"liquidations (got {stats.liquidation_count})",
            stats=stats,
            breakdown={},
        )

    if stats.max_drawdown < -float(objective.max_drawdown_limit):
        return ScoreResult(
            score=float("-inf"),
            rejected=True,
            reason=f"max_drawdown<{-float(objective.max_drawdown_limit):.2%}",
            stats=stats,
            breakdown={},
        )

    if stats.worst_day_return < -float(objective.worst_day_limit):
        return ScoreResult(
            score=float("-inf"),
            rejected=True,
            reason=f"worst_day<{-float(objective.worst_day_limit):.2%}",
            stats=stats,
            breakdown={},
        )

    if stats.turnover > float(objective.turnover_cap):
        return ScoreResult(
            score=float("-inf"),
            rejected=True,
            reason=f"turnover>{float(objective.turnover_cap):.2f}",
            stats=stats,
            breakdown={},
        )

    dd_mag = abs(float(stats.max_drawdown))
    tail_mag = abs(min(0.0, float(stats.worst_day_return)))
    turnover = float(stats.turnover)

    breakdown = {
        "total_return": float(stats.total_return) * float(objective.w_total_return),
        "sharpe_daily": float(stats.sharpe_daily) * float(objective.w_sharpe_daily),
        "sharpe": float(stats.sharpe) * float(objective.w_sharpe),
        "positive_trading_days": float(stats.positive_trading_day_frac)
        * float(objective.w_positive_trading_days),
        "drawdown_penalty": -dd_mag * float(objective.w_drawdown),
        "turnover_penalty": -turnover * float(objective.w_turnover),
        "worst_day_penalty": -tail_mag * float(objective.w_worst_day),
    }
    score = float(sum(breakdown.values()))
    return ScoreResult(
        score=score, rejected=False, reason="", stats=stats, breakdown=breakdown
    )


@dataclass(frozen=True)
class TrialRecord:
    segment: int
    trial: int
    phase: str
    params: dict[str, Any]
    score: float
    rejected: bool
    reject_reason: str
    stats: dict[str, Any]
    breakdown: dict[str, float]


@dataclass(frozen=True)
class SegmentSelection:
    segment: int
    params: dict[str, Any]
    selection_score: float
    train: dict[str, Any]
    validate: dict[str, Any]
    test: dict[str, Any]


@dataclass(frozen=True)
class TuneResult:
    run_dir: Path
    strategy: str
    market: str
    symbols: list[str]
    config: dict[str, Any]
    segments: list[dict[str, Any]]
    selections: list[SegmentSelection]
    best_params_latest: dict[str, Any]
    best_params_stable: dict[str, Any]
    stability: dict[str, Any]
    elapsed_s: float


@dataclass(frozen=True)
class TuneProgress:
    segment: int
    n_segments: int
    trial: int
    trials_per_segment: int
    phase: str
    best_selection_score: float
    best_params: dict[str, Any]
    last_score: float
    last_rejected: bool
    last_reject_reason: str


def _slice_bars(
    bars_by_symbol: dict[str, pd.DataFrame],
    *,
    start: datetime,
    end: datetime,
) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    for sym, df in bars_by_symbol.items():
        out[sym] = df[(df.index >= start_ts) & (df.index < end_ts)]
    return out


def _slice_bars_with_warmup(
    bars_by_symbol: dict[str, pd.DataFrame],
    common_index: pd.DatetimeIndex,
    *,
    score_start: datetime,
    score_end: datetime,
    warmup_bars: int,
) -> dict[str, pd.DataFrame]:
    start_ts = pd.Timestamp(score_start)
    end_ts = pd.Timestamp(score_end)

    start_pos = int(common_index.searchsorted(start_ts, side="left"))
    end_pos = int(common_index.searchsorted(end_ts, side="left"))
    warm_start_pos = max(0, start_pos - int(max(0, warmup_bars)))

    idx_slice = common_index[warm_start_pos:end_pos]
    return {sym: df.loc[idx_slice] for sym, df in bars_by_symbol.items()}


def _build_strategy_instance(
    *,
    strategy_name: str,
    symbols: list[str],
    params: dict[str, Any],
) -> Strategy:
    return build_strategy(
        name=strategy_name,
        params_path=None,
        symbols=symbols,
        fast_window=10,
        slow_window=30,
        params=params,
    )


def _run_backtest_for_market(
    *,
    market: Market,
    bars_by_symbol: dict[str, pd.DataFrame],
    strategy: Strategy,
    cfg: BacktestConfig,
    run_dir: Path,
    output_mode: str = "full",
    score_start: Optional[datetime] = None,
    score_end: Optional[datetime] = None,
    no_trade_before: Optional[datetime] = None,
) -> BacktestOutputs:
    if market == Market.DERIVATIVES:
        return run_derivatives_backtest(
            bars_by_symbol=bars_by_symbol,
            strategy=strategy,
            cfg=cfg,
            run_dir=run_dir,
            output_mode=output_mode,
            score_start=score_start,
            score_end=score_end,
            no_trade_before=no_trade_before,
        )
    return run_backtest(
        bars_by_symbol=bars_by_symbol,
        strategy=strategy,
        cfg=cfg,
        run_dir=run_dir,
        output_mode=output_mode,
        score_start=score_start,
        score_end=score_end,
        no_trade_before=no_trade_before,
    )


def build_walk_forward_segments(
    *,
    start: datetime,
    end: datetime,
    cfg: WalkForwardConfig,
) -> list[WalkForwardSegment]:
    train_td = cfg.train_td()
    val_td = cfg.validate_td()
    test_td = cfg.test_td()
    step_td = cfg.step_td()

    if train_td <= timedelta(0) or val_td <= timedelta(0) or test_td <= timedelta(0):
        raise ValueError("train/validate/test durations must be > 0")
    if step_td <= timedelta(0):
        raise ValueError("step must be > 0")

    segments: list[WalkForwardSegment] = []
    cursor = start
    while True:
        train_start = cursor
        train_end = train_start + train_td
        val_start = train_end
        val_end = val_start + val_td
        test_start = val_end
        test_end = test_start + test_td

        if test_end > end:
            break

        segments.append(
            WalkForwardSegment(
                train=Window(start=train_start, end=train_end),
                validate=Window(start=val_start, end=val_end),
                test=Window(start=test_start, end=test_end),
            )
        )
        cursor = cursor + step_td
        if cursor >= end:
            break

    if not segments:
        raise ValueError("walk-forward config produced 0 segments (window too small?)")
    return segments


def _stability_summary(selections: list[SegmentSelection]) -> dict[str, Any]:
    if not selections:
        return {"segments": 0}

    params_list = [sel.params for sel in selections]
    keys = sorted({k for p in params_list for k in p.keys()})
    out: dict[str, Any] = {"segments": len(selections), "params": {}}

    for k in keys:
        vals = [p.get(k) for p in params_list if k in p]
        if not vals:
            continue
        if all(isinstance(v, (int, float)) for v in vals):
            floats = [float(v) for v in vals]
            out["params"][k] = {
                "min": float(min(floats)),
                "max": float(max(floats)),
                "mean": float(statistics.mean(floats)),
                "stdev": float(statistics.pstdev(floats)) if len(floats) >= 2 else 0.0,
            }
        else:
            counts: dict[str, int] = {}
            for v in vals:
                key = str(v)
                counts[key] = counts.get(key, 0) + 1
            out["params"][k] = {"counts": counts}

    # Overall best by test score.
    best = max(selections, key=lambda s: float(s.test.get("score", float("-inf"))))
    out["best_segment_by_test_score"] = int(best.segment)
    out["best_test_score"] = float(best.test.get("score", float("-inf")))
    return out


def _stable_params_median(
    selections: list[SegmentSelection],
    *,
    last_n: int = 10,
) -> dict[str, Any]:
    if not selections:
        return {}
    n = max(1, min(int(last_n), len(selections)))
    window = selections[-n:]

    keys = sorted({k for sel in window for k in sel.params.keys()})
    out: dict[str, Any] = {}
    for k in keys:
        vals = [sel.params.get(k) for sel in window if k in sel.params]
        if not vals:
            continue
        if all(isinstance(v, int) for v in vals):
            out[k] = int(round(statistics.median([int(v) for v in vals])))
        elif all(isinstance(v, (int, float)) for v in vals):
            out[k] = float(statistics.median([float(v) for v in vals]))
        else:
            # Best-effort for categorical-ish values: most common string form.
            counts: dict[str, int] = {}
            for v in vals:
                key = str(v)
                counts[key] = counts.get(key, 0) + 1
            out[k] = max(counts, key=counts.get) if counts else vals[-1]
    return out


def tune_walk_forward(
    *,
    bars_by_symbol: dict[str, pd.DataFrame],
    market: str,
    symbols: list[str],
    strategy: str,
    backtest_cfg: BacktestConfig,
    tune_cfg: TuneConfig,
    run_dir: Path,
    base_params: Optional[dict[str, Any]] = None,
    stop_event: Optional[Event] = None,
    on_progress: Optional[Callable[[TuneProgress], None]] = None,
) -> TuneResult:
    market_enum = parse_market(market)
    strategy = (strategy or "").strip().lower().replace("-", "_")
    symbols = [s.strip().upper() for s in symbols if s.strip()]
    if not symbols:
        raise ValueError("symbols must be non-empty")

    # Align bars across symbols for consistent slicing.
    common_index: Optional[pd.DatetimeIndex] = None
    for sym in symbols:
        idx = bars_by_symbol[sym].index
        common_index = idx if common_index is None else common_index.intersection(idx)
    if common_index is None or len(common_index) < 3:
        raise ValueError("backtest window has too few aligned bars")
    common_index = common_index.sort_values()
    bars_by_symbol = {s: bars_by_symbol[s].loc[common_index] for s in symbols}

    start_ts = pd.Timestamp(common_index[0]).to_pydatetime()
    end_ts = pd.Timestamp(common_index[-1]).to_pydatetime()

    segments = build_walk_forward_segments(
        start=start_ts, end=end_ts, cfg=tune_cfg.walk_forward
    )

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "segments.json").write_text(
        json.dumps([s.to_dict() for s in segments], indent=2)
    )
    (run_dir / "config.json").write_text(
        json.dumps(asdict(tune_cfg), indent=2, default=str)
    )

    trials_path = run_dir / "trials.jsonl"
    space = get_search_space(strategy)
    rng = random.Random(int(tune_cfg.seed))

    selections: list[SegmentSelection] = []
    fixed_params = dict(base_params or {})
    incumbent = dict(base_params or {})

    jobs = int(tune_cfg.jobs)
    if jobs <= 0:
        jobs = int(os.cpu_count() or 1)
    jobs = max(1, jobs)

    executor: Optional[concurrent.futures.Executor] = None
    if jobs > 1:
        try:
            executor = concurrent.futures.ProcessPoolExecutor(
                max_workers=int(jobs),
                initializer=_init_tune_worker,
                initargs=(bars_by_symbol, common_index),
            )
        except Exception as exc:
            logger.warning(
                "Parallel tuning: ProcessPoolExecutor unavailable (%s). Falling back to threads.",
                exc,
            )
            _init_tune_worker(bars_by_symbol, common_index)
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=int(jobs))

    t0 = time.perf_counter()
    try:
        with trials_path.open("w") as f_trials:
            for seg_i, seg in enumerate(segments):
                if stop_event is not None and stop_event.is_set():
                    break

                best_params: Optional[dict[str, Any]] = None
                best_selection_score = float("-inf")
                best_train: dict[str, Any] = {}
                best_val: dict[str, Any] = {}

                seg_tmp_dir = run_dir / "tmp" / f"seg{seg_i:03d}"
                train_tmp = seg_tmp_dir / "train"
                val_tmp = seg_tmp_dir / "val"
                train_tmp.mkdir(parents=True, exist_ok=True)
                val_tmp.mkdir(parents=True, exist_ok=True)

                # Always evaluate incumbent as a candidate if provided.
                incumbent_selection_score = float("-inf")
                if incumbent:
                    incumbent_params = dict(fixed_params)
                    incumbent_params.update(incumbent)
                    warmup = _build_strategy_instance(
                        strategy_name=strategy,
                        symbols=symbols,
                        params=incumbent_params,
                    ).warmup_bars()

                    train_bars = _slice_bars_with_warmup(
                        bars_by_symbol,
                        common_index,
                        score_start=seg.train.start,
                        score_end=seg.train.end,
                        warmup_bars=warmup,
                    )
                    _run_backtest_for_market(
                        market=market_enum,
                        bars_by_symbol=train_bars,
                        strategy=_build_strategy_instance(
                            strategy_name=strategy,
                            symbols=symbols,
                            params=incumbent_params,
                        ),
                        cfg=backtest_cfg,
                        run_dir=train_tmp,
                        output_mode="minimal",
                        score_start=seg.train.start,
                        score_end=seg.train.end,
                        no_trade_before=seg.train.start,
                    )
                    train_scored = score_run(
                        train_tmp,
                        objective=tune_cfg.objective,
                        score_start=seg.train.start,
                        score_end=seg.train.end,
                    )

                    val_bars = _slice_bars_with_warmup(
                        bars_by_symbol,
                        common_index,
                        score_start=seg.validate.start,
                        score_end=seg.validate.end,
                        warmup_bars=warmup,
                    )
                    _run_backtest_for_market(
                        market=market_enum,
                        bars_by_symbol=val_bars,
                        strategy=_build_strategy_instance(
                            strategy_name=strategy,
                            symbols=symbols,
                            params=incumbent_params,
                        ),
                        cfg=backtest_cfg,
                        run_dir=val_tmp,
                        output_mode="minimal",
                        score_start=seg.validate.start,
                        score_end=seg.validate.end,
                        no_trade_before=seg.validate.start,
                    )
                    val_scored = score_run(
                        val_tmp,
                        objective=tune_cfg.objective,
                        score_start=seg.validate.start,
                        score_end=seg.validate.end,
                    )

                    incumbent_selection_score = 0.25 * float(
                        train_scored.score
                    ) + 0.75 * float(val_scored.score)

                if executor is None:
                    for trial_i in range(int(tune_cfg.trials_per_segment)):
                        if stop_event is not None and stop_event.is_set():
                            break

                        params = sample_params(
                            strategy=strategy,
                            rng=rng,
                            space=space,
                            incumbent=incumbent if incumbent else None,
                            drift_frac=tune_cfg.drift_frac,
                        )
                        full_params = dict(fixed_params)
                        full_params.update(params)
                        warmup = _build_strategy_instance(
                            strategy_name=strategy,
                            symbols=symbols,
                            params=full_params,
                        ).warmup_bars()
                        train_bars = _slice_bars_with_warmup(
                            bars_by_symbol,
                            common_index,
                            score_start=seg.train.start,
                            score_end=seg.train.end,
                            warmup_bars=warmup,
                        )
                        _run_backtest_for_market(
                            market=market_enum,
                            bars_by_symbol=train_bars,
                            strategy=_build_strategy_instance(
                                strategy_name=strategy,
                                symbols=symbols,
                                params=full_params,
                            ),
                            cfg=backtest_cfg,
                            run_dir=train_tmp,
                            output_mode="minimal",
                            score_start=seg.train.start,
                            score_end=seg.train.end,
                            no_trade_before=seg.train.start,
                        )
                        train_score = score_run(
                            train_tmp,
                            objective=tune_cfg.objective,
                            score_start=seg.train.start,
                            score_end=seg.train.end,
                        )

                        val_bars = _slice_bars_with_warmup(
                            bars_by_symbol,
                            common_index,
                            score_start=seg.validate.start,
                            score_end=seg.validate.end,
                            warmup_bars=warmup,
                        )
                        _run_backtest_for_market(
                            market=market_enum,
                            bars_by_symbol=val_bars,
                            strategy=_build_strategy_instance(
                                strategy_name=strategy,
                                symbols=symbols,
                                params=full_params,
                            ),
                            cfg=backtest_cfg,
                            run_dir=val_tmp,
                            output_mode="minimal",
                            score_start=seg.validate.start,
                            score_end=seg.validate.end,
                            no_trade_before=seg.validate.start,
                        )
                        val_score = score_run(
                            val_tmp,
                            objective=tune_cfg.objective,
                            score_start=seg.validate.start,
                            score_end=seg.validate.end,
                        )

                        # Selection score: favor validation, but require it isn't a train-only mirage.
                        selection_score = 0.25 * float(train_score.score) + 0.75 * float(
                            val_score.score
                        )

                        record = TrialRecord(
                            segment=int(seg_i),
                            trial=int(trial_i),
                            phase="train",
                            params=dict(full_params),
                            score=float(train_score.score),
                            rejected=bool(train_score.rejected),
                            reject_reason=str(train_score.reason or ""),
                            stats=asdict(train_score.stats),
                            breakdown=dict(train_score.breakdown),
                        )
                        f_trials.write(json.dumps(asdict(record)) + "\n")
                        record = TrialRecord(
                            segment=int(seg_i),
                            trial=int(trial_i),
                            phase="validate",
                            params=dict(full_params),
                            score=float(val_score.score),
                            rejected=bool(val_score.rejected),
                            reject_reason=str(val_score.reason or ""),
                            stats=asdict(val_score.stats),
                            breakdown=dict(val_score.breakdown),
                        )
                        f_trials.write(json.dumps(asdict(record)) + "\n")

                        # Prefer candidates that pass both windows.
                        rejected = bool(train_score.rejected or val_score.rejected)
                        if not rejected and float(selection_score) > float(
                            best_selection_score
                        ):
                            best_selection_score = float(selection_score)
                            best_params = dict(full_params)
                            best_train = {
                                "score": float(train_score.score),
                                "stats": asdict(train_score.stats),
                                "breakdown": dict(train_score.breakdown),
                            }
                            best_val = {
                                "score": float(val_score.score),
                                "stats": asdict(val_score.stats),
                                "breakdown": dict(val_score.breakdown),
                            }

                        if on_progress is not None:
                            on_progress(
                                TuneProgress(
                                    segment=int(seg_i),
                                    n_segments=int(len(segments)),
                                    trial=int(trial_i + 1),
                                    trials_per_segment=int(tune_cfg.trials_per_segment),
                                    phase="search",
                                    best_selection_score=float(best_selection_score),
                                    best_params=dict(best_params or {}),
                                    last_score=float(selection_score),
                                    last_rejected=bool(rejected),
                                    last_reject_reason=str(
                                        train_score.reason or val_score.reason or ""
                                    ),
                                )
                            )
                else:
                    params_by_trial: list[dict[str, Any]] = []
                    for trial_i in range(int(tune_cfg.trials_per_segment)):
                        if stop_event is not None and stop_event.is_set():
                            break
                        params = sample_params(
                            strategy=strategy,
                            rng=rng,
                            space=space,
                            incumbent=incumbent if incumbent else None,
                            drift_frac=tune_cfg.drift_frac,
                        )
                        full_params = dict(fixed_params)
                        full_params.update(params)
                        params_by_trial.append(full_params)

                    futures: dict[int, concurrent.futures.Future] = {}
                    for trial_i, full_params in enumerate(params_by_trial):
                        futures[trial_i] = executor.submit(
                            _evaluate_train_validate_trial,
                            market=market_enum.value,
                            symbols=symbols,
                            strategy=strategy,
                            backtest_cfg=backtest_cfg,
                            objective=tune_cfg.objective,
                            segment=int(seg_i),
                            trial=int(trial_i),
                            params=dict(full_params),
                            train_start=seg.train.start,
                            train_end=seg.train.end,
                            validate_start=seg.validate.start,
                            validate_end=seg.validate.end,
                        )

                    results_by_trial: dict[int, dict[str, Any]] = {}
                    completed = 0
                    for fut in concurrent.futures.as_completed(futures.values()):
                        res = fut.result()
                        results_by_trial[int(res["trial"])] = dict(res)
                        completed += 1

                        if on_progress is not None:
                            best_so_far_score = float("-inf")
                            best_so_far_params: dict[str, Any] = {}
                            for trial_j in sorted(results_by_trial.keys()):
                                r = results_by_trial[trial_j]
                                if bool(r.get("rejected", False)):
                                    continue
                                score = float(r.get("selection_score", float("-inf")))
                                if score > best_so_far_score:
                                    best_so_far_score = float(score)
                                    best_so_far_params = dict(r.get("params") or {})
                            on_progress(
                                TuneProgress(
                                    segment=int(seg_i),
                                    n_segments=int(len(segments)),
                                    trial=int(completed),
                                    trials_per_segment=int(tune_cfg.trials_per_segment),
                                    phase="search",
                                    best_selection_score=float(best_so_far_score),
                                    best_params=dict(best_so_far_params),
                                    last_score=float(res.get("selection_score", float("-inf"))),
                                    last_rejected=bool(res.get("rejected", False)),
                                    last_reject_reason=str(res.get("reject_reason", "")),
                                )
                            )

                    for trial_i in range(len(params_by_trial)):
                        res = results_by_trial.get(trial_i)
                        if res is None:
                            continue

                        train = dict(res.get("train") or {})
                        validate = dict(res.get("validate") or {})
                        selection_score = float(res.get("selection_score", float("-inf")))

                        record = TrialRecord(
                            segment=int(seg_i),
                            trial=int(trial_i),
                            phase="train",
                            params=dict(res.get("params") or {}),
                            score=float(train.get("score", float("-inf"))),
                            rejected=bool(train.get("rejected", False)),
                            reject_reason=str(train.get("reject_reason", "")),
                            stats=dict(train.get("stats") or {}),
                            breakdown=dict(train.get("breakdown") or {}),
                        )
                        f_trials.write(json.dumps(asdict(record)) + "\n")
                        record = TrialRecord(
                            segment=int(seg_i),
                            trial=int(trial_i),
                            phase="validate",
                            params=dict(res.get("params") or {}),
                            score=float(validate.get("score", float("-inf"))),
                            rejected=bool(validate.get("rejected", False)),
                            reject_reason=str(validate.get("reject_reason", "")),
                            stats=dict(validate.get("stats") or {}),
                            breakdown=dict(validate.get("breakdown") or {}),
                        )
                        f_trials.write(json.dumps(asdict(record)) + "\n")

                        rejected = bool(res.get("rejected", False))
                        if not rejected and float(selection_score) > float(
                            best_selection_score
                        ):
                            best_selection_score = float(selection_score)
                            best_params = dict(res.get("params") or {})
                            best_train = {
                                "score": float(train.get("score", float("-inf"))),
                                "stats": dict(train.get("stats") or {}),
                                "breakdown": dict(train.get("breakdown") or {}),
                            }
                            best_val = {
                                "score": float(validate.get("score", float("-inf"))),
                                "stats": dict(validate.get("stats") or {}),
                                "breakdown": dict(validate.get("breakdown") or {}),
                            }

                # If we didn't find a feasible set, fall back to incumbent (or defaults).
                incumbent_params = dict(fixed_params)
                incumbent_params.update(incumbent)
                chosen_params = dict(best_params or incumbent_params or fixed_params or {})
                chosen_score = float(best_selection_score)

                if chosen_params and incumbent_params:
                    # Enforce improvement margin vs incumbent (optional).
                    if float(chosen_score) < float(incumbent_selection_score) + float(
                        tune_cfg.improvement_margin
                    ):
                        chosen_params = dict(incumbent_params)
                        chosen_score = float(incumbent_selection_score)

                # Evaluate out-of-sample on the test window and keep the run outputs.
                seg_out_dir = run_dir / f"segment_{seg_i:03d}"
                test_run_dir = seg_out_dir / "test"
                test_run_dir.mkdir(parents=True, exist_ok=True)

                warmup = _build_strategy_instance(
                    strategy_name=strategy,
                    symbols=symbols,
                    params=chosen_params,
                ).warmup_bars()
                test_bars = _slice_bars_with_warmup(
                    bars_by_symbol,
                    common_index,
                    score_start=seg.test.start,
                    score_end=seg.test.end,
                    warmup_bars=warmup,
                )
                _run_backtest_for_market(
                    market=market_enum,
                    bars_by_symbol=test_bars,
                    strategy=_build_strategy_instance(
                        strategy_name=strategy,
                        symbols=symbols,
                        params=chosen_params,
                    ),
                    cfg=backtest_cfg,
                    run_dir=test_run_dir,
                    score_start=seg.test.start,
                    score_end=seg.test.end,
                    no_trade_before=seg.test.start,
                )
                test_score = score_run(
                    test_run_dir,
                    objective=tune_cfg.objective,
                    score_start=seg.test.start,
                    score_end=seg.test.end,
                )

                # Also record one row for test in trials log (trial=-1).
                record = TrialRecord(
                    segment=int(seg_i),
                    trial=-1,
                    phase="test",
                    params=dict(chosen_params),
                    score=float(test_score.score),
                    rejected=bool(test_score.rejected),
                    reject_reason=str(test_score.reason or ""),
                    stats=asdict(test_score.stats),
                    breakdown=dict(test_score.breakdown),
                )
                f_trials.write(json.dumps(asdict(record)) + "\n")

                selection = SegmentSelection(
                    segment=int(seg_i),
                    params=dict(chosen_params),
                    selection_score=float(chosen_score),
                    train=dict(best_train or {}),
                    validate=dict(best_val or {}),
                    test={
                        "score": float(test_score.score),
                        "rejected": bool(test_score.rejected),
                        "reject_reason": str(test_score.reason or ""),
                        "stats": asdict(test_score.stats),
                        "breakdown": dict(test_score.breakdown),
                        "run_dir": str(test_run_dir),
                    },
                )
                selections.append(selection)
                incumbent = dict(chosen_params)

                if on_progress is not None:
                    on_progress(
                        TuneProgress(
                            segment=int(seg_i),
                            n_segments=int(len(segments)),
                            trial=int(tune_cfg.trials_per_segment),
                            trials_per_segment=int(tune_cfg.trials_per_segment),
                            phase="segment_done",
                            best_selection_score=float(chosen_score),
                            best_params=dict(chosen_params),
                            last_score=float(test_score.score),
                            last_rejected=bool(test_score.rejected),
                            last_reject_reason=str(test_score.reason or ""),
                        )
                    )

                shutil.rmtree(seg_tmp_dir, ignore_errors=True)

        elapsed_s = time.perf_counter() - t0
    finally:
        if executor is not None:
            executor.shutdown(cancel_futures=True)

    stability = _stability_summary(selections)
    (run_dir / "selections.json").write_text(
        json.dumps([asdict(s) for s in selections], indent=2, default=str)
    )
    (run_dir / "stability.json").write_text(json.dumps(stability, indent=2))

    best_latest = dict(selections[-1].params) if selections else dict(base_params or {})
    best_params_path = run_dir / "best_params.json"
    best_params_path.write_text(json.dumps({strategy: best_latest}, indent=2))

    stable_window = min(10, len(selections)) if selections else 0
    best_stable = (
        _stable_params_median(selections, last_n=stable_window)
        if stable_window
        else dict(best_latest)
    )
    if best_stable and not validate_params(strategy, best_stable):
        window = selections[-stable_window:] if stable_window else selections
        if window:
            best_by_test = max(
                window, key=lambda s: float((s.test or {}).get("score", float("-inf")))
            )
            best_stable = dict(best_by_test.params)
        else:
            best_stable = dict(best_latest)

    (run_dir / "best_params_stable.json").write_text(
        json.dumps({strategy: best_stable}, indent=2)
    )

    return TuneResult(
        run_dir=run_dir,
        strategy=strategy,
        market=market_enum.value,
        symbols=symbols,
        config={
            "tune": asdict(tune_cfg),
            "backtest": asdict(backtest_cfg),
            "generated_at": datetime.now().isoformat(),
        },
        segments=[s.to_dict() for s in segments],
        selections=selections,
        best_params_latest=best_latest,
        best_params_stable=best_stable,
        stability=stability,
        elapsed_s=float(elapsed_s),
    )
