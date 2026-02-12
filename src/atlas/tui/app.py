from __future__ import annotations

import json
import math
import os
import queue
import time
import csv
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from threading import Event, Thread
from typing import Any, Optional

import pandas as pd
from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box
from rich.align import Align
from textual import events
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Header, Input, Log, Static

from atlas.backtest.derivatives_engine import run_derivatives_backtest
from atlas.backtest.engine import BacktestConfig, BacktestProgress, run_backtest
from atlas.backtest.plots import write_equity_vs_benchmark_artifacts
from atlas.backtest.window_analysis import rolling_window_summary
from atlas.config import get_alpaca_settings, get_default_max_position_notional_usd
from atlas.data.benchmarks import spy_total_return
from atlas.data.bars import parse_bar_timeframe
from atlas.data.universe import load_universe_bars
from atlas.market import Market, coerce_symbols_for_market, default_symbols, parse_market
from atlas.ml.tune import (
    ObjectiveConfig,
    TuneConfig,
    TuneProgress,
    WalkForwardConfig,
    parse_duration_spec,
    tune_walk_forward,
)
from atlas.paper.runner import PaperConfig, run_paper_loop
from atlas.strategies.registry import build_strategy, list_strategy_names
from atlas.utils.time import NY_TZ, now_ny, parse_iso_datetime


@dataclass
class TuiState:
    market: str = "equity"
    symbols: str = "SPY"
    data_source: str = "sample"
    csv_path: Optional[str] = None
    start: Optional[str] = None
    end: Optional[str] = None
    timeframe: Optional[str] = None
    bar_timeframe: str = "1Min"
    alpaca_feed: str = "delayed_sip"
    paper_feed: str = "iex"
    strategy: str = "spy_open_close"
    fast_window: int = 10
    slow_window: int = 30
    strategy_params: dict[str, dict[str, Any]] = field(default_factory=dict)
    initial_cash: float = 100_000.0
    max_position_notional_usd: float = 10_000.0
    slippage_bps: float = 0.0
    taker_fee_bps: float = 0.0
    allow_short: bool = False
    debug: bool = False
    paper_lookback_bars: int = 200
    paper_poll_seconds: int = 60
    paper_max_position_notional_usd: float = 1_000.0
    paper_regular_hours_only: bool = True
    paper_allow_trading_when_closed: bool = False
    paper_limit_offset_bps: float = 5.0
    paper_dry_run: bool = False
    tune_trials_per_segment: int = 60
    tune_jobs: int = 1
    tune_seed: int = 7
    tune_train: str = "30d"
    tune_validate: str = "7d"
    tune_test: str = "7d"
    tune_step: str = "7d"
    tune_drift_frac: float = 0.50
    tune_improvement_margin: float = 0.0
    tune_last_run_dir: Optional[str] = None
    tune_best_params: dict[str, dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: dict) -> "TuiState":
        fields = {f.name for f in cls.__dataclass_fields__.values()}
        data = {k: v for k, v in raw.items() if k in fields}
        return cls(**data)

    def to_dict(self) -> dict:
        return asdict(self)


STRATEGY_PARAM_SPECS: dict[str, dict[str, type]] = {
    "spy_open_close": {},
    "no_trade": {},
    "ema_crossover": {
        "fast_window": int,
        "slow_window": int,
    },
    "ma_crossover": {
        "fast_window": int,
        "slow_window": int,
    },
    "nec_x": {
        "M": int,
        "V": int,
        "Wcorr": int,
        "rho_min": float,
        "strength_entry": float,
        "strength_exit": float,
        "H_max": int,
        "k_cost": float,
        "spread_floor_bps": float,
        "slip_bps": float,
        "daily_loss_limit": float,
        "kill_switch": float,
    },
    "nec_pdt": {
        "M": int,
        "V": int,
        "eps": float,
        "H": int,
        "base_thr_bps": float,
        "budget_step_bps": float,
        "atr_lookback_bars": int,
        "stop_atr_mult": float,
        "trail_atr_mult": float,
        "min_hold_bars": int,
        "flip_confirm_bars": int,
        "max_day_trades_per_rolling_5_days": int,
        "half_spread_bps": float,
        "slippage_bps": float,
        "fee_bps": float,
    },
    "orb_trend": {
        "orb_minutes": int,
        "orb_breakout_bps": float,
        "confirm_bars": int,
        "atr_window": int,
        "er_window": int,
        "er_min": float,
        "expected_hold_bars": int,
        "k_cost": float,
        "slippage_bps": float,
        "min_hold_bars": int,
        "daily_loss_limit": float,
        "kill_switch": float,
    },
    "crypto_ensemble": {
        "market_symbol": str,
        "ema_fast": int,
        "ema_slow": int,
        "atr_window": int,
        "er_window": int,
        "breakout_window": int,
        "momentum_window": int,
        "er_trend_min": float,
        "er_range_max": float,
        "trend_z_min": float,
        "min_atr_bps": float,
        "meanrev_ewm_span": int,
        "meanrev_entry_z": float,
        "meanrev_exit_z": float,
        "rsi_window": int,
        "rsi_oversold": float,
        "rsi_overbought": float,
        "require_vwap_alignment_for_trend": bool,
        "meanrev_disable_cost_rt_bps": float,
        "meanrev_allow_bear_trend_long_only": bool,
        "meanrev_setup_max_bars": int,
        "meanrev_reversal_min_bps": float,
        "meanrev_size_mult": float,
        "meanrev_stop_atr_mult": float,
        "meanrev_trail_atr_mult": float,
        "meanrev_max_hold_bars": int,
        "breakout_buffer_bps": float,
        "confirm_bars": int,
        "max_positions": int,
        "max_gross_exposure": float,
        "max_exposure_per_symbol": float,
        "risk_budget": float,
        "stop_atr_mult": float,
        "trail_atr_mult": float,
        "take_profit_atr_mult": float,
        "max_hold_bars": int,
        "min_hold_bars": int,
        "cooldown_bars": int,
        "flip_confirm_bars": int,
        "min_dollar_volume_ewma": float,
        "dv_ewm_span": int,
        "rebalance_exposure_threshold": float,
        "min_trade_notional_usd": float,
        "slippage_bps": float,
        "taker_fee_bps": float,
        "edge_floor_bps": float,
        "k_cost": float,
        "daily_loss_limit": float,
        "kill_switch": float,
        "kill_switch_cooldown_days": int,
        "market_drawdown_off": float,
        "market_drawdown_reduce": float,
        "market_vol_off_bps": float,
        "market_vol_reduce_bps": float,
        "market_peak_halflife_bars": int,
        "market_mom_bars": int,
        "market_mom_off": float,
        "market_mom_reduce": float,
        "heartbeat_every_bars": int,
        "heartbeat_notional_usd": float,
    },
    "crypto_tsm": {
        "market_symbol": str,
        "ema_fast": int,
        "ema_slow": int,
        "atr_window": int,
        "momentum_window": int,
        "confirm_bars": int,
        "exit_confirm_bars": int,
        "max_positions": int,
        "max_gross_exposure": float,
        "max_exposure_per_symbol": float,
        "risk_budget": float,
        "stop_atr_mult": float,
        "trail_atr_mult": float,
        "take_profit_atr_mult": float,
        "max_hold_bars": int,
        "min_hold_bars": int,
        "cooldown_bars": int,
        "rebalance_interval_bars": int,
        "rebalance_exposure_threshold": float,
        "min_dollar_volume_ewma": float,
        "dv_ewm_span": int,
        "min_trade_notional_usd": float,
        "slippage_bps": float,
        "taker_fee_bps": float,
        "edge_floor_bps": float,
        "k_cost": float,
        "daily_loss_limit": float,
        "kill_switch": float,
        "kill_switch_cooldown_days": int,
        "market_drawdown_off": float,
        "market_drawdown_reduce": float,
        "market_vol_off_bps": float,
        "market_vol_reduce_bps": float,
        "market_peak_halflife_bars": int,
    },
    "crypto_rotation": {
        "market_symbol": str,
        "rebalance_interval_bars": int,
        "min_trade_notional_usd": float,
        "rebalance_exposure_threshold": float,
        "mom_short_bars": int,
        "mom_med_bars": int,
        "mom_long_bars": int,
        "w_mom_short": float,
        "w_mom_med": float,
        "w_mom_long": float,
        "vol_window_bars": int,
        "vol_target_bps_per_bar": float,
        "max_total_exposure": float,
        "max_exposure_per_symbol": float,
        "top_k": int,
        "score_floor": float,
        "dv_ewm_span": int,
        "min_dollar_volume_ewma": float,
        "slippage_bps": float,
        "taker_fee_bps": float,
        "k_cost": float,
        "edge_floor_bps": float,
        "daily_loss_limit": float,
        "kill_switch": float,
        "kill_switch_cooldown_days": int,
        "market_drawdown_off": float,
        "market_drawdown_reduce": float,
        "market_vol_off_bps": float,
        "market_vol_reduce_bps": float,
        "market_peak_halflife_bars": int,
        "market_mom_bars": int,
        "market_mom_off": float,
        "market_mom_reduce": float,
        "heartbeat_every_bars": int,
        "heartbeat_notional_usd": float,
    },
    "crypto_momentum": {
        "market_symbol": str,
        "momentum_window_bars": int,
        "max_total_exposure": float,
        "max_exposure_per_symbol": float,
        "rebalance_interval_bars": int,
        "rebalance_exposure_threshold": float,
        "min_trade_notional_usd": float,
        "heartbeat_every_bars": int,
        "heartbeat_notional_usd": float,
    },
    "crypto_meta": {
        "market_symbol": str,
        "regime_mom_bars": int,
        "regime_mom_threshold": float,
        "regime_er_bars": int,
        "regime_er_min": float,
        "rotation_weight_trending": float,
        "rotation_weight_ranging": float,
        "ensemble_params_file": str,
        "rotation_params_file": str,
        "ensemble_symbols": str,
        "rotation_symbols": str,
    },
    "perp_flare": {
        "atr_window": int,
        "ema_fast": int,
        "ema_slow": int,
        "er_window": int,
        "breakout_window": int,
        "er_min": float,
        "taker_fee_bps": float,
        "half_spread_bps": float,
        "base_slippage_bps": float,
        "edge_floor_bps": float,
        "k_cost": float,
        "risk_per_trade": float,
        "stop_atr_mult": float,
        "trail_atr_mult": float,
        "max_margin_utilization": float,
        "max_leverage": float,
        "sizing_mode": str,
        "target_leverage": float,
        "maintenance_margin_rate": float,
        "min_liq_buffer_atr": float,
    },
    "perp_hawk": {
        "atr_window": int,
        "ema_fast": int,
        "ema_slow": int,
        "er_window": int,
        "breakout_window": int,
        "breakout_buffer_bps": float,
        "er_min": float,
        "trend_z_min": float,
        "min_atr_bps": float,
        "allow_trend_entry_without_breakout": bool,
        "risk_budget": float,
        "stop_atr_mult": float,
        "trail_atr_mult": float,
        "max_positions": int,
        "rebalance_exposure_threshold": float,
        "max_leverage": float,
        "max_margin_utilization": float,
        "funding_entry_bps_per_day": float,
        "funding_exit_bps_per_day": float,
        "daily_loss_limit": float,
        "kill_switch": float,
        "min_hold_bars": int,
        "flip_confirm_bars": int,
        "cooldown_bars": int,
    },
    "perp_scalp": {
        "atr_window": int,
        "ema_fast": int,
        "ema_slow": int,
        "er_window": int,
        "breakout_window": int,
        "breakout_buffer_bps": float,
        "er_min": float,
        "trend_z_min": float,
        "min_atr_bps": float,
        "edge_floor_bps": float,
        "k_cost": float,
        "taker_fee_bps": float,
        "slippage_bps": float,
        "funding_entry_bps_per_day": float,
        "funding_exit_bps_per_day": float,
        "risk_per_trade": float,
        "stop_atr_mult": float,
        "trail_atr_mult": float,
        "take_profit_atr_mult": float,
        "max_hold_bars": int,
        "min_hold_bars": int,
        "flip_confirm_bars": int,
        "cooldown_bars": int,
        "sizing_mode": str,
        "target_leverage": float,
        "max_leverage": float,
        "max_margin_utilization": float,
        "maintenance_margin_rate": float,
        "min_liq_buffer_atr": float,
        "daily_loss_limit": float,
        "kill_switch": float,
    },
    "perp_weekly_trend_reset": {
        "lookback_days": int,
        "momentum_threshold_bps": float,
        "ema_fast": int,
        "ema_slow": int,
        "require_ema_confirmation": bool,
        "target_leverage": float,
        "max_margin_utilization": float,
        "maintenance_margin_rate": float,
        "stop_atr_mult": float,
        "trail_atr_mult": float,
        "atr_window": int,
        "min_liq_buffer_atr": float,
        "use_stops": bool,
        "rebalance_weekday_utc": int,
        "rebalance_hour_utc": int,
        "rebalance_minute_utc": int,
        "weekly_nudge_exposure": float,
        "min_trade_notional_usd": float,
        "heartbeat_exposure": float,
        "heartbeat_hold_bars": int,
    },
    "perp_weekly_profit_chase": {
        "rebalance_weekday_utc": int,
        "rebalance_hour_utc": int,
        "rebalance_minute_utc": int,
        "weekly_profit_target": float,
        "weekly_chase_k": float,
        "atr_window": int,
        "opening_range_minutes": int,
        "breakout_buffer_bps": float,
        "lookback_short_days": float,
        "lookback_long_days": float,
        "momentum_threshold_bps": float,
        "min_atr_bps": float,
        "sizing_mode": str,
        "risk_per_trade": float,
        "base_leverage": float,
        "max_leverage": float,
        "max_margin_utilization": float,
        "maintenance_margin_rate": float,
        "stop_atr_mult": float,
        "min_liq_buffer_atr": float,
        "min_trade_notional_usd": float,
        "weekly_heartbeat_exposure": float,
        "weekly_heartbeat_hold_bars": int,
        "weekly_nudge_exposure": float,
        "max_flips_per_day": int,
    },
    "basis_carry": {
        "funding_ema_alpha": float,
        "funding_entry_bps_per_day": float,
        "funding_exit_bps_per_day": float,
        "edge_horizon_hours": float,
        "min_basis_bps": float,
        "min_basis_exit_bps": float,
        "basis_mean_bps": float,
        "basis_halflife_hours": float,
        "basis_momentum_window_bars": int,
        "max_basis_widening_bps_per_hour": float,
        "basis_vol_window_bars": int,
        "lambda_basis_vol": float,
        "edge_saturation_bps": float,
        "collateral_buffer_frac": float,
        "z_sigma_daily": float,
        "spot_vol_window_bars": int,
        "max_leverage": float,
        "max_margin_utilization": float,
        "maintenance_margin_rate": float,
        "rebalance_drift_frac": float,
        "rebalance_min_notional_usd": float,
        "min_trade_notional_usd": float,
        "allow_reverse": bool,
        "require_funding_rate": bool,
    },
    "hedge": {
        "edge_horizon_hours": float,
        "funding_ema_alpha": float,
        "basis_halflife_hours": float,
        "theta_intercept_bps": float,
        "theta_funding_beta": float,
        "include_expected_rebalance_costs": bool,
        "cov_window_bars": int,
        "rebalance_delta_max": float,
        "rebalance_turnover_frac_per_unit_delta": float,
        "spot_financing_rate_per_hour": float,
        "z_risk": float,
        "lambda_risk": float,
        "z_liq": float,
        "collateral_buffer_frac": float,
        "max_leverage": float,
        "max_margin_utilization": float,
        "maintenance_margin_rate": float,
        "min_trade_notional_usd": float,
        "rebalance_min_notional_usd": float,
        "flip_hysteresis_bps": float,
        "require_funding_rate": bool,
    },
}

STRATEGY_DEFAULT_PARAMS: dict[str, dict[str, Any]] = {
    "spy_open_close": {},
    "no_trade": {},
    "ema_crossover": {
        "fast_window": 10,
        "slow_window": 30,
    },
    "ma_crossover": {
        "fast_window": 10,
        "slow_window": 30,
    },
    "nec_x": {
        "M": 6,
        "V": 12,
        "Wcorr": 12,
        "rho_min": 0.60,
        "strength_entry": 0.80,
        "strength_exit": 0.20,
        "H_max": 6,
        "k_cost": 1.25,
        "spread_floor_bps": 0.50,
        "slip_bps": 0.75,
        "daily_loss_limit": 0.010,
        "kill_switch": 0.025,
    },
    "nec_pdt": {
        "M": 6,
        "V": 12,
        "eps": 1e-8,
        "H": 12,
        "base_thr_bps": 10.0,
        "budget_step_bps": 4.0,
        "atr_lookback_bars": 12,
        "stop_atr_mult": 2.0,
        "trail_atr_mult": 2.5,
        "min_hold_bars": 4,
        "flip_confirm_bars": 3,
        "max_day_trades_per_rolling_5_days": 3,
        "half_spread_bps": 1.5,
        "slippage_bps": 2.0,
        "fee_bps": 0.3,
    },
    "orb_trend": {
        "orb_minutes": 30,
        "orb_breakout_bps": 4.0,
        "confirm_bars": 2,
        "atr_window": 20,
        "er_window": 12,
        "er_min": 0.35,
        "expected_hold_bars": 12,
        "k_cost": 2.0,
        "slippage_bps": 1.25,
        "min_hold_bars": 3,
        "kill_switch": 0.025,
    },
    "crypto_ensemble": {
        "market_symbol": "BTC/USD",
        "ema_fast": 20,
        "ema_slow": 80,
        "atr_window": 20,
        "er_window": 40,
        "breakout_window": 60,
        "momentum_window": 240,
        "er_trend_min": 0.35,
        "er_range_max": 0.20,
        "trend_z_min": 0.20,
        "min_atr_bps": 6.0,
        "meanrev_ewm_span": 120,
        "meanrev_entry_z": 1.75,
        "meanrev_exit_z": 0.50,
        "rsi_window": 14,
        "rsi_oversold": 35.0,
        "rsi_overbought": 65.0,
        "require_vwap_alignment_for_trend": True,
        "meanrev_disable_cost_rt_bps": 30.0,
        "meanrev_allow_bear_trend_long_only": False,
        "meanrev_setup_max_bars": 8,
        "meanrev_reversal_min_bps": 0.0,
        "meanrev_size_mult": 0.35,
        "meanrev_stop_atr_mult": 0.0,
        "meanrev_trail_atr_mult": 0.0,
        "meanrev_max_hold_bars": 0,
        "breakout_buffer_bps": 2.0,
        "confirm_bars": 2,
        "max_positions": 3,
        "max_gross_exposure": 1.0,
        "max_exposure_per_symbol": 1.0,
        "risk_budget": 0.02,
        "stop_atr_mult": 2.0,
        "trail_atr_mult": 3.0,
        "take_profit_atr_mult": 0.0,
        "max_hold_bars": 0,
        "min_hold_bars": 3,
        "cooldown_bars": 6,
        "flip_confirm_bars": 3,
        "min_dollar_volume_ewma": 50_000.0,
        "dv_ewm_span": 60,
        "rebalance_exposure_threshold": 0.05,
        "min_trade_notional_usd": 25.0,
        "slippage_bps": 3.0,
        "taker_fee_bps": 25.0,
        "edge_floor_bps": 4.0,
        "k_cost": 2.0,
        "daily_loss_limit": 0.03,
        "kill_switch": 0.12,
        "kill_switch_cooldown_days": 7,
        "market_drawdown_off": 0.15,
        "market_drawdown_reduce": 0.08,
        "market_vol_off_bps": 250.0,
        "market_vol_reduce_bps": 150.0,
        "market_peak_halflife_bars": 240,
        "heartbeat_every_bars": 0,
        "heartbeat_notional_usd": 25.0,
    },
    "crypto_tsm": {
        "market_symbol": "BTC/USD",
        "ema_fast": 24,
        "ema_slow": 120,
        "atr_window": 24,
        "momentum_window": 240,
        "confirm_bars": 3,
        "exit_confirm_bars": 3,
        "max_positions": 2,
        "max_gross_exposure": 1.0,
        "max_exposure_per_symbol": 1.0,
        "risk_budget": 0.05,
        "stop_atr_mult": 3.0,
        "trail_atr_mult": 5.0,
        "take_profit_atr_mult": 0.0,
        "max_hold_bars": 0,
        "min_hold_bars": 6,
        "cooldown_bars": 12,
        "rebalance_interval_bars": 4,
        "rebalance_exposure_threshold": 0.05,
        "min_dollar_volume_ewma": 100_000.0,
        "dv_ewm_span": 60,
        "min_trade_notional_usd": 25.0,
        "slippage_bps": 3.0,
        "taker_fee_bps": 25.0,
        "edge_floor_bps": 8.0,
        "k_cost": 2.0,
        "daily_loss_limit": 0.05,
        "kill_switch": 0.20,
        "kill_switch_cooldown_days": 7,
        "market_drawdown_off": 0.20,
        "market_drawdown_reduce": 0.10,
        "market_vol_off_bps": 300.0,
        "market_vol_reduce_bps": 180.0,
        "market_peak_halflife_bars": 240,
    },
    "crypto_rotation": {
        "market_symbol": "BTC/USD",
        "rebalance_interval_bars": 28,
        "min_trade_notional_usd": 25.0,
        "rebalance_exposure_threshold": 0.02,
        "mom_short_bars": 28,
        "mom_med_bars": 120,
        "mom_long_bars": 360,
        "w_mom_short": 0.20,
        "w_mom_med": 0.30,
        "w_mom_long": 0.50,
        "vol_window_bars": 120,
        "vol_target_bps_per_bar": 80.0,
        "max_total_exposure": 1.0,
        "max_exposure_per_symbol": 0.60,
        "top_k": 2,
        "score_floor": 0.0,
        "dv_ewm_span": 60,
        "min_dollar_volume_ewma": 50_000.0,
        "slippage_bps": 3.0,
        "taker_fee_bps": 25.0,
        "k_cost": 1.0,
        "edge_floor_bps": 0.0,
        "daily_loss_limit": 0.05,
        "kill_switch": 0.25,
        "kill_switch_cooldown_days": 7,
        "market_drawdown_off": 0.25,
        "market_drawdown_reduce": 0.12,
        "market_vol_off_bps": 300.0,
        "market_vol_reduce_bps": 180.0,
        "market_peak_halflife_bars": 240,
        "market_mom_bars": 0,
        "market_mom_off": 0.0,
        "market_mom_reduce": 0.0,
        "heartbeat_every_bars": 0,
        "heartbeat_notional_usd": 25.0,
    },
    "crypto_meta": {
        "market_symbol": "BTC/USD",
        "regime_mom_bars": 168,
        "regime_mom_threshold": 0.0,
        "regime_er_bars": 84,
        "regime_er_min": 0.25,
        "rotation_weight_trending": 1.0,
        "rotation_weight_ranging": 0.0,
        "ensemble_params_file": "strategy_params/crypto_ensemble_ultra_6h_coinbase_heartbeat.json",
        "rotation_params_file": "strategy_params/crypto_rotation_2022_candidate_r2_momfilter_v11_6h_coinbase_nohb.json",
        "ensemble_symbols": "BTC/USD,ETH/USD",
        "rotation_symbols": "",
    },
    "perp_flare": {
        "atr_window": 14,
        "ema_fast": 12,
        "ema_slow": 24,
        "er_window": 10,
        "breakout_window": 20,
        "er_min": 0.35,
        "taker_fee_bps": 3.0,
        "half_spread_bps": 1.0,
        "base_slippage_bps": 1.5,
        "edge_floor_bps": 5.0,
        "k_cost": 1.5,
        "risk_per_trade": 0.01,
        "stop_atr_mult": 2.0,
        "trail_atr_mult": 3.0,
        "max_margin_utilization": 0.65,
        "max_leverage": 10.0,
        "sizing_mode": "risk",
        "target_leverage": 10.0,
        "maintenance_margin_rate": 0.05,
        "min_liq_buffer_atr": 3.0,
    },
    "perp_hawk": {
        "atr_window": 14,
        "ema_fast": 20,
        "ema_slow": 60,
        "er_window": 20,
        "breakout_window": 20,
        "breakout_buffer_bps": 2.0,
        "er_min": 0.30,
        "trend_z_min": 0.25,
        "min_atr_bps": 5.0,
        "allow_trend_entry_without_breakout": True,
        "risk_budget": 0.010,
        "stop_atr_mult": 2.2,
        "trail_atr_mult": 3.2,
        "max_positions": 2,
        "rebalance_exposure_threshold": 0.05,
        "max_leverage": 3.0,
        "max_margin_utilization": 0.35,
        "funding_entry_bps_per_day": 25.0,
        "funding_exit_bps_per_day": 60.0,
        "daily_loss_limit": 0.02,
        "kill_switch": 0.10,
        "min_hold_bars": 3,
        "flip_confirm_bars": 3,
        "cooldown_bars": 5,
    },
    "perp_scalp": {
        "atr_window": 14,
        "ema_fast": 8,
        "ema_slow": 21,
        "er_window": 10,
        "breakout_window": 8,
        "breakout_buffer_bps": 1.0,
        "er_min": 0.25,
        "trend_z_min": 0.15,
        "min_atr_bps": 8.0,
        "edge_floor_bps": 3.0,
        "k_cost": 1.5,
        "taker_fee_bps": 3.0,
        "slippage_bps": 1.5,
        "funding_entry_bps_per_day": 40.0,
        "funding_exit_bps_per_day": 80.0,
        "risk_per_trade": 0.005,
        "stop_atr_mult": 1.2,
        "trail_atr_mult": 1.8,
        "take_profit_atr_mult": 1.5,
        "max_hold_bars": 12,
        "min_hold_bars": 2,
        "flip_confirm_bars": 2,
        "cooldown_bars": 4,
        "sizing_mode": "risk",
        "target_leverage": None,
        "max_leverage": 5.0,
        "max_margin_utilization": 0.40,
        "maintenance_margin_rate": 0.05,
        "min_liq_buffer_atr": 2.5,
        "daily_loss_limit": 0.02,
        "kill_switch": 0.10,
    },
    "perp_weekly_trend_reset": {
        "lookback_days": 14,
        "momentum_threshold_bps": 0.0,
        "ema_fast": 12,
        "ema_slow": 48,
        "require_ema_confirmation": False,
        "target_leverage": 2.0,
        "max_margin_utilization": 0.80,
        "maintenance_margin_rate": 0.05,
        "stop_atr_mult": 2.5,
        "trail_atr_mult": 4.0,
        "atr_window": 14,
        "min_liq_buffer_atr": 4.0,
        "use_stops": False,
        "rebalance_weekday_utc": 0,
        "rebalance_hour_utc": 0,
        "rebalance_minute_utc": 5,
        "weekly_nudge_exposure": 0.002,
        "min_trade_notional_usd": 10.0,
        "heartbeat_exposure": 0.01,
        "heartbeat_hold_bars": 12,
    },
    "perp_weekly_profit_chase": {
        "rebalance_weekday_utc": 0,
        "rebalance_hour_utc": 0,
        "rebalance_minute_utc": 5,
        "weekly_profit_target": 0.01,
        "weekly_chase_k": 0.0,
        "atr_window": 14,
        "opening_range_minutes": 60,
        "breakout_buffer_bps": 8.0,
        "lookback_short_days": 1.0,
        "lookback_long_days": 7.0,
        "momentum_threshold_bps": 0.0,
        "min_atr_bps": 2.0,
        "sizing_mode": "risk",
        "risk_per_trade": 0.03,
        "base_leverage": 10.0,
        "max_leverage": 25.0,
        "max_margin_utilization": 0.80,
        "maintenance_margin_rate": 0.05,
        "stop_atr_mult": 2.0,
        "min_liq_buffer_atr": 3.0,
        "min_trade_notional_usd": 10.0,
        "weekly_heartbeat_exposure": 0.01,
        "weekly_heartbeat_hold_bars": 1,
        "weekly_nudge_exposure": 0.002,
        "max_flips_per_day": 3,
    },
    "basis_carry": {
        "funding_ema_alpha": 0.20,
        "funding_entry_bps_per_day": 10.0,
        "funding_exit_bps_per_day": 0.0,
        "edge_horizon_hours": 8.0,
        "min_basis_bps": 5.0,
        "min_basis_exit_bps": 0.0,
        "basis_mean_bps": 0.0,
        "basis_halflife_hours": 24.0,
        "basis_momentum_window_bars": 30,
        "max_basis_widening_bps_per_hour": 10.0,
        "basis_vol_window_bars": 120,
        "lambda_basis_vol": 1.0,
        "edge_saturation_bps": 50.0,
        "collateral_buffer_frac": 0.10,
        "z_sigma_daily": 3.0,
        "spot_vol_window_bars": 120,
        "max_leverage": 3.0,
        "max_margin_utilization": 0.50,
        "maintenance_margin_rate": 0.05,
        "rebalance_drift_frac": 0.02,
        "rebalance_min_notional_usd": 100.0,
        "min_trade_notional_usd": 200.0,
        "allow_reverse": False,
        "require_funding_rate": False,
    },
    "hedge": {
        "edge_horizon_hours": 8.0,
        "funding_ema_alpha": 0.20,
        "basis_halflife_hours": 24.0,
        "theta_intercept_bps": 0.0,
        "theta_funding_beta": 0.25,
        "include_expected_rebalance_costs": True,
        "cov_window_bars": 240,
        "rebalance_delta_max": 0.02,
        "rebalance_turnover_frac_per_unit_delta": 0.50,
        "spot_financing_rate_per_hour": 0.0,
        "z_risk": 1.0,
        "lambda_risk": 8.0,
        "z_liq": 2.33,
        "collateral_buffer_frac": 0.10,
        "max_leverage": 3.0,
        "max_margin_utilization": 0.50,
        "maintenance_margin_rate": 0.05,
        "min_trade_notional_usd": 200.0,
        "rebalance_min_notional_usd": 100.0,
        "flip_hysteresis_bps": 2.0,
        "require_funding_rate": False,
    },
}


def _parse_relative_timeframe(spec: str) -> timedelta:
    spec = spec.strip().lower()
    if spec.endswith("min"):
        return timedelta(minutes=int(spec[:-3]))
    if spec.endswith("h"):
        return timedelta(hours=int(spec[:-1]))
    if spec.endswith("d"):
        return timedelta(days=int(spec[:-1]))
    if spec.endswith("w"):
        return timedelta(weeks=int(spec[:-1]))
    if spec.endswith("m"):
        return timedelta(days=30 * int(spec[:-1]))
    if spec.endswith("y"):
        return timedelta(days=365 * int(spec[:-1]))
    raise ValueError("unsupported timeframe spec")


def _parse_bool_arg(value: str) -> Optional[bool]:
    value = value.strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    return None


def _infer_bar_minutes(index: pd.DatetimeIndex) -> float:
    if len(index) < 3:
        return 0.0
    diffs = index.to_series().diff().dropna().dt.total_seconds() / 60.0
    median = float(diffs.median())
    return median if median > 0 else 0.0


def _resolve_time_window(
    *,
    bars: pd.DataFrame,
    timeframe: Optional[str],
    start: Optional[str],
    end: Optional[str],
) -> tuple[pd.DataFrame, Optional[str], Optional[str]]:
    start_dt = parse_iso_datetime(start) if start else None
    end_dt = parse_iso_datetime(end) if end else None

    if timeframe:
        delta = _parse_relative_timeframe(timeframe)
        end_dt = pd.Timestamp(bars.index[-1]).to_pydatetime()
        start_dt = end_dt - delta

    if start_dt is not None:
        bars = bars[bars.index >= start_dt]
    if end_dt is not None:
        bars = bars[bars.index <= end_dt]

    return (
        bars,
        start_dt.isoformat() if start_dt else None,
        end_dt.isoformat() if end_dt else None,
    )


class AtlasTui(App):
    SUGGESTION_MAX_LINES = 8
    COMMAND_ALIASES = {
        "/?": "/help",
        "/bars": "/bar",
        "/strategy": "/algorithm",
        "/aglorithm": "/algorithm",
        "/algorithim": "/algorithm",
        "/initialcash": "/cash",
        "/initial_cash": "/cash",
        "/max": "/maxnotional",
        "/notional": "/maxnotional",
        "/max_notional": "/maxnotional",
        "/slip": "/slippage",
        "/fee": "/takerfee",
        "/taker_fee": "/takerfee",
        "/allowshort": "/short",
        "/allow_short": "/short",
        "/paperlookbackbars": "/paperlookback",
        "/paperpolls": "/paperpoll",
        "/papermax": "/papermaxnotional",
        "/equity": "/stock",
        "/stocks": "/stock",
        "/stocks": "/stock",
        "/cryptos": "/crypto",
        "/derivatives": "/derivative",
        "/perp": "/derivative",
        "/perps": "/derivative",
        "/future": "/derivative",
        "/futures": "/derivative",
        "/train": "/tune",
        "/tuning": "/tune",
        "/optimize": "/tune",
        "/presets": "/preset",
    }
    BASE_COMMANDS = [
        "/help",
        "/?",
        "/stock",
        "/stock",
        "/crypto",
        "/derivative",
        "/backtest",
        "/tune",
        "/paper",
        "/timeframe",
        "/bar",
        "/bars",
        "/algorithm",
        "/strategy",
        "/param",
        "/params",
        "/preset",
        "/fast",
        "/slow",
        "/cash",
        "/initialcash",
        "/maxnotional",
        "/slippage",
        "/takerfee",
        "/short",
        "/debug",
        "/data",
        "/feed",
        "/paperfeed",
        "/paperlookback",
        "/paperpoll",
        "/papermaxnotional",
        "/paperclosed",
        "/paperrth",
        "/paperlimitbps",
        "/paperdry",
        "/csv",
        "/symbol",
        "/symbols",
        "/start",
        "/end",
        "/save",
        "/load",
    ]
    CSS = """
    Screen { layout: vertical; background: $surface; }
    #body { height: 1fr; }
    #settings { width: 36%; border: round $accent; padding: 1 2; }
    #results { width: 64%; border: round $accent; padding: 1 2; }
    #lower { height: 18; border: round $accent; padding: 1 2; }
    #log { height: 1fr; background: $surface; }
    #suggestions {
        height: 0;
        max-height: 10;
        border: round $accent;
        padding: 0 1;
        color: $text-muted;
        text-style: dim;
        background: $surface;
        overflow-y: auto;
    }
    #input { height: 3; border: round $accent; padding: 0 1; }
    Input > .input--suggestion { color: $text-muted; text-style: dim; }
    """

    def __init__(self) -> None:
        super().__init__()
        self.state = TuiState(
            max_position_notional_usd=get_default_max_position_notional_usd(
                mode="backtest"
            ),
            paper_max_position_notional_usd=get_default_max_position_notional_usd(
                mode="paper"
            ),
        )
        self._ctrl_c_armed_at: Optional[float] = None
        self._backtest_thread: Optional[Thread] = None
        self._backtest_events: "queue.Queue[tuple[str, object]]" = queue.Queue()
        self._backtest_status: str = ""
        self._backtest_progress: Optional[BacktestProgress] = None
        self._backtest_run_dir: Optional[Path] = None
        self._backtest_started_at: Optional[float] = None
        self._backtest_initial_cash: float = float(self.state.initial_cash)
        self._backtest_max_notional: float = float(self.state.max_position_notional_usd)
        self._backtest_equity_spark: deque[float] = deque(maxlen=120)
        self._paper_thread: Optional[Thread] = None
        self._paper_stop: Optional[Event] = None
        self._paper_run_dir: Optional[Path] = None
        self._tune_thread: Optional[Thread] = None
        self._tune_stop: Optional[Event] = None
        self._tune_events: "queue.Queue[tuple[str, object]]" = queue.Queue()
        self._tune_status: str = ""
        self._tune_progress: Optional[TuneProgress] = None
        self._tune_run_dir: Optional[Path] = None
        self._tune_started_at: Optional[float] = None
        self._tune_last_result_dir: Optional[Path] = None
        self._tune_score_spark: deque[float] = deque(maxlen=120)
        self._tune_best_score_spark: deque[float] = deque(maxlen=120)
        self._tune_recent_trials: deque[tuple[int, int, float, bool, str]] = deque(maxlen=12)
        self._tune_trial_total: int = 0
        self._tune_trial_rejected: int = 0
        self._tune_segment_index: int = -1
        self._tune_segment_trial_total: int = 0
        self._tune_segment_trial_rejected: int = 0
        self._last_run_dir: Optional[Path] = None
        self._last_live_decision_ts: Optional[str] = None
        self._suggestion_matches: list[str] = []
        self._suggestion_index: int = 0
        self._suggestion_scroll: int = 0
        self._config_path = Path(os.getenv("ATLAS_TUI_CONFIG", ".atlas_tui.json"))
        self._autosave_enabled = True
        self._config_load_failed = False

    def _load_config(self) -> None:
        path = self._config_path
        if not path.exists():
            self._config_load_failed = False
            return
        try:
            raw = json.loads(path.read_text())
            self.state = TuiState.from_dict(raw)
            self._config_load_failed = False
        except Exception as exc:
            self._config_load_failed = True
            self._autosave_enabled = False
            self._write_log(f"failed to load config ({path}): {exc}")

    def _save_config(self) -> None:
        if not self._autosave_enabled or self._config_load_failed:
            return
        path = self._config_path
        try:
            path.write_text(json.dumps(self.state.to_dict(), indent=2))
        except Exception as exc:
            self._autosave_enabled = False
            self._write_log(f"failed to save config ({path}): {exc}")

    def _strategy_params_dir(self) -> Path:
        return Path(os.getenv("ATLAS_STRATEGY_PARAMS_DIR", "strategy_params"))

    def _unwrap_params_json(self, raw: object) -> object:
        if isinstance(raw, dict) and "params" in raw and isinstance(raw["params"], dict):
            return raw["params"]
        if isinstance(raw, dict) and "parameters" in raw and isinstance(raw["parameters"], dict):
            return raw["parameters"]
        return raw

    def _extract_params_for_strategy(
        self, raw: object, strategy: str
    ) -> Optional[dict[str, Any]]:
        raw = self._unwrap_params_json(raw)
        if not isinstance(raw, dict):
            return None

        canonical = strategy.replace("-", "_")
        if canonical in raw and isinstance(raw[canonical], dict):
            return dict(raw[canonical])
        if strategy in raw and isinstance(raw[strategy], dict):
            return dict(raw[strategy])

        # Heuristic: if the file is a direct param dict and it contains at least one known key.
        spec = self._strategy_param_spec(strategy)
        if spec and any(k in spec for k in raw.keys()):
            return dict(raw)

        return None

    def _extract_preset_profile(self, raw: object) -> Optional[dict[str, Any]]:
        if not isinstance(raw, dict):
            return None

        for key in ("atlas_profile", "_atlas_profile", "preset_profile", "profile"):
            candidate = raw.get(key)
            if isinstance(candidate, dict):
                return dict(candidate)
        return None

    def _profile_bool(self, value: Any, *, key: str) -> Optional[bool]:
        if isinstance(value, bool):
            return value
        parsed = _parse_bool_arg(str(value))
        if parsed is None:
            self._write_log(f"preset profile ignored invalid boolean for {key}: {value!r}")
        return parsed

    def _apply_preset_profile(self, profile: dict[str, Any], *, strategy: str) -> None:
        changed: list[str] = []

        market_raw = profile.get("market")
        if isinstance(market_raw, str) and market_raw.strip():
            try:
                self._set_market(market_raw)
                changed.append(f"market={self.state.market}")
            except Exception:
                self._write_log(f"preset profile ignored invalid market: {market_raw!r}")

        data_source_raw = profile.get("data_source")
        if isinstance(data_source_raw, str) and data_source_raw.strip():
            value = data_source_raw.strip().lower()
            if value in {"sample", "csv", "alpaca", "coinbase"}:
                self.state.data_source = value
                changed.append(f"data_source={value}")
            else:
                self._write_log(f"preset profile ignored invalid data_source: {data_source_raw!r}")

        bar_raw = profile.get("bar_timeframe", profile.get("bar"))
        if bar_raw is not None:
            value = str(bar_raw).strip()
            if value:
                self.state.bar_timeframe = value
                changed.append(f"bar_timeframe={value}")

        timeframe_raw = profile.get("timeframe")
        if timeframe_raw is not None:
            value = str(timeframe_raw).strip()
            self.state.timeframe = value if value else None
            changed.append(f"timeframe={self.state.timeframe or 'clear'}")

        start_raw = profile.get("start")
        if start_raw is not None:
            value = str(start_raw).strip()
            self.state.start = value if value else None
            changed.append("start")

        end_raw = profile.get("end")
        if end_raw is not None:
            value = str(end_raw).strip()
            self.state.end = value if value else None
            changed.append("end")

        if (
            self.state.timeframe
            and self.state.data_source in {"alpaca", "coinbase"}
            and (start_raw is None and end_raw is None)
        ):
            try:
                delta = _parse_relative_timeframe(self.state.timeframe)
                end_dt = now_ny()
                start_dt = end_dt - delta
                self.state.start = start_dt.isoformat()
                self.state.end = end_dt.isoformat()
                changed.append("start/end(from timeframe)")
            except Exception:
                self._write_log(
                    f"preset profile timeframe not parsed: {self.state.timeframe!r} "
                    "(use a relative spec like 180d, 30d, 6h)"
                )

        symbols_raw = profile.get("symbols", profile.get("symbol"))
        if symbols_raw is not None:
            raw_symbols: list[str] = []
            if isinstance(symbols_raw, str):
                chunks = [c.strip() for c in symbols_raw.split(",") if c.strip()]
                raw_symbols.extend(chunks)
            elif isinstance(symbols_raw, list):
                for item in symbols_raw:
                    if item is None:
                        continue
                    text = str(item).strip()
                    if text:
                        raw_symbols.append(text)
            else:
                text = str(symbols_raw).strip()
                if text:
                    raw_symbols.append(text)

            try:
                mkt = parse_market(self.state.market)
                coerced = coerce_symbols_for_market(raw_symbols, mkt)
                if coerced:
                    self.state.symbols = ",".join(coerced)
                    changed.append(f"symbols={self.state.symbols}")
            except Exception:
                self._write_log(f"preset profile ignored invalid symbols: {symbols_raw!r}")

        cash_raw = profile.get("initial_cash", profile.get("cash"))
        if cash_raw is not None:
            try:
                cash = float(cash_raw)
                if cash > 0:
                    self.state.initial_cash = cash
                    changed.append(f"initial_cash={cash:g}")
                else:
                    self._write_log(f"preset profile ignored non-positive initial_cash: {cash_raw!r}")
            except Exception:
                self._write_log(f"preset profile ignored invalid initial_cash: {cash_raw!r}")

        max_notional_raw = profile.get(
            "max_position_notional_usd", profile.get("max_notional_usd", profile.get("maxnotional"))
        )
        if max_notional_raw is not None:
            try:
                max_notional = float(max_notional_raw)
                if max_notional > 0:
                    self.state.max_position_notional_usd = max_notional
                    changed.append(f"max_position_notional_usd={max_notional:g}")
                else:
                    self._write_log(
                        f"preset profile ignored non-positive max_position_notional_usd: {max_notional_raw!r}"
                    )
            except Exception:
                self._write_log(
                    f"preset profile ignored invalid max_position_notional_usd: {max_notional_raw!r}"
                )

        slippage_raw = profile.get("slippage_bps")
        if slippage_raw is not None:
            try:
                slippage = float(slippage_raw)
                if slippage < 0:
                    raise ValueError("slippage must be >= 0")
                self.state.slippage_bps = slippage
                if "slippage_bps" in self._strategy_param_spec(strategy):
                    self._ensure_strategy_params(strategy)
                    self.state.strategy_params[strategy]["slippage_bps"] = float(slippage)
                changed.append(f"slippage_bps={slippage:g}")
            except Exception:
                self._write_log(f"preset profile ignored invalid slippage_bps: {slippage_raw!r}")

        taker_fee_raw = profile.get("taker_fee_bps")
        if taker_fee_raw is not None:
            try:
                fee = float(taker_fee_raw)
                if fee < 0:
                    raise ValueError("taker_fee must be >= 0")
                self.state.taker_fee_bps = fee
                if "taker_fee_bps" in self._strategy_param_spec(strategy):
                    self._ensure_strategy_params(strategy)
                    self.state.strategy_params[strategy]["taker_fee_bps"] = float(fee)
                changed.append(f"taker_fee_bps={fee:g}")
            except Exception:
                self._write_log(f"preset profile ignored invalid taker_fee_bps: {taker_fee_raw!r}")

        allow_short_raw = profile.get("allow_short")
        if allow_short_raw is not None:
            value = self._profile_bool(allow_short_raw, key="allow_short")
            if value is not None:
                self.state.allow_short = value
                changed.append(f"allow_short={str(value).lower()}")

        paper_max_notional_raw = profile.get("paper_max_position_notional_usd", profile.get("papermaxnotional"))
        if paper_max_notional_raw is not None:
            try:
                paper_notional = float(paper_max_notional_raw)
                if paper_notional > 0:
                    self.state.paper_max_position_notional_usd = paper_notional
                    changed.append(f"paper_max_position_notional_usd={paper_notional:g}")
            except Exception:
                self._write_log(
                    f"preset profile ignored invalid paper_max_position_notional_usd: {paper_max_notional_raw!r}"
                )

        paper_poll_raw = profile.get("paper_poll_seconds", profile.get("paperpoll"))
        if paper_poll_raw is not None:
            try:
                paper_poll = int(paper_poll_raw)
                if paper_poll > 0:
                    self.state.paper_poll_seconds = paper_poll
                    changed.append(f"paper_poll_seconds={paper_poll}")
            except Exception:
                self._write_log(f"preset profile ignored invalid paper_poll_seconds: {paper_poll_raw!r}")

        paper_lookback_raw = profile.get("paper_lookback_bars", profile.get("paperlookback"))
        if paper_lookback_raw is not None:
            try:
                paper_lookback = int(paper_lookback_raw)
                if paper_lookback > 0:
                    self.state.paper_lookback_bars = paper_lookback
                    changed.append(f"paper_lookback_bars={paper_lookback}")
            except Exception:
                self._write_log(f"preset profile ignored invalid paper_lookback_bars: {paper_lookback_raw!r}")

        for bool_key in ("paper_regular_hours_only", "paper_allow_trading_when_closed", "paper_dry_run"):
            if bool_key not in profile:
                continue
            value = self._profile_bool(profile.get(bool_key), key=bool_key)
            if value is None:
                continue
            setattr(self.state, bool_key, value)
            changed.append(f"{bool_key}={str(value).lower()}")

        self._render_settings()
        if changed:
            self._write_log("applied preset profile: " + ", ".join(changed))

    def _detect_strategy_keys_in_params_file(self, raw: object) -> list[str]:
        raw = self._unwrap_params_json(raw)
        if not isinstance(raw, dict):
            return []

        known = set(list_strategy_names())
        out: set[str] = set()
        for key, value in raw.items():
            if not isinstance(key, str) or not isinstance(value, dict):
                continue
            canonical = key.strip().replace("-", "_")
            if canonical in known:
                out.add(canonical)
        return sorted(out)

    def _strategy_preset_paths(self, strategy: str) -> list[Path]:
        preset_dir = self._strategy_params_dir()
        if not preset_dir.exists() or not preset_dir.is_dir():
            return []

        paths = sorted(preset_dir.glob("*.json"))
        out: list[Path] = []
        for path in paths:
            try:
                raw = json.loads(path.read_text())
            except Exception:
                continue
            params = self._extract_params_for_strategy(raw, strategy)
            if params is not None:
                out.append(path)
        return out

    def _canonicalize_strategy_name(self, name: str) -> str:
        name = name.strip()
        alias_map = {
            "nec-x": "nec_x",
            "nec-pdt": "nec_pdt",
            "orb-trend": "orb_trend",
            "ema-crossover": "ema_crossover",
            "spy-open-close": "spy_open_close",
            "no-trade": "no_trade",
            "perp-flare": "perp_flare",
            "perp-hawk": "perp_hawk",
            "perp-scalp": "perp_scalp",
            "basis-carry": "basis_carry",
            "cash-and-carry": "basis_carry",
            "hedge": "hedge",
            "hedge-implementation": "hedge",
            "hedge-impl": "hedge",
            "hedge_impl": "hedge",
            "crypto-ensemble": "crypto_ensemble",
            "crypto-ens": "crypto_ensemble",
            "crypto-rotation": "crypto_rotation",
            "crypto-rot": "crypto_rotation",
            "crypto-tsm": "crypto_tsm",
            "crypto-trend": "crypto_tsm",
            "crypto-meta": "crypto_meta",
        }
        if name in alias_map:
            return alias_map[name]
        return name

    def _strategy_param_spec(self, strategy: str) -> dict[str, type]:
        return STRATEGY_PARAM_SPECS.get(strategy, {})

    def _ensure_strategy_params(self, strategy: str) -> None:
        alias_map = {
            "nec_x": "nec-x",
            "nec_pdt": "nec-pdt",
            "orb_trend": "orb-trend",
            "ema_crossover": "ema-crossover",
            "spy_open_close": "spy-open-close",
            "spy_open_close": "spy-open-close",
            "no_trade": "no-trade",
            "perp_flare": "perp-flare",
            "perp_hawk": "perp-hawk",
            "perp_scalp": "perp-scalp",
            "basis_carry": "basis-carry",
            "hedge": "hedge-implementation",
            "crypto_ensemble": "crypto-ensemble",
            "crypto_tsm": "crypto-tsm",
            "crypto_rotation": "crypto-rotation",
            "crypto_meta": "crypto-meta",
        }
        alias = alias_map.get(strategy)
        if alias and alias in self.state.strategy_params and strategy not in self.state.strategy_params:
            self.state.strategy_params[strategy] = self.state.strategy_params.pop(alias)

        if strategy == "ma_crossover":
            if not self.state.strategy_params:
                defaults = {
                    "fast_window": self.state.fast_window,
                    "slow_window": self.state.slow_window,
                }
            else:
                defaults = STRATEGY_DEFAULT_PARAMS.get(strategy, {})
        else:
            defaults = STRATEGY_DEFAULT_PARAMS.get(strategy, {})

        if strategy not in self.state.strategy_params:
            self.state.strategy_params[strategy] = dict(defaults)
        else:
            params = self.state.strategy_params[strategy]
            for key, value in defaults.items():
                params.setdefault(key, value)

        if strategy in {"ma_crossover", "ema_crossover"}:
            params = self.state.strategy_params.get(strategy, {})
            if "fast_window" in params:
                self.state.fast_window = int(params["fast_window"])
            if "slow_window" in params:
                self.state.slow_window = int(params["slow_window"])

    def _normalize_param_key(self, strategy: str, key: str) -> str:
        spec = self._strategy_param_spec(strategy)
        if spec:
            for candidate in spec:
                if candidate.lower() == key.lower():
                    return candidate
        return key

    def _format_param_value(self, value: Any) -> str:
        if isinstance(value, float):
            return f"{value:.6g}"
        return str(value)

    def _format_strategy_params(self, strategy: str, params: dict[str, Any]) -> str:
        if not params:
            return "-"
        spec = self._strategy_param_spec(strategy)
        ordered = list(spec.keys())
        for key in params:
            if key not in ordered:
                ordered.append(key)
        parts = [f"{k}={self._format_param_value(params[k])}" for k in ordered if k in params]
        per_line = 4
        lines = [
            "  ".join(parts[i : i + per_line])
            for i in range(0, len(parts), per_line)
        ]
        return "\n".join(lines)

    def _parse_int_value(self, raw: str) -> int:
        try:
            return int(raw)
        except ValueError:
            value = float(raw)
            if not value.is_integer():
                raise ValueError("expected integer") from None
            return int(value)

    def _parse_param_value(self, strategy: str, key: str, raw: str) -> Any:
        spec = self._strategy_param_spec(strategy)
        if spec and key in spec:
            expected = spec[key]
            if expected is int:
                return self._parse_int_value(raw)
            if expected is bool:
                value = _parse_bool_arg(raw)
                if value is None:
                    raise ValueError("expected boolean")
                return bool(value)
            if expected is str:
                return str(raw)
            return float(raw)
        try:
            return self._parse_int_value(raw)
        except Exception:
            try:
                return float(raw)
            except Exception:
                return raw

    def _canonicalize_command(self, token: str) -> str:
        token = token.strip().lower()
        return self.COMMAND_ALIASES.get(token, token)

    def _set_market(self, market: str) -> None:
        try:
            mkt = parse_market(market)
        except ValueError as exc:
            self._write_log(str(exc))
            return

        self.state.market = mkt.value
        if mkt == Market.EQUITY:
            self.state.slippage_bps = 0.0
            self.state.taker_fee_bps = 0.0
        elif mkt == Market.CRYPTO:
            self.state.slippage_bps = 3.0
            self.state.taker_fee_bps = 25.0
        else:
            self.state.slippage_bps = 1.25
            self.state.taker_fee_bps = 3.0

        if mkt == Market.EQUITY:
            self.state.paper_regular_hours_only = True
            self.state.paper_allow_trading_when_closed = False
        else:
            self.state.paper_regular_hours_only = False
            self.state.paper_allow_trading_when_closed = True

        strategy = self._canonicalize_strategy_name(self.state.strategy)
        self._ensure_strategy_params(strategy)
        spec = self._strategy_param_spec(strategy)
        if "slippage_bps" in spec:
            self.state.strategy_params[strategy]["slippage_bps"] = float(self.state.slippage_bps)
        if "taker_fee_bps" in spec:
            self.state.strategy_params[strategy]["taker_fee_bps"] = float(self.state.taker_fee_bps)
        current_symbols = [
            s.strip() for s in str(self.state.symbols or "").split(",") if s.strip()
        ]
        if not current_symbols:
            if strategy in {"basis_carry", "hedge"} and mkt in {Market.CRYPTO, Market.DERIVATIVES}:
                current_symbols = ["BTC/USD", "BTC-PERP"]
            else:
                default_count = (
                    2
                    if strategy in {"nec_x", "nec_pdt", "basis_carry", "hedge", "crypto_ensemble", "crypto_tsm"}
                    else 1
                )
                current_symbols = default_symbols(mkt, count=default_count)
        else:
            if mkt == Market.EQUITY and any("/" in s for s in current_symbols):
                default_count = 2 if strategy in {"nec_x", "nec_pdt"} else 1
                current_symbols = default_symbols(mkt, count=default_count)
            else:
                current_symbols = coerce_symbols_for_market(current_symbols, mkt)

        if strategy in {"nec_x", "nec_pdt", "basis_carry", "hedge"}:
            if len(current_symbols) < 2:
                if strategy in {"basis_carry", "hedge"} and mkt in {Market.CRYPTO, Market.DERIVATIVES}:
                    current_symbols = ["BTC/USD", "BTC-PERP"]
                else:
                    current_symbols = default_symbols(mkt, count=2)
            elif len(current_symbols) > 2:
                current_symbols = current_symbols[:2]

        self.state.symbols = ",".join(current_symbols)
        self._render_settings()

    def _arg_options(self, command: str) -> list[str]:
        cmd = self._canonicalize_command(command)
        if cmd == "/paper":
            return ["start", "stop"]
        if cmd == "/fast":
            return ["5", "10", "20"]
        if cmd == "/slow":
            return ["20", "30", "50", "100"]
        if cmd == "/cash":
            return ["10000", "50000", "100000", "250000"]
        if cmd == "/maxnotional":
            return ["1000", "5000", "10000", "25000", "100000"]
        if cmd == "/slippage":
            return ["0", "0.5", "1.25", "2.0", "5.0"]
        if cmd == "/short":
            return ["true", "false"]
        if cmd == "/debug":
            return ["on", "off"]
        if cmd == "/data":
            return ["sample", "csv", "alpaca", "coinbase"]
        if cmd in {"/feed", "/paperfeed"}:
            return ["iex", "delayed_sip", "sip"]
        if cmd == "/paperlookback":
            return ["50", "100", "200", "500"]
        if cmd == "/paperpoll":
            return ["5", "15", "30", "60"]
        if cmd == "/papermaxnotional":
            return ["1000", "5000", "10000"]
        if cmd in {"/paperclosed", "/paperrth", "/paperdry"}:
            return ["true", "false"]
        if cmd == "/paperlimitbps":
            return ["0", "1", "5", "10", "25"]
        if cmd == "/bar":
            return ["1Min", "5Min", "15Min", "30Min", "60Min", "4H", "6H", "1D"]
        if cmd == "/algorithm":
            return list_strategy_names()
        if cmd == "/tune":
            return ["start", "stop", "apply", "trials", "jobs", "seed", "train", "validate", "test", "step", "drift", "margin"]
        if cmd == "/param":
            strategy = self._canonicalize_strategy_name(self.state.strategy)
            spec = self._strategy_param_spec(strategy)
            return list(spec.keys()) if spec else []
        if cmd == "/preset":
            strategy = self._canonicalize_strategy_name(self.state.strategy)
            names = [p.stem for p in self._strategy_preset_paths(strategy)]
            out: list[str] = ["list", "load"]
            for name in names:
                if name not in out:
                    out.append(name)
            return out
        if cmd == "/timeframe":
            return ["6h", "7d", "30d", "180d", "1y", "clear"]
        if cmd == "/csv":
            return ["data/sample"]
        if cmd == "/symbol":
            return ["SPY", "QQQ"]
        if cmd == "/symbols":
            return ["SPY,QQQ"]
        if cmd in {"/derivative", "/derivatives", "/perp", "/perps", "/future", "/futures"}:
            return default_symbols(Market.DERIVATIVES, count=2)
        return []

    def _compute_suggestions(self, raw: str) -> list[str]:
        raw = raw.rstrip("\n")
        raw_l = raw.lstrip()
        if not raw_l.startswith("/"):
            return []

        ends_with_ws = bool(raw_l) and raw_l[-1].isspace()
        tokens = raw_l.strip().split()
        if not tokens:
            return []

        cmd_token = tokens[0]
        cmd = self._canonicalize_command(cmd_token)
        base_prefix = cmd_token.strip().lower()
        known_base = {self._canonicalize_command(c) for c in self.BASE_COMMANDS}

        # If user has typed more than one arg and then whitespace, we intentionally stop.
        if len(tokens) >= 2 and ends_with_ws:
            return []

        # Completing the command itself (no args yet).
        if len(tokens) == 1 and cmd not in known_base:
            return [c for c in self.BASE_COMMANDS if c.startswith(base_prefix)]

        # Completing the first arg for a known command.
        options = self._arg_options(cmd)
        if not options:
            return []

        arg_prefix = ""
        if len(tokens) >= 2:
            arg_prefix = tokens[1].strip().lower()

        matches = [o for o in options if o.lower().startswith(arg_prefix)]
        return [f"{cmd_token.strip()} {m}" for m in matches]

    def on_key(self, event: events.Key) -> None:
        input_box = self.query_one("#input", Input)
        if not input_box.has_focus:
            return

        if not self._suggestion_matches:
            return

        if event.key in {"up", "down"}:
            delta = -1 if event.key == "up" else 1
            self._suggestion_index = (self._suggestion_index + delta) % len(
                self._suggestion_matches
            )
            self._render_suggestions()
            event.prevent_default()
            event.stop()
            return

        if event.key == "tab":
            selected = self._suggestion_matches[self._suggestion_index]
            input_box.value = selected
            try:
                input_box.cursor_position = len(selected)
            except Exception:
                pass
            self._update_suggestions(selected)
            event.prevent_default()
            event.stop()
            return

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="body"):
            settings = Static(id="settings")
            settings.border_title = "Settings"
            yield settings
            results = Static(id="results")
            results.border_title = "Results"
            yield results
        lower = Vertical(id="lower")
        lower.border_title = "Console"
        with lower:
            yield Log(id="log")
            yield Static(id="suggestions")
            cmd = Input(
                id="input",
                placeholder="/help",
            )
            cmd.border_title = "Command"
            yield cmd

    def on_mount(self) -> None:
        log_widget = self.query_one("#log", Log)
        log_widget.auto_scroll = True
        log_widget.display = False
        suggestions = self.query_one("#suggestions", Static)
        suggestions.display = False
        self.query_one("#input", Input).focus()
        self._load_config()
        self.state.strategy = self._canonicalize_strategy_name(self.state.strategy)
        self._ensure_strategy_params(self.state.strategy)
        self._render_settings()
        self._render_results(None)
        self.set_interval(0.5, self._refresh_live_view)
        self.set_interval(0.2, self._refresh_backtest_view)
        self.set_interval(0.2, self._refresh_tune_view)

    def on_unmount(self) -> None:
        self._save_config()

    def action_help_quit(self) -> None:
        input_box = self.query_one("#input", Input)
        if input_box.value:
            input_box.value = ""
            self._update_suggestions("")
            self._ctrl_c_armed_at = None
            return

        now = time.monotonic()
        if self._ctrl_c_armed_at and now - self._ctrl_c_armed_at < 2.0:
            self.exit()
            return
        self._ctrl_c_armed_at = now
        self._write_log("press ctrl+c again to quit")

    def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        event.input.value = ""
        self._update_suggestions("")
        if not text:
            return
        if not text.startswith("/"):
            self._write_log("commands must start with /")
            return
        self._handle_command(text)

    def on_input_changed(self, event: Input.Changed) -> None:
        self._update_suggestions(event.value)

    def _clear_suggestions(self) -> None:
        suggestions = self.query_one("#suggestions", Static)
        self._suggestion_matches = []
        self._suggestion_index = 0
        self._suggestion_scroll = 0
        suggestions.styles.height = 0
        suggestions.update("")
        suggestions.display = False

    def _render_suggestions(self) -> None:
        suggestions = self.query_one("#suggestions", Static)
        if not self._suggestion_matches:
            self._clear_suggestions()
            return

        matches = self._suggestion_matches
        self._suggestion_index = max(0, min(int(self._suggestion_index), len(matches) - 1))

        max_lines = int(self.SUGGESTION_MAX_LINES)
        if self._suggestion_index < self._suggestion_scroll:
            self._suggestion_scroll = self._suggestion_index
        if self._suggestion_index >= self._suggestion_scroll + max_lines:
            self._suggestion_scroll = self._suggestion_index - max_lines + 1
        self._suggestion_scroll = max(0, min(int(self._suggestion_scroll), max(len(matches) - 1, 0)))

        window = matches[self._suggestion_scroll : self._suggestion_scroll + max_lines]

        lines: list[str] = []
        for idx, cmd in enumerate(window):
            global_idx = self._suggestion_scroll + idx
            prefix = "> " if global_idx == self._suggestion_index else "  "
            lines.append(prefix + cmd)

        suggestions.display = True
        suggestions.styles.height = len(lines) + 2
        suggestions.update("\n".join(lines))

    def _update_suggestions(self, value: str) -> None:
        raw = value
        if not raw.lstrip().startswith("/"):
            self._clear_suggestions()
            return

        matches = self._compute_suggestions(raw)
        if not matches:
            self._clear_suggestions()
            return
        if matches != self._suggestion_matches:
            self._suggestion_index = 0
            self._suggestion_scroll = 0
        self._suggestion_matches = matches
        self._render_suggestions()

    def _handle_command(self, text: str) -> None:
        parts = text.split()
        cmd = self._canonicalize_command(parts[0])
        args = parts[1:]

        if cmd in {"/help", "/?"}:
            self._write_log(
                "commands: /stock, /crypto, /backtest, /paper start|stop, /timeframe <7d|6h|1m|1y|clear>, "
                "/bar <1Min|5Min|15Min|30Min|60Min|4H|6H|1D>, /algorithm <name>, /data <sample|csv|alpaca|coinbase>, "
                "/tune start|stop|apply|trials|jobs|seed|train|validate|test|step|drift|margin, "
                "/param <key> <value>, /params, /preset list|load <name|path>, "
                "/fast <int>, /slow <int>, /cash <usd> (/initialcash <usd>), /maxnotional <usd>, /slippage <bps>, /takerfee <bps>, /short <true|false>, /debug <true|false>, "
                "/feed <iex|delayed_sip|sip>, /paperfeed <iex|delayed_sip|sip>, /csv <path>, "
                "/paperlookback <bars>, /paperpoll <seconds>, /papermaxnotional <usd>, "
                "/paperclosed <true|false>, /paperrth <true|false>, /paperlimitbps <float>, /paperdry <true|false>, "
                "/symbol <SPY>, /symbols <SPY,QQQ>, /start <iso>, /end <iso>, /save [path], /load [path]"
            )
            return

        if cmd == "/crypto":
            self._set_market("crypto")
            self._write_log("market set to crypto")
            return

        if cmd == "/stock":
            self._set_market("equity")
            self._write_log("market set to equity")
            return

        if cmd == "/derivative":
            self._set_market("derivatives")
            self._write_log("market set to derivatives")
            if args:
                coerced = coerce_symbols_for_market([args[0]], Market.DERIVATIVES)
                self.state.symbols = coerced[0] if coerced else ""
                self._render_settings()
            return

        if cmd == "/symbol" and args:
            mkt = parse_market(self.state.market)
            coerced = coerce_symbols_for_market([args[0]], mkt)
            self.state.symbols = coerced[0] if coerced else ""
            self._render_settings()
            return

        if cmd == "/symbols" and args:
            mkt = parse_market(self.state.market)
            raw = " ".join(args)
            parts: list[str] = []
            for chunk in raw.split(","):
                parts.extend([p.strip() for p in chunk.split() if p.strip()])
            coerced = coerce_symbols_for_market(parts, mkt)
            self.state.symbols = ",".join(coerced)
            self._render_settings()
            return

        if cmd == "/data" and args:
            value = args[0].lower()
            if value not in {"sample", "csv", "alpaca", "coinbase"}:
                self._write_log("data source must be sample|csv|alpaca|coinbase")
                return
            self.state.data_source = value
            if self.state.timeframe and self.state.data_source in {"alpaca", "coinbase"}:
                try:
                    delta = _parse_relative_timeframe(self.state.timeframe)
                except Exception:
                    pass
                else:
                    end_dt = now_ny()
                    start_dt = end_dt - delta
                    self.state.start = start_dt.isoformat()
                    self.state.end = end_dt.isoformat()
            self._render_settings()
            return

        if cmd == "/feed" and args:
            value = args[0].lower()
            if value not in {"iex", "delayed_sip", "sip"}:
                self._write_log("feed must be one of: iex | delayed_sip | sip")
                return
            self.state.alpaca_feed = value
            self._render_settings()
            return

        if cmd == "/paperfeed" and args:
            value = args[0].lower()
            if value not in {"iex", "delayed_sip", "sip"}:
                self._write_log("paperfeed must be one of: iex | delayed_sip | sip")
                return
            self.state.paper_feed = value
            self._render_settings()
            return

        if cmd == "/fast" and args:
            try:
                value = int(args[0])
            except ValueError:
                self._write_log("fast must be an integer")
                return
            if value <= 0:
                self._write_log("fast must be > 0")
                return
            active = self._canonicalize_strategy_name(self.state.strategy)
            target = "ema_crossover" if active == "ema_crossover" else "ma_crossover"
            self._ensure_strategy_params(target)
            if active in {"ma_crossover", "ema_crossover"}:
                self.state.fast_window = value
            self.state.strategy_params[target]["fast_window"] = value
            self._render_settings()
            return

        if cmd == "/slow" and args:
            try:
                value = int(args[0])
            except ValueError:
                self._write_log("slow must be an integer")
                return
            if value <= 0:
                self._write_log("slow must be > 0")
                return
            active = self._canonicalize_strategy_name(self.state.strategy)
            target = "ema_crossover" if active == "ema_crossover" else "ma_crossover"
            self._ensure_strategy_params(target)
            if active in {"ma_crossover", "ema_crossover"}:
                self.state.slow_window = value
            self.state.strategy_params[target]["slow_window"] = value
            self._render_settings()
            return

        if cmd == "/cash" and args:
            try:
                value = float(args[0])
            except ValueError:
                self._write_log("cash must be a number (usd)")
                return
            if value <= 0:
                self._write_log("cash must be > 0")
                return
            prev_cash = float(self.state.initial_cash) if self.state.initial_cash else 0.0
            prev_max_notional = float(self.state.max_position_notional_usd)
            ratio = (
                (prev_max_notional / prev_cash)
                if prev_cash > 0 and prev_max_notional > 0
                else (get_default_max_position_notional_usd(mode="backtest") / 100_000.0)
            )
            self.state.initial_cash = value
            self.state.max_position_notional_usd = max(float(value) * float(ratio), 1.0)
            self._render_settings()
            return

        if cmd == "/maxnotional" and args:
            try:
                value = float(args[0])
            except ValueError:
                self._write_log("maxnotional must be a number (usd)")
                return
            if value <= 0:
                self._write_log("maxnotional must be > 0")
                return
            self.state.max_position_notional_usd = value
            self._render_settings()
            return

        if cmd == "/slippage" and args:
            try:
                value = float(args[0])
            except ValueError:
                self._write_log("slippage must be a number (bps)")
                return
            if value < 0:
                self._write_log("slippage must be >= 0")
                return
            self.state.slippage_bps = value
            strategy = self._canonicalize_strategy_name(self.state.strategy)
            spec = self._strategy_param_spec(strategy)
            if "slippage_bps" in spec:
                self._ensure_strategy_params(strategy)
                self.state.strategy_params[strategy]["slippage_bps"] = float(value)
            elif strategy == "nec_x":
                self._write_log(
                    "note: for nec_x, slippage is derived from spread_floor_bps + slip_bps (edit via /param to keep gating consistent)"
                )
            elif strategy == "nec_pdt":
                self._write_log(
                    "note: for nec_pdt, slippage is derived from half_spread_bps + slippage_bps + fee_bps (edit via /param to keep gating consistent)"
                )
            self._render_settings()
            return

        if cmd == "/takerfee" and args:
            try:
                value = float(args[0])
            except ValueError:
                self._write_log("takerfee must be a number (bps)")
                return
            if value < 0:
                self._write_log("takerfee must be >= 0")
                return
            self.state.taker_fee_bps = value

            # Keep strategy-side admission logic aligned when the strategy exposes the param.
            strategy = self._canonicalize_strategy_name(self.state.strategy)
            spec = self._strategy_param_spec(strategy)
            if "taker_fee_bps" in spec:
                self._ensure_strategy_params(strategy)
                self.state.strategy_params[strategy]["taker_fee_bps"] = float(value)
            elif strategy == "nec_pdt":
                self._write_log(
                    "note: nec_pdt models fees via its fee_bps param (edit via /param fee_bps ...); "
                    "this takerfee setting affects the backtest engine only"
                )

            self._render_settings()
            return

        if cmd == "/short" and args:
            value = _parse_bool_arg(args[0])
            if value is None:
                self._write_log("short must be true|false")
                return
            self.state.allow_short = value
            self._render_settings()
            return

        if cmd == "/debug":
            if not args:
                self._write_log(f"debug is {'on' if self.state.debug else 'off'}")
                return
            value = _parse_bool_arg(args[0])
            if value is None:
                self._write_log("debug must be true|false (or on|off)")
                return
            self.state.debug = bool(value)
            self._render_settings()
            self._write_log(f"debug set to {'on' if self.state.debug else 'off'}")
            return

        if cmd == "/paperclosed" and args:
            value = _parse_bool_arg(args[0])
            if value is None:
                self._write_log("paperclosed must be true|false")
                return
            self.state.paper_allow_trading_when_closed = value
            self._render_settings()
            return

        if cmd == "/paperrth" and args:
            value = _parse_bool_arg(args[0])
            if value is None:
                self._write_log("paperrth must be true|false")
                return
            self.state.paper_regular_hours_only = value
            self._render_settings()
            return

        if cmd == "/paperlookback" and args:
            try:
                value = int(args[0])
            except ValueError:
                self._write_log("paperlookback must be an integer (bars)")
                return
            if value <= 0:
                self._write_log("paperlookback must be > 0")
                return
            self.state.paper_lookback_bars = value
            self._render_settings()
            return

        if cmd == "/paperpoll" and args:
            try:
                value = int(args[0])
            except ValueError:
                self._write_log("paperpoll must be an integer (seconds)")
                return
            if value <= 0:
                self._write_log("paperpoll must be > 0")
                return
            self.state.paper_poll_seconds = value
            self._render_settings()
            return

        if cmd == "/papermaxnotional" and args:
            try:
                value = float(args[0])
            except ValueError:
                self._write_log("papermaxnotional must be a number (usd)")
                return
            if value <= 0:
                self._write_log("papermaxnotional must be > 0")
                return
            self.state.paper_max_position_notional_usd = value
            self._render_settings()
            return

        if cmd == "/paperlimitbps" and args:
            try:
                value = float(args[0])
            except ValueError:
                self._write_log("paperlimitbps must be a number (bps)")
                return
            if value < 0:
                self._write_log("paperlimitbps must be >= 0")
                return
            self.state.paper_limit_offset_bps = value
            self._render_settings()
            return

        if cmd == "/paperdry" and args:
            value = _parse_bool_arg(args[0])
            if value is None:
                self._write_log("paperdry must be true|false")
                return
            self.state.paper_dry_run = value
            self._render_settings()
            return

        if cmd == "/csv" and args:
            self.state.csv_path = " ".join(args)
            self._render_settings()
            return

        if cmd == "/timeframe" and args:
            if args[0].lower() == "clear":
                self.state.timeframe = None
                self._render_settings()
                return

            try:
                delta = _parse_relative_timeframe(args[0])
            except Exception:
                self._write_log("unsupported timeframe spec (examples: 7d, 6h, 1m, 1y)")
                return
            self.state.timeframe = args[0]
            if self.state.data_source in {"alpaca", "coinbase"}:
                end_dt = now_ny()
                start_dt = end_dt - delta
                self.state.start = start_dt.isoformat()
                self.state.end = end_dt.isoformat()
            self._render_settings()
            return

        if cmd in {"/bar", "/bars"} and args:
            self.state.bar_timeframe = args[0]
            self._render_settings()
            return

        if cmd == "/start" and args:
            self.state.start = " ".join(args)
            self._render_settings()
            return

        if cmd == "/end" and args:
            self.state.end = " ".join(args)
            self._render_settings()
            return

        if cmd in {"/algorithm", "/strategy", "/aglorithm", "/algorithim"} and args:
            strategy = self._canonicalize_strategy_name(args[0])
            self.state.strategy = strategy
            self._ensure_strategy_params(strategy)
            mkt = parse_market(self.state.market)
            if strategy in {"basis_carry", "hedge"}:
                current_symbols = [
                    s.strip().upper() for s in self.state.symbols.split(",") if s.strip()
                ]
                if len(current_symbols) < 2:
                    if mkt in {Market.CRYPTO, Market.DERIVATIVES}:
                        self.state.symbols = "BTC/USD,BTC-PERP"
                    else:
                        self.state.symbols = ",".join(default_symbols(mkt, count=2))
                    self._write_log(
                        f"{strategy} is pair-based (spot + perp). "
                        f"Defaulting to {self.state.symbols}. Set via /symbols BASE/USD,BASE-PERP"
                    )
                elif len(current_symbols) > 2:
                    self.state.symbols = ",".join(current_symbols[:2])
                    self._write_log(
                        f"{strategy} is pair-based (2 symbols). Using first two: {self.state.symbols}"
                    )
            if strategy in {"nec_x", "nec_pdt", "orb_trend"}:
                current_symbols = [
                    s.strip().upper() for s in self.state.symbols.split(",") if s.strip()
                ]
                if strategy in {"nec_x", "nec_pdt"}:
                    if len(current_symbols) < 2:
                        self.state.symbols = ",".join(default_symbols(mkt, count=2))
                        self._write_log(
                            f"{strategy} is pair-based (2 symbols). Defaulting to {self.state.symbols}. "
                            "Set via /symbols AAPL,TSLA"
                        )
                    elif len(current_symbols) > 2:
                        self.state.symbols = ",".join(current_symbols[:2])
                        self._write_log(
                            f"{strategy} is pair-based (2 symbols). Using first two: {self.state.symbols}"
                        )
                else:
                    if (
                        not current_symbols
                        or current_symbols == ["SPY"]
                        or current_symbols == ["BTC/USD"]
                    ):
                        self.state.symbols = ",".join(default_symbols(mkt, count=2))
                self.state.bar_timeframe = "5Min"
                params = self.state.strategy_params.get(strategy, {})
                if strategy == "nec_x":
                    self.state.slippage_bps = float(params.get("spread_floor_bps", 0.0)) + float(
                        params.get("slip_bps", 0.0)
                    )
                elif strategy == "nec_pdt":
                    self.state.slippage_bps = float(params.get("half_spread_bps", 0.0)) + float(
                        params.get("slippage_bps", 0.0)
                    ) + float(params.get("fee_bps", 0.0))
                else:
                    self.state.slippage_bps = float(params.get("slippage_bps", 1.25))
            elif strategy == "spy_open_close":
                self.state.symbols = ",".join(default_symbols(mkt, count=1))
            self._render_settings()
            return

        if cmd in {"/param", "/params"}:
            strategy = self._canonicalize_strategy_name(self.state.strategy)
            self._ensure_strategy_params(strategy)
            params = self.state.strategy_params.get(strategy, {})
            if cmd == "/params" or not args:
                rendered = self._format_strategy_params(strategy, params)
                self._write_log(f"{strategy} params: {rendered}")
                return
            if len(args) != 2:
                self._write_log("param usage: /param <key> <value>")
                return
            key = self._normalize_param_key(strategy, args[0])
            spec = self._strategy_param_spec(strategy)
            if strategy in STRATEGY_PARAM_SPECS and not spec:
                self._write_log(f"{strategy} has no editable params")
                return
            if spec and key not in spec:
                valid = ", ".join(spec.keys())
                self._write_log(f"unknown param for {strategy}: {args[0]} (valid: {valid})")
                return
            try:
                value = self._parse_param_value(strategy, key, args[1])
            except ValueError:
                self._write_log(f"invalid value for {key}")
                return
            params[key] = value
            if strategy in {"ma_crossover", "ema_crossover"}:
                if key == "fast_window":
                    self.state.fast_window = int(value)
                if key == "slow_window":
                    self.state.slow_window = int(value)
            if strategy == "orb_trend" and key == "slippage_bps":
                self.state.slippage_bps = float(value)
            if strategy == "nec_x" and key in {"spread_floor_bps", "slip_bps"}:
                self.state.slippage_bps = float(params.get("spread_floor_bps", 0.0)) + float(
                    params.get("slip_bps", 0.0)
                )
            if strategy == "nec_pdt" and key in {"half_spread_bps", "slippage_bps", "fee_bps"}:
                self.state.slippage_bps = float(params.get("half_spread_bps", 0.0)) + float(
                    params.get("slippage_bps", 0.0)
                ) + float(params.get("fee_bps", 0.0))
            self._render_settings()
            return

        if cmd == "/tune":
            action = args[0].lower() if args else "start"
            if action in {"start"}:
                self._start_tune()
                return
            if action == "stop":
                self._stop_tune()
                return
            if action == "apply":
                self._apply_tuned_params()
                return

            if len(args) != 2:
                self._write_log(
                    "tune usage: /tune start|stop|apply OR /tune trials|jobs|seed|train|validate|test|step|drift|margin <value>"
                )
                return

            key = action
            value_raw = args[1]
            if key == "trials":
                try:
                    value = int(value_raw)
                except ValueError:
                    self._write_log("tune trials must be an integer")
                    return
                if value <= 0:
                    self._write_log("tune trials must be > 0")
                    return
                self.state.tune_trials_per_segment = value
            elif key == "jobs":
                if value_raw.strip().lower() == "auto":
                    value = 0
                else:
                    try:
                        value = int(value_raw)
                    except ValueError:
                        self._write_log("tune jobs must be an integer (or 'auto')")
                        return
                if int(value) < 0:
                    self._write_log("tune jobs must be >= 0 (0=auto)")
                    return
                self.state.tune_jobs = int(value)
            elif key == "seed":
                try:
                    self.state.tune_seed = int(value_raw)
                except ValueError:
                    self._write_log("tune seed must be an integer")
                    return
            elif key in {"train", "validate", "test", "step"}:
                try:
                    _ = parse_duration_spec(value_raw)
                except Exception:
                    self._write_log("tune window must be like 30d/6h/1y")
                    return
                if key == "train":
                    self.state.tune_train = value_raw
                elif key == "validate":
                    self.state.tune_validate = value_raw
                elif key == "test":
                    self.state.tune_test = value_raw
                else:
                    self.state.tune_step = value_raw
            elif key == "drift":
                try:
                    value = float(value_raw)
                except ValueError:
                    self._write_log("tune drift must be a number")
                    return
                if value < 0:
                    self._write_log("tune drift must be >= 0")
                    return
                self.state.tune_drift_frac = float(value)
            elif key == "margin":
                try:
                    value = float(value_raw)
                except ValueError:
                    self._write_log("tune margin must be a number")
                    return
                self.state.tune_improvement_margin = float(value)
            else:
                self._write_log(
                    "unknown tune setting (use: trials, jobs, seed, train, validate, test, step, drift, margin)"
                )
                return

            self._render_settings()
            return

        if cmd == "/preset":
            if not args or args[0].lower() == "list":
                strategy = self._canonicalize_strategy_name(self.state.strategy)
                presets = self._strategy_preset_paths(strategy)
                if not presets:
                    self._write_log(
                        f"no presets found for {strategy} (expected json under {self._strategy_params_dir()})"
                    )
                    return
                names = ", ".join([p.stem for p in presets])
                self._write_log(f"presets for {strategy}: {names}")
                return

            action = args[0].lower()
            query = " ".join(args[1:]) if action == "load" else " ".join(args)
            query = query.strip()
            if not query:
                self._write_log("preset usage: /preset list | /preset load <name|path> | /preset <name>")
                return

            strategy = self._canonicalize_strategy_name(self.state.strategy)
            preset_dir = self._strategy_params_dir()

            selected: Optional[Path] = None
            raw_query_path = Path(query)
            if raw_query_path.exists():
                selected = raw_query_path
            else:
                rel = preset_dir / query
                if rel.exists():
                    selected = rel
                elif not query.lower().endswith(".json"):
                    rel_json = preset_dir / f"{query}.json"
                    if rel_json.exists():
                        selected = rel_json

            if selected is None:
                needle = query.lower()
                candidates = self._strategy_preset_paths(strategy)
                matches = [p for p in candidates if needle in p.stem.lower()]
                if len(matches) == 1:
                    selected = matches[0]
                elif not matches:
                    self._write_log(f"no matching presets for {strategy}: {query}")
                    return
                else:
                    names = ", ".join([p.stem for p in matches[:12]])
                    suffix = "" if len(matches) <= 12 else f" (+{len(matches) - 12} more)"
                    self._write_log(f"multiple presets match {query!r}: {names}{suffix}")
                    return

            try:
                raw = json.loads(selected.read_text())
            except Exception as exc:
                self._write_log(f"failed to read preset {selected}: {exc}")
                return

            params = self._extract_params_for_strategy(raw, strategy)
            if params is None:
                keys = self._detect_strategy_keys_in_params_file(raw)
                if len(keys) == 1:
                    strategy = keys[0]
                    self.state.strategy = strategy
                    self._ensure_strategy_params(strategy)
                    params = self._extract_params_for_strategy(raw, strategy)

            if params is None:
                self._write_log(
                    f"preset {selected} does not include params for {self._canonicalize_strategy_name(self.state.strategy)}"
                )
                return

            self.state.strategy_params[strategy] = dict(params)
            profile = self._extract_preset_profile(raw)

            # If a crypto preset is loaded, switch market to crypto for convenience.
            if strategy in {
                "crypto_tsm",
                "crypto_ensemble",
                "crypto_rotation",
                "crypto_momentum",
                "crypto_meta",
            }:
                self._set_market("crypto")
            else:
                self._render_settings()

            if profile is not None:
                self._apply_preset_profile(profile, strategy=strategy)
            self._write_log(f"loaded preset for {strategy}: {selected}")
            return

        if cmd == "/backtest":
            self._run_backtest()
            return

        if cmd == "/paper":
            action = args[0].lower() if args else "start"
            if action == "start":
                self._start_paper()
            elif action == "stop":
                self._stop_paper()
            else:
                self._write_log("paper command must be /paper start|stop")
            return

        if cmd == "/save":
            path = Path(" ".join(args)) if args else self._config_path
            self._config_path = path
            self._autosave_enabled = True
            self._config_load_failed = False
            try:
                path.write_text(json.dumps(self.state.to_dict(), indent=2))
            except Exception as exc:
                self._write_log(f"failed to save config ({path}): {exc}")
                return
            self._write_log(f"saved config: {path}")
            return

        if cmd == "/load":
            path = Path(" ".join(args)) if args else self._config_path
            self._config_path = path
            if not path.exists():
                self._write_log(f"config not found: {path}")
                return
            try:
                raw = json.loads(path.read_text())
                self.state = TuiState.from_dict(raw)
            except Exception as exc:
                self._write_log(f"failed to load config ({path}): {exc}")
                return
            self.state.strategy = self._canonicalize_strategy_name(self.state.strategy)
            self._ensure_strategy_params(self.state.strategy)
            self._autosave_enabled = True
            self._config_load_failed = False
            self._render_settings()
            self._write_log(f"loaded config: {path}")
            return

        self._write_log(f"unknown command: {cmd}")

    def _write_log(self, message: str) -> None:
        log_widget = self.query_one("#log", Log)
        if not log_widget.display:
            log_widget.display = True
        log_widget.write_line(message)

    def _render_settings(self) -> None:
        try:
            self.state.market = parse_market(self.state.market).value
        except Exception:
            self.state.market = "equity"
        self.state.strategy = self._canonicalize_strategy_name(self.state.strategy)
        self._ensure_strategy_params(self.state.strategy)
        table = Table(show_header=False, box=box.SIMPLE, expand=True, pad_edge=False)
        table.add_column("k", style="bold", no_wrap=True)
        table.add_column("v")

        def _section(label: str) -> None:
            table.add_row(Text(label, style="bold cyan"), "")

        _section("Data")
        table.add_row("market", self.state.market)
        table.add_row("symbols", self.state.symbols)
        table.add_row("data_source", self.state.data_source)
        table.add_row("alpaca_feed", self.state.alpaca_feed)
        table.add_row("csv_path", self.state.csv_path or "-")
        table.add_row("timeframe", self.state.timeframe or "-")
        table.add_row("bar_timeframe", self.state.bar_timeframe)
        table.add_row("start", self.state.start or "-")
        table.add_row("end", self.state.end or "-")

        _section("Strategy")
        table.add_row(
            "strategy",
            (
                f"{self.state.strategy} (fast={self.state.fast_window} slow={self.state.slow_window})"
                if self.state.strategy in {"ma_crossover", "ema_crossover"}
                else self.state.strategy
            ),
        )
        params = self.state.strategy_params.get(self.state.strategy, {})
        if params:
            table.add_row("params", self._format_strategy_params(self.state.strategy, params))
        table.add_row("slippage_bps", f"{self.state.slippage_bps:.2f}")
        table.add_row("taker_fee_bps", f"{self.state.taker_fee_bps:.2f}")
        table.add_row(
            "allow_short",
            Text("true", style="green") if self.state.allow_short else Text("false", style="red"),
        )

        _section("Sizing")
        table.add_row("initial_cash", f"{self.state.initial_cash:.2f}")
        table.add_row("max_notional", f"{self.state.max_position_notional_usd:.2f}")

        _section("Debug")
        table.add_row(
            "debug",
            Text("true", style="green") if self.state.debug else Text("false", style="red"),
        )

        _section("Paper")
        table.add_row(
            "paper_running",
            Text("true", style="green") if (self._paper_thread is not None) else Text("false", style="red"),
        )
        table.add_row("paper_lookback", str(self.state.paper_lookback_bars))
        table.add_row("paper_poll_s", str(self.state.paper_poll_seconds))
        table.add_row("paper_max_notional", f"{self.state.paper_max_position_notional_usd:.2f}")
        table.add_row("paper_feed", self.state.paper_feed)
        table.add_row("paper_rth_only", str(self.state.paper_regular_hours_only))
        table.add_row("paper_when_closed", str(self.state.paper_allow_trading_when_closed))
        table.add_row("paper_limit_bps", f"{self.state.paper_limit_offset_bps:.2f}")
        table.add_row("paper_dry_run", str(self.state.paper_dry_run))

        _section("Tune")
        table.add_row(
            "tune_running",
            Text("true", style="green") if (self._tune_thread is not None) else Text("false", style="red"),
        )
        table.add_row("tune_trials", str(self.state.tune_trials_per_segment))
        table.add_row(
            "tune_jobs",
            "auto" if int(self.state.tune_jobs) == 0 else str(int(self.state.tune_jobs)),
        )
        table.add_row("tune_seed", str(self.state.tune_seed))
        table.add_row(
            "tune_windows",
            f"train={self.state.tune_train} val={self.state.tune_validate} test={self.state.tune_test} step={self.state.tune_step}",
        )
        table.add_row("tune_drift", f"{float(self.state.tune_drift_frac):.2f}")
        table.add_row("tune_margin", f"{float(self.state.tune_improvement_margin):.4g}")
        table.add_row("tune_last_run", self.state.tune_last_run_dir or "-")
        best_for_strategy = self.state.tune_best_params.get(self.state.strategy)
        table.add_row(
            "tune_best",
            Text("available", style="green") if best_for_strategy else Text("none", style="red"),
        )

        _section("Config")
        table.add_row("config", str(self._config_path))

        self.query_one("#settings", Static).update(table)
        self._save_config()

    def _render_results(self, summary: Optional[Table]) -> None:
        widget = self.query_one("#results", Static)
        if summary is None:
            widget.border_title = "Results"
            table = Table(show_header=False, box=box.SIMPLE)
            table.add_column("k", style="bold")
            table.add_column("v")
            table.add_row("status", "no backtest yet")
            table.add_row("hint", "run /backtest to generate a summary")
            widget.update(Panel(table, title="Welcome", border_style="cyan", box=box.ROUNDED))
            return
        widget.update(summary)

    def _tail_last_jsonl(self, path: Path) -> Optional[dict]:
        if not path.exists():
            return None
        try:
            with path.open("rb") as f:
                f.seek(0, 2)
                size = f.tell()
                if size <= 0:
                    return None
                read_size = min(size, 131072)
                f.seek(-read_size, 2)
                chunk = f.read().decode("utf-8", errors="ignore")
        except OSError:
            chunk = path.read_text(errors="ignore")
        lines = [ln for ln in chunk.splitlines() if ln.strip()]
        if not lines:
            return None
        try:
            return json.loads(lines[-1])
        except json.JSONDecodeError:
            return None

    def _tail_jsonl_items(self, path: Path, *, max_items: int) -> list[dict]:
        max_items = max(1, int(max_items))
        if not path.exists():
            return []
        try:
            with path.open("rb") as f:
                f.seek(0, 2)
                size = f.tell()
                if size <= 0:
                    return []
                read_size = min(size, 262144)
                f.seek(-read_size, 2)
                chunk = f.read().decode("utf-8", errors="ignore")
        except OSError:
            chunk = path.read_text(errors="ignore")

        out: list[dict] = []
        for ln in [ln for ln in chunk.splitlines() if ln.strip()][-max_items:]:
            try:
                item = json.loads(ln)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                out.append(item)
        return out

    def _tail_equity_curve(self, path: Path, *, max_rows: int) -> list[dict[str, str]]:
        max_rows = max(1, int(max_rows))
        if not path.exists():
            return []
        try:
            with path.open("rb") as f:
                f.seek(0, 2)
                size = f.tell()
                if size <= 0:
                    return []
                read_size = min(size, 262144)
                f.seek(-read_size, 2)
                chunk = f.read().decode("utf-8", errors="ignore")
        except OSError:
            chunk = path.read_text(errors="ignore")

        lines = [ln for ln in chunk.splitlines() if ln.strip()]
        rows: list[dict[str, str]] = []
        reader = csv.reader(lines)
        for cols in reader:
            if not cols:
                continue
            if cols[0].strip().lower() == "timestamp":
                continue
            if len(cols) < 2:
                continue
            row: dict[str, str] = {
                "timestamp": str(cols[0]).strip(),
                "equity": str(cols[1]).strip(),
            }
            if len(cols) >= 3:
                row["day_return"] = str(cols[2]).strip()
            rows.append(row)
        return rows[-max_rows:]

    def _render_paper_live(self, decision: dict) -> None:
        results = self.query_one("#results", Static)
        results.border_title = "Paper (live)"

        now = pd.Timestamp.now(tz=NY_TZ)
        reason = str(decision.get("reason") or "-")
        decision_ts_raw = str(decision.get("timestamp", "")) or ""
        decision_ts: Optional[pd.Timestamp]
        try:
            decision_ts = pd.Timestamp(decision_ts_raw)
            if decision_ts.tzinfo is None:
                decision_ts = decision_ts.tz_localize(NY_TZ)
            else:
                decision_ts = decision_ts.tz_convert(NY_TZ)
        except Exception:
            decision_ts = None

        targets = decision.get("targets", {}) or {}
        positions = decision.get("positions", {}) or {}
        debug = decision.get("debug", {}) or {}

        try:
            tf = parse_bar_timeframe(self.state.bar_timeframe)
            tf_minutes = max(int(tf.minutes), 1)
        except Exception:
            tf_minutes = 1

        bar_open = now.floor(f"{tf_minutes}min")
        next_open = bar_open + pd.Timedelta(minutes=tf_minutes)
        to_next_s = max(float((next_open - now).total_seconds()), 0.0)
        since_decision_s = (
            max(float((now - decision_ts).total_seconds()), 0.0) if decision_ts is not None else None
        )

        mkt = parse_market(self.state.market)
        if mkt == Market.CRYPTO:
            bars_label = "Alpaca crypto (paper_feed ignored)"
        else:
            bars_label = f"Alpaca stocks (feed={self.state.paper_feed})"

        reason_style = "cyan"
        if reason in {"enter", "hold"}:
            reason_style = "green"
        elif reason in {"no_trade", "outside_rth", "entry_cutoff", "long_only_abstain"}:
            reason_style = "yellow"
        elif reason in {"kill_switch", "daily_loss_limit", "risk_disabled", "forced_flat"}:
            reason_style = "red"

        equity_rows: list[dict[str, str]] = []
        if self._paper_run_dir:
            equity_rows = self._tail_equity_curve(self._paper_run_dir / "equity_curve.csv", max_rows=120)
        equity_vals: list[float] = []
        last_equity: Optional[float] = None
        last_day_return: Optional[float] = None
        for r in equity_rows:
            try:
                equity_vals.append(float(r.get("equity") or 0.0))
            except Exception:
                continue
        if equity_vals:
            last_equity = float(equity_vals[-1])
        if equity_rows:
            try:
                last_day_return = float(equity_rows[-1].get("day_return") or 0.0)
            except Exception:
                last_day_return = None

        equity_fallback = float(decision.get("equity") or 0.0) if "equity" in decision else None
        cash_fallback = float(decision.get("cash") or 0.0) if "cash" in decision else None
        equity_display = last_equity if last_equity is not None else equity_fallback
        cash_display = cash_fallback

        day_style = (
            "green"
            if (last_day_return is not None and last_day_return >= 0)
            else "red"
            if (last_day_return is not None and last_day_return < 0)
            else "cyan"
        )
        eq_style = (
            "green"
            if len(equity_vals) >= 2 and equity_vals[-1] >= equity_vals[-2]
            else "red"
            if len(equity_vals) >= 2
            else "cyan"
        )

        kpis = Table.grid(expand=True)
        kpis.add_column(ratio=1)
        kpis.add_column(ratio=1)
        kpis.add_column(ratio=1)
        kpis.add_row(
            Panel(
                Align.center(
                    Text(
                        f"${equity_display:,.2f}" if equity_display is not None else "-",
                        style=f"bold {eq_style}",
                    )
                ),
                title="Equity",
                border_style=eq_style,
                box=box.ROUNDED,
                padding=(1, 2),
            ),
            Panel(
                Align.center(
                    Text(
                        f"{last_day_return:+.2%}" if last_day_return is not None else "-",
                        style=f"bold {day_style}",
                    )
                ),
                title="Day Return",
                border_style=day_style,
                box=box.ROUNDED,
                padding=(1, 2),
            ),
            Panel(
                Align.center(
                    Text(
                        f"{to_next_s:,.1f}s",
                        style="bold cyan",
                    )
                ),
                title="Next Bar",
                border_style="cyan",
                box=box.ROUNDED,
                padding=(1, 2),
            ),
        )

        decision_banner = Table.grid(expand=True)
        decision_banner.add_column(ratio=3)
        decision_banner.add_column(ratio=5)
        reason_display = reason.replace("_", " ").upper()
        decision_banner.add_row(
            Text(f"Reason: {reason_display}", style=f"bold {reason_style}"),
            Text(f"Bars: {bars_label}", style="dim"),
        )
        if decision_ts is not None:
            decision_banner.add_row(
                Text(
                    f"Decision: {decision_ts.strftime('%Y-%m-%d %H:%M:%S %Z')}",
                    style="dim",
                ),
                Text(
                    f"Age: {since_decision_s:,.1f}s" if since_decision_s is not None else "",
                    style="dim",
                ),
            )

        no_trade_breakdown: Optional[Table] = None
        cand_rows: list[tuple[str, str, str]] = []
        params = self.state.strategy_params.get(self.state.strategy, {}) or {}
        for key, raw in debug.items():
            if not (isinstance(key, str) and key.startswith("cand_") and isinstance(raw, dict)):
                continue
            sym = str(raw.get("symbol") or key[len("cand_") :]).strip().upper()
            tag = str(raw.get("reason_tag") or "-")

            detail = ""
            if tag == "gate_er":
                try:
                    er = float(raw.get("er") or 0.0)
                except Exception:
                    er = 0.0
                er_min = params.get("er_min")
                if er_min is not None:
                    try:
                        detail = f"er={er:.2f} < {float(er_min):.2f}"
                    except Exception:
                        detail = f"er={er:.2f}"
                else:
                    detail = f"er={er:.2f}"
            elif tag == "no_breakout":
                parts: list[str] = []
                for k in ("close", "vwap", "th_up", "th_dn"):
                    v = raw.get(k)
                    if v is None:
                        continue
                    try:
                        parts.append(f"{k}={float(v):.2f}")
                    except Exception:
                        continue
                detail = "  ".join(parts)
            elif tag == "net_edge_not_positive":
                v = raw.get("net_edge_bps")
                if v is not None:
                    try:
                        detail = f"net_edge_bps={float(v):+.2f}"
                    except Exception:
                        detail = ""
            elif tag == "orb_not_ready":
                orb_end = raw.get("orb_end")
                if orb_end:
                    detail = f"orb_end={orb_end}"

            cand_rows.append((sym, tag, detail))

        if reason in {"no_trade", "long_only_abstain"} and cand_rows:
            cand_rows.sort(key=lambda r: r[0])
            breakdown = Table(box=box.SIMPLE, show_header=True, header_style="bold")
            breakdown.add_column("symbol", style="bold")
            breakdown.add_column("tag")
            breakdown.add_column("detail")
            for sym, tag, detail in cand_rows:
                style = "cyan"
                if tag in {"candidate_ok"}:
                    style = "green"
                elif tag in {"gate_er", "no_breakout", "confirm_wait", "orb_not_ready"}:
                    style = "yellow"
                elif tag in {"insufficient_bars", "too_few_today"}:
                    style = "red"
                breakdown.add_row(sym, Text(tag, style=style), Text(detail or "-", style="dim"))
            no_trade_breakdown = breakdown

        last_orders = []
        last_fills = []
        if self._paper_run_dir:
            last_orders = self._tail_jsonl_items(self._paper_run_dir / "orders.jsonl", max_items=6)
            last_fills = self._tail_jsonl_items(self._paper_run_dir / "fills.jsonl", max_items=6)
        last_order = last_orders[-1] if last_orders else None
        last_fill = last_fills[-1] if last_fills else None

        meta = Table(show_header=False, box=box.SIMPLE)
        meta.add_column("k", style="bold")
        meta.add_column("v")
        meta.add_row("market", mkt.value)
        if self._paper_run_dir is not None:
            meta.add_row("run_dir", str(self._paper_run_dir))
        meta.add_row("strategy", self.state.strategy)
        meta.add_row("symbols", self.state.symbols)
        meta.add_row("bar_timeframe", self.state.bar_timeframe)
        if decision_ts is not None:
            meta.add_row("decision_ts", decision_ts.isoformat())
        if since_decision_s is not None:
            meta.add_row("age", f"{since_decision_s:,.1f}s")
        meta.add_row("next_open", next_open.isoformat())
        if cash_display is not None:
            meta.add_row("cash", f"${float(cash_display):,.2f}")
        if "chosen" in debug:
            meta.add_row("chosen", str(debug.get("chosen")))
        if "chosen_netEdge_bps" in debug:
            meta.add_row("netEdge_bps", f"{float(debug.get('chosen_netEdge_bps') or 0.0):+.2f}")
        if last_order:
            meta.add_row(
                "last_order",
                f"{last_order.get('symbol')} {last_order.get('side')} qty={last_order.get('qty')} id={last_order.get('order_id')}",
            )
        if last_fill:
            meta.add_row(
                "last_fill",
                f"{last_fill.get('symbol')} {last_fill.get('side')} qty={last_fill.get('filled_qty')} px={last_fill.get('filled_avg_price')} status={last_fill.get('status')}",
            )

        pos_table = Table(box=box.SIMPLE, show_header=True, header_style="bold")
        pos_table.add_column("symbol", style="bold")
        pos_table.add_column("target", justify="right")
        pos_table.add_column("pos_qty", justify="right")

        symbols = sorted(set(list(targets.keys()) + list(positions.keys())))
        if not symbols:
            pos_table.add_row("-", "0.00", "0.0000")
        else:
            for sym in symbols:
                tgt = float(targets.get(sym, 0.0) or 0.0)
                qty = float(positions.get(sym, 0.0) or 0.0)
                s = "green" if qty > 0 else "red" if qty < 0 else "cyan"
                pos_table.add_row(sym, Text(f"{tgt:+.2f}", style=s), Text(f"{qty:+.4f}", style=s))

        tape = Table(box=box.SIMPLE, show_header=True, header_style="bold")
        tape.add_column("ts", no_wrap=True)
        tape.add_column("type", style="bold")
        tape.add_column("symbol", style="bold")
        tape.add_column("side")
        tape.add_column("qty", justify="right")
        tape.add_column("px/status", justify="right")

        events: list[tuple[pd.Timestamp, dict]] = []
        for o in last_orders:
            try:
                t = pd.Timestamp(str(o.get("timestamp") or ""))
            except Exception:
                continue
            events.append((t, {"type": "order", **o}))
        for f in last_fills:
            try:
                t = pd.Timestamp(str(f.get("timestamp") or ""))
            except Exception:
                continue
            events.append((t, {"type": "fill", **f}))
        events.sort(key=lambda x: x[0])
        events = events[-12:]

        if not events:
            tape.add_row("-", "-", "-", "-", "-", "-")
        else:
            for t, e in events:
                kind = str(e.get("type"))
                sym = str(e.get("symbol") or "-")
                side = str(e.get("side") or "-")
                if kind == "fill":
                    qty = str(e.get("filled_qty") or "-")
                    px = str(e.get("filled_avg_price") or e.get("status") or "-")
                    kind_style = "green"
                else:
                    qty = str(e.get("qty") or "-")
                    px = str(e.get("order_id") or ("dry_run" if e.get("dry_run") else "-"))
                    kind_style = "cyan"
                tape.add_row(
                    t.tz_convert(NY_TZ).strftime("%H:%M:%S") if t.tzinfo else str(t),
                    Text(kind, style=f"bold {kind_style}"),
                    sym,
                    side,
                    qty,
                    px,
                )

        spark = self._sparkline(equity_vals, width=80)
        spark_style = eq_style
        spark_panel = Panel(
            Align.center(Text(spark or "-", style=spark_style)),
            title="Equity (recent)",
            border_style=spark_style,
            box=box.ROUNDED,
            padding=(1, 2),
        )

        decision_group_items: list[object] = [decision_banner]
        if no_trade_breakdown is not None:
            decision_group_items.extend(["", no_trade_breakdown])
        decision_group_items.extend(["", meta])

        results.update(
            Group(
                kpis,
                "",
                Panel(Group(*decision_group_items), title="Decision", border_style="cyan", box=box.ROUNDED),
                "",
                Panel(pos_table, title="Targets / Positions", border_style="magenta", box=box.ROUNDED),
                "",
                Panel(tape, title="Tape (recent orders/fills)", border_style="yellow", box=box.ROUNDED),
                "",
                spark_panel,
            )
        )

    def _refresh_live_view(self) -> None:
        if self._backtest_thread is not None:
            return
        if self._paper_run_dir is None:
            return
        decision = self._tail_last_jsonl(self._paper_run_dir / "decisions.jsonl")
        if not decision:
            if self._paper_thread is not None:
                results = self.query_one("#results", Static)
                results.border_title = "Paper (live)"
                body = Table(show_header=False, box=box.SIMPLE)
                body.add_column("k", style="bold")
                body.add_column("v")
                body.add_row("status", "waiting for first decision…")

                try:
                    tf = parse_bar_timeframe(self.state.bar_timeframe)
                    tf_minutes = max(int(tf.minutes), 1)
                except Exception:
                    tf_minutes = 1

                now = pd.Timestamp.now(tz=NY_TZ)
                bar_open = now.floor(f"{tf_minutes}min")
                next_open = bar_open + pd.Timedelta(minutes=tf_minutes)
                to_next_s = max(float((next_open - now).total_seconds()), 0.0)

                body.add_row("now", now.strftime("%Y-%m-%d %H:%M:%S %Z"))
                body.add_row("next_bar", next_open.strftime("%Y-%m-%d %H:%M:%S %Z"))
                body.add_row("in", f"{to_next_s:.0f}s")
                body.add_row("bar_timeframe", str(self.state.bar_timeframe))
                body.add_row("run_dir", str(self._paper_run_dir))
                results.update(Panel(body, title="Paper", border_style="cyan", box=box.ROUNDED))
            return
        ts = str(decision.get("timestamp", ""))
        if self._paper_thread is None and ts and ts == self._last_live_decision_ts:
            return
        self._last_live_decision_ts = ts
        self._render_paper_live(decision)

    def _progress_bar(self, pct: float, *, width: int = 26) -> str:
        pct = max(0.0, min(float(pct), 1.0))
        width = max(int(width), 8)
        filled = int(round(pct * width))
        filled = max(0, min(filled, width))
        return "█" * filled + "░" * (width - filled)

    def _sparkline(self, values: list[float], *, width: int = 72) -> str:
        blocks = "▁▂▃▄▅▆▇█"
        if not values:
            return ""
        if width <= 0:
            width = 1

        step = max(1, len(values) // width)
        sampled = [float(v) for v in values[::step]]
        if len(sampled) > width:
            sampled = sampled[-width:]

        # Guard against NaN/inf, which can happen when plotting sentinel scores.
        has_finite = any(math.isfinite(v) for v in sampled)
        if not has_finite:
            return ""
        last_finite: Optional[float] = None
        for idx, v in enumerate(sampled):
            if math.isfinite(v):
                last_finite = v
            elif last_finite is not None:
                sampled[idx] = last_finite
        try:
            first_finite = next(v for v in sampled if math.isfinite(v))
        except StopIteration:
            return ""
        for idx, v in enumerate(sampled):
            if not math.isfinite(v):
                sampled[idx] = first_finite

        lo = min(sampled)
        hi = max(sampled)
        if not (hi > lo):
            return blocks[0] * len(sampled)
        span = hi - lo
        out: list[str] = []
        for v in sampled:
            idx = int((float(v) - float(lo)) / float(span) * (len(blocks) - 1))
            idx = max(0, min(idx, len(blocks) - 1))
            out.append(blocks[idx])
        return "".join(out)

    def _render_backtest_live(self) -> None:
        results = self.query_one("#results", Static)
        results.border_title = "Backtest (running)"

        status = self._backtest_status or "running…"
        run_dir = self._backtest_run_dir
        started_at = self._backtest_started_at or time.monotonic()
        elapsed_s = max(time.monotonic() - started_at, 0.0)

        progress = self._backtest_progress
        if progress is None:
            body = Table(show_header=False, box=box.SIMPLE)
            body.add_column("k", style="bold")
            body.add_column("v")
            body.add_row("status", status)
            if run_dir is not None:
                body.add_row("run_dir", str(run_dir))
            body.add_row("elapsed", f"{elapsed_s:.1f}s")
            results.update(Panel(body, title="Backtest", border_style="cyan", box=box.ROUNDED))
            return

        pct = float(progress.i) / float(progress.n) if progress.n > 0 else 0.0
        bar = self._progress_bar(pct, width=28)

        rate = float(progress.i) / elapsed_s if elapsed_s > 0 else 0.0
        eta_s = float(progress.n - progress.i) / rate if rate > 0 else 0.0
        eta_text = "-" if eta_s <= 0 else f"{eta_s/60.0:.1f}m"

        equity = float(progress.equity)
        total_return = float(progress.total_return)
        drawdown = float(progress.drawdown)
        day_pnl = float(progress.day_pnl)

        return_style = "green" if total_return >= 0 else "red"
        dd_style = "red" if drawdown < 0 else "green"
        day_style = "green" if day_pnl >= 0 else "red"

        kpis = Table.grid(expand=True)
        kpis.add_column(ratio=1)
        kpis.add_column(ratio=1)
        kpis.add_column(ratio=1)
        kpis.add_row(
            Panel(
                Align.center(Text(f"${equity:,.2f}", style=f"bold {return_style}")),
                title="Equity",
                border_style=return_style,
                box=box.ROUNDED,
                padding=(1, 2),
            ),
            Panel(
                Align.center(Text(f"{total_return:+.2%}", style=f"bold {return_style}")),
                title="Total Return",
                border_style=return_style,
                box=box.ROUNDED,
                padding=(1, 2),
            ),
            Panel(
                Align.center(Text(f"{drawdown:.2%}", style=f"bold {dd_style}")),
                title="Drawdown",
                border_style=dd_style,
                box=box.ROUNDED,
                padding=(1, 2),
            ),
        )

        meta = Table(show_header=False, box=box.SIMPLE)
        meta.add_column("k", style="bold")
        meta.add_column("v")
        meta.add_row("status", status)
        meta.add_row("progress", f"{bar}  {pct*100:5.1f}%  ({progress.i}/{progress.n})")
        meta.add_row("ts", str(progress.timestamp))
        meta.add_row("elapsed", f"{elapsed_s:.1f}s  rate={rate:.0f} bars/s  eta≈{eta_text}")
        if run_dir is not None:
            meta.add_row("run_dir", str(run_dir))
        meta.add_row("cash", f"${float(progress.cash):,.2f}")
        meta.add_row("day_pnl", Text(f"{day_pnl:+,.2f}", style=f"bold {day_style}"))
        meta.add_row("fills", str(int(progress.fills)))
        if progress.last_trade:
            t = progress.last_trade
            meta.add_row(
                "last_fill",
                f"{t.get('symbol')} {t.get('side')} qty={t.get('qty')} px={t.get('fill_price')} reason={t.get('strategy_reason')}",
            )

        positions = Table(box=box.SIMPLE, show_header=True, header_style="bold")
        positions.add_column("symbol", style="bold")
        positions.add_column("qty", justify="right")
        positions.add_column("last", justify="right")
        positions.add_column("notional", justify="right")
        positions.add_column("exposure", justify="right")

        any_pos = False
        for sym, qty in sorted(progress.positions.items()):
            qty = float(qty)
            if abs(qty) <= 1e-8:
                continue
            any_pos = True
            last = float(progress.closes.get(sym, 0.0))
            notional = qty * last
            exposure = (
                notional / float(self._backtest_max_notional)
                if float(self._backtest_max_notional) > 0
                else 0.0
            )
            s = "green" if qty > 0 else "red"
            positions.add_row(
                sym,
                Text(f"{qty:,.2f}", style=s),
                f"{last:,.2f}",
                Text(f"{notional:,.2f}", style=s),
                Text(f"{exposure:+.2f}x", style=s),
            )
        if not any_pos:
            positions.add_row("-", "0", "-", "0", "0x")

        spark_values = list(self._backtest_equity_spark)
        spark = self._sparkline(spark_values, width=80)
        spark_style = "green" if len(spark_values) >= 2 and spark_values[-1] >= spark_values[0] else "red"
        spark_panel = Panel(
            Align.center(Text(spark, style=spark_style)),
            title="Equity (recent)",
            border_style=spark_style,
            box=box.ROUNDED,
            padding=(1, 2),
        )

        results.update(
            Group(
                kpis,
                "",
                Panel(meta, title="Progress", border_style="cyan", box=box.ROUNDED),
                "",
                Panel(positions, title="Positions", border_style="magenta", box=box.ROUNDED),
                "",
                spark_panel,
            )
        )

    def _refresh_backtest_view(self) -> None:
        if self._backtest_thread is None:
            return

        updated = False
        while True:
            try:
                kind, payload = self._backtest_events.get_nowait()
            except queue.Empty:
                break

            if kind == "status":
                self._backtest_status = str(payload)
                updated = True
                continue

            if kind == "progress":
                progress = payload if isinstance(payload, BacktestProgress) else None
                if progress is not None:
                    self._backtest_progress = progress
                    self._backtest_equity_spark.append(float(progress.equity))
                    updated = True
                continue

            if kind == "done":
                run_dir, summary = payload  # type: ignore[misc]
                self._backtest_thread = None
                self._backtest_run_dir = None
                self._backtest_started_at = None
                self._backtest_status = ""
                self._backtest_progress = None
                self._backtest_equity_spark.clear()
                self._last_run_dir = run_dir
                self.query_one("#results", Static).border_title = "Backtest summary"
                self._render_results(summary)
                self._write_log(f"backtest complete: {run_dir}")
                return

            if kind == "error":
                self._backtest_thread = None
                self._backtest_run_dir = None
                self._backtest_started_at = None
                self._backtest_status = ""
                self._backtest_progress = None
                self._backtest_equity_spark.clear()
                self._write_log(f"backtest error: {payload}")
                err = Table(show_header=False, box=box.SIMPLE)
                err.add_column("k", style="bold red")
                err.add_column("v")
                err.add_row("status", "backtest failed")
                err.add_row("error", str(payload))
                results = self.query_one("#results", Static)
                results.border_title = "Backtest failed"
                results.update(
                    Panel(err, title="Backtest", border_style="red", box=box.ROUNDED)
                )
                return

        if updated:
            self._render_backtest_live()

    def _render_tune_live(self) -> None:
        results = self.query_one("#results", Static)
        results.border_title = "Tune (running)"

        status = self._tune_status or "running…"
        run_dir = self._tune_run_dir
        started_at = self._tune_started_at or time.monotonic()
        elapsed_s = max(time.monotonic() - started_at, 0.0)

        progress = self._tune_progress
        segment_label = "-"
        trial_label = "-"
        phase_label = "-"
        best_score_text = "-"
        last_score_text = "-"
        best_style = "cyan"
        last_style = "cyan"
        verdict_text = "-"
        verdict_style = "cyan"
        last_kpi_title = "Last"
        bar = self._progress_bar(0.0, width=28)
        pct = 0.0
        done_trials = 0
        total_trials = 0
        eta_text = "-"
        params_text = "-"

        if progress is not None:
            seg_idx = int(progress.segment)
            seg_total = max(int(progress.n_segments), 0)
            trials_per_seg = max(int(progress.trials_per_segment), 0)
            trial_idx = int(progress.trial)

            segment_label = (
                f"{seg_idx + 1}/{seg_total}" if seg_total > 0 else f"{seg_idx + 1}/-"
            )
            trial_label = (
                f"{trial_idx}/{trials_per_seg}" if trials_per_seg > 0 else f"{trial_idx}/-"
            )
            phase_label = str(progress.phase or "-")
            if str(progress.phase) == "segment_done":
                last_kpi_title = "Test"

            total_trials = max(seg_total * trials_per_seg, 0)
            done_trials = max(seg_idx * trials_per_seg + trial_idx, 0)
            pct = float(done_trials) / float(total_trials) if total_trials > 0 else 0.0
            bar = self._progress_bar(pct, width=28)

            rate = float(done_trials) / elapsed_s if elapsed_s > 0 else 0.0
            eta_s = float(total_trials - done_trials) / rate if rate > 0 else 0.0
            eta_text = "-" if eta_s <= 0 else f"{eta_s/60.0:.1f}m"

            best_score = float(progress.best_selection_score)
            last_score = float(progress.last_score)

            if math.isfinite(best_score):
                best_score_text = f"{best_score:.6g}"
                best_style = "green" if best_score >= 0 else "red"
            else:
                best_style = "cyan"

            if math.isfinite(last_score):
                last_score_text = f"{last_score:.6g}"
            else:
                last_score_text = "-"
            if progress.last_rejected:
                last_style = "red"
                verdict_text = "rejected"
                verdict_style = "red"
            else:
                verdict_text = "accepted"
                verdict_style = "green"
                if not math.isfinite(last_score):
                    last_style = "cyan"
                else:
                    last_style = "green" if last_score >= 0 else "yellow"

            strategy = self._canonicalize_strategy_name(self.state.strategy)
            if progress.best_params:
                params_text = self._format_strategy_params(strategy, progress.best_params)

        kpis = Table.grid(expand=True)
        kpis.add_column(ratio=1)
        kpis.add_column(ratio=1)
        kpis.add_column(ratio=1)
        kpis.add_column(ratio=1)
        kpis.add_row(
            Panel(
                Align.center(Text(segment_label, style="bold cyan")),
                title="Segment",
                border_style="cyan",
                box=box.ROUNDED,
                padding=(0, 1),
            ),
            Panel(
                Align.center(Text(trial_label, style="bold cyan")),
                title="Trial",
                border_style="cyan",
                box=box.ROUNDED,
                padding=(0, 1),
            ),
            Panel(
                Align.center(Text(best_score_text, style=f"bold {best_style}")),
                title="Best (seg)",
                border_style=best_style,
                box=box.ROUNDED,
                padding=(0, 1),
            ),
            Panel(
                Align.center(Text(last_score_text, style=f"bold {last_style}")),
                title=last_kpi_title,
                border_style=last_style,
                box=box.ROUNDED,
                padding=(0, 1),
            ),
        )

        meta = Table(show_header=False, box=box.SIMPLE)
        meta.add_column("k", style="bold")
        meta.add_column("v")
        meta.add_row("status", status)
        meta.add_row("phase", phase_label)
        progress_suffix = (
            f"({done_trials}/{total_trials})" if total_trials > 0 else "(-/-)"
        )
        meta.add_row("progress", f"{bar}  {pct*100:5.1f}%  {progress_suffix}")
        meta.add_row("elapsed", f"{elapsed_s:.1f}s  eta≈{eta_text}")
        if run_dir is not None:
            meta.add_row("run_dir", str(run_dir))

        if self._tune_trial_total > 0:
            accepted = int(self._tune_trial_total - self._tune_trial_rejected)
            meta.add_row(
                "accept_rate",
                f"{accepted}/{self._tune_trial_total} ({accepted/float(self._tune_trial_total):.1%})",
            )

        if self._tune_segment_trial_total > 0:
            seg_accepted = int(self._tune_segment_trial_total - self._tune_segment_trial_rejected)
            meta.add_row(
                "segment_accept",
                f"{seg_accepted}/{self._tune_segment_trial_total} ({seg_accepted/float(self._tune_segment_trial_total):.1%})",
            )

        if progress is not None:
            meta.add_row("last_verdict", Text(verdict_text, style=f"bold {verdict_style}"))
            if progress.last_rejected:
                meta.add_row(
                    "reject_reason",
                    Text(str(progress.last_reject_reason or "-"), style="red"),
                )

        recent = Table(box=box.SIMPLE, show_header=True, header_style="bold")
        recent.add_column("seg", justify="right")
        recent.add_column("trial", justify="right")
        recent.add_column("score", justify="right")
        recent.add_column("verdict", no_wrap=True)
        recent.add_column("reason")

        if not self._tune_recent_trials:
            recent.add_row("-", "-", "-", "-", "-")
        else:
            for seg_i, trial_i, score, rejected, reason in list(self._tune_recent_trials)[-8:]:
                verdict = "rejected" if rejected else "accepted"
                verdict_s = "red" if rejected else "green"
                reason_text = str(reason or "-")
                if len(reason_text) > 64:
                    reason_text = reason_text[:61] + "..."
                score_text = f"{float(score):.6g}" if math.isfinite(float(score)) else "-"
                recent.add_row(
                    str(int(seg_i) + 1),
                    str(int(trial_i)),
                    Text(score_text, style=f"bold {verdict_s}"),
                    Text(verdict, style=f"bold {verdict_s}"),
                    Text(reason_text, style="red") if rejected else Text("-", style="dim"),
                )

        best_spark_values = list(self._tune_best_score_spark)
        last_spark_values = list(self._tune_score_spark)
        best_spark = self._sparkline(best_spark_values, width=80)
        last_spark = self._sparkline(last_spark_values, width=80)
        best_spark_style = (
            "green"
            if len(best_spark_values) >= 2 and best_spark_values[-1] >= best_spark_values[0]
            else "red"
        )
        last_spark_style = (
            "green"
            if len(last_spark_values) >= 2 and last_spark_values[-1] >= last_spark_values[0]
            else "red"
        )

        score_table = Table(show_header=False, box=box.SIMPLE)
        score_table.add_column("k", style="bold", no_wrap=True)
        score_table.add_column("v")
        score_table.add_row("best(seg)", Text(best_spark or "-", style=best_spark_style))
        score_table.add_row("last", Text(last_spark or "-", style=last_spark_style))

        activity = Group(
            recent,
            "",
            score_table,
        )

        results.update(
            Group(
                kpis,
                "",
                Panel(meta, title="Progress", border_style="cyan", box=box.ROUNDED),
                "",
                Panel(
                    Text(params_text or "-", style=""),
                    title="Best Params (segment)",
                    border_style="magenta",
                    box=box.ROUNDED,
                ),
                "",
                Panel(activity, title="Search (recent)", border_style="yellow", box=box.ROUNDED),
            )
        )

    def _refresh_tune_view(self) -> None:
        if self._tune_thread is None:
            return

        updated = False
        while True:
            try:
                kind, payload = self._tune_events.get_nowait()
            except queue.Empty:
                break

            if kind == "status":
                self._tune_status = str(payload)
                updated = True
                continue

            if kind == "progress":
                progress = payload if isinstance(payload, TuneProgress) else None
                if progress is not None:
                    self._tune_progress = progress
                    try:
                        last_score = float(progress.last_score)
                        if math.isfinite(last_score):
                            self._tune_score_spark.append(last_score)
                    except Exception:
                        pass

                    if str(progress.phase) == "search":
                        seg_idx = int(progress.segment)
                        if seg_idx != int(self._tune_segment_index):
                            self._tune_segment_index = seg_idx
                            self._tune_segment_trial_total = 0
                            self._tune_segment_trial_rejected = 0
                            self._tune_best_score_spark.clear()

                        self._tune_trial_total += 1
                        self._tune_segment_trial_total += 1
                        if bool(progress.last_rejected):
                            self._tune_trial_rejected += 1
                            self._tune_segment_trial_rejected += 1

                        try:
                            best_score = float(progress.best_selection_score)
                            if math.isfinite(best_score):
                                self._tune_best_score_spark.append(best_score)
                        except Exception:
                            pass

                        try:
                            self._tune_recent_trials.append(
                                (
                                    int(progress.segment),
                                    int(progress.trial),
                                    float(progress.last_score),
                                    bool(progress.last_rejected),
                                    str(progress.last_reject_reason or ""),
                                )
                            )
                        except Exception:
                            pass
                    updated = True
                continue

            if kind == "done":
                run_dir, summary, result = payload  # type: ignore[misc]
                self._tune_thread = None
                self._tune_stop = None
                self._tune_run_dir = None
                self._tune_started_at = None
                self._tune_status = ""
                self._tune_progress = None
                self._tune_last_result_dir = run_dir

                try:
                    if hasattr(result, "strategy") and hasattr(result, "best_params_latest"):
                        strategy_name = str(result.strategy)
                        self.state.tune_last_run_dir = str(run_dir)
                        self.state.tune_best_params[strategy_name] = dict(result.best_params_latest)
                        self._render_settings()
                except Exception:
                    pass

                results = self.query_one("#results", Static)
                results.border_title = "Tune summary"
                results.update(summary)
                self._write_log(f"tune complete: {run_dir}")
                return

            if kind == "error":
                self._tune_thread = None
                self._tune_stop = None
                self._tune_run_dir = None
                self._tune_started_at = None
                self._tune_status = ""
                self._tune_progress = None
                self._write_log(f"tune error: {payload}")
                err = Table(show_header=False, box=box.SIMPLE)
                err.add_column("k", style="bold red")
                err.add_column("v")
                err.add_row("status", "tune failed")
                err.add_row("error", str(payload))
                results = self.query_one("#results", Static)
                results.border_title = "Tune failed"
                results.update(Panel(err, title="Tune", border_style="red", box=box.ROUNDED))
                return

        if updated:
            self._render_tune_live()

    def _start_tune(self) -> None:
        if self._tune_thread is not None:
            if self._tune_thread.is_alive():
                self._write_log("tune already running")
                return
            self._tune_thread = None

        strategy = self._canonicalize_strategy_name(self.state.strategy)
        self._ensure_strategy_params(strategy)

        run_dir = Path("outputs") / "tuning" / f"tui_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self._tune_run_dir = run_dir
        self._tune_started_at = time.monotonic()
        self._tune_status = "preparing…"
        self._tune_progress = None
        self._tune_last_result_dir = None
        self._tune_score_spark.clear()
        self._tune_best_score_spark.clear()
        self._tune_recent_trials.clear()
        self._tune_trial_total = 0
        self._tune_trial_rejected = 0
        self._tune_segment_index = -1
        self._tune_segment_trial_total = 0
        self._tune_segment_trial_rejected = 0

        while True:
            try:
                _ = self._tune_events.get_nowait()
            except queue.Empty:
                break

        self._render_tune_live()
        self._write_log(f"starting tune: {run_dir}")

        snapshot = self.state.to_dict()
        snapshot["strategy"] = strategy

        stop_event = Event()
        self._tune_stop = stop_event

        def _worker() -> None:
            try:
                run_dir.mkdir(parents=True, exist_ok=True)

                mkt = parse_market(str(snapshot.get("market", "equity")))

                raw_symbols = [s.strip() for s in str(snapshot.get("symbols", "")).split(",") if s.strip()]
                canonical_strategy = self._canonicalize_strategy_name(str(snapshot.get("strategy", strategy)))
                if canonical_strategy in {"nec_x", "nec_pdt"} and len(raw_symbols) < 2:
                    raw_symbols = default_symbols(mkt, count=2)

                symbols = coerce_symbols_for_market(raw_symbols, mkt)
                if not symbols:
                    raise ValueError("symbols not set")

                tf = parse_bar_timeframe(str(snapshot.get("bar_timeframe", "1Min")))
                start_dt = parse_iso_datetime(snapshot.get("start")) if snapshot.get("start") else None
                end_dt = parse_iso_datetime(snapshot.get("end")) if snapshot.get("end") else None

                alpaca_settings = None
                timeframe = snapshot.get("timeframe")
                data_source = str(snapshot.get("data_source", "sample"))
                if data_source in {"alpaca", "coinbase"} and timeframe:
                    try:
                        delta = _parse_relative_timeframe(str(timeframe))
                    except Exception as exc:
                        raise ValueError(f"invalid timeframe: {timeframe}") from exc
                    end_dt = now_ny()
                    start_dt = end_dt - delta

                if data_source == "alpaca":
                    if not (start_dt and end_dt):
                        raise ValueError("start/end required for alpaca data")
                    self._tune_events.put(("status", "authenticating alpaca…"))
                    alpaca_settings = get_alpaca_settings(require_keys=True)
                elif data_source == "coinbase":
                    if not (start_dt and end_dt):
                        raise ValueError("start/end required for coinbase data")

                csv_path = Path(snapshot["csv_path"]) if snapshot.get("csv_path") else None
                csv_dir = csv_path if (csv_path and csv_path.is_dir()) else None
                csv_file = csv_path if (csv_path and csv_path.is_file()) else None

                self._tune_events.put(("status", "loading bars…"))
                universe = load_universe_bars(
                    symbols=symbols,
                    data_source=data_source,
                    timeframe=tf,
                    start=start_dt,
                    end=end_dt,
                    csv_path=csv_file,
                    csv_dir=csv_dir,
                    alpaca_settings=alpaca_settings,
                    alpaca_feed=str(snapshot.get("alpaca_feed", "delayed_sip")),
                    market=mkt.value,
                )
                data_hint = universe.hint
                bars_by_symbol = universe.bars_by_symbol

                if timeframe and data_source != "alpaca":
                    delta = _parse_relative_timeframe(str(timeframe))
                    end_dt = min(pd.Timestamp(df.index[-1]).to_pydatetime() for df in bars_by_symbol.values())
                    start_dt = end_dt - delta
                    for s in list(bars_by_symbol):
                        df = bars_by_symbol[s]
                        df = df[(df.index >= start_dt) & (df.index <= end_dt)]
                        bars_by_symbol[s] = df

                self._tune_events.put(("status", "tuning (walk-forward)…"))

                backtest_cfg = BacktestConfig(
                    symbols=symbols,
                    initial_cash=float(snapshot.get("initial_cash", 100_000.0)),
                    max_position_notional_usd=float(snapshot.get("max_position_notional_usd", 10_000.0)),
                    slippage_bps=float(snapshot.get("slippage_bps", 0.0)),
                    taker_fee_bps=float(snapshot.get("taker_fee_bps", 0.0)),
                    allow_short=bool(snapshot.get("allow_short", False)),
                )
                tune_cfg = TuneConfig(
                    trials_per_segment=int(snapshot.get("tune_trials_per_segment", 60)),
                    jobs=int(snapshot.get("tune_jobs", 1)),
                    seed=int(snapshot.get("tune_seed", 7)),
                    drift_frac=(
                        None
                        if float(snapshot.get("tune_drift_frac", 0.50) or 0.0) == 0.0
                        else float(snapshot.get("tune_drift_frac", 0.50))
                    ),
                    improvement_margin=float(snapshot.get("tune_improvement_margin", 0.0) or 0.0),
                    objective=ObjectiveConfig(),
                    walk_forward=WalkForwardConfig(
                        train=str(snapshot.get("tune_train", "30d")),
                        validate=str(snapshot.get("tune_validate", "7d")),
                        test=str(snapshot.get("tune_test", "7d")),
                        step=str(snapshot.get("tune_step", "7d")),
                    ),
                    keep_best_test_runs=True,
                )

                base_params = dict((snapshot.get("strategy_params") or {}).get(strategy, {}))

                def _progress(p: TuneProgress) -> None:
                    self._tune_events.put(("progress", p))

                result = tune_walk_forward(
                    bars_by_symbol=bars_by_symbol,
                    market=mkt.value,
                    symbols=symbols,
                    strategy=strategy,
                    backtest_cfg=backtest_cfg,
                    tune_cfg=tune_cfg,
                    run_dir=run_dir,
                    base_params=base_params,
                    stop_event=stop_event,
                    on_progress=_progress,
                )

                self._tune_events.put(("status", "finalizing…"))

                summary = Table(title="Tune summary", show_header=False, box=box.SIMPLE)
                summary.add_column("k", style="bold")
                summary.add_column("v")
                summary.add_row("run_dir", str(run_dir))
                summary.add_row("market", mkt.value)
                summary.add_row("symbols", ",".join(symbols))
                summary.add_row("data", f"{data_source} ({data_hint})")
                summary.add_row("strategy", strategy)
                summary.add_row("segments", str(len(result.selections)))
                if result.selections:
                    last = result.selections[-1]
                    test_stats = (last.test or {}).get("stats") or {}
                    summary.add_row("latest_test_return", f"{float(test_stats.get('total_return', 0.0)):.4%}")
                    summary.add_row("latest_test_dd", f"{float(test_stats.get('max_drawdown', 0.0)):.4%}")
                    summary.add_row("latest_test_sharpe", f"{float(test_stats.get('sharpe', 0.0)):.2f}")
                    summary.add_row("latest_test_trades", str(int(test_stats.get('trades', 0) or 0)))
                    summary.add_row(
                        "latest_test_liqs",
                        str(int(test_stats.get("liquidation_count", 0) or 0)),
                    )
                    summary.add_row(
                        "best_params_latest",
                        self._format_strategy_params(strategy, result.best_params_latest),
                    )
                    summary.add_row(
                        "best_params_stable",
                        self._format_strategy_params(strategy, result.best_params_stable),
                    )
                    summary.add_row("apply", "run /tune apply to copy tuned params into settings")
                summary.add_row("best_params_file", str(run_dir / "best_params.json"))
                summary.add_row("best_params_stable_file", str(run_dir / "best_params_stable.json"))
                summary.add_row("stability_file", str(run_dir / "stability.json"))

                self._tune_events.put(("done", (run_dir, summary, result)))
            except Exception as exc:
                self._tune_events.put(("error", str(exc)))

        self._tune_thread = Thread(target=_worker, daemon=True)
        self._tune_thread.start()
        self._render_settings()

    def _stop_tune(self) -> None:
        if self._tune_stop is None:
            self._write_log("tune is not running")
            return
        self._tune_stop.set()
        self._write_log("tune stop requested")

    def _apply_tuned_params(self) -> None:
        strategy = self._canonicalize_strategy_name(self.state.strategy)
        tuned = self.state.tune_best_params.get(strategy)
        if not tuned:
            self._write_log(f"no tuned params available for {strategy} (run /tune first)")
            return
        self._ensure_strategy_params(strategy)
        merged = dict(self.state.strategy_params.get(strategy, {}))
        merged.update(dict(tuned))
        self.state.strategy_params[strategy] = merged
        self._ensure_strategy_params(strategy)
        self._render_settings()
        self._write_log(f"applied tuned params to {strategy}")

    def _run_backtest(self) -> None:
        if self._backtest_thread is not None:
            if self._backtest_thread.is_alive():
                self._write_log("backtest already running")
                return
            self._backtest_thread = None

        strategy = self._canonicalize_strategy_name(self.state.strategy)
        self._ensure_strategy_params(strategy)

        run_dir = (
            Path("outputs")
            / "backtests"
            / f"tui_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

        self._backtest_run_dir = run_dir
        self._backtest_started_at = time.monotonic()
        self._backtest_status = "preparing…"
        self._backtest_progress = None
        self._backtest_equity_spark.clear()
        self._backtest_initial_cash = float(self.state.initial_cash)
        self._backtest_max_notional = float(self.state.max_position_notional_usd)

        while True:
            try:
                _ = self._backtest_events.get_nowait()
            except queue.Empty:
                break

        self._render_backtest_live()
        self._write_log(f"starting backtest: {run_dir}")

        snapshot = self.state.to_dict()
        snapshot["strategy"] = strategy

        def _worker() -> None:
            try:
                run_dir.mkdir(parents=True, exist_ok=True)

                mkt = parse_market(str(snapshot.get("market", "equity")))

                raw_symbols = [s.strip() for s in str(snapshot.get("symbols", "")).split(",") if s.strip()]
                canonical_strategy = self._canonicalize_strategy_name(str(snapshot.get("strategy", strategy)))
                if canonical_strategy in {"nec_x", "nec_pdt"} and len(raw_symbols) < 2:
                    raw_symbols = default_symbols(mkt, count=2)

                symbols = coerce_symbols_for_market(raw_symbols, mkt)
                if not symbols:
                    raise ValueError("symbols not set")

                tf = parse_bar_timeframe(str(snapshot["bar_timeframe"]))
                start_dt = parse_iso_datetime(snapshot.get("start")) if snapshot.get("start") else None
                end_dt = parse_iso_datetime(snapshot.get("end")) if snapshot.get("end") else None

                alpaca_settings = None
                timeframe = snapshot.get("timeframe")
                data_source = str(snapshot.get("data_source", "sample"))
                if data_source in {"alpaca", "coinbase"} and timeframe:
                    try:
                        delta = _parse_relative_timeframe(str(timeframe))
                    except Exception as exc:
                        raise ValueError(f"invalid timeframe: {timeframe}") from exc
                    end_dt = now_ny()
                    start_dt = end_dt - delta

                if data_source == "alpaca":
                    if not (start_dt and end_dt):
                        raise ValueError("start/end required for alpaca data")
                    self._backtest_events.put(("status", "authenticating alpaca…"))
                    alpaca_settings = get_alpaca_settings(require_keys=True)
                elif data_source == "coinbase":
                    if not (start_dt and end_dt):
                        raise ValueError("start/end required for coinbase data")

                csv_path = Path(snapshot["csv_path"]) if snapshot.get("csv_path") else None
                csv_dir = csv_path if (csv_path and csv_path.is_dir()) else None
                csv_file = csv_path if (csv_path and csv_path.is_file()) else None

                self._backtest_events.put(("status", "loading bars…"))
                universe = load_universe_bars(
                    symbols=symbols,
                    data_source=data_source,
                    timeframe=tf,
                    start=start_dt,
                    end=end_dt,
                    csv_path=csv_file,
                    csv_dir=csv_dir,
                    alpaca_settings=alpaca_settings,
                    alpaca_feed=str(snapshot.get("alpaca_feed", "delayed_sip")),
                    market=mkt.value,
                )
                data_hint = universe.hint
                bars_by_symbol = universe.bars_by_symbol

                if timeframe and data_source != "alpaca":
                    delta = _parse_relative_timeframe(str(timeframe))
                    end_dt = min(pd.Timestamp(df.index[-1]).to_pydatetime() for df in bars_by_symbol.values())
                    start_dt = end_dt - delta
                    for s in list(bars_by_symbol):
                        df = bars_by_symbol[s]
                        df = df[(df.index >= start_dt) & (df.index <= end_dt)]
                        bars_by_symbol[s] = df

                common_index: Optional[pd.DatetimeIndex] = None
                for s in symbols:
                    idx = bars_by_symbol[s].index
                    common_index = idx if common_index is None else common_index.intersection(idx)
                if common_index is None or len(common_index) < 3:
                    raise ValueError("backtest window has too few aligned bars")
                common_index = common_index.sort_values()

                self._backtest_events.put(("status", "building strategy…"))
                strat = build_strategy(
                    name=str(snapshot.get("strategy", strategy)),
                    params_path=None,
                    symbols=symbols,
                    fast_window=int(snapshot.get("fast_window", 10)),
                    slow_window=int(snapshot.get("slow_window", 30)),
                    params=dict((snapshot.get("strategy_params") or {}).get(strategy, {})),
                )

                cfg = BacktestConfig(
                    symbols=symbols,
                    initial_cash=float(snapshot.get("initial_cash", 100_000.0)),
                    max_position_notional_usd=float(snapshot.get("max_position_notional_usd", 10_000.0)),
                    slippage_bps=float(snapshot.get("slippage_bps", 0.0)),
                    taker_fee_bps=float(snapshot.get("taker_fee_bps", 0.0)),
                    allow_short=bool(snapshot.get("allow_short", False)),
                )

                self._backtest_events.put(("status", "running backtest…"))

                def _progress(p: BacktestProgress) -> None:
                    self._backtest_events.put(("progress", p))

                if mkt == Market.DERIVATIVES:
                    run_derivatives_backtest(
                        bars_by_symbol=bars_by_symbol,
                        strategy=strat,
                        cfg=cfg,
                        run_dir=run_dir,
                        progress=_progress,
                        progress_interval_s=0.25,
                        debug=bool(snapshot.get("debug", False)),
                    )
                else:
                    run_backtest(
                        bars_by_symbol=bars_by_symbol,
                        strategy=strat,
                        cfg=cfg,
                        run_dir=run_dir,
                        progress=_progress,
                        progress_interval_s=0.25,
                        debug=bool(snapshot.get("debug", False)),
                    )

                self._backtest_events.put(("status", "finalizing…"))

                metrics = json.loads((run_dir / "metrics.json").read_text())
                final_equity = float(cfg.initial_cash) * (1.0 + float(metrics["total_return"]))
                bar_minutes = _infer_bar_minutes(common_index)
                sessions = int(pd.Series(common_index.date).nunique())
                duration = pd.Timestamp(common_index[-1]) - pd.Timestamp(common_index[0])

                trades_path = run_dir / "trades.csv"
                trades = pd.read_csv(trades_path) if trades_path.exists() else pd.DataFrame()
                gross_notional = float(trades["notional"].sum()) if len(trades) and "notional" in trades.columns else 0.0

                summary = Table(title="Backtest summary", show_header=False, box=box.SIMPLE)
                summary.add_column("k", style="bold")
                summary.add_column("v")
                summary.add_row("run_dir", str(run_dir))
                if bool(snapshot.get("debug", False)):
                    summary.add_row("debug", "enabled")
                    summary.add_row("trade_debug", str(run_dir / "trade_debug.jsonl"))
                summary.add_row("symbols", ",".join(symbols))
                summary.add_row("data", f"{data_source} ({data_hint})")
                summary.add_row(
                    "window",
                    f"{(start_dt.isoformat() if start_dt else common_index[0].isoformat())} -> {(end_dt.isoformat() if end_dt else common_index[-1].isoformat())}  |  bars={len(common_index)}  sessions={sessions}  bar={bar_minutes:.2f}m",
                )
                summary.add_row("duration", str(duration))
                strategy_name = str(snapshot.get("strategy", strategy))
                if strategy_name in {"ma_crossover", "ema_crossover"}:
                    summary.add_row(
                        "strategy",
                        f"{strategy_name} (fast={int(snapshot.get('fast_window', 10))} slow={int(snapshot.get('slow_window', 30))})",
                    )
                else:
                    summary.add_row("strategy", strategy_name)
                summary.add_row("warmup_bars", str(strat.warmup_bars()))
                summary.add_row(
                    "config",
                    "  ".join(
                        [
                            f"initial_cash={cfg.initial_cash:.2f}",
                            f"max_notional={cfg.max_position_notional_usd:.2f}",
                            f"slippage_bps={cfg.slippage_bps:.2f}",
                            f"allow_short={cfg.allow_short}",
                        ]
                    ),
                )
                summary.add_row(
                    "results",
                    "  ".join(
                        [
                            f"final_equity={final_equity:.2f}",
                            f"total_return={float(metrics['total_return']):.4%}",
                            f"max_drawdown={float(metrics['max_drawdown']):.4%}",
                            f"sharpe={float(metrics['sharpe']):.2f}",
                            f"sharpe_daily={float(metrics.get('sharpe_daily', 0.0)):.2f}",
                            f"fills={int(metrics['trades'])}",
                            f"gross_notional={gross_notional:.2f}",
                        ]
                    ),
                )

                self._backtest_events.put(("done", (run_dir, summary)))
            except Exception as exc:
                self._backtest_events.put(("error", str(exc)))

        self._backtest_thread = Thread(target=_worker, daemon=True)
        self._backtest_thread.start()

    def _run_backtest_sync(self) -> tuple[Path, Table]:
        run_dir = (
            Path("outputs")
            / "backtests"
            / f"tui_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        run_dir.mkdir(parents=True, exist_ok=True)

        mkt = parse_market(self.state.market)
        raw_symbols = [s.strip() for s in self.state.symbols.split(",") if s.strip()]
        canonical_strategy = self._canonicalize_strategy_name(self.state.strategy)
        if canonical_strategy in {"nec_x", "nec_pdt"} and len(raw_symbols) < 2:
            raw_symbols = default_symbols(mkt, count=2)
        symbols = coerce_symbols_for_market(raw_symbols, mkt)
        if not symbols:
            raise ValueError("symbols not set")

        tf = parse_bar_timeframe(self.state.bar_timeframe)
        start_dt = parse_iso_datetime(self.state.start) if self.state.start else None
        end_dt = parse_iso_datetime(self.state.end) if self.state.end else None

        alpaca_settings = None
        if self.state.data_source in {"alpaca", "coinbase"} and self.state.timeframe:
            try:
                delta = _parse_relative_timeframe(self.state.timeframe)
            except Exception as exc:
                raise ValueError(f"invalid timeframe: {self.state.timeframe}") from exc
            end_dt = now_ny()
            start_dt = end_dt - delta
            self.state.start = start_dt.isoformat()
            self.state.end = end_dt.isoformat()

        if self.state.data_source == "alpaca":
            if not (start_dt and end_dt):
                raise ValueError("start/end required for alpaca data")
            alpaca_settings = get_alpaca_settings(require_keys=True)
        elif self.state.data_source == "coinbase":
            if not (start_dt and end_dt):
                raise ValueError("start/end required for coinbase data")

        csv_path = Path(self.state.csv_path) if self.state.csv_path else None
        csv_dir = csv_path if (csv_path and csv_path.is_dir()) else None
        csv_file = csv_path if (csv_path and csv_path.is_file()) else None

        universe = load_universe_bars(
            symbols=symbols,
            data_source=self.state.data_source,
            timeframe=tf,
            start=start_dt,
            end=end_dt,
            csv_path=csv_file,
            csv_dir=csv_dir,
            alpaca_settings=alpaca_settings,
            alpaca_feed=self.state.alpaca_feed,
            market=mkt.value,
        )
        data_hint = universe.hint
        bars_by_symbol = universe.bars_by_symbol

        if self.state.timeframe and self.state.data_source != "alpaca":
            delta = _parse_relative_timeframe(self.state.timeframe)
            end_dt = min(pd.Timestamp(df.index[-1]).to_pydatetime() for df in bars_by_symbol.values())
            start_dt = end_dt - delta
            for s in list(bars_by_symbol):
                df = bars_by_symbol[s]
                df = df[(df.index >= start_dt) & (df.index <= end_dt)]
                bars_by_symbol[s] = df

        common_index: Optional[pd.DatetimeIndex] = None
        for s in symbols:
            idx = bars_by_symbol[s].index
            common_index = idx if common_index is None else common_index.intersection(idx)
        if common_index is None or len(common_index) < 3:
            raise ValueError("backtest window has too few aligned bars")
        common_index = common_index.sort_values()

        strat = build_strategy(
            name=self.state.strategy,
            params_path=None,
            symbols=symbols,
            fast_window=self.state.fast_window,
            slow_window=self.state.slow_window,
            params=self.state.strategy_params.get(self.state.strategy),
        )

        cfg = BacktestConfig(
            symbols=symbols,
            initial_cash=self.state.initial_cash,
            max_position_notional_usd=self.state.max_position_notional_usd,
            slippage_bps=self.state.slippage_bps,
            taker_fee_bps=self.state.taker_fee_bps,
            allow_short=self.state.allow_short,
        )

        if mkt == Market.DERIVATIVES:
            run_derivatives_backtest(
                bars_by_symbol=bars_by_symbol,
                strategy=strat,
                cfg=cfg,
                run_dir=run_dir,
                debug=bool(self.state.debug),
            )
        else:
            run_backtest(
                bars_by_symbol=bars_by_symbol,
                strategy=strat,
                cfg=cfg,
                run_dir=run_dir,
                debug=bool(self.state.debug),
            )

        metrics = json.loads((run_dir / "metrics.json").read_text())
        equity_curve = pd.read_csv(
            run_dir / "equity_curve.csv", parse_dates=["timestamp"]
        )
        final_equity = float(equity_curve["equity"].iloc[-1])
        bar_minutes = _infer_bar_minutes(common_index)
        sessions = int(pd.Series(common_index.date).nunique())
        duration = pd.Timestamp(common_index[-1]) - pd.Timestamp(common_index[0])
        trades = pd.read_csv(run_dir / "trades.csv")
        gross_notional = float(trades["notional"].sum()) if len(trades) else 0.0

        summary = Table(title="Backtest summary", show_header=False)
        summary.add_column("k", style="bold")
        summary.add_column("v")
        summary.add_row("run_dir", str(run_dir))
        if bool(self.state.debug):
            summary.add_row("debug", "enabled")
            summary.add_row("trade_debug", str(run_dir / "trade_debug.jsonl"))
        summary.add_row("symbols", ",".join(symbols))
        summary.add_row("data", f"{self.state.data_source} ({data_hint})")
        summary.add_row(
            "window",
            f"{(start_dt.isoformat() if start_dt else common_index[0].isoformat())} -> {(end_dt.isoformat() if end_dt else common_index[-1].isoformat())}  |  bars={len(common_index)}  sessions={sessions}  bar={bar_minutes:.2f}m",
        )
        summary.add_row("duration", str(duration))
        summary.add_row(
            "strategy",
            (
                f"{self.state.strategy} (fast={self.state.fast_window} slow={self.state.slow_window})"
                if self.state.strategy in {"ma_crossover", "ema_crossover"}
                else self.state.strategy
            ),
        )
        summary.add_row("warmup_bars", str(strat.warmup_bars()))
        summary.add_row(
            "config",
            "  ".join(
                [
                    f"initial_cash={cfg.initial_cash:.2f}",
                    f"max_notional={cfg.max_position_notional_usd:.2f}",
                    f"slippage_bps={cfg.slippage_bps:.2f}",
                    f"taker_fee_bps={cfg.taker_fee_bps:.2f}",
                    f"allow_short={cfg.allow_short}",
                ]
            ),
        )
        summary.add_row(
            "results",
            "  ".join(
                [
                    f"final_equity={final_equity:.2f}",
                    f"total_return={metrics['total_return']:.4%}",
                    f"max_drawdown={metrics['max_drawdown']:.4%}",
                    f"sharpe={metrics['sharpe']:.2f}",
                    f"sharpe_daily={float(metrics.get('sharpe_daily', 0.0)):.2f}",
                    f"fills={metrics['trades']}",
                    f"gross_notional={gross_notional:.2f}",
                ]
            ),
        )

        try:
            start_ts = pd.Timestamp(common_index[0]).to_pydatetime()
            end_ts = pd.Timestamp(common_index[-1]).to_pydatetime()
            spy = spy_total_return(start=start_ts, end=end_ts)
            if spy is not None:
                (run_dir / "benchmark.json").write_text(json.dumps({"spy": spy.to_dict()}, indent=2))
                summary.add_row(
                    "benchmark",
                    f"SPY {float(spy.total_return):.4%} ({spy.start_observed} → {spy.end_observed})",
                )
                alpha = float(metrics.get("total_return", 0.0) or 0.0) - float(spy.total_return)
                summary.add_row("vs_spy", f"alpha={alpha:.4%}  beat_spy={alpha > 0.0}")
        except Exception:
            # Benchmark is optional (network/data fetch can fail); omit if unavailable.
            pass

        try:
            csv_path, png_path = write_equity_vs_benchmark_artifacts(
                run_dir=run_dir,
                benchmark="spy.us",
            )
            if csv_path is not None:
                summary.add_row("plot_csv", str(csv_path))
            if png_path is not None:
                summary.add_row("plot_png", str(png_path))
        except Exception:
            # Plotting is optional (matplotlib/network can fail); omit if unavailable.
            pass

        try:
            weekly, _rows = rolling_window_summary(
                run_dir=run_dir, window=timedelta(days=7), step=timedelta(days=7), benchmark="spy.us"
            )
            summary.add_row("weekly_mean_return", f"{float(weekly.mean_return):.4%}")
            summary.add_row("weekly_trade_window_frac", f"{float(weekly.trade_window_frac):.2%}")
        except Exception:
            # Window analysis is optional (depends on equity/trades files + benchmark data).
            pass

        return run_dir, summary

    def _start_paper(self) -> None:
        if self._paper_thread is not None:
            self._write_log("paper loop already running")
            return
        settings = get_alpaca_settings(require_keys=True)
        mkt = parse_market(self.state.market)
        run_dir = (
            Path("outputs")
            / "paper"
            / f"tui_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        run_dir.mkdir(parents=True, exist_ok=True)
        self._paper_run_dir = run_dir
        self._last_live_decision_ts = None

        placeholder = Table(show_header=False)
        placeholder.add_column("k", style="bold")
        placeholder.add_column("v")
        placeholder.add_row("status", "starting paper loop…")
        placeholder.add_row("run_dir", str(run_dir))
        placeholder.add_row("market", mkt.value)
        placeholder.add_row("symbols", self.state.symbols)
        placeholder.add_row("bar_timeframe", self.state.bar_timeframe)
        placeholder.add_row("strategy", self.state.strategy)
        results = self.query_one("#results", Static)
        results.border_title = "Paper (live)"
        results.update(placeholder)

        raw_symbols = [s.strip() for s in self.state.symbols.split(",") if s.strip()]
        canonical_strategy = self._canonicalize_strategy_name(self.state.strategy)
        if canonical_strategy in {"nec_x", "nec_pdt"} and len(raw_symbols) < 2:
            raw_symbols = default_symbols(mkt, count=2)
        symbols = coerce_symbols_for_market(raw_symbols, mkt)

        strat = build_strategy(
            name=self.state.strategy,
            params_path=None,
            symbols=symbols,
            fast_window=self.state.fast_window,
            slow_window=self.state.slow_window,
            params=self.state.strategy_params.get(self.state.strategy),
        )
        cfg = PaperConfig(
            symbols=symbols,
            bar_timeframe=self.state.bar_timeframe,
            alpaca_feed=self.state.paper_feed,
            lookback_bars=self.state.paper_lookback_bars,
            poll_seconds=self.state.paper_poll_seconds,
            max_position_notional_usd=self.state.paper_max_position_notional_usd,
            slippage_bps=float(self.state.slippage_bps),
            taker_fee_bps=float(self.state.taker_fee_bps),
            allow_short=self.state.allow_short,
            regular_hours_only=self.state.paper_regular_hours_only,
            allow_trading_when_closed=self.state.paper_allow_trading_when_closed,
            limit_offset_bps=self.state.paper_limit_offset_bps,
            dry_run=self.state.paper_dry_run,
            market=mkt.value,
        )

        self._paper_stop = Event()

        def _runner() -> None:
            try:
                run_paper_loop(
                    settings=settings,
                    strategy=strat,
                    cfg=cfg,
                    run_dir=run_dir,
                    max_loops=None,
                    stop_event=self._paper_stop,
                )
            except Exception as exc:
                self.call_from_thread(self._write_log, f"paper loop error: {exc}")
            finally:
                self._paper_thread = None
                self._paper_stop = None
                self.call_from_thread(self._render_settings)

        self._paper_thread = Thread(target=_runner, daemon=True)
        self._paper_thread.start()
        self._write_log(f"paper loop started: {run_dir}")
        self._render_settings()

    def _stop_paper(self) -> None:
        if self._paper_stop is None:
            self._write_log("paper loop is not running")
            return
        self._paper_stop.set()
        self._write_log("paper loop stop requested")


def run_tui() -> None:
    AtlasTui().run()
