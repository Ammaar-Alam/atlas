from __future__ import annotations

import json
import math
import random
import shutil
import statistics
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


def get_search_space(strategy: str) -> list[Param]:
    strategy = (strategy or "").strip().lower().replace("-", "_")
    if strategy == "perp_flare":
        return _perp_flare_space()
    if strategy == "orb_trend":
        return _orb_trend_space()
    if strategy == "hedge":
        return _hedge_space()
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
    if strategy == "orb_trend":
        return _validate_orb_trend_params(params)
    if strategy == "hedge":
        return _validate_hedge_params(params)
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
    w_sharpe: float = 0.10
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
    seed: int = 7
    drift_frac: Optional[float] = 0.50
    improvement_margin: float = 0.0
    objective: ObjectiveConfig = field(default_factory=ObjectiveConfig)
    walk_forward: WalkForwardConfig = field(default_factory=WalkForwardConfig)
    keep_best_test_runs: bool = True


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
        trade_count = int(metrics.trades)
    else:
        total_return = 0.0
        max_drawdown = 0.0
        sharpe = 0.0
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
        out[sym] = df[(df.index >= start_ts) & (df.index < end_ts)].copy()
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
    return {sym: df.loc[idx_slice].copy() for sym, df in bars_by_symbol.items()}


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
) -> BacktestOutputs:
    if market == Market.DERIVATIVES:
        return run_derivatives_backtest(
            bars_by_symbol=bars_by_symbol,
            strategy=strategy,
            cfg=cfg,
            run_dir=run_dir,
        )
    return run_backtest(
        bars_by_symbol=bars_by_symbol, strategy=strategy, cfg=cfg, run_dir=run_dir
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
    bars_by_symbol = {s: bars_by_symbol[s].loc[common_index].copy() for s in symbols}

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

    t0 = time.perf_counter()
    with trials_path.open("w") as f_trials:
        for seg_i, seg in enumerate(segments):
            if stop_event is not None and stop_event.is_set():
                break

            best_params: Optional[dict[str, Any]] = None
            best_selection_score = float("-inf")
            best_train: dict[str, Any] = {}
            best_val: dict[str, Any] = {}

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

                train_tmp = run_dir / "tmp" / f"seg{seg_i}_incumbent_train"
                val_tmp = run_dir / "tmp" / f"seg{seg_i}_incumbent_val"
                train_tmp.mkdir(parents=True, exist_ok=True)
                val_tmp.mkdir(parents=True, exist_ok=True)
                try:
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
                finally:
                    shutil.rmtree(train_tmp, ignore_errors=True)
                    shutil.rmtree(val_tmp, ignore_errors=True)

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

                train_dir = run_dir / "tmp" / f"seg{seg_i}_trial{trial_i}_train"
                val_dir = run_dir / "tmp" / f"seg{seg_i}_trial{trial_i}_val"
                train_dir.mkdir(parents=True, exist_ok=True)
                val_dir.mkdir(parents=True, exist_ok=True)

                try:
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
                        run_dir=train_dir,
                    )
                    train_score = score_run(
                        train_dir,
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
                        run_dir=val_dir,
                    )
                    val_score = score_run(
                        val_dir,
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
                finally:
                    shutil.rmtree(train_dir, ignore_errors=True)
                    shutil.rmtree(val_dir, ignore_errors=True)

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

    elapsed_s = time.perf_counter() - t0

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
