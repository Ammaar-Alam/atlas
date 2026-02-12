from __future__ import annotations

import json
import logging
import secrets
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from atlas.backtest.derivatives_engine import run_derivatives_backtest
from atlas.backtest.engine import BacktestConfig, run_backtest
from atlas.backtest.window_analysis import write_window_analysis_json
from atlas.config import get_alpaca_settings
from atlas.data.benchmarks import spy_total_return
from atlas.data.bars import parse_bar_timeframe
from atlas.data.universe import load_universe_bars
from atlas.market import Market, coerce_symbols_for_market, default_symbols, parse_market
from atlas.ml.tune import ObjectiveConfig, WalkForwardConfig, parse_duration_spec
from atlas.ml.validate import walk_forward_evaluate
from atlas.strategies.registry import build_strategy, list_strategy_names

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CandidateSpec:
    strategy: str
    market: str
    symbols: list[str]
    data_source: str
    bar_timeframe: str
    start: str
    end: str
    allow_short: bool
    strategy_params: Optional[str] = None


@dataclass(frozen=True)
class EvaluationConfig:
    initial_cash: float = 500.0
    max_position_notional_usd: float = 500.0
    prewarm: str = "90d"
    baseline_slippage_bps: float = 3.0
    baseline_taker_fee_bps: float = 25.0
    stress_slippage_grid: tuple[float, ...] = (3.0, 5.0, 8.0)
    stress_taker_fee_grid: tuple[float, ...] = (25.0, 40.0, 60.0)
    stress_min_mean_return: float = -0.0025
    stress_min_positive_segment_frac: float = 0.45
    stress_min_accepted_segment_frac: float = 0.40
    top_n_validate: int = 8
    validate_train: str = "180d"
    validate_validate: str = "30d"
    validate_test: str = "30d"
    validate_step: str = "30d"
    validate_min_trades: int = 1
    validate_max_drawdown_limit: float = 0.20
    validate_worst_day_limit: float = 0.20
    validate_turnover_cap: float = 250.0
    gate_max_drawdown: float = -0.20
    gate_min_positive_week_frac: float = 0.70
    gate_min_stress_pass_frac: float = 0.66
    require_beat_spy: bool = True
    equity_fallback_sample: bool = True


@dataclass
class CandidateEvaluation:
    candidate_id: str
    strategy: str
    market: str
    symbols: list[str]
    data_source: str
    bar_timeframe: str
    start: str
    end: str
    strategy_params: Optional[str]
    baseline_run_dir: Optional[str] = None
    validation_run_dir: Optional[str] = None
    total_return: Optional[float] = None
    sharpe_daily: Optional[float] = None
    max_drawdown: Optional[float] = None
    trades: Optional[int] = None
    spy_total_return: Optional[float] = None
    alpha_vs_spy: Optional[float] = None
    beat_spy: Optional[bool] = None
    weekly_windows: int = 0
    weekly_positive_windows: int = 0
    weekly_positive_frac: float = 0.0
    weekly_trade_frac: float = 0.0
    baseline_score: float = -1e9
    validation_scenarios: int = 0
    stress_pass_scenarios: int = 0
    stress_pass_frac: float = 0.0
    stress_mean_return_median: Optional[float] = None
    stress_mean_return_worst: Optional[float] = None
    gate_drawdown: bool = False
    gate_weekly: bool = False
    gate_benchmark: bool = False
    gate_stress: bool = False
    gate_pass: bool = False
    robust_score: float = -1e9
    notes: list[str] = field(default_factory=list)
    validation_rows: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["symbols"] = list(self.symbols)
        payload["notes"] = list(self.notes)
        payload["validation_rows"] = list(self.validation_rows)
        return payload


@dataclass(frozen=True)
class EvaluationResult:
    run_dir: str
    generated_at: str
    config: dict[str, Any]
    total_candidates: int
    validated_candidates: int
    passed_candidates: int
    algorithm_a_candidate_id: Optional[str]
    algorithm_b_candidate_id: Optional[str]
    winner_candidate_id: Optional[str]
    candidates: list[dict[str, Any]]
    leaderboard_csv: str
    leaderboard_json: str
    full_json: str


def _run_suffix() -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return f"{ts}_{secrets.token_hex(2)}"


def _strategy_params_path_for(strategy: str) -> Optional[str]:
    mapping = {
        "crypto_momentum": "strategy_params/crypto_momentum_btc_eth_6h_alpaca_hb28.json",
        "crypto_ensemble": "strategy_params/crypto_ensemble_ultra_micro_6h_alpaca_hb28.json",
        "crypto_rotation": "strategy_params/crypto_rotation_2022_candidate_v16a_diversify_6h_coinbase_hb28.json",
        "crypto_tsm": "strategy_params/crypto_tsm_stable_6h_coinbase.json",
        "crypto_meta": "strategy_params/crypto_meta_r2_blend_6h_coinbase.json",
        "crypto_regime_vol_target": "strategy_params/crypto_regime_vol_target_tuned_6h_coinbase.json",
        "crypto_regime_fusion": "strategy_params/crypto_regime_fusion_tuned_v2_6h_coinbase.json",
        "crypto_vol_squeeze": "strategy_params/crypto_vol_squeeze_tuned_v2_6h_coinbase.json",
        "crypto_7d_positive_gate": "strategy_params/crypto_7d_positive_gate_v1_6h_coinbase.json",
        "crypto_weekly_lock_momentum": "strategy_params/crypto_weekly_lock_momentum_tuned_6h_coinbase.json",
        "perp_weekly_profit_chase": "strategy_params/perp_weekly_profit_chase_btc_perp_5min_v2_risk05.json",
        "perp_weekly_trend_reset": "strategy_params/perp_weekly_trend_reset_btc_perp_5min_longlookback_v2.json",
        "perp_trend_vol_guard": "strategy_params/perp_trend_vol_guard_tuned_1h.json",
        "perp_quant_fusion": "strategy_params/perp_quant_fusion_tuned_v3_1h.json",
        "perp_weekly_carry_shield": "strategy_params/perp_weekly_carry_shield_tuned_v3_1h.json",
    }
    raw = mapping.get(strategy)
    if raw is None:
        return None
    path = Path(raw)
    return str(path) if path.exists() else None


def _normalize_token(raw: str) -> str:
    return str(raw).strip().lower().replace("-", "_")


def _strategy_param_variants_for(strategy: str, prefixes: Optional[list[str]]) -> list[str]:
    root = Path("strategy_params")
    if not root.exists() or not root.is_dir():
        return []

    canonical = _normalize_token(strategy)
    normalized_prefixes = [
        _normalize_token(Path(p).stem)
        for p in (prefixes or [])
        if str(p).strip()
    ]

    variants: list[str] = []
    for path in sorted(root.glob("*.json")):
        stem = _normalize_token(path.stem)
        if not stem.startswith(canonical):
            continue
        if normalized_prefixes and not any(stem.startswith(prefix) for prefix in normalized_prefixes):
            continue
        variants.append(str(path))
    return variants


def _market_for_strategy(strategy: str) -> Market:
    s = (strategy or "").strip().lower().replace("-", "_")
    if s.startswith("crypto_"):
        return Market.CRYPTO
    if s.startswith("perp_") or s in {"hedge", "basis_carry"}:
        return Market.DERIVATIVES
    return Market.EQUITY


def _symbols_for_strategy(strategy: str, market: Market) -> list[str]:
    s = (strategy or "").strip().lower().replace("-", "_")
    if s in {"nec_x", "nec_pdt"}:
        return default_symbols(market, count=2)
    if s in {"basis_carry", "hedge"}:
        return ["BTC/USD", "BTC-PERP"]
    if s in {"crypto_rotation", "crypto_meta"}:
        return ["BTC/USD", "ETH/USD", "SOL/USD", "ADA/USD", "LTC/USD", "BCH/USD"]
    if s in {"crypto_vol_squeeze"}:
        return ["BTC/USD", "ETH/USD", "SOL/USD", "AVAX/USD"]
    if s in {"crypto_weekly_lock_momentum"}:
        return ["BTC/USD", "ETH/USD", "SOL/USD", "AVAX/USD"]
    if s in {"crypto_regime_vol_target"}:
        return ["BTC/USD", "ETH/USD", "SOL/USD", "AVAX/USD"]
    if s.startswith("crypto_"):
        return ["BTC/USD", "ETH/USD"]
    if s in {"perp_scalp", "perp_weekly_profit_chase", "perp_weekly_trend_reset"}:
        return ["BTC-PERP"]
    if s.startswith("perp_"):
        return ["BTC-PERP", "ETH-PERP"]
    if s in {"orb_trend", "spy_open_close", "ma_crossover", "ema_crossover", "no_trade"}:
        return ["SPY"]
    return default_symbols(market, count=1)


def _data_source_for_market(market: Market) -> str:
    if market == Market.EQUITY:
        return "alpaca"
    return "coinbase"


def _timeframe_for_strategy(strategy: str, market: Market) -> str:
    s = (strategy or "").strip().lower().replace("-", "_")
    if market == Market.EQUITY:
        return "15Min"
    if market == Market.CRYPTO:
        return "6H"
    if s in {"perp_scalp", "perp_weekly_profit_chase", "perp_weekly_trend_reset", "perp_flare"}:
        return "15Min"
    return "1H"


def _date_range_for_market(market: Market) -> tuple[str, str]:
    if market == Market.EQUITY:
        return ("2024-01-01T00:00:00+00:00", "2026-01-01T00:00:00+00:00")
    if market == Market.CRYPTO:
        return ("2018-01-01T00:00:00+00:00", "2026-01-01T00:00:00+00:00")
    return ("2022-01-01T00:00:00+00:00", "2026-01-01T00:00:00+00:00")


def _allow_short_for_market(market: Market) -> bool:
    return market == Market.DERIVATIVES


def _parse_iso(value: str) -> datetime:
    raw = str(value).strip()
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    ts = pd.Timestamp(raw)
    if ts.tz is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.to_pydatetime()


def _actual_score_bounds_from_run(run_dir: Path) -> tuple[Optional[datetime], Optional[datetime]]:
    """
    Infer the realized score window from equity_curve timestamps.

    This prevents benchmark mismatch when requested windows exceed locally available
    market data (common for newly listed instruments).
    """
    path = run_dir / "equity_curve.csv"
    if not path.exists():
        return None, None
    try:
        frame = pd.read_csv(path, usecols=["timestamp"])
    except Exception:
        return None, None
    if "timestamp" not in frame.columns or frame.empty:
        return None, None

    ts = pd.to_datetime(frame["timestamp"], errors="coerce", utc=True).dropna()
    if ts.empty:
        return None, None

    start = pd.Timestamp(ts.min())
    end = pd.Timestamp(ts.max()) + pd.Timedelta(days=1)
    return start.to_pydatetime(), end.to_pydatetime()


def _candidate_id(spec: CandidateSpec) -> str:
    sym = ",".join(spec.symbols)
    params_tag = Path(spec.strategy_params).name if spec.strategy_params else "default"
    return f"{spec.strategy}|{spec.market}|{sym}|{spec.bar_timeframe}|{spec.data_source}|{params_tag}"


def default_candidate_specs(
    *,
    strategies: Optional[list[str]] = None,
    strategy_params_prefixes: Optional[list[str]] = None,
) -> list[CandidateSpec]:
    if strategies:
        raw_names = [s.strip().lower().replace("-", "_") for s in strategies if s.strip()]
    else:
        raw_names = list_strategy_names()

    specs: list[CandidateSpec] = []
    for strategy in raw_names:
        market = _market_for_strategy(strategy)
        start, end = _date_range_for_market(market)
        symbols = _symbols_for_strategy(strategy, market)
        data_source = _data_source_for_market(market)
        bar_timeframe = _timeframe_for_strategy(strategy, market)
        allow_short = _allow_short_for_market(market)

        variant_paths = _strategy_param_variants_for(strategy, strategy_params_prefixes)
        if variant_paths:
            for params_path in variant_paths:
                specs.append(
                    CandidateSpec(
                        strategy=strategy,
                        market=market.value,
                        symbols=symbols,
                        data_source=data_source,
                        bar_timeframe=bar_timeframe,
                        start=start,
                        end=end,
                        allow_short=allow_short,
                        strategy_params=params_path,
                    )
                )
            continue

        specs.append(
            CandidateSpec(
                strategy=strategy,
                market=market.value,
                symbols=symbols,
                data_source=data_source,
                bar_timeframe=bar_timeframe,
                start=start,
                end=end,
                allow_short=allow_short,
                strategy_params=_strategy_params_path_for(strategy),
            )
        )

    deduped: list[CandidateSpec] = []
    seen: set[str] = set()
    for spec in specs:
        key = _candidate_id(spec)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(spec)
    return deduped


def _load_strategy_params_for_name(path: Optional[Path], strategy_name: str) -> dict[str, Any]:
    if path is None:
        return {}
    raw = json.loads(path.read_text())
    if isinstance(raw, dict) and "params" in raw and isinstance(raw["params"], dict):
        raw = raw["params"]
    if isinstance(raw, dict) and "parameters" in raw and isinstance(raw["parameters"], dict):
        raw = raw["parameters"]
    if not isinstance(raw, dict):
        raise ValueError("strategy params json must be an object")

    canonical = strategy_name.replace("-", "_")
    if canonical in raw and isinstance(raw[canonical], dict):
        return dict(raw[canonical])
    if strategy_name in raw and isinstance(raw[strategy_name], dict):
        return dict(raw[strategy_name])
    return dict(raw)


def _load_universe_for_spec(
    *,
    spec: CandidateSpec,
    cfg: EvaluationConfig,
) -> tuple[dict[str, pd.DataFrame], list[str], datetime, datetime, datetime, str, list[str]]:
    market = parse_market(spec.market)
    tf = parse_bar_timeframe(spec.bar_timeframe)
    start_dt = _parse_iso(spec.start)
    end_dt = _parse_iso(spec.end)
    score_start = start_dt
    score_end = end_dt
    prewarm_delta = parse_duration_spec(cfg.prewarm)
    load_start = start_dt - prewarm_delta

    symbols = coerce_symbols_for_market(spec.symbols, market)
    if not symbols:
        raise ValueError("candidate symbols are empty after normalization")

    notes: list[str] = []
    source_used = spec.data_source
    alpaca_settings = get_alpaca_settings(require_keys=True) if source_used == "alpaca" else None

    try:
        universe = load_universe_bars(
            symbols=symbols,
            data_source=source_used,
            timeframe=tf,
            start=load_start,
            end=end_dt,
            alpaca_settings=alpaca_settings,
            market=market.value,
        )
    except Exception as exc:
        if market == Market.EQUITY and source_used == "alpaca" and cfg.equity_fallback_sample:
            source_used = "sample"
            notes.append(f"alpaca_load_failed_fallback_to_sample: {exc}")
            universe = load_universe_bars(
                symbols=symbols,
                data_source=source_used,
                timeframe=parse_bar_timeframe("1Min"),
                start=None,
                end=None,
                alpaca_settings=None,
                market=market.value,
            )
            sample_index = None
            for sym in symbols:
                idx = universe.bars_by_symbol[sym].index
                sample_index = idx if sample_index is None else sample_index.intersection(idx)
            if sample_index is None or len(sample_index) < 3:
                raise ValueError("equity sample fallback has too few aligned bars")
            sample_index = sample_index.sort_values()
            score_start = pd.Timestamp(sample_index[0]).to_pydatetime()
            score_end = pd.Timestamp(sample_index[-1]).to_pydatetime()
            load_start = score_start
            symbols = list(universe.bars_by_symbol.keys())
        else:
            raise

    return universe.bars_by_symbol, symbols, load_start, score_start, score_end, source_used, notes


def _baseline_score(*, total_return: float, sharpe_daily: float, max_drawdown: float, alpha: Optional[float]) -> float:
    alpha_term = float(alpha if alpha is not None else total_return)
    dd_penalty = abs(min(0.0, float(max_drawdown)))
    return float(alpha_term + 0.30 * float(sharpe_daily) - 0.75 * dd_penalty)


def _run_candidate_baseline(
    *,
    spec: CandidateSpec,
    cfg: EvaluationConfig,
    run_dir: Path,
) -> CandidateEvaluation:
    candidate = CandidateEvaluation(
        candidate_id=_candidate_id(spec),
        strategy=spec.strategy,
        market=spec.market,
        symbols=list(spec.symbols),
        data_source=spec.data_source,
        bar_timeframe=spec.bar_timeframe,
        start=spec.start,
        end=spec.end,
        strategy_params=spec.strategy_params,
    )

    bars_by_symbol, symbols, _load_start, score_start, score_end, source_used, notes = _load_universe_for_spec(
        spec=spec,
        cfg=cfg,
    )
    candidate.symbols = list(symbols)
    candidate.data_source = source_used
    candidate.notes.extend(notes)

    strat = build_strategy(
        name=spec.strategy,
        params_path=Path(spec.strategy_params) if spec.strategy_params else None,
        symbols=symbols,
        fast_window=10,
        slow_window=30,
    )

    market = parse_market(spec.market)
    run_dir.mkdir(parents=True, exist_ok=True)

    bt_cfg = BacktestConfig(
        symbols=list(symbols),
        initial_cash=float(cfg.initial_cash),
        max_position_notional_usd=float(cfg.max_position_notional_usd),
        slippage_bps=float(cfg.baseline_slippage_bps),
        allow_short=bool(spec.allow_short),
        taker_fee_bps=float(cfg.baseline_taker_fee_bps),
    )

    if market == Market.DERIVATIVES:
        run_derivatives_backtest(
            bars_by_symbol=bars_by_symbol,
            strategy=strat,
            cfg=bt_cfg,
            run_dir=run_dir,
            output_mode="full",
            score_start=score_start,
            score_end=score_end,
            no_trade_before=score_start,
        )
    else:
        run_backtest(
            bars_by_symbol=bars_by_symbol,
            strategy=strat,
            cfg=bt_cfg,
            run_dir=run_dir,
            output_mode="full",
            score_start=score_start,
            score_end=score_end,
            no_trade_before=score_start,
        )

    candidate.baseline_run_dir = str(run_dir)
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"missing metrics file: {metrics_path}")
    metrics = json.loads(metrics_path.read_text())

    candidate.total_return = float(metrics.get("total_return", 0.0) or 0.0)
    candidate.sharpe_daily = float(metrics.get("sharpe_daily", 0.0) or 0.0)
    candidate.max_drawdown = float(metrics.get("max_drawdown", 0.0) or 0.0)
    candidate.trades = int(metrics.get("trades", 0) or 0)

    benchmark_start = score_start
    benchmark_end = score_end
    actual_start, actual_end = _actual_score_bounds_from_run(run_dir)
    if actual_start is not None and actual_end is not None:
        benchmark_start = actual_start
        benchmark_end = actual_end
        if (actual_start != score_start) or (actual_end != score_end):
            candidate.notes.append(
                "benchmark_window_adjusted_to_realized_score_range"
            )

    try:
        spy = spy_total_return(start=benchmark_start, end=benchmark_end)
        if spy is not None:
            candidate.spy_total_return = float(spy.total_return)
            candidate.alpha_vs_spy = float(candidate.total_return - candidate.spy_total_return)
            candidate.beat_spy = bool(candidate.alpha_vs_spy > 0.0)
            (run_dir / "benchmark.json").write_text(
                json.dumps(
                    {
                        "score_window": {
                            "start": pd.Timestamp(benchmark_start).isoformat(),
                            "end": pd.Timestamp(benchmark_end).isoformat(),
                        },
                        "spy": spy.to_dict(),
                    },
                    indent=2,
                )
            )
    except Exception as exc:
        candidate.notes.append(f"spy_benchmark_failed: {exc}")

    candidate.baseline_score = _baseline_score(
        total_return=float(candidate.total_return),
        sharpe_daily=float(candidate.sharpe_daily),
        max_drawdown=float(candidate.max_drawdown),
        alpha=candidate.alpha_vs_spy,
    )

    try:
        out_path = write_window_analysis_json(
            run_dir=run_dir,
            window=parse_duration_spec("7d"),
            step=parse_duration_spec("7d"),
            benchmark="spy.us",
        )
        payload = json.loads(out_path.read_text())
        windows = payload.get("windows") or []
        candidate.weekly_windows = int(len(windows))
        if candidate.weekly_windows > 0:
            positive = int(sum(1 for row in windows if float(row.get("return", 0.0) or 0.0) > 0.0))
            candidate.weekly_positive_windows = int(positive)
            candidate.weekly_positive_frac = float(positive / candidate.weekly_windows)
        summary = payload.get("summary") or {}
        candidate.weekly_trade_frac = float(summary.get("trade_window_frac", 0.0) or 0.0)
    except Exception as exc:
        candidate.notes.append(f"weekly_window_analysis_failed: {exc}")

    return candidate


def _stress_rows_to_summary(rows: list[dict[str, Any]]) -> tuple[int, int, float, Optional[float], Optional[float]]:
    total = int(len(rows))
    passed = int(sum(1 for row in rows if bool(row.get("scenario_pass", False))))
    frac = float(passed / total) if total > 0 else 0.0
    mean_returns = [float(row.get("mean_return", 0.0) or 0.0) for row in rows]
    if not mean_returns:
        return total, passed, frac, None, None
    mean_series = pd.Series(mean_returns, dtype=float)
    return total, passed, frac, float(mean_series.median()), float(mean_series.min())


def _run_candidate_validation(
    *,
    candidate: CandidateEvaluation,
    spec: CandidateSpec,
    cfg: EvaluationConfig,
    run_dir: Path,
) -> CandidateEvaluation:
    bars_by_symbol, symbols, _load_start, _score_start, _score_end, source_used, notes = _load_universe_for_spec(
        spec=spec,
        cfg=cfg,
    )
    candidate.symbols = list(symbols)
    candidate.data_source = source_used
    for note in notes:
        if note not in candidate.notes:
            candidate.notes.append(note)

    params = _load_strategy_params_for_name(
        Path(spec.strategy_params) if spec.strategy_params else None,
        spec.strategy,
    )
    wf = WalkForwardConfig(
        train=str(cfg.validate_train),
        validate=str(cfg.validate_validate),
        test=str(cfg.validate_test),
        step=str(cfg.validate_step),
    )
    objective = ObjectiveConfig(
        min_trades=int(cfg.validate_min_trades),
        max_drawdown_limit=float(cfg.validate_max_drawdown_limit),
        worst_day_limit=float(cfg.validate_worst_day_limit),
        turnover_cap=float(cfg.validate_turnover_cap),
    )

    run_dir.mkdir(parents=True, exist_ok=True)
    scenario_rows: list[dict[str, Any]] = []

    for slip in cfg.stress_slippage_grid:
        for fee in cfg.stress_taker_fee_grid:
            scenario_dir = run_dir / f"slip{slip:g}_fee{fee:g}"
            bt_cfg = BacktestConfig(
                symbols=list(symbols),
                initial_cash=float(cfg.initial_cash),
                max_position_notional_usd=float(cfg.max_position_notional_usd),
                slippage_bps=float(slip),
                allow_short=bool(spec.allow_short),
                taker_fee_bps=float(fee),
            )
            result = walk_forward_evaluate(
                bars_by_symbol=bars_by_symbol,
                market=spec.market,
                symbols=list(symbols),
                strategy=spec.strategy,
                params=params,
                backtest_cfg=bt_cfg,
                walk_forward=wf,
                objective=objective,
                run_dir=scenario_dir,
                keep_test_runs=False,
            )
            summary = dict(result.summary or {})
            mean_ret = float(summary.get("mean_return", 0.0) or 0.0)
            pos_seg_frac = float(summary.get("positive_segment_frac", 0.0) or 0.0)
            accepted = int(summary.get("accepted", 0) or 0)
            segments = int(summary.get("segments", 0) or 0)
            accepted_frac = float(accepted / segments) if segments > 0 else 0.0
            scenario_pass = bool(
                mean_ret >= float(cfg.stress_min_mean_return)
                and pos_seg_frac >= float(cfg.stress_min_positive_segment_frac)
                and accepted_frac >= float(cfg.stress_min_accepted_segment_frac)
            )
            scenario_rows.append(
                {
                    "slippage_bps": float(slip),
                    "taker_fee_bps": float(fee),
                    "segments": segments,
                    "accepted": accepted,
                    "accepted_frac": accepted_frac,
                    "mean_return": mean_ret,
                    "median_return": float(summary.get("median_return", 0.0) or 0.0),
                    "positive_segment_frac": pos_seg_frac,
                    "scenario_pass": scenario_pass,
                    "run_dir": str(scenario_dir),
                }
            )

    total, passed, frac, median_mean, worst_mean = _stress_rows_to_summary(scenario_rows)
    candidate.validation_run_dir = str(run_dir)
    candidate.validation_rows = scenario_rows
    candidate.validation_scenarios = int(total)
    candidate.stress_pass_scenarios = int(passed)
    candidate.stress_pass_frac = float(frac)
    candidate.stress_mean_return_median = median_mean
    candidate.stress_mean_return_worst = worst_mean
    return candidate


def _apply_gates(candidate: CandidateEvaluation, cfg: EvaluationConfig) -> CandidateEvaluation:
    max_dd = float(candidate.max_drawdown if candidate.max_drawdown is not None else -1.0)
    candidate.gate_drawdown = bool(max_dd >= float(cfg.gate_max_drawdown))
    candidate.gate_weekly = bool(float(candidate.weekly_positive_frac) >= float(cfg.gate_min_positive_week_frac))
    if cfg.require_beat_spy:
        candidate.gate_benchmark = bool(candidate.beat_spy is True)
    else:
        candidate.gate_benchmark = True
    candidate.gate_stress = bool(float(candidate.stress_pass_frac) >= float(cfg.gate_min_stress_pass_frac))
    candidate.gate_pass = bool(
        candidate.gate_drawdown
        and candidate.gate_weekly
        and candidate.gate_benchmark
        and candidate.gate_stress
    )

    stress_term = float(candidate.stress_mean_return_median or -1.0)
    dd_penalty = abs(min(0.0, max_dd))
    sharpe = float(candidate.sharpe_daily or 0.0)
    candidate.robust_score = float(stress_term + 0.25 * sharpe - 0.50 * dd_penalty + 0.20 * candidate.baseline_score)
    return candidate


def _leaderboard_rows(candidates: list[CandidateEvaluation]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for c in candidates:
        row = {
            "candidate_id": c.candidate_id,
            "strategy": c.strategy,
            "market": c.market,
            "symbols": ",".join(c.symbols),
            "data_source": c.data_source,
            "bar_timeframe": c.bar_timeframe,
            "strategy_params": c.strategy_params,
            "total_return": c.total_return,
            "sharpe_daily": c.sharpe_daily,
            "max_drawdown": c.max_drawdown,
            "trades": c.trades,
            "spy_total_return": c.spy_total_return,
            "alpha_vs_spy": c.alpha_vs_spy,
            "beat_spy": c.beat_spy,
            "weekly_windows": c.weekly_windows,
            "weekly_positive_windows": c.weekly_positive_windows,
            "weekly_positive_frac": c.weekly_positive_frac,
            "weekly_trade_frac": c.weekly_trade_frac,
            "baseline_score": c.baseline_score,
            "validation_scenarios": c.validation_scenarios,
            "stress_pass_scenarios": c.stress_pass_scenarios,
            "stress_pass_frac": c.stress_pass_frac,
            "stress_mean_return_median": c.stress_mean_return_median,
            "stress_mean_return_worst": c.stress_mean_return_worst,
            "gate_drawdown": c.gate_drawdown,
            "gate_weekly": c.gate_weekly,
            "gate_benchmark": c.gate_benchmark,
            "gate_stress": c.gate_stress,
            "gate_pass": c.gate_pass,
            "robust_score": c.robust_score,
            "baseline_run_dir": c.baseline_run_dir,
            "validation_run_dir": c.validation_run_dir,
            "notes": " | ".join(c.notes),
        }
        rows.append(row)
    return rows


def _save_outputs(
    *,
    run_dir: Path,
    cfg: EvaluationConfig,
    ranked: list[CandidateEvaluation],
    winner: Optional[CandidateEvaluation],
    algorithm_a: Optional[CandidateEvaluation],
    algorithm_b: Optional[CandidateEvaluation],
) -> EvaluationResult:
    run_dir.mkdir(parents=True, exist_ok=True)

    leaderboard_rows = _leaderboard_rows(ranked)
    df = pd.DataFrame(leaderboard_rows)

    leaderboard_csv = run_dir / "leaderboard.csv"
    leaderboard_json = run_dir / "leaderboard.json"
    full_json = run_dir / "evaluation_result.json"
    state_json = Path("outputs") / "evaluations" / "latest_state.json"
    deployment_json = run_dir / "deployment_feasibility.json"

    if len(df):
        df.to_csv(leaderboard_csv, index=False)
        leaderboard_json.write_text(df.to_json(orient="records", indent=2))
    else:
        leaderboard_csv.write_text("")
        leaderboard_json.write_text("[]")

    deployment_payload = {
        "summary": {
            "winner_candidate_id": winner.candidate_id if winner else None,
            "algorithm_a_candidate_id": algorithm_a.candidate_id if algorithm_a else None,
            "algorithm_b_candidate_id": algorithm_b.candidate_id if algorithm_b else None,
            "note": "Winner is selected by robustness gates + score. Venue/legal checks must be re-verified before live deployment.",
        },
        "us_feasibility_checks": [
            "PDT constraints for US equities remain applicable under FINRA rules.",
            "Leverage and shorting permissions are venue/account-tier specific.",
            "For non-Alpaca winners, a concrete execution adapter is required before paper/live trading.",
        ],
    }
    deployment_json.write_text(json.dumps(deployment_payload, indent=2))

    payload = {
        "run_dir": str(run_dir),
        "generated_at": datetime.now().isoformat(),
        "config": asdict(cfg),
        "total_candidates": int(len(ranked)),
        "validated_candidates": int(sum(1 for c in ranked if c.validation_scenarios > 0)),
        "passed_candidates": int(sum(1 for c in ranked if c.gate_pass)),
        "algorithm_a_candidate_id": algorithm_a.candidate_id if algorithm_a else None,
        "algorithm_b_candidate_id": algorithm_b.candidate_id if algorithm_b else None,
        "winner_candidate_id": winner.candidate_id if winner else None,
        "candidates": [c.to_dict() for c in ranked],
        "leaderboard_csv": str(leaderboard_csv),
        "leaderboard_json": str(leaderboard_json),
        "deployment_feasibility_json": str(deployment_json),
    }
    full_json.write_text(json.dumps(payload, indent=2))

    state_json.parent.mkdir(parents=True, exist_ok=True)
    state_json.write_text(json.dumps(payload, indent=2))

    return EvaluationResult(
        run_dir=str(run_dir),
        generated_at=str(payload["generated_at"]),
        config=payload["config"],
        total_candidates=int(payload["total_candidates"]),
        validated_candidates=int(payload["validated_candidates"]),
        passed_candidates=int(payload["passed_candidates"]),
        algorithm_a_candidate_id=payload["algorithm_a_candidate_id"],
        algorithm_b_candidate_id=payload["algorithm_b_candidate_id"],
        winner_candidate_id=payload["winner_candidate_id"],
        candidates=payload["candidates"],
        leaderboard_csv=str(leaderboard_csv),
        leaderboard_json=str(leaderboard_json),
        full_json=str(full_json),
    )


def write_execution_state_markdown(
    *,
    path: Path,
    result: EvaluationResult,
    title: str = "Atlas Strategy Search Execution State",
) -> None:
    payload = {
        "run_dir": result.run_dir,
        "winner_candidate_id": result.winner_candidate_id,
        "algorithm_a_candidate_id": result.algorithm_a_candidate_id,
        "algorithm_b_candidate_id": result.algorithm_b_candidate_id,
        "passed_candidates": result.passed_candidates,
        "total_candidates": result.total_candidates,
        "leaderboard_csv": result.leaderboard_csv,
        "leaderboard_json": result.leaderboard_json,
        "full_json": result.full_json,
    }

    lines = [
        f"# {title}",
        "",
        "## Objective",
        "Find the single strongest strategy under pessimistic costs for a $500 account, with PDT-safe constraints and SPY outperformance.",
        "",
        "## Hard Gates",
        "- max_drawdown >= -20%",
        f"- weekly_positive_frac >= {float(result.config.get('gate_min_positive_week_frac', 0.70)):.0%}",
        "- beat SPY over matched dates",
        "- stress_pass_frac >= configured threshold",
        "",
        "## Latest Run",
        f"- generated_at: {result.generated_at}",
        f"- run_dir: `{result.run_dir}`",
        f"- total_candidates: {result.total_candidates}",
        f"- validated_candidates: {result.validated_candidates}",
        f"- passed_candidates: {result.passed_candidates}",
        f"- algorithm_a_candidate_id: `{result.algorithm_a_candidate_id}`",
        f"- algorithm_b_candidate_id: `{result.algorithm_b_candidate_id}`",
        f"- winner_candidate_id: `{result.winner_candidate_id}`",
        "",
        "## Artifacts",
        f"- leaderboard_csv: `{result.leaderboard_csv}`",
        f"- leaderboard_json: `{result.leaderboard_json}`",
        f"- full_json: `{result.full_json}`",
        "- latest_state_json: `outputs/evaluations/latest_state.json`",
        "",
        "## Resume Checklist",
        "1. Open `outputs/evaluations/latest_state.json` and identify the current winner and failed gates.",
        "2. Re-run `atlas evaluate-all` with tighter strategy filters or revised costs if needed.",
        "3. Re-check deployment feasibility for the winner before moving to live paper/live execution.",
        "",
        "## Snapshot",
        "```json",
        json.dumps(payload, indent=2),
        "```",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def run_evaluate_all(
    *,
    run_dir: Path,
    cfg: EvaluationConfig,
    strategies: Optional[list[str]] = None,
    strategy_params_prefixes: Optional[list[str]] = None,
) -> EvaluationResult:
    specs = default_candidate_specs(
        strategies=strategies,
        strategy_params_prefixes=strategy_params_prefixes,
    )
    if not specs:
        raise ValueError("no candidate strategies to evaluate")

    logger.info("evaluate-all: %d candidates", len(specs))

    baseline_root = run_dir / "baselines"
    validation_root = run_dir / "validations"
    baseline_root.mkdir(parents=True, exist_ok=True)
    validation_root.mkdir(parents=True, exist_ok=True)

    by_id: dict[str, CandidateEvaluation] = {}
    spec_by_id: dict[str, CandidateSpec] = {}

    for i, spec in enumerate(specs, start=1):
        cid = _candidate_id(spec)
        spec_by_id[cid] = spec
        baseline_dir = baseline_root / f"{spec.strategy}_{_run_suffix()}"
        logger.info("baseline %d/%d: %s", i, len(specs), cid)
        try:
            out = _run_candidate_baseline(spec=spec, cfg=cfg, run_dir=baseline_dir)
        except Exception as exc:
            out = CandidateEvaluation(
                candidate_id=cid,
                strategy=spec.strategy,
                market=spec.market,
                symbols=list(spec.symbols),
                data_source=spec.data_source,
                bar_timeframe=spec.bar_timeframe,
                start=spec.start,
                end=spec.end,
                strategy_params=spec.strategy_params,
                notes=[f"baseline_failed: {exc}"],
            )
        by_id[cid] = out

    ranked_baseline = sorted(
        by_id.values(),
        key=lambda c: float(c.baseline_score),
        reverse=True,
    )

    shortlist: list[CandidateEvaluation] = []
    for c in ranked_baseline:
        if c.total_return is None:
            continue
        shortlist.append(c)
        if len(shortlist) >= int(max(0, cfg.top_n_validate)):
            break

    for i, base in enumerate(shortlist, start=1):
        cid = base.candidate_id
        spec = spec_by_id[cid]
        val_dir = validation_root / f"{spec.strategy}_{_run_suffix()}"
        logger.info("validation %d/%d: %s", i, len(shortlist), cid)
        try:
            by_id[cid] = _run_candidate_validation(
                candidate=base,
                spec=spec,
                cfg=cfg,
                run_dir=val_dir,
            )
        except Exception as exc:
            if f"validation_failed: {exc}" not in by_id[cid].notes:
                by_id[cid].notes.append(f"validation_failed: {exc}")

    for cid in list(by_id.keys()):
        by_id[cid] = _apply_gates(by_id[cid], cfg)

    ranked = sorted(
        by_id.values(),
        key=lambda c: (
            int(bool(c.gate_pass)),
            float(c.robust_score),
            float(c.baseline_score),
            float(c.total_return or -1e9),
        ),
        reverse=True,
    )

    derivatives_ranked = [c for c in ranked if c.market == Market.DERIVATIVES.value]
    non_derivatives_ranked = [c for c in ranked if c.market != Market.DERIVATIVES.value]

    algorithm_a = derivatives_ranked[0] if derivatives_ranked else None
    algorithm_b = non_derivatives_ranked[0] if non_derivatives_ranked else None
    winner = ranked[0] if ranked else None

    result = _save_outputs(
        run_dir=run_dir,
        cfg=cfg,
        ranked=ranked,
        winner=winner,
        algorithm_a=algorithm_a,
        algorithm_b=algorithm_b,
    )

    write_execution_state_markdown(
        path=Path("docs") / "EXECUTION_STATE_STRATEGY_SEARCH.md",
        result=result,
    )

    return result
