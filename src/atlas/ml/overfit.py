from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import NormalDist
from typing import Any, Optional

import numpy as np
import pandas as pd

from atlas.utils.time import NY_TZ


def _load_equity_curve(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "equity_curve.csv"
    if not path.exists():
        return pd.DataFrame()
    equity = pd.read_csv(path, parse_dates=["timestamp"])
    equity["timestamp"] = pd.to_datetime(equity["timestamp"], errors="coerce", utc=True)
    equity = equity.dropna(subset=["timestamp"]).set_index("timestamp").sort_index()
    if len(equity) and equity.index.tz is None:
        equity.index = pd.to_datetime(equity.index, utc=True)
    return equity


def _daily_returns_from_equity(equity: pd.DataFrame) -> pd.Series:
    if equity.empty:
        return pd.Series(dtype=float)

    if "day_return" in equity.columns:
        daily = equity["day_return"].astype(float).groupby(equity.index.date).last()
        return daily.dropna()

    if "equity" not in equity.columns:
        return pd.Series(dtype=float)

    eq = equity[["equity"]].copy()
    idx = eq.index
    if idx.tz is None:
        idx = idx.tz_localize(NY_TZ)
    else:
        idx = idx.tz_convert(NY_TZ)
    eq.index = idx
    eq = eq.sort_index()

    has_weekend = bool((eq.index.dayofweek >= 5).any())
    if not has_weekend:
        eq = eq[eq.index.dayofweek < 5]
        if eq.empty:
            return pd.Series(dtype=float)
        try:
            session = eq.between_time("09:30", "16:00", include_start=True, include_end=True)
        except TypeError:
            session = eq.between_time("09:30", "16:00")
        daily = session.resample("1D").last() if not session.empty else eq.resample("1D").last()
    else:
        daily = eq.resample("1D").last()

    daily = daily.dropna(subset=["equity"])
    return daily["equity"].pct_change().dropna()


def daily_returns_from_run(run_dir: Path) -> pd.Series:
    equity = _load_equity_curve(run_dir)
    return _daily_returns_from_equity(equity)


def _annualization_factor(daily_index: Optional[pd.Index]) -> float:
    if daily_index is None:
        return 252.0
    try:
        idx = pd.to_datetime(daily_index)
        has_weekend = bool((idx.dayofweek >= 5).any())
    except Exception:
        has_weekend = False
    return 365.0 if has_weekend else 252.0


def sharpe_from_returns(returns: np.ndarray, *, annualization: float) -> float:
    if returns.size < 2:
        return 0.0
    mean = float(np.nanmean(returns))
    std = float(np.nanstd(returns, ddof=1))
    if std == 0.0 or math.isnan(std):
        return 0.0
    return float((mean / std) * math.sqrt(float(annualization)))


def bootstrap_summary(
    returns: np.ndarray,
    *,
    annualization: float,
    n_boot: int = 1000,
    block: int = 5,
    seed: Optional[int] = 7,
) -> dict[str, float]:
    rets = np.asarray(returns, dtype=float)
    rets = rets[np.isfinite(rets)]
    if rets.size < 2:
        return {
            "n": float(rets.size),
            "mean_sharpe": 0.0,
            "p05_sharpe": 0.0,
            "p50_sharpe": 0.0,
            "p95_sharpe": 0.0,
            "p_sharpe_gt_0": 0.0,
            "mean_total_return": 0.0,
            "p05_total_return": 0.0,
            "p50_total_return": 0.0,
            "p95_total_return": 0.0,
        }

    rng = np.random.default_rng(seed)
    n = int(rets.size)
    block = max(1, min(int(block), n))
    n_blocks = int(math.ceil(n / block))

    boot_sharpe = np.zeros(int(n_boot))
    boot_ret = np.zeros(int(n_boot))
    for i in range(int(n_boot)):
        idx = []
        for _ in range(n_blocks):
            start = int(rng.integers(0, n - block + 1))
            idx.extend(range(start, start + block))
        sample = rets[np.array(idx[:n])]
        boot_sharpe[i] = sharpe_from_returns(sample, annualization=annualization)
        boot_ret[i] = float(np.prod(1.0 + sample) - 1.0)

    return {
        "n": float(n),
        "mean_sharpe": float(np.mean(boot_sharpe)),
        "p05_sharpe": float(np.quantile(boot_sharpe, 0.05)),
        "p50_sharpe": float(np.quantile(boot_sharpe, 0.50)),
        "p95_sharpe": float(np.quantile(boot_sharpe, 0.95)),
        "p_sharpe_gt_0": float(np.mean(boot_sharpe > 0.0)),
        "mean_total_return": float(np.mean(boot_ret)),
        "p05_total_return": float(np.quantile(boot_ret, 0.05)),
        "p50_total_return": float(np.quantile(boot_ret, 0.50)),
        "p95_total_return": float(np.quantile(boot_ret, 0.95)),
    }


def deflated_sharpe_ratio(
    returns: np.ndarray,
    *,
    trials: int,
    annualization: float,
) -> dict[str, float]:
    rets = np.asarray(returns, dtype=float)
    rets = rets[np.isfinite(rets)]
    n = int(rets.size)
    if n < 3:
        return {"n": float(n), "sr": 0.0, "sr0": 0.0, "sigma_sr": 0.0, "dsr": 0.0}

    mean = float(np.mean(rets))
    std = float(np.std(rets, ddof=1))
    if std == 0.0 or not math.isfinite(std):
        return {"n": float(n), "sr": 0.0, "sr0": 0.0, "sigma_sr": 0.0, "dsr": 0.0}

    sr = float((mean / std) * math.sqrt(float(annualization)))

    centered = rets - mean
    m2 = float(np.mean(centered**2))
    m3 = float(np.mean(centered**3))
    m4 = float(np.mean(centered**4))
    if m2 <= 0:
        return {"n": float(n), "sr": sr, "sr0": 0.0, "sigma_sr": 0.0, "dsr": 0.0}

    skew = float(m3 / (m2 ** 1.5))
    kurt = float(m4 / (m2 ** 2))
    sigma_sr = math.sqrt(
        max(0.0, (1.0 - skew * sr + ((kurt - 1.0) / 4.0) * (sr**2)) / (n - 1))
    )

    trials = max(1, int(trials))
    if sigma_sr == 0.0:
        sr0 = 0.0
        dsr = 1.0 if sr > 0.0 else 0.0
    else:
        norm = NormalDist()
        sr0 = float(sigma_sr * norm.inv_cdf(1.0 - (1.0 / float(trials))))
        dsr = float(norm.cdf((sr - sr0) / sigma_sr))

    return {
        "n": float(n),
        "sr": float(sr),
        "sr0": float(sr0),
        "sigma_sr": float(sigma_sr),
        "dsr": float(dsr),
    }


def _extract_metric(record: dict[str, Any], metric: str) -> Optional[float]:
    if metric == "score":
        return float(record.get("score", float("nan")))
    if metric.startswith("stats."):
        key = metric.split(".", 1)[1]
        stats = record.get("stats") or {}
        if key in stats:
            return float(stats.get(key))
    return None


def _params_key(params: dict[str, Any]) -> str:
    try:
        return json.dumps(params, sort_keys=True, default=str)
    except Exception:
        return str(params)


def cscv_pbo_from_trials(
    trials_path: Path,
    *,
    metric: str = "score",
    phase_is: str = "train",
    phase_oos: str = "validate",
    drop_rejected: bool = True,
) -> dict[str, Any]:
    by_segment: dict[int, dict[str, dict[str, Any]]] = {}
    if not trials_path.exists():
        return {
            "segments": 0,
            "segments_used": 0,
            "pbo": 0.0,
            "lambda_mean": 0.0,
            "lambda_median": 0.0,
        }

    with trials_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            phase = rec.get("phase")
            if phase not in {phase_is, phase_oos}:
                continue
            if drop_rejected and bool(rec.get("rejected", False)):
                continue
            metric_val = _extract_metric(rec, metric)
            if metric_val is None or not math.isfinite(metric_val):
                continue
            seg = int(rec.get("segment", -1))
            params = rec.get("params") or {}
            key = _params_key(params)
            by_segment.setdefault(seg, {}).setdefault(key, {"params": params})[phase] = float(metric_val)

    lambdas: list[float] = []
    for seg, params_map in by_segment.items():
        candidates = []
        for key, data in params_map.items():
            if phase_is in data and phase_oos in data:
                candidates.append((key, float(data[phase_is]), float(data[phase_oos])))
        if len(candidates) < 2:
            continue

        candidates.sort(key=lambda x: float(x[1]), reverse=True)
        best_key = candidates[0][0]
        oos_scores = np.array([c[2] for c in candidates], dtype=float)
        order = np.argsort(-oos_scores)
        ranks = np.empty_like(order)
        ranks[order] = np.arange(1, len(order) + 1)

        best_idx = next(i for i, c in enumerate(candidates) if c[0] == best_key)
        rank = int(ranks[best_idx])
        n = len(candidates)
        lam = float((rank - 1) / float(n - 1)) if n > 1 else 0.0
        lambdas.append(lam)

    if not lambdas:
        return {
            "segments": int(len(by_segment)),
            "segments_used": 0,
            "pbo": 0.0,
            "lambda_mean": 0.0,
            "lambda_median": 0.0,
        }

    lambdas_arr = np.array(lambdas, dtype=float)
    return {
        "segments": int(len(by_segment)),
        "segments_used": int(len(lambdas)),
        "pbo": float(np.mean(lambdas_arr > 0.5)),
        "lambda_mean": float(np.mean(lambdas_arr)),
        "lambda_median": float(np.median(lambdas_arr)),
    }


def overfit_report_from_walk_forward(
    run_dir: Path,
    *,
    trials_path: Optional[Path] = None,
    trials: Optional[int] = None,
    n_boot: int = 1000,
    block: int = 5,
    seed: Optional[int] = 7,
) -> dict[str, Any]:
    wf_path = run_dir / "walk_forward_eval.json"
    payload = json.loads(wf_path.read_text()) if wf_path.exists() else {}
    tests = payload.get("tests") or []

    segment_rows: list[dict[str, Any]] = []
    all_returns: list[pd.Series] = []
    for test in tests:
        test_dir = Path(str(test.get("run_dir") or ""))
        if not test_dir.exists():
            continue
        daily = daily_returns_from_run(test_dir)
        if daily.empty:
            continue
        annualization = _annualization_factor(daily.index)
        segment_rows.append(
            {
                "segment": int(test.get("segment", 0)),
                "n_days": int(daily.size),
                "total_return": float(np.prod(1.0 + daily.values) - 1.0),
                "sharpe": float(sharpe_from_returns(daily.values, annualization=annualization)),
            }
        )
        all_returns.append(daily)

    combined = pd.concat(all_returns).sort_index() if all_returns else pd.Series(dtype=float)
    annualization = _annualization_factor(combined.index) if not combined.empty else 252.0

    report: dict[str, Any] = {
        "segments": int(len(tests)),
        "segments_used": int(len(segment_rows)),
        "segment_stats": segment_rows,
    }

    if combined.empty:
        report["daily_returns"] = {"n": 0}
    else:
        rets = combined.values.astype(float)
        report["daily_returns"] = {
            "n": int(rets.size),
            "total_return": float(np.prod(1.0 + rets) - 1.0),
            "sharpe": float(sharpe_from_returns(rets, annualization=annualization)),
            "bootstrap": bootstrap_summary(
                rets,
                annualization=annualization,
                n_boot=int(n_boot),
                block=int(block),
                seed=seed,
            ),
        }

        if trials is not None:
            report["deflated_sharpe"] = deflated_sharpe_ratio(
                rets,
                trials=int(trials),
                annualization=annualization,
            )

    if trials_path is not None:
        report["cscv_pbo"] = cscv_pbo_from_trials(trials_path)

    out_path = run_dir / "overfit_report.json"
    out_path.write_text(json.dumps(report, indent=2))
    return report
