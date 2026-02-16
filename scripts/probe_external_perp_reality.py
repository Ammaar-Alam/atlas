#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import requests

from atlas.backtest.derivatives_engine import run_derivatives_backtest
from atlas.backtest.engine import BacktestConfig
from atlas.backtest.window_analysis import rolling_window_summary
from atlas.coinbase.client import CoinbaseClient
from atlas.data.benchmarks import spy_total_return
from atlas.strategies.registry import build_strategy


@dataclass(frozen=True)
class WindowSpec:
    year: int
    start: datetime
    end: datetime
    length_days: int


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Probe external BTC perpetual data sources against Coinbase BTC-PERP, "
            "then evaluate cand08 realism on the closest source."
        )
    )
    p.add_argument(
        "--launch-windows-json",
        default="outputs/evaluations/coinbase_perp_rolling_180d_20260213/windows.json",
        help="Launch-era windows JSON",
    )
    p.add_argument(
        "--random-windows-json",
        default="outputs/evaluations/ab_random_year_windows_20260211_020934/windows.json",
        help="Random-year windows JSON",
    )
    p.add_argument(
        "--strategy",
        default="perp_trend_vol_guard",
        help="Strategy name in registry",
    )
    p.add_argument(
        "--params-file",
        default="strategy_params/trend_guard_manual_chase_grid2/cand08.json",
        help="Params file containing strategy params",
    )
    p.add_argument(
        "--symbol",
        default="BTC-PERP",
        help="Internal symbol name used by strategy/backtest",
    )
    p.add_argument(
        "--overlap-start",
        default="2025-07-18T00:00:00+00:00",
        help="Start of source-comparison overlap window",
    )
    p.add_argument(
        "--overlap-end",
        default="",
        help="End of source-comparison overlap window (default: now UTC)",
    )
    p.add_argument("--initial-cash", type=float, default=500.0)
    p.add_argument("--max-notional", type=float, default=5000.0)
    p.add_argument("--slippage-bps", type=float, default=1.5)
    p.add_argument("--taker-fee-bps", type=float, default=10.0)
    p.add_argument("--fixed-fee-per-contract-usd", type=float, default=0.15)
    p.add_argument("--contract-size-units", type=float, default=0.01)
    p.add_argument(
        "--allow-short",
        dest="allow_short",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument("--prewarm-days", type=int, default=45)
    p.add_argument(
        "--force-source",
        default="auto",
        choices=["auto", "okx_btc_usdt_swap", "deribit_btc_perpetual"],
        help="Force chosen source instead of automatic overlap ranking.",
    )
    p.add_argument(
        "--out-root",
        default="outputs/evaluations/external_perp_probe",
        help="Output directory root",
    )
    return p.parse_args()


def _to_dt_utc(value: str) -> datetime:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.to_pydatetime()


def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    out = df.copy()
    idx = pd.DatetimeIndex(out.index)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    else:
        idx = idx.tz_convert("UTC")
    out.index = idx.floor("1h")
    out = out.sort_index()
    out = (
        out.groupby(out.index)
        .agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )
        .dropna(subset=["open", "high", "low", "close"])
        .sort_index()
    )
    return out[["open", "high", "low", "close", "volume"]].astype(float)


def _fetch_coinbase_hourly(start: datetime, end: datetime) -> pd.DataFrame:
    client = CoinbaseClient()
    df = client.get_product_candles(
        product_id="BTC-PERP",
        start=start,
        end=end,
        granularity="ONE_HOUR",
    )
    return _normalize_ohlcv(df)


def _fetch_deribit_hourly(start: datetime, end: datetime) -> pd.DataFrame:
    all_frames: list[pd.DataFrame] = []
    chunk = timedelta(days=170)
    cur = start
    while cur < end:
        cur_end = min(cur + chunk, end)
        params = {
            "instrument_name": "BTC-PERPETUAL",
            "resolution": "60",
            "start_timestamp": int(cur.timestamp() * 1000),
            "end_timestamp": int(cur_end.timestamp() * 1000),
        }
        resp = requests.get(
            "https://www.deribit.com/api/v2/public/get_tradingview_chart_data",
            params=params,
            timeout=30,
        )
        resp.raise_for_status()
        payload = resp.json()
        result = payload.get("result") or {}
        ticks = result.get("ticks") or []
        opens = result.get("open") or []
        highs = result.get("high") or []
        lows = result.get("low") or []
        closes = result.get("close") or []
        volumes = result.get("volume") or []
        n = min(len(ticks), len(opens), len(highs), len(lows), len(closes), len(volumes))
        if n > 0:
            frame = pd.DataFrame(
                {
                    "timestamp": pd.to_datetime(ticks[:n], unit="ms", utc=True),
                    "open": pd.to_numeric(opens[:n], errors="coerce"),
                    "high": pd.to_numeric(highs[:n], errors="coerce"),
                    "low": pd.to_numeric(lows[:n], errors="coerce"),
                    "close": pd.to_numeric(closes[:n], errors="coerce"),
                    "volume": pd.to_numeric(volumes[:n], errors="coerce"),
                }
            ).dropna()
            if not frame.empty:
                frame = frame.set_index("timestamp").sort_index()
                all_frames.append(frame)
        cur = cur_end
    if not all_frames:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    df = pd.concat(all_frames, axis=0).sort_index()
    return _normalize_ohlcv(df)


def _fetch_okx_hourly(start: datetime, end: datetime) -> pd.DataFrame:
    inst_id = "BTC-USDT-SWAP"
    url = "https://www.okx.com/api/v5/market/history-candles"
    rows: list[list[Any]] = []
    after: Optional[int] = None
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    while True:
        params: dict[str, str] = {
            "instId": inst_id,
            "bar": "1H",
            "limit": "300",
        }
        if after is not None:
            params["after"] = str(after)
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        payload = resp.json()
        if str(payload.get("code", "")) != "0":
            raise RuntimeError(f"OKX error: {payload}")
        data = payload.get("data") or []
        if not data:
            break
        parsed = [[int(r[0]), r[1], r[2], r[3], r[4], r[5]] for r in data]
        rows.extend(parsed)
        oldest_ts = min(r[0] for r in parsed)
        if oldest_ts <= start_ms:
            break
        after = oldest_ts
        if len(rows) > 500_000:
            break
    if not rows:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    frame = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    frame = frame.drop_duplicates(subset=["ts"], keep="first")
    frame["timestamp"] = pd.to_datetime(frame["ts"], unit="ms", utc=True)
    frame = frame.set_index("timestamp").sort_index()
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    if start_ts.tzinfo is None:
        start_ts = start_ts.tz_localize("UTC")
    else:
        start_ts = start_ts.tz_convert("UTC")
    if end_ts.tzinfo is None:
        end_ts = end_ts.tz_localize("UTC")
    else:
        end_ts = end_ts.tz_convert("UTC")
    frame = frame[(frame.index >= start_ts) & (frame.index <= end_ts)]
    for c in ["open", "high", "low", "close", "volume"]:
        frame[c] = pd.to_numeric(frame[c], errors="coerce")
    frame = frame.dropna(subset=["open", "high", "low", "close"])
    return _normalize_ohlcv(frame[["open", "high", "low", "close", "volume"]])


def _compare_source(
    *,
    source_name: str,
    ref: pd.DataFrame,
    candidate: pd.DataFrame,
) -> dict[str, Any]:
    best: Optional[dict[str, Any]] = None
    for shift_h in range(-2, 3):
        cmp_df = candidate.copy()
        if shift_h != 0:
            cmp_df.index = cmp_df.index + pd.Timedelta(hours=int(shift_h))
        joined = ref[["close"]].rename(columns={"close": "ref_close"}).join(
            cmp_df[["close"]].rename(columns={"close": "cmp_close"}),
            how="inner",
        )
        joined = joined.dropna()
        if len(joined) < 200:
            continue
        r_ref = np.log(joined["ref_close"]).diff().dropna()
        r_cmp = np.log(joined["cmp_close"]).diff().dropna()
        common = pd.concat([r_ref.rename("ref"), r_cmp.rename("cmp")], axis=1).dropna()
        if len(common) < 150:
            continue
        corr = float(common["ref"].corr(common["cmp"]))
        ret_mae_bps = float(np.mean(np.abs(common["cmp"] - common["ref"])) * 10_000.0)
        sign_match = float(np.mean(np.sign(common["ref"]) == np.sign(common["cmp"])))
        level_spread = (joined["cmp_close"] / joined["ref_close"] - 1.0) * 10_000.0
        level_mae_bps = float(np.mean(np.abs(level_spread)))
        score = float(corr - 0.0001 * ret_mae_bps - 0.00003 * level_mae_bps + 0.05 * sign_match)
        row = {
            "source": source_name,
            "best_shift_hours": int(shift_h),
            "overlap_bars": int(len(joined)),
            "overlap_returns": int(len(common)),
            "return_corr": corr,
            "return_mae_bps": ret_mae_bps,
            "level_mae_bps": level_mae_bps,
            "sign_match_frac": sign_match,
            "score": score,
            "overlap_start": str(joined.index.min()),
            "overlap_end": str(joined.index.max()),
        }
        if best is None or float(row["score"]) > float(best["score"]):
            best = row
    if best is None:
        return {
            "source": source_name,
            "best_shift_hours": None,
            "overlap_bars": 0,
            "overlap_returns": 0,
            "return_corr": None,
            "return_mae_bps": None,
            "level_mae_bps": None,
            "sign_match_frac": None,
            "score": float("-inf"),
            "overlap_start": None,
            "overlap_end": None,
        }
    return best


def _parse_windows(path: Path) -> list[WindowSpec]:
    raw = json.loads(path.read_text())
    out: list[WindowSpec] = []
    for item in raw:
        start = pd.Timestamp(item["start"])
        end = pd.Timestamp(item["end"])
        if start.tz is None:
            start = start.tz_localize("UTC")
        else:
            start = start.tz_convert("UTC")
        if end.tz is None:
            end = end.tz_localize("UTC")
        else:
            end = end.tz_convert("UTC")
        out.append(
            WindowSpec(
                year=int(item.get("year", start.year)),
                start=start.to_pydatetime(),
                end=end.to_pydatetime(),
                length_days=int(item.get("length_days", (end - start).days)),
            )
        )
    return out


def _load_params(path: Path, strategy: str) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if strategy in payload and isinstance(payload[strategy], dict):
        return dict(payload[strategy])
    raise ValueError(f"params file missing '{strategy}' object: {path}")


def _evaluate_windows(
    *,
    label: str,
    bars: pd.DataFrame,
    windows: list[WindowSpec],
    params: dict[str, Any],
    strategy_name: str,
    symbol: str,
    cfg: BacktestConfig,
    out_dir: Path,
    prewarm_days: int,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    runs_root = out_dir / f"runs_{label}"
    runs_root.mkdir(parents=True, exist_ok=True)
    for i, w in enumerate(windows, start=1):
        load_start = pd.Timestamp(w.start).tz_convert("UTC") - pd.Timedelta(days=int(prewarm_days))
        end = pd.Timestamp(w.end).tz_convert("UTC")
        slice_df = bars[(bars.index >= load_start) & (bars.index < end)].copy()
        if len(slice_df) < 300:
            continue
        run_dir = runs_root / f"w{i:02d}_{w.year}_{pd.Timestamp(w.start).date()}_{pd.Timestamp(w.end).date()}"
        run_dir.mkdir(parents=True, exist_ok=True)
        strat = build_strategy(
            name=str(strategy_name),
            params_path=None,
            symbols=[symbol],
            fast_window=10,
            slow_window=30,
            params=params,
        )
        run_derivatives_backtest(
            bars_by_symbol={symbol: slice_df},
            strategy=strat,
            cfg=cfg,
            run_dir=run_dir,
            debug=False,
            score_start=w.start,
            score_end=w.end,
            no_trade_before=w.start,
        )
        metrics = json.loads((run_dir / "metrics.json").read_text())
        weekly_summary, weekly_rows = rolling_window_summary(
            run_dir=run_dir,
            window=timedelta(days=7),
            step=timedelta(days=7),
            benchmark="spy.us",
        )
        weeks_positive = int(sum(1 for r in weekly_rows if float(r.get("return", 0.0)) > 0.0))
        weeks_total = int(weekly_summary.windows)
        spy = spy_total_return(start=w.start, end=w.end)
        spy_ret = float(spy.total_return) if spy is not None else None
        total_return = float(metrics.get("total_return", 0.0))
        row = {
            "window_index": int(i),
            "window_year": int(w.year),
            "window_start": str(pd.Timestamp(w.start).tz_convert("UTC")),
            "window_end": str(pd.Timestamp(w.end).tz_convert("UTC")),
            "total_return": total_return,
            "max_drawdown": float(metrics.get("max_drawdown", 0.0)),
            "trades": int(metrics.get("trade_count", metrics.get("trades", 0) or 0)),
            "weeks_total": int(weeks_total),
            "weeks_positive": int(weeks_positive),
            "weekly_positive_frac": float(weeks_positive / max(1, weeks_total)),
            "mean_weekly_return": float(weekly_summary.mean_return),
            "beat_spy_weekly_frac": float(weekly_summary.beat_benchmark_frac or 0.0),
            "spy_total_return": spy_ret,
            "alpha_vs_spy": (float(total_return - spy_ret) if spy_ret is not None else None),
            "run_dir": str(run_dir),
        }
        row["run_profitable_and_beat_spy"] = bool(
            (row["total_return"] > 0.0)
            and (row["alpha_vs_spy"] is not None)
            and (float(row["alpha_vs_spy"]) > 0.0)
        )
        rows.append(row)

    rows_csv = out_dir / f"{label}_window_rows.csv"
    pd.DataFrame(rows).to_csv(rows_csv, index=False)

    if not rows:
        return {
            "label": label,
            "runs": 0,
            "rows_csv": str(rows_csv),
        }

    runs = len(rows)
    rets = np.array([float(r["total_return"]) for r in rows], dtype=float)
    dds = np.array([float(r["max_drawdown"]) for r in rows], dtype=float)
    weekly_gate_runs = int(sum(1 for r in rows if float(r["weekly_positive_frac"]) >= 0.70))
    profitable_runs = int(sum(1 for r in rows if float(r["total_return"]) > 0.0))
    beat_spy_runs = int(
        sum(1 for r in rows if (r["alpha_vs_spy"] is not None and float(r["alpha_vs_spy"]) > 0.0))
    )
    profitable_and_beat_spy_runs = int(sum(1 for r in rows if bool(r["run_profitable_and_beat_spy"])))
    weeks_total = int(sum(int(r["weeks_total"]) for r in rows))
    weeks_positive = int(sum(int(r["weeks_positive"]) for r in rows))

    return {
        "label": label,
        "runs": int(runs),
        "profitable_runs": int(profitable_runs),
        "profitable_run_frac": float(profitable_runs / max(1, runs)),
        "beat_spy_runs": int(beat_spy_runs),
        "beat_spy_run_frac": float(beat_spy_runs / max(1, runs)),
        "profitable_and_beat_spy_runs": int(profitable_and_beat_spy_runs),
        "profitable_and_beat_spy_run_frac": float(profitable_and_beat_spy_runs / max(1, runs)),
        "weekly_gate_runs": int(weekly_gate_runs),
        "weekly_gate_run_frac": float(weekly_gate_runs / max(1, runs)),
        "weeks_total": int(weeks_total),
        "weeks_positive": int(weeks_positive),
        "aggregate_weekly_positive_frac": float(weeks_positive / max(1, weeks_total)),
        "mean_total_return": float(rets.mean()),
        "median_total_return": float(np.median(rets)),
        "worst_total_return": float(rets.min()),
        "best_total_return": float(rets.max()),
        "mean_max_drawdown": float(dds.mean()),
        "worst_max_drawdown": float(dds.min()),
        "rows_csv": str(rows_csv),
    }


def main() -> int:
    args = _parse_args()
    now = datetime.now(timezone.utc)
    overlap_start = _to_dt_utc(str(args.overlap_start))
    overlap_end = _to_dt_utc(str(args.overlap_end)) if str(args.overlap_end).strip() else now
    if overlap_end <= overlap_start:
        raise ValueError("overlap_end must be after overlap_start")

    out_dir = Path(args.out_root) / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/5] Fetching Coinbase overlap bars {overlap_start.isoformat()} -> {overlap_end.isoformat()}")
    coinbase_overlap = _fetch_coinbase_hourly(overlap_start, overlap_end)
    if coinbase_overlap.empty:
        raise RuntimeError("no Coinbase overlap bars fetched")
    coinbase_overlap.to_csv(out_dir / "coinbase_overlap_1h.csv")

    print("[2/5] Fetching external overlap bars (Deribit, OKX)")
    deribit_overlap = _fetch_deribit_hourly(overlap_start, overlap_end)
    okx_overlap = _fetch_okx_hourly(overlap_start, overlap_end)
    deribit_overlap.to_csv(out_dir / "deribit_overlap_1h.csv")
    okx_overlap.to_csv(out_dir / "okx_overlap_1h.csv")

    similarity_rows = [
        _compare_source(source_name="deribit_btc_perpetual", ref=coinbase_overlap, candidate=deribit_overlap),
        _compare_source(source_name="okx_btc_usdt_swap", ref=coinbase_overlap, candidate=okx_overlap),
    ]
    similarity_rows = sorted(similarity_rows, key=lambda r: float(r["score"]), reverse=True)
    similarity_json = out_dir / "source_similarity.json"
    similarity_json.write_text(json.dumps(similarity_rows, indent=2))
    auto_best_source = similarity_rows[0]["source"]
    best_source = str(args.force_source)
    if best_source == "auto":
        best_source = str(auto_best_source)

    print(f"[3/5] Best overlap source = {auto_best_source}; selected source = {best_source}")

    launch_windows = _parse_windows(Path(args.launch_windows_json))
    random_windows = _parse_windows(Path(args.random_windows_json))
    all_windows = launch_windows + random_windows
    min_start = min(pd.Timestamp(w.start).tz_convert("UTC") for w in all_windows)
    max_end = max(pd.Timestamp(w.end).tz_convert("UTC") for w in all_windows)
    fetch_start = (min_start - pd.Timedelta(days=int(args.prewarm_days))).to_pydatetime()
    fetch_end = max_end.to_pydatetime()

    print(f"[4/5] Fetching full chosen-source bars {fetch_start.isoformat()} -> {fetch_end.isoformat()}")
    if best_source == "deribit_btc_perpetual":
        chosen = _fetch_deribit_hourly(fetch_start, fetch_end)
    elif best_source == "okx_btc_usdt_swap":
        chosen = _fetch_okx_hourly(fetch_start, fetch_end)
    else:
        raise RuntimeError(f"unsupported best source: {best_source}")
    if chosen.empty:
        raise RuntimeError("chosen source returned no bars for evaluation range")
    chosen_path = out_dir / f"{best_source}_full_1h.csv"
    chosen.to_csv(chosen_path)

    strategy_name = str(args.strategy).strip()
    params = _load_params(Path(args.params_file), strategy_name)
    cfg = BacktestConfig(
        symbols=[str(args.symbol)],
        initial_cash=float(args.initial_cash),
        max_position_notional_usd=float(args.max_notional),
        slippage_bps=float(args.slippage_bps),
        taker_fee_bps=float(args.taker_fee_bps),
        fixed_fee_per_contract_usd=float(args.fixed_fee_per_contract_usd),
        contract_size_units=float(args.contract_size_units),
        allow_short=bool(args.allow_short),
        maintenance_margin_rate=0.05,
        liquidation_fee_rate=0.005,
    )

    print("[5/5] Evaluating cand08 on launch + random-year windows")
    launch_summary = _evaluate_windows(
        label="launch_windows",
        bars=chosen,
        windows=launch_windows,
        params=params,
        strategy_name=strategy_name,
        symbol=str(args.symbol),
        cfg=cfg,
        out_dir=out_dir,
        prewarm_days=int(args.prewarm_days),
    )
    random_summary = _evaluate_windows(
        label="random_windows",
        bars=chosen,
        windows=random_windows,
        params=params,
        strategy_name=strategy_name,
        symbol=str(args.symbol),
        cfg=cfg,
        out_dir=out_dir,
        prewarm_days=int(args.prewarm_days),
    )

    result = {
        "out_dir": str(out_dir),
        "similarity_json": str(similarity_json),
        "best_source": best_source,
        "auto_best_source": auto_best_source,
        "chosen_source_csv": str(chosen_path),
        "config": {
            "strategy": strategy_name,
            "params_file": str(args.params_file),
            "symbol": str(args.symbol),
            "initial_cash": float(args.initial_cash),
            "max_notional": float(args.max_notional),
            "slippage_bps": float(args.slippage_bps),
            "taker_fee_bps": float(args.taker_fee_bps),
            "fixed_fee_per_contract_usd": float(args.fixed_fee_per_contract_usd),
            "contract_size_units": float(args.contract_size_units),
            "allow_short": bool(args.allow_short),
            "prewarm_days": int(args.prewarm_days),
        },
        "launch_summary": launch_summary,
        "random_summary": random_summary,
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
