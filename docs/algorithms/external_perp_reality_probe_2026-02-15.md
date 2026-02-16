# External Perp Reality Probe (cand08) — 2026-02-15

## Goal
- Identify which external BTC perpetual feed is closest to Coinbase `BTC-PERP` (launch overlap).
- Evaluate `perp_trend_vol_guard` `cand08` on that external feed to check overfit risk with realistic Coinbase costs.

## Strategy Under Test
- strategy: `perp_trend_vol_guard`
- params: `strategy_params/trend_guard_manual_chase_grid2/cand08.json`
- sizing/costs:
  - `initial_cash=500`
  - `max_notional=5000`
  - `slippage_bps=1.5`
  - `taker_fee_bps=10.0`
  - `fixed_fee_per_contract_usd=0.15`
  - `contract_size_units=0.01`
  - `allow_short=true`

## Script
- `scripts/probe_external_perp_reality.py`

## Source Similarity Test (Coinbase overlap)
Overlap used: `2025-07-18` to `2026-02-15` on 1H candles.

Run:
- `outputs/evaluations/external_perp_probe/20260215_165618/source_similarity.json`

Result:
- `okx_btc_usdt_swap`
  - return corr: `0.9927747527`
  - return MAE: `3.43998 bps`
  - level MAE: `7.52116 bps`
  - sign match: `0.94599`
- `deribit_btc_perpetual`
  - return corr: `0.9927098602`
  - return MAE: `3.46942 bps`
  - level MAE: `3.67792 bps`
  - sign match: `0.94360`

Automatic selection from overlap score: `okx_btc_usdt_swap`.

## cand08 on Closest Source (OKX)
Run:
- `outputs/evaluations/external_perp_probe/20260215_165618/result.json`

Launch windows (12 x 180d, 2025+):
- profitable runs: `5/12`
- weekly gate runs (`>=0.70`): `0/12`
- aggregate weekly positive frac: `0.3067`
- mean total return: `-7.10%`

Random windows (available subset on OKX history, 6 windows):
- profitable runs: `0/6`
- weekly gate runs: `0/6`
- aggregate weekly positive frac: `0.0867`
- mean total return: `-27.21%`

## Control: cand08 on Deribit (longer history)
Run:
- `outputs/evaluations/external_perp_probe/20260215_165955/result.json`

Launch windows (12 x 180d):
- profitable runs: `12/12`
- weekly gate runs (`>=0.70`): `4/12`
- aggregate weekly positive frac: `0.6567`
- mean total return: `+21.95%`

Random windows (available subset on Deribit history, 8 windows from 2018+):
- profitable runs: `0/8`
- weekly gate runs: `0/8`
- aggregate weekly positive frac: `0.0576`
- mean total return: `-25.48%`

## Interpretation
- `cand08` remains highly regime-sensitive:
  - can look very strong on launch-era windows,
  - fails decisively on pre-launch multi-year perp windows.
- This confirms high overfit risk and does not meet deployment robustness criteria.

## Best Current Status
- Under strict realistic Coinbase fee model and multi-year random windows, no candidate currently satisfies:
  - durable profitability,
  - weekly-positive gate,
  - robust SPY outperformance.
