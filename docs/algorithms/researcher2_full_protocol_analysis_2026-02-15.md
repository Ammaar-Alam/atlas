# Researcher 2 Full Protocol Analysis (2026-02-15)

## Scope Completed
This run executed the full researcher2 protocol for `perp_research_vol_momentum` under realistic Coinbase cost modeling:
- `slippage_bps=1.5`
- `taker_fee_bps=10.0`
- `coinbase_fee_model=true`
- `fixed_fee_per_contract_usd=0.15`
- `contract_size_units=0.01`
- `initial_cash=500`
- `max_position_notional_usd=5000`

## 1) Pre-flight checks

### 1.1 Rebalance timestamp compatibility
- Strategy/profile uses `rebalance_minute_utc=0` for 1H bars.
- Verified in registry default wiring for this strategy: `src/atlas/strategies/registry.py:651`.

### 1.2 Fixed fee included in gating/debug
- Cost fields are now present in decision debug (`fixed_bps_side`, `cost_side_bps`, `cost_rt_bps`).
- Verified in live debug output from:
  - `outputs/backtests/backtest_20260215_144421_782494_62941_3e19/trade_debug.jsonl`
- Example observed at ~30.5k BTC:
  - `fixed_bps_side=4.9212`
  - `cost_side_bps=16.4212`
  - `cost_rt_bps=32.8424`

### 1.3 Fixed fee bps sanity check
- Formula: `fixed_bps_side = (0.15 / (0.01 * price)) * 10000`
- At `price=50,000`: `fixed_bps_side=3.0 bps` (expected).
- Metrics artifact: `docs/algorithms/researcher2_protocol_metrics_2026-02-15.json`.

### 1.4 Contract quantization checks
- `contracts` and quantized sizing metadata are emitted in debug under `selected_meta`.
- Verified same run above (`contracts=1.0` for executed entries), confirming contract-lot realism.

## 2) Evaluation suites run

### 2.1 Launch-era rolling suite (12 windows)
- Output: `outputs/evaluations/strategy_eval/research_vm_r2_launch12_cb10fix015_full_20260215_192008`
- Result: all profiles no-trade (`0.00%` mean, `0/12` profitable, `0/12` weekly gate).

### 2.2 Random30 stratified suite (required)
- Generated stratified windows (10 high-vol, 10 low/mod-vol, 10 random):
  - `outputs/evaluations/ab_random30_regime_windows_20260215_191958/windows.json`
  - `outputs/evaluations/ab_random30_regime_windows_20260215_191958/summary.json`
- Evaluation output:
  - `outputs/evaluations/strategy_eval/research_vm_r2_random30_cb10fix015_full_20260215_192335`

### 2.3 External perp reality probes (3 probes)
Executed for each R2 profile on launch + random30 windows (external source selected by overlap match):
- Aggressive:
  - `outputs/evaluations/external_perp_probe_r2_aggressive/20260215_193441`
- Balanced:
  - `outputs/evaluations/external_perp_probe_r2_balanced/20260215_193925`
- Conservative:
  - `outputs/evaluations/external_perp_probe_r2_conservative/20260215_193925`

Overlap similarity (aggressive probe):
- `source_similarity.json` chose `okx_btc_usdt_swap` as closest to Coinbase with:
  - return corr `0.9928`
  - sign match `0.9460`

## 3) Primary robustness gates (researcher2)

Computed from `random30` run for best participating profile (`aggressive`) and protocol metrics in:
- `docs/algorithms/researcher2_protocol_metrics_2026-02-15.json`

### Gate 1: mean and median 180d return > 0
- Mean: `+0.12699%` (pass)
- Median: `0.00%` (fail)
- **Status: FAIL**

### Gate 2: weekly positivity
- Overall weekly positive fraction: `0.00667` (target `>=0.70`)
- Windows with weekly positive frac >=0.65: `0.00` (target `>=0.60`)
- **Status: FAIL**

### Gate 3: SPY comparison
- Beat SPY window fraction: `0.10` (target `>=0.60`)
- Mean excess return vs SPY: `-9.312%` per 180d window (target `>=+2%`)
- **Status: FAIL**

## 4) Risk/operational sanity gates

### Gate 4: drawdown
- Median max drawdown: `0.00%` (<15%)
- Worst-window max drawdown: `-3.28%` (> -25%)
- **Status: PASS**

### Gate 5: turnover / hold / fee drag
- Median roundtrips per 180d: `0.0` (<=12, pass)
- Median avg hold: `0.25 days` (>=5 days, fail)
- Median fee fraction of gross PnL: `45.09%` (<35%, fail)
- **Status: FAIL**

### Gate 6: contract realism
- Sub-contract sizing not observed in executed entries.
- Contract-count metadata present in debug and aligned with lot size.
- **Status: PASS**

## 5) Anti-overfit checks

### 5.1 Sensitivity test (8 perturbations, ±20%)
Runs:
- `outputs/evaluations/strategy_eval/research_vm_r2_random30_sensA_cb10fix015_20260215_200537`
- `outputs/evaluations/strategy_eval/research_vm_r2_random30_sensB_cb10fix015_20260215_200537`

Results:
- Mean return > 0 in `8/8` perturbations (passes first criterion)
- Weekly positive fraction >=0.65 in `0/8` perturbations (fails second criterion)
- **Status: FAIL**

### 5.2 One-shot holdout (60/40 split)
Selection:
- `outputs/evaluations/strategy_eval/research_vm_r2_random30_selection60_cb10fix015_20260215_201840`
- Winner: aggressive

Holdout:
- `outputs/evaluations/strategy_eval/research_vm_r2_random30_holdout40_cb10fix015_20260215_202427`
- Holdout mean return: `-0.0569%`
- Holdout weekly positive fraction: `0.00333`
- **Status: FAIL**

### 5.3 Cross-venue direction stability
- Mean return sign (aggressive):
  - Coinbase proxy random30: positive (`+0.12699%`)
  - External (OKX) random windows: positive (`+0.04584%`)
- Entry-side distribution:
  - Coinbase: long 4 / short 2 (long fraction `0.667`)
  - External OKX: long 2 / short 1 (long fraction `0.667`)
- **Status: PASS** (sign consistent; no side inversion)

## 6) Final protocol verdict
- Full researcher2 protocol was executed.
- Structural implementation is correct and preflight checks pass.
- Robustness/performance gates are not met.
- **Deployment status: NOT READY**.

## 7) Likely profitability assessment
With current researcher2 profile set and realistic Coinbase fees, the algorithm is **not likely to be reliably profitable** in live deployment under the required standards. The dominant failure mode is not catastrophic drawdown; it is low-quality participation and weak weekly edge after full cost modeling.
