# Perp Quant Fusion `mut_05` (Fee 12.5 bps) - Multiyear Validation

Date: 2026-02-12  
Status: current strongest candidate for `>=70%` profitable 180d windows under baseline costs.

## Objective

Find a derivatives strategy (Coinbase data, BTC/USD signal stream, derivatives engine) that is profitable in at least 70% of multiyear 180-day windows using pessimistic transaction costs:

- `slippage_bps=1.5` (per side)
- `taker_fee_bps=12.5` (per side)
- `initial_cash=500`
- `max_position_notional_usd=2500`

## Selected Strategy

- Strategy: `perp_quant_fusion`
- Preset: `strategy_params/perp_quant_fusion_mut05_fee12p5_coinbase_profile.json`
- Candidate source run:
  - `outputs/evaluations/perp_quant_fusion_search_fee12p5_20260212_064647`
  - Candidate id: `mut_05`

## Canonical 10-Window Result (baseline cost 1.5 / 12.5)

From search and revalidation:

- Profitable runs: `7/10` (meets 70% run-profit gate)
- Mean 180d return: `+0.4996%`
- Median 180d return: `+0.3773%`
- Worst 180d return: `-0.5373%`
- Worst max drawdown: `-1.3085%`

Artifacts:

- Search summary:
  - `outputs/evaluations/perp_quant_fusion_search_fee12p5_20260212_064647/summary.json`
- Stress bundle (includes canonical subset):
  - `outputs/evaluations/perp_quant_fusion_mut05_stress_20260212_071821/summary.json`

## Out-of-Sample 12-Window Result (fresh random windows)

Validation run:

- `outputs/evaluations/perp_quant_fusion_oos_validation_20260212_070906`

Result for `mut_05`:

- Profitable runs: `9/12` (`75%`)
- Mean 180d return: `+0.2764%`
- Median 180d return: `+0.4896%`
- Worst 180d return: `-1.0533%`
- Worst max drawdown: `-1.6989%`

## Cost Stress Results (`mut_05`)

From:

- `outputs/evaluations/perp_quant_fusion_mut05_stress_20260212_071821/summary.json`

### Scenario A: `1.5 / 12.5` (baseline)

- Canonical 10: `7/10` profitable
- OOS 12: `9/12` profitable
- All 22 combined: `16/22` profitable (`72.7%`)

### Scenario B: `2.0 / 15.0` (moderate stress)

- Canonical 10: `8/10` profitable
- OOS 12: `9/12` profitable
- All 22 combined: `17/22` profitable (`77.3%`)

### Scenario C: `3.0 / 20.0` (harsh stress)

- Canonical 10: `6/10` profitable
- OOS 12: `7/12` profitable
- All 22 combined: `13/22` profitable (`59.1%`)

## Important Notes

- This candidate satisfies the current run-profit criterion (>=70% profitable 180d windows) on baseline and moderate-stress scenarios.
- It does **not** satisfy a weekly-positive-window >=70% criterion (weekly gate remains low due low-turnover behavior).
- Symbols are configured as `BTC/USD` in the profile so the backtest can use long history; for Coinbase derivatives execution, broker conversion maps to `BTC-PERP` order products.

## Exact Reproduction Commands

### Load preset in TUI

```text
/algorithm perp_quant_fusion
/preset load perp_quant_fusion_mut05_fee12p5_coinbase_profile
```

### CLI baseline backtest (single 180d recent window)

```bash
atlas backtest \
  --market derivatives \
  --symbols BTC/USD \
  --data-source coinbase \
  --bar-timeframe 1H \
  --timeframe 180d \
  --strategy perp_quant_fusion \
  --strategy-params strategy_params/perp_quant_fusion_mut05_fee12p5_coinbase_profile.json \
  --slippage-bps 1.5 \
  --taker-fee-bps 12.5 \
  --allow-short
```

### Paper loop (dry-run first)

```bash
atlas paper \
  --market derivatives \
  --symbols BTC/USD \
  --data-source coinbase \
  --execution-venue coinbase \
  --bar-timeframe 1H \
  --strategy perp_quant_fusion \
  --strategy-params strategy_params/perp_quant_fusion_mut05_fee12p5_coinbase_profile.json \
  --initial-cash 500 \
  --max-position-notional-usd 2500 \
  --slippage-bps 1.5 \
  --taker-fee-bps 12.5 \
  --allow-short \
  --dry-run
```
