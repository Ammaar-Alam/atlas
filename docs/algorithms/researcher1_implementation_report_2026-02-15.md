# Researcher 1 Implementation Report (2026-02-15)

## Scope
Implemented researcher1 feedback on `perp_research_vol_momentum`, added requested profiles, and ran launch/random robustness evaluations under realistic Coinbase fee model.

## Implemented Changes

### Strategy code
- Updated: `src/atlas/strategies/perp_research_vol_momentum.py`
- Added features and gates from reviewer direction:
  - cost-aware edge gate with expected-return proxy:
    - `expected_hold_bars`
    - `signal_decay_factor`
    - `min_net_edge_bps`
  - trend consistency filter:
    - `trend_consistency_min`
    - `trend_consistency_subwindows`
  - volatility percentile regime filter:
    - `vol_pctl_low`
    - `vol_pctl_high`
  - multi-day rebalance support:
    - `rebalance_days_utc`
  - equity-aware stop cap:
    - `max_loss_per_trade_pct`
- Also implemented tradable lot guard so minimum target notional respects contract lot size when fixed per-contract fees are active.

### Registry wiring
- Updated: `src/atlas/strategies/registry.py`
- Added parsing and wiring for new params, including list parsing for `rebalance_days_utc`.

### TUI summary clarity (from same session)
- Updated: `src/atlas/tui/app.py`
- Backtest summary now shows both:
  - `requested_window`
  - `observed_data_window`

## Added Profile Files (as requested)
- `strategy_params/perp_research_vol_momentum_reviewer1/conservative.json`
- `strategy_params/perp_research_vol_momentum_reviewer1/balanced.json`
- `strategy_params/perp_research_vol_momentum_reviewer1/aggressive.json`

## Validation Runs

### Quick smoke runs (single 180d launch-era range)
- Conservative: `outputs/backtests/backtest_20260215_133748_678182_38640_552a`
- Balanced: `outputs/backtests/backtest_20260215_134451_950175_41308_e5c5`
- Aggressive: `outputs/backtests/backtest_20260215_134516_600730_41535_8caa`

### Full launch/random evaluations (3 profiles)
- Launch windows (BTC-PERP):
  - `outputs/evaluations/strategy_eval/research_vm_r1_launch12_cb10fix015_v2_20260215_184535`
- Random windows (BTC/USD proxy):
  - `outputs/evaluations/strategy_eval/research_vm_r1_random10_cb10fix015_v2_20260215_184535`

## Results Summary

### Launch windows (12 runs)
From `.../research_vm_r1_launch12_cb10fix015_v2_.../leaderboard.csv`:
- Balanced: mean return `0.00%`, profitable runs `0/12`, weekly gate runs `0/12` (no trading)
- Conservative: mean return `0.00%`, profitable runs `0/12`, weekly gate runs `0/12` (no trading)
- Aggressive: mean return `-10.04%`, profitable runs `0/12`, weekly gate runs `0/12`

### Random windows (10 runs)
From `.../research_vm_r1_random10_cb10fix015_v2_.../leaderboard.csv`:
- Conservative: mean return `-0.98%`, profitable runs `1/10`, weekly gate runs `0/10`, weekly positive frac `0.036`
- Balanced: mean return `-2.57%`, profitable runs `1/10`, weekly gate runs `0/10`, weekly positive frac `0.056`
- Aggressive: mean return `-8.94%`, profitable runs `2/10`, weekly gate runs `0/10`, weekly positive frac `0.168`

## Before vs After (against previous baseline)
Previous baseline (v1 profile):
- Launch12 mean return: `-1.43%` (`outputs/evaluations/strategy_eval/research_vm_launch12_20260215_172307`)
- Random10 mean return: `-8.17%` (`outputs/evaluations/strategy_eval/research_vm_random10_20260215_172307`)

After researcher1 implementation:
- Best launch12 mean return: `0.00%` (balanced/conservative; effectively no-trade)
- Best random10 mean return: `-0.98%` (conservative), but weekly consistency gate still fails badly.

Interpretation:
- Return improved numerically on random mean for conservative profile vs prior baseline.
- However, all profiles fail profitability/weekly-gate criteria.
- Conservative and balanced became too restrictive (near no-trade behavior).

## Deployment Readiness
Not ready for deployment.

Reasons:
- No candidate meets robust profitability criteria.
- Weekly positive gate (`>=0.70`) not met by any candidate.
- Launch-era behavior either no-trade or negative.

## Recommended Next Step for Researcher 2
- Keep cost-aware lot sizing logic.
- Loosen admission model from static expected-return proxy to calibrated conditional expectancy by regime bucket.
- Explicitly optimize trade frequency vs fixed-fee burden for `$500` account (lot-size constrained).
- Retune around a target of fewer but higher-quality entries while preserving non-zero participation.
