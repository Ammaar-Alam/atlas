# Response to Researcher 1 (2026-02-15)

## Purpose
This document is a direct response to your recommendations in `researcher1.md`.
It records exactly what was implemented, what was tested, and the measured outcomes under realistic Coinbase fee assumptions.

## Environment and Constraints Used
- Venue: Coinbase derivatives (BTC perpetual)
- Starting equity: `$500`
- Cost model in all tests:
  - `slippage_bps = 1.5`
  - `taker_fee_bps = 10.0`
  - `fixed_fee_per_contract_usd = 0.15`
  - `contract_size_units = 0.01`
  - `coinbase_fee_model = true`
- Market mode: `derivatives`
- Bars: `1H`

## What We Implemented from Your Feedback

### 1) Multi-day rebalance schedule
Implemented support for `rebalance_days_utc` instead of single-day scheduling.
- Code: `src/atlas/strategies/perp_research_vol_momentum.py`
- Wiring: `src/atlas/strategies/registry.py`

### 2) Cost-aware expected-return admission gate
Implemented new gate using:
- `expected_hold_bars`
- `signal_decay_factor`
- `min_net_edge_bps`
- Explicit fixed-fee conversion into effective bps at the intended trade size.

### 3) Regime and consistency filters
Implemented:
- `vol_pctl_low`, `vol_pctl_high`
- `trend_consistency_min`, `trend_consistency_subwindows`

### 4) Equity-aware stop cap
Implemented `max_loss_per_trade_pct` to cap stop width relative to equity-at-risk.

### 5) Lot-size-aware minimum tradable notional
Added guard so candidate exposure must clear one-contract notional when fixed per-contract fees and contract-size lot constraints are active.

### 6) Reviewer profile sets
Created three profile files per your template:
- `strategy_params/perp_research_vol_momentum_reviewer1/conservative.json`
- `strategy_params/perp_research_vol_momentum_reviewer1/balanced.json`
- `strategy_params/perp_research_vol_momentum_reviewer1/aggressive.json`

## Validation Protocol Run

### A) Smoke backtests
Executed representative single-window backtests for all three profiles:
- `outputs/backtests/backtest_20260215_133748_678182_38640_552a`
- `outputs/backtests/backtest_20260215_134451_950175_41308_e5c5`
- `outputs/backtests/backtest_20260215_134516_600730_41535_8caa`

### B) Launch-era robustness set (12 windows)
- Output dir: `outputs/evaluations/strategy_eval/research_vm_r1_launch12_cb10fix015_v2_20260215_184535`
- Source windows: `outputs/evaluations/coinbase_perp_rolling_180d_20260213/windows.json`

### C) Random multi-year windows (10 windows)
- Output dir: `outputs/evaluations/strategy_eval/research_vm_r1_random10_cb10fix015_v2_20260215_184535`
- Source windows: `outputs/evaluations/ab_random_year_windows_20260211_020934/windows.json`

## Results

### Launch windows (12 runs)
From leaderboard:
- **Balanced**: mean return `0.00%`, profitable `0/12`, weekly-gate `0/12` (no trades)
- **Conservative**: mean return `0.00%`, profitable `0/12`, weekly-gate `0/12` (no trades)
- **Aggressive**: mean return `-10.04%`, profitable `0/12`, weekly-gate `0/12`

### Random windows (10 runs)
From leaderboard:
- **Conservative**: mean return `-0.98%`, profitable `1/10`, weekly positive frac `0.036`, weekly-gate `0/10`
- **Balanced**: mean return `-2.57%`, profitable `1/10`, weekly positive frac `0.056`, weekly-gate `0/10`
- **Aggressive**: mean return `-8.94%`, profitable `2/10`, weekly positive frac `0.168`, weekly-gate `0/10`

## Comparison vs Previous Baseline
Previous strategy baseline (pre-researcher1 patch):
- Launch12 mean: `-1.43%`
- Random10 mean: `-8.17%`

After implementing your recommendations:
- Best launch12 mean: `0.00%` (but no-trade behavior)
- Best random10 mean: `-0.98%` (numerically better than prior random mean, but still negative)
- Weekly gate target (>=0.70): not met by any profile

## Conclusions
1. Your recommendations were implemented at code level and executed under realistic fee assumptions.
2. The resulting profiles still **do not satisfy deployment criteria**.
3. Main observed failure modes:
   - conservative/balanced profiles become too restrictive (near no participation),
   - aggressive profile participates but loses across launch and random sets,
   - weekly consistency target remains far below requirement.

## Deployment Status
- **Not ready for live deployment.**

## Proposed Follow-up for Researcher 2
- Keep fixed-fee/lot-size aware admission and cost model.
- Replace static expected-return proxy with calibrated conditional expectancy by regime bucket.
- Retune to avoid no-trade collapse while preserving cost-aware admission.
- Explicitly optimize for weekly consistency and not only mean return.
