# Perp Weekly Profit Chase Tuning Round (2026-02-11)

## Objective
Tune around Algorithms A/B to maximize net-profit consistency across random multi-year 180-day windows, while keeping weekly hit-rate high.

## Data / Evaluation Frame
- Windows: one random 180-day window per year (2016..2025), seed `20260211`
- Window file: `outputs/evaluations/ab_random_year_windows_20260211_020934/windows.json`
- Market mode: `derivatives`
- Symbol for long history: `BTC/USD`
- Bars: Coinbase `15Min`
- Baseline costs for tuning and selection:
  - `slippage_bps=1.5`
  - `taker_fee_bps=6`
- Risk envelope:
  - `initial_cash=500`
  - `max_position_notional_usd=2500`
  - `allow_short=true`

## Search Batches
1. W2 (B-centered, regime subset `2,4,5,9`)
- Run: `outputs/evaluations/perp_weekly_profit_chase_search/w2_b_20260211_024734_seed307`

2. W3 (A-centered, regime subset `3,7,9,10`)
- Run: `outputs/evaluations/perp_weekly_profit_chase_search/w3_mix_20260211_024734_seed409`

3. W1-fast (A-centered, regime subset `1,3,6,8`)
- Run: `outputs/evaluations/perp_weekly_profit_chase_search/w1_a_fast_20260211_034654_seed211`

## Full 10-Window Finalist Validation
Run root:
- `outputs/evaluations/perp_weekly_profit_chase_finalists_full10_20260211`

Leaderboard:
- `outputs/evaluations/perp_weekly_profit_chase_finalists_full10_20260211/leaderboard.csv`

Top outcomes:

1) `c03_w2_b_seed307_rank01_prof4of4_wgate4of4`
- Params file: `outputs/evaluations/perp_weekly_profit_chase_search/w2_b_20260211_024734_seed307/candidates/w2_b_seed307_rank01_prof4of4_wgate4of4.json`
- `profitable_runs`: `6/10`
- `weekly_gate_runs (>=70%)`: `8/10`
- `aggregate_weekly_positive_frac`: `82.8%`
- `mean_total_return`: `-0.58%`
- `median_total_return`: `+7.20%`
- `worst_max_drawdown`: `-55.37%`

2) `c07_w1_a_fast_seed211_rank01_prof2of4_wgate3of4`
- Params file: `outputs/evaluations/perp_weekly_profit_chase_search/w1_a_fast_20260211_034654_seed211/candidates/w1_a_fast_seed211_rank01_prof2of4_wgate3of4.json`
- `profitable_runs`: `5/10`
- `weekly_gate_runs (>=70%)`: `8/10`
- `aggregate_weekly_positive_frac`: `85.2%`
- `mean_total_return`: `+2.51%`
- `median_total_return`: `+1.22%`
- `worst_max_drawdown`: `-48.18%`

## Additional Exploitation Round (Full 10 Windows)
Run:
- `outputs/evaluations/perp_weekly_profit_chase_search/full10_exploit_c03_20260211_070415_seed777`

Result:
- No mutation beat the incumbent C03 on the primary score.
- Top candidate remained base `cand_000` (same C03 params).

## Moderate-Cost Check (3 bps / 10 bps)
Run:
- `outputs/evaluations/perp_weekly_profit_chase_costcheck_20260211/summary.json`

Results:
- `c03_best_prof_runs`: profitable `4/10`, weekly-gate `8/10`, mean return `-14.96%`, worst DD `-73.60%`
- `c07_best_mean_return`: profitable `5/10`, weekly-gate `8/10`, mean return `-12.24%`, worst DD `-67.48%`

Interpretation:
- Under stricter costs, both degrade materially.
- `c07` degrades less in this test.

## Promoted Tuned Parameter Files
### 1) Profit-Window Maximizer
- `strategy_params/perp_weekly_profit_chase_algo_profit_windows_max_15m.json`
- Source: C03 winner from full-10-window validation
- Strength: best profitable-window count (`6/10`) at baseline costs

### 2) Growth-Balance Profile
- `strategy_params/perp_weekly_profit_chase_algo_growth_balance_15m.json`
- Source: C07 finalist
- Strength: better average return and lower drawdown among finalists (`mean +2.51%`, `worst DD -48.18%`) at baseline costs

## Current Recommendation
- If optimizing for *count of profitable 180-day windows*: use `strategy_params/perp_weekly_profit_chase_algo_profit_windows_max_15m.json`.
- If optimizing for *return profile quality and cost resilience*: use `strategy_params/perp_weekly_profit_chase_algo_growth_balance_15m.json`.
