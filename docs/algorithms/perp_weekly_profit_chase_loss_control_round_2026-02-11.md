# Perp Weekly Profit Chase Loss-Control Round (2026-02-11)

## Objective
- Keep the current high weekly-hit-rate profile.
- Improve total return and test explicit downside controls (hard stops, cooldown, trailing/breakeven exits).
- Validate on the same random multi-year 180-day windows (`2016..2025`) under baseline and stressed costs.

## Code Changes
- Strategy risk controls added:
  - `src/atlas/strategies/perp_weekly_profit_chase.py`
  - New params: `daily_loss_hard_stop`, `weekly_loss_hard_stop`, `cooldown_bars_after_exit`, `trailing_stop_atr_mult`, `break_even_trigger_atr`, `max_hold_bars`
  - New behavior: daily/weekly flatten lock, cooldown after exits, trailing/breakeven stop management, max-hold timeout exit.
- Strategy registry wiring:
  - `src/atlas/strategies/registry.py`
- Tuning space + validation updates:
  - `src/atlas/ml/tune.py`
- Tuner mutation/score updates:
  - `scripts/optimize_perp_weekly_profit_chase_random_windows.py`
- New candidate evaluator script:
  - `scripts/evaluate_perp_weekly_profit_chase_candidates.py`

## Key Run Artifacts
- Subset searches (fast discovery):
  - `outputs/evaluations/perp_weekly_profit_chase_search/riskctrl_wA_gb_20260211_140057_seed2201`
  - `outputs/evaluations/perp_weekly_profit_chase_search/riskctrl_wB_gb_20260211_140057_seed2203`
  - `outputs/evaluations/perp_weekly_profit_chase_search/riskctrl_wA_pw_20260211_145158_seed2307`
  - `outputs/evaluations/perp_weekly_profit_chase_search/riskctrl_wB_pw_20260211_145158_seed2311`
- Full 10-window focused evaluation (baseline costs):
  - `outputs/evaluations/perp_weekly_profit_chase_candidate_eval/riskctrl_full10_focus_baseline_costs_20260211_154545/leaderboard.csv`
- Additional full-10 local refinements:
  - `outputs/evaluations/perp_weekly_profit_chase_search/riskrefine_full10_r1_20260211_174251_seed4101`
  - `outputs/evaluations/perp_weekly_profit_chase_search/riskrefine_full10_gb_20260211_185623_seed4201`
  - `outputs/evaluations/perp_weekly_profit_chase_search/riskrefine_full10_lowdd_20260211_200755_seed4301`
- Structured loss-guard sweep:
  - `outputs/evaluations/perp_weekly_profit_chase_candidate_eval/lossguard_sweep_full10_basecost_20260211_214027/leaderboard.csv`
- Cost stress check (3 bps slippage, 10 bps taker fee):
  - `outputs/evaluations/perp_weekly_profit_chase_candidate_eval/riskctrl_coststress_focus_20260211_211650/leaderboard.csv`

## Full-10 Baseline-Cost Winner
- Params:
  - `strategy_params/perp_weekly_profit_chase_algo_profit_plus_15m.json`
- Source candidate:
  - `outputs/evaluations/perp_weekly_profit_chase_search/riskctrl_wB_pw_20260211_145158_seed2311/candidates/riskctrl_wB_pw_seed2311_rank01_prof3of4_wgate4of4.json`
- Metrics (`slippage=1.5`, `fee=6`):
  - `profitable_runs`: `6/10`
  - `weekly_gate_runs (>=70%)`: `8/10`
  - `aggregate_weekly_positive_frac`: `0.852`
  - `mean_total_return`: `+0.0431`
  - `median_total_return`: `+0.2362`
  - `worst_max_drawdown`: `-0.5817`

## Loss-Control Sweep Result
- Best-performing variant in sweep was baseline-like `lossguard_v1`:
  - `strategy_params/perp_weekly_profit_chase_lossguard_v1_baseline_15m.json`
- Hard-stop variants reduced risk in some cases but consistently reduced profitability:
  - Example lower-DD variant:
    - `strategy_params/perp_weekly_profit_chase_algo_low_dd_experimental_15m.json`
    - Full-10 metrics: lower tail DD but negative mean return.

## Stress-Cost Result (3/10)
- Winner remained `algo_profit_plus_15m`, but all tested variants had negative mean return under this harsher assumption.
- Comparative outcome (mean 180d return):
  - `algo_profit_plus_15m`: `-0.1165`
  - `algo_profit_windows_max_15m`: `-0.1354`
  - `algo_growth_balance_15m`: `-0.1289`

## Promoted / Saved Files
- Primary promoted file:
  - `strategy_params/perp_weekly_profit_chase_algo_profit_plus_15m.json`
- Experimental low-drawdown reference:
  - `strategy_params/perp_weekly_profit_chase_algo_low_dd_experimental_15m.json`
- Sweep variants:
  - `strategy_params/perp_weekly_profit_chase_lossguard_v1_baseline_15m.json`
  - `strategy_params/perp_weekly_profit_chase_lossguard_v2_day2_week8_15m.json`
  - `strategy_params/perp_weekly_profit_chase_lossguard_v3_day1p5_week6_15m.json`
  - `strategy_params/perp_weekly_profit_chase_lossguard_v4_day2_week8_trail1_be0p5_15m.json`
  - `strategy_params/perp_weekly_profit_chase_lossguard_v5_day1p5_week5_trail0p8_be0p4_15m.json`
  - `strategy_params/perp_weekly_profit_chase_lossguard_v6_lowerlev_day1p5_week7_15m.json`
  - `strategy_params/perp_weekly_profit_chase_lossguard_v7_lowerlev_trail1_day2_week9_15m.json`
  - `strategy_params/perp_weekly_profit_chase_lossguard_v8_softtrail_day1_week5_15m.json`

## Repro Commands
```bash
# Focused full-10 candidate evaluation at baseline costs
python3 scripts/evaluate_perp_weekly_profit_chase_candidates.py \
  --label riskctrl_full10_focus_baseline_costs \
  --windows-json outputs/evaluations/ab_random_year_windows_20260211_020934/windows.json \
  --params-file outputs/evaluations/perp_weekly_profit_chase_search/riskctrl_wB_pw_20260211_145158_seed2311/candidates/riskctrl_wB_pw_seed2311_rank01_prof3of4_wgate4of4.json \
  --params-file strategy_params/perp_weekly_profit_chase_algo_growth_balance_15m.json \
  --params-file strategy_params/perp_weekly_profit_chase_algo_profit_windows_max_15m.json \
  --slippage-bps 1.5 \
  --taker-fee-bps 6.0

# Stress-cost comparison
python3 scripts/evaluate_perp_weekly_profit_chase_candidates.py \
  --label riskctrl_coststress_focus \
  --windows-json outputs/evaluations/ab_random_year_windows_20260211_020934/windows.json \
  --params-file strategy_params/perp_weekly_profit_chase_algo_profit_plus_15m.json \
  --params-file strategy_params/perp_weekly_profit_chase_algo_growth_balance_15m.json \
  --params-file strategy_params/perp_weekly_profit_chase_algo_profit_windows_max_15m.json \
  --slippage-bps 3.0 \
  --taker-fee-bps 10.0
```
