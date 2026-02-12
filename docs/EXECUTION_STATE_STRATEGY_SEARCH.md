# Atlas Strategy Search Execution State

Last updated: 2026-02-11

## Current Objective
Optimize `perp_weekly_profit_chase` for maximum practical profitability on random multi-year 180-day windows while keeping weekly consistency high and testing realistic costs.

## Current Primary Candidate
- Stable params file:
  - `strategy_params/perp_weekly_profit_chase_algo_profit_plus_15m.json`
- Source candidate:
  - `outputs/evaluations/perp_weekly_profit_chase_search/riskctrl_wB_pw_20260211_145158_seed2311/candidates/riskctrl_wB_pw_seed2311_rank01_prof3of4_wgate4of4.json`

## Baseline-Cost Full-10 Result (1.5 / 6)
- Run dir:
  - `outputs/evaluations/perp_weekly_profit_chase_candidate_eval/riskctrl_full10_focus_baseline_costs_20260211_154545`
- Leaderboard:
  - `outputs/evaluations/perp_weekly_profit_chase_candidate_eval/riskctrl_full10_focus_baseline_costs_20260211_154545/leaderboard.csv`
- Winner metrics:
  - `profitable_runs = 6/10`
  - `weekly_gate_runs = 8/10` (>=70% weekly-positive gate)
  - `aggregate_weekly_positive_frac = 0.852`
  - `mean_total_return = +0.0431`
  - `median_total_return = +0.2362`
  - `worst_max_drawdown = -0.5817`

## Stress-Cost Snapshot (3 / 10)
- Run dir:
  - `outputs/evaluations/perp_weekly_profit_chase_candidate_eval/riskctrl_coststress_focus_20260211_211650`
- Leaderboard:
  - `outputs/evaluations/perp_weekly_profit_chase_candidate_eval/riskctrl_coststress_focus_20260211_211650/leaderboard.csv`
- Winner remained `algo_profit_plus_15m`, but mean return turned negative (`-0.1165`) under this harsher assumption.

## Loss-Control Experiment State
- Strategy-level risk controls have been implemented in code (`daily/weekly hard stops`, `cooldown`, `trailing/breakeven`, `max-hold`).
- Structured loss-control sweep run:
  - `outputs/evaluations/perp_weekly_profit_chase_candidate_eval/lossguard_sweep_full10_basecost_20260211_214027`
- Outcome:
  - Hard-stop variants reduced risk in some cases but degraded mean return materially.
  - Best full-10 baseline-cost score remained the baseline-like profit-plus profile.

## Key Docs
- Master log:
  - `docs/STRATEGY_RESEARCH_MASTER_LOG.md`
- Detailed round writeup:
  - `docs/algorithms/perp_weekly_profit_chase_loss_control_round_2026-02-11.md`

## Resume Checklist
1. If continuing search, start from `strategy_params/perp_weekly_profit_chase_algo_profit_plus_15m.json`.
2. Use `scripts/optimize_perp_weekly_profit_chase_random_windows.py` for local search.
3. Validate shortlisted files with `scripts/evaluate_perp_weekly_profit_chase_candidates.py` on:
   - baseline costs (1.5/6)
   - stress costs (3/10)
4. Only promote a new file if it improves the chosen tradeoff profile (profit + weekly gate + downside) on full-10 windows.

## Snapshot
```json
{
  "primary_params_file": "strategy_params/perp_weekly_profit_chase_algo_profit_plus_15m.json",
  "baseline_full10_eval": "outputs/evaluations/perp_weekly_profit_chase_candidate_eval/riskctrl_full10_focus_baseline_costs_20260211_154545",
  "stress_eval": "outputs/evaluations/perp_weekly_profit_chase_candidate_eval/riskctrl_coststress_focus_20260211_211650",
  "lossguard_sweep": "outputs/evaluations/perp_weekly_profit_chase_candidate_eval/lossguard_sweep_full10_basecost_20260211_214027",
  "winner_metrics_baseline": {
    "profitable_runs": "6/10",
    "weekly_gate_runs": "8/10",
    "aggregate_weekly_positive_frac": 0.852,
    "mean_total_return": 0.04306415134232977,
    "median_total_return": 0.23619272806498892,
    "worst_max_drawdown": -0.5817481164858711
  }
}
```
