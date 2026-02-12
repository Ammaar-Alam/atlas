# `perp_weekly_profit_chase_algo_low_dd_experimental_15m`

## File
- `strategy_params/perp_weekly_profit_chase_algo_low_dd_experimental_15m.json`

## Purpose
- Experimental lower-drawdown profile from the loss-control refinement round.
- Intended as a downside-control reference point, not current primary deployment candidate.

## Source
- `outputs/evaluations/perp_weekly_profit_chase_search/riskrefine_full10_gb_20260211_185623_seed4201/candidates/riskrefine_full10_gb_seed4201_rank02_prof5of10_wgate6of10.json`

## Parameter Traits
- Lower leverage and tighter loss constraints than the primary profit-plus profile:
  - `base_leverage=2.016`
  - `max_leverage=3.608`
  - `max_margin_utilization=0.4039`
  - `daily_loss_hard_stop=0.025`
  - `weekly_loss_hard_stop=0.025`
  - `max_flips_per_day=1`

## Full-10 Metrics
From:
- `outputs/evaluations/perp_weekly_profit_chase_search/riskrefine_full10_gb_20260211_185623_seed4201/leaderboard.json`

Metrics:
- `profitable_runs = 5/10`
- `weekly_gate_runs (>=70%) = 6/10`
- `mean_total_return = -0.0302`
- `worst_max_drawdown = -0.2580`

## Interpretation
- Drawdown profile is materially better than aggressive profiles in this round.
- Return profile was not acceptable for primary selection (negative mean over tested windows).
