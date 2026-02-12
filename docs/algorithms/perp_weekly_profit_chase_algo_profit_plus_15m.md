# `perp_weekly_profit_chase_algo_profit_plus_15m`

## File
- `strategy_params/perp_weekly_profit_chase_algo_profit_plus_15m.json`
- `strategy_params/perp_weekly_profit_chase_algo_profit_plus_15m_coinbase_profile.json` (same params + TUI session profile)

## Strategy
- `perp_weekly_profit_chase`
- Market/data context used for tuning/validation:
  - `market=derivatives`
  - `symbol=BTC/USD`
  - `data_source=coinbase`
  - `bar_timeframe=15Min`
  - `initial_cash=500`
  - `max_position_notional_usd=2500`

## Parameter Summary
- Core behavior:
  - ORB breakout with weekly heartbeat entry logic.
  - Leverage-mode sizing with margin-utilization and liquidation-buffer caps.
- Key tuned values:
  - `base_leverage=2.601`
  - `max_leverage=3.919`
  - `max_margin_utilization=0.4176`
  - `stop_atr_mult=2.348`
  - `max_hold_bars=64`
  - `daily_loss_hard_stop=0.0`
  - `weekly_loss_hard_stop=0.0`
  - `trailing_stop_atr_mult=0.0`

## Baseline-Cost Full-10 Metrics
From:
- `outputs/evaluations/perp_weekly_profit_chase_candidate_eval/riskctrl_full10_focus_baseline_costs_20260211_154545/leaderboard.csv`

Metrics:
- `profitable_runs = 6/10`
- `weekly_gate_runs (>=70%) = 8/10`
- `aggregate_weekly_positive_frac = 0.852`
- `mean_total_return = +0.0431`
- `median_total_return = +0.2362`
- `worst_total_return = -0.5104`
- `worst_max_drawdown = -0.5817`

## Stress-Cost Metrics (3/10)
From:
- `outputs/evaluations/perp_weekly_profit_chase_candidate_eval/riskctrl_coststress_focus_20260211_211650/leaderboard.csv`

Metrics:
- `profitable_runs = 5/10`
- `weekly_gate_runs (>=70%) = 8/10`
- `aggregate_weekly_positive_frac = 0.824`
- `mean_total_return = -0.1165`
- `median_total_return = +0.0375`
- `worst_total_return = -0.7112`
- `worst_max_drawdown = -0.7695`

## Interpretation
- This is currently the best-performing full-10 baseline-cost profile among tested variants.
- It improved mean return versus prior growth-balance profile in this round, but downside tails remain large.
