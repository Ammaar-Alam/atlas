# Atlas Strategy Research Master Log

Last updated: 2026-02-11

## Objective
- Find deployable algorithms under pessimistic assumptions for a small account (`initial_cash=500`) with:
1. `max_drawdown >= -20%`
2. `weekly_positive_frac >= 70%`
3. `beat_spy = true`
4. `stress_pass_frac >= threshold`

## Key Fix Applied (Critical)
- Benchmark alignment bug fixed in `src/atlas/evaluation/orchestrator.py`.
- Previous runs compared strategy return on truncated available data (e.g., ~2025-07 to 2026-01 for perps) against SPY over the full configured window (e.g., 2022-2026), creating unfair alpha.
- New behavior infers benchmark window from realized `equity_curve.csv` timestamps for each candidate baseline run and writes that interval to `benchmark.json`.

## Major Completed Runs (Newest)
1. Derivatives matrix after benchmark fix:
- `outputs/evaluations/evaluate_all_20260210_140404_248608_86950_c859`
- `total=20`, `validated=14`, `passed=0`
- Best weekly improved but no all-gates passer.

2. Focused curated search (`perp_weekly_profit_chase`, BTC-PERP, 15m):
- 30 new curated parameter variants generated:
  - `strategy_params/perp_weekly_profit_chase_curated_01_15m.json` ... `strategy_params/perp_weekly_profit_chase_curated_30_15m.json`
- Evaluation run:
  - `outputs/evaluations/evaluate_all_20260210_160518_327211_29307_31b3`
  - `total=30`, `validated=10`, `passed=2`

3. Final promotion run (stable param filenames):
- `outputs/evaluations/evaluate_all_20260210_162617_071340_36010_81a5`
- `total=2`, `validated=2`, `passed=2`
- Confirms both promoted algorithm files still pass all gates.

## Passing Candidates (All Gates True)
### Passer A (winner)
- Candidate ID:
  - `perp_weekly_profit_chase|derivatives|BTC-PERP|15Min|coinbase|perp_weekly_profit_chase_algo_a_intraday_winner_15m.json`
- Params:
  - `strategy_params/perp_weekly_profit_chase_algo_a_intraday_winner_15m.json`
- Metrics:
  - `total_return = 0.24772730545334376`
  - `spy_total_return = 0.08658657063641284`
  - `alpha_vs_spy = 0.16114073481693092`
  - `max_drawdown = -0.07252932336278828`
  - `weekly_positive_frac = 0.9565217391304348`
  - `stress_pass_frac = 1.0`
  - `sharpe_daily = 2.3407446634894673`
  - `trades = 134`

### Passer B (secondary)
- Candidate ID:
  - `perp_weekly_profit_chase|derivatives|BTC-PERP|15Min|coinbase|perp_weekly_profit_chase_algo_b_growth_15m.json`
- Params:
  - `strategy_params/perp_weekly_profit_chase_algo_b_growth_15m.json`
- Metrics:
  - `total_return = 0.1991151819485728`
  - `spy_total_return = 0.08658657063641284`
  - `alpha_vs_spy = 0.11252861131215997`
  - `max_drawdown = -0.1210599823741797`
  - `weekly_positive_frac = 0.8695652173913043`
  - `stress_pass_frac = 1.0`
  - `sharpe_daily = 1.7187160015871645`
  - `trades = 132`

## Recommended Two-Algorithm Set (Current)
1. Intraday / active strategy:
- `perp_weekly_profit_chase_algo_a_intraday_winner_15m.json` (winner; best robust score and profit)

2. Portfolio-growth / lower-risk alternative:
- `perp_weekly_profit_chase_algo_b_growth_15m.json` (still passes all gates; lower return and higher DD than A, but valid fallback)

## Current Best Pick
- Winner: `perp_weekly_profit_chase_algo_a_intraday_winner_15m.json`
- Selection basis: highest `robust_score` among all gate passers in latest run.

## Repro Commands
```bash
# Re-run final promoted pair
.venv/bin/python -m atlas.cli evaluate-all \
  --strategies perp_weekly_profit_chase \
  --strategy-params-prefixes perp_weekly_profit_chase_algo_a_intraday_winner_15m,perp_weekly_profit_chase_algo_b_growth_15m \
  --initial-cash 500 \
  --max-position-notional-usd 2500 \
  --baseline-slippage-bps 1.5 \
  --baseline-taker-fee-bps 6 \
  --stress-slippage-grid 1.5,3,5 \
  --stress-taker-fee-grid 6,10,15 \
  --stress-min-mean-return -0.0025 \
  --stress-min-positive-segment-frac 0.45 \
  --stress-min-accepted-segment-frac 0.40 \
  --gate-min-stress-pass-frac 0.44 \
  --gate-min-positive-week-frac 0.70 \
  --top-n-validate 2 \
  --validate-train 45d \
  --validate-validate 14d \
  --validate-test 14d \
  --validate-step 14d
```

## Artifacts
- Latest state:
  - `outputs/evaluations/latest_state.json`
  - `docs/EXECUTION_STATE_STRATEGY_SEARCH.md`
- Winning run:
  - `outputs/evaluations/evaluate_all_20260210_162617_071340_36010_81a5/leaderboard.csv`
  - `outputs/evaluations/evaluate_all_20260210_162617_071340_36010_81a5/evaluation_result.json`

## Additional Post-Selection Testing
- Extended 16-scenario walk-forward stress (45/14/14):
  - Algo A: `outputs/validation/validate_20260210_194030_186715_91252_44e7`
  - Algo B: `outputs/validation/validate_20260210_194030_186865_91251_039b`
- Alternate-segmentation walk-forward stress (60/21/21):
  - Algo A: `outputs/validation/validate_20260210_195155_584244_93813_c707`
  - Algo B: `outputs/validation/validate_20260210_195155_584370_93812_84dc`
- Notional/symbol robustness matrix CSV:
  - `outputs/evaluations/algo_ab_robustness_matrix_20260210.csv`
- Time-split stability CSV:
  - `outputs/evaluations/algo_ab_time_split_20260210.csv`
- Harsh-cost failure boundary CSV:
  - `outputs/evaluations/algo_ab_harsh_cost_backtest_20260210.csv`
- Random multi-year windows (one random 180d window per year, 2016..2025, seed 20260211):
  - Run root: `outputs/evaluations/ab_random_year_windows_20260211_020934`
  - Per-window rows: `outputs/evaluations/ab_random_year_windows_20260211_020934/ab_random_year_windows_results.csv`
  - Aggregate summary: `outputs/evaluations/ab_random_year_windows_20260211_020934/ab_random_year_windows_summary.json`

## Detailed Algorithm Docs
- `docs/algorithms/perp_weekly_profit_chase_algo_a_intraday_winner_15m.md`
- `docs/algorithms/perp_weekly_profit_chase_algo_b_growth_15m.md`
- `docs/algorithms/perp_weekly_profit_chase_validation_deep_dive_2026-02-10.md`
- `docs/algorithms/perp_weekly_profit_chase_ab_random_year_windows_2026-02-11.md`
- `docs/algorithms/perp_weekly_profit_chase_tuning_round_2026-02-11.md`
- `docs/algorithms/perp_weekly_profit_chase_loss_control_round_2026-02-11.md`
- `docs/algorithms/perp_weekly_profit_chase_algo_profit_plus_15m.md`
- `docs/algorithms/perp_weekly_profit_chase_algo_low_dd_experimental_15m.md`

## New Tuned Candidates (2026-02-11)
- Profit-window maximizer:
  - `strategy_params/perp_weekly_profit_chase_algo_profit_windows_max_15m.json`
- Growth-balance profile:
  - `strategy_params/perp_weekly_profit_chase_algo_growth_balance_15m.json`

## Full-Window Tuning Artifacts (2026-02-11)
- Search batches:
  - `outputs/evaluations/perp_weekly_profit_chase_search/w2_b_20260211_024734_seed307`
  - `outputs/evaluations/perp_weekly_profit_chase_search/w3_mix_20260211_024734_seed409`
  - `outputs/evaluations/perp_weekly_profit_chase_search/w1_a_fast_20260211_034654_seed211`
- Finalist full 10-window validation:
  - `outputs/evaluations/perp_weekly_profit_chase_finalists_full10_20260211/leaderboard.csv`
- Exploitation pass around current winner:
  - `outputs/evaluations/perp_weekly_profit_chase_search/full10_exploit_c03_20260211_070415_seed777`
- Moderate-cost check (3/10):
  - `outputs/evaluations/perp_weekly_profit_chase_costcheck_20260211/summary.json`

## Loss-Control Round (2026-02-11, Later)
- Detailed doc:
  - `docs/algorithms/perp_weekly_profit_chase_loss_control_round_2026-02-11.md`

### New code for downside controls and evaluation tooling
- `src/atlas/strategies/perp_weekly_profit_chase.py` (daily/weekly hard stop, cooldown, trailing/breakeven stops, max-hold exit)
- `src/atlas/strategies/registry.py` (new params wiring)
- `src/atlas/ml/tune.py` (search-space + validation updates)
- `scripts/optimize_perp_weekly_profit_chase_random_windows.py` (risk-control mutation + score penalties)
- `scripts/evaluate_perp_weekly_profit_chase_candidates.py` (new reusable evaluator)

### New promoted parameter file
- `strategy_params/perp_weekly_profit_chase_algo_profit_plus_15m.json`

### Full 10-window baseline-cost winner (among tuned candidates)
- Source leaderboard:
  - `outputs/evaluations/perp_weekly_profit_chase_candidate_eval/riskctrl_full10_focus_baseline_costs_20260211_154545/leaderboard.csv`
- Winner summary:
  - `profitable_runs = 6/10`
  - `weekly_gate_runs (>=70%) = 8/10`
  - `aggregate_weekly_positive_frac = 0.852`
  - `mean_total_return = +0.0431`
  - `median_total_return = +0.2362`
  - `worst_max_drawdown = -0.5817`

### Structured loss-guard sweep outcome
- Run:
  - `outputs/evaluations/perp_weekly_profit_chase_candidate_eval/lossguard_sweep_full10_basecost_20260211_214027/leaderboard.csv`
- Finding:
  - Hard-stop/trailing variants reduced downside in some cases but consistently reduced profitability.
  - Baseline-like variant remained best on full-10 baseline-cost score.

### Stress-cost comparison (3/10)
- Run:
  - `outputs/evaluations/perp_weekly_profit_chase_candidate_eval/riskctrl_coststress_focus_20260211_211650/leaderboard.csv`
- Finding:
  - All tested variants had negative mean return under this harsher cost regime.
  - `algo_profit_plus_15m` still ranked highest among tested set.

## Notes
- Results are historical backtests under configured assumptions; they are not guarantees of future profits.
- If continuing research, next efficient step is to mutate around the two passing parameter files only (small local neighborhoods), not broad all-strategy sweeps.
