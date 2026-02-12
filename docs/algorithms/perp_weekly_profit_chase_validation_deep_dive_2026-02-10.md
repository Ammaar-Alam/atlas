# Perp Weekly Profit Chase: Deep Validation Report (2026-02-10)

## Why This Exists
This report answers two practical questions:
1. Were the two promoted algorithms tuned?  
2. How far and how hard were they tested?

## Short Answer
- Tuned: **Yes**, both were selected from a curated parameter search and then re-validated in a final promoted head-to-head run.
- Date coverage: `2025-07-18 22:00:00+00:00` to `2026-01-01 00:00:00+00:00` on `BTC-PERP` 15-minute bars (Coinbase derivatives data available locally).

## Primary Selection Evidence
Run: `outputs/evaluations/evaluate_all_20260210_162617_071340_36010_81a5`

| Algorithm | Params File | Return | SPY Return | Alpha | Max DD | Weekly+ Frac | Stress Pass | Sharpe Daily | Trades |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A (winner) | `strategy_params/perp_weekly_profit_chase_algo_a_intraday_winner_15m.json` | 24.77% | 8.66% | 16.11% | -7.25% | 95.65% | 1.00 | 2.3407 | 134 |
| B | `strategy_params/perp_weekly_profit_chase_algo_b_growth_15m.json` | 19.91% | 8.66% | 11.25% | -12.11% | 86.96% | 1.00 | 1.7187 | 132 |

## Weekly Window Behavior (7d/7d)
From baseline run window analysis JSON files.

| Algorithm | Mean Weekly | Median Weekly | Worst Week | Best Week | Trade Window Frac | Beat-SPY Weekly Frac |
|---|---:|---:|---:|---:|---:|---:|
| A | 0.9106% | 0.7668% | -4.7391% | 3.2880% | 100.00% | 73.91% |
| B | 0.7396% | 0.7148% | -7.5281% | 11.1326% | 100.00% | 73.91% |

## Extended Walk-Forward Stress (Added After Selection)
### Matrix 1: 45d / 14d / 14d, step 14d
- Algorithm A run: `outputs/validation/validate_20260210_194030_186715_91252_44e7`
- Algorithm B run: `outputs/validation/validate_20260210_194030_186865_91251_039b`
- Cost grid: slippage `1.5,3,5,8` x fee `6,10,15,25` (16 scenarios each)

Using evaluator stress thresholds (`mean_return>=-0.25%`, `positive_segment_frac>=0.45`, `accepted_frac>=0.40`):
- A: `12/16` scenarios pass, worst mean scenario `-11.13%` at `8/25`.
- B: `13/16` scenarios pass, worst mean scenario `-6.47%` at `8/25`.

Interpretation:
- Both remain robust in realistic-mid cost bands.
- Both break under the harshest `8/25` configuration.

### Matrix 2: 60d / 21d / 21d, step 21d
- Algorithm A run: `outputs/validation/validate_20260210_195155_584244_93813_c707`
- Algorithm B run: `outputs/validation/validate_20260210_195155_584370_93812_84dc`
- Cost grid: slippage `1.5,3,5` x fee `6,10,15` (9 scenarios each)

Stress-threshold pass counts:
- A: `9/9`
- B: `9/9`

Interpretation:
- Under this segmentation and moderate cost grid, both are consistently positive.

## Sensitivity and Transfer Tests (Added After Selection)
Source CSV: `outputs/evaluations/algo_ab_robustness_matrix_20260210.csv`

### Notional sensitivity on BTC (base cost 1.5/6)
- A: 22.70% (`1500`) -> 24.77% (`2500`) -> 24.77% (`4000`)
- B: 19.53% (`1500`) -> 19.91% (`2500`) -> 19.88% (`4000`)

### Cross-symbol transfer
- A on ETH-only: -32.73% (DD -39.25%)
- B on ETH-only: -18.26% (DD -32.48%)
- A on BTC+ETH: +23.21% but DD -31.32%
- B on BTC+ETH: -24.38% and DD -40.97%

Interpretation:
- Promoted configs are BTC-specific and not safe to transfer directly.

## Time-Split Stability Test
Source CSV: `outputs/evaluations/algo_ab_time_split_20260210.csv`

| Algorithm | Early (2025-07-18 to 2025-10-15) | Late (2025-10-15 to 2026-01-01) |
|---|---|---|
| A | +6.90%, alpha +1.38% | -2.64%, alpha -5.16% |
| B | +3.31%, alpha -2.22% | +16.65%, alpha +14.14% |

Interpretation:
- Regime dependency exists; results are not uniform by sub-period.

## Explicit Failure Boundary (Harsh Costs)
Source CSV: `outputs/evaluations/algo_ab_harsh_cost_backtest_20260210.csv`

At slippage `8` bps and fee `25` bps:
- A: -90.42%, DD -90.80%
- B: -68.86%, DD -71.82%

Interpretation:
- These strategies are not robust to very adverse execution cost regimes.

## Final Ranking (Current)
1. `strategy_params/perp_weekly_profit_chase_algo_a_intraday_winner_15m.json`
2. `strategy_params/perp_weekly_profit_chase_algo_b_growth_15m.json`

## Practical Conclusion
- These are genuinely tuned and repeatedly validated candidates, not one-off picks.
- They pass the current hard gates on the available BTC-PERP data.
- They are not universally robust: performance degrades materially in harsh-cost environments and on symbol transfer.
- If you want broader confidence, the next step is not more random tuning; it is new data coverage (longer perp history / additional venues) and repeated rolling forward re-validation.
