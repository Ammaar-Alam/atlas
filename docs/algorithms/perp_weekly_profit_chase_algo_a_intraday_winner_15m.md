# Algorithm A: `perp_weekly_profit_chase_algo_a_intraday_winner_15m`

## Status
- Strategy family: `perp_weekly_profit_chase`
- File: `strategy_params/perp_weekly_profit_chase_algo_a_intraday_winner_15m.json`
- Current role: primary intraday candidate (winner among validated candidates)

## Was This Tuned?
Yes.

Tuning/selection path:
1. Generated large candidate banks for `perp_weekly_profit_chase` under realistic-cost assumptions.
2. Ran curated variant sweep and stress-validated top candidates.
3. Promoted the best candidate to stable filename.
4. Re-ran final head-to-head confirmation on promoted files.

Primary selection runs:
- Curated sweep: `outputs/evaluations/evaluate_all_20260210_160518_327211_29307_31b3`
- Final confirmation: `outputs/evaluations/evaluate_all_20260210_162617_071340_36010_81a5`

## Backtest Data Coverage
- Market: derivatives
- Symbol: `BTC-PERP`
- Data source: `coinbase`
- Bar timeframe: `15Min`
- Actual bars available: `2025-07-18 22:00:00+00:00` to `2026-01-01 00:00:00+00:00`

Notes:
- This is a hard data boundary for this instrument in the local dataset.
- Benchmark (`SPY`) is aligned to realized strategy window in evaluator logic.

## Parameter Set
```json
{
  "perp_weekly_profit_chase": {
    "rebalance_weekday_utc": 0,
    "rebalance_hour_utc": 0,
    "rebalance_minute_utc": 5,
    "atr_window": 14,
    "opening_range_minutes": 30,
    "breakout_buffer_bps": 3.0,
    "lookback_short_days": 1.0,
    "lookback_long_days": 7.0,
    "momentum_threshold_bps": 0.0,
    "min_atr_bps": 3.0,
    "sizing_mode": "leverage",
    "risk_per_trade": 0.01,
    "base_leverage": 3.0,
    "max_leverage": 3.0,
    "max_margin_utilization": 0.4,
    "maintenance_margin_rate": 0.05,
    "stop_atr_mult": 1.2,
    "min_liq_buffer_atr": 5.0,
    "min_trade_notional_usd": 10.0,
    "weekly_heartbeat_exposure": 0.001,
    "weekly_heartbeat_hold_bars": 1,
    "weekly_nudge_exposure": 0.0,
    "max_flips_per_day": 2,
    "weekly_profit_target": 0.0065,
    "weekly_chase_k": 0.0
  }
}
```

## Final Gate-Passing Metrics
Source run: `outputs/evaluations/evaluate_all_20260210_162617_071340_36010_81a5`

- `total_return`: `+24.77%`
- `spy_total_return`: `+8.66%`
- `alpha_vs_spy`: `+16.11%`
- `max_drawdown`: `-7.25%`
- `weekly_positive_frac`: `95.65%` (22/23 windows)
- `stress_pass_frac`: `1.00` (9/9 scenarios passed)
- `sharpe_daily`: `2.3407`
- `trades`: `134`

## Weekly Window Diagnostics
Source: `outputs/evaluations/evaluate_all_20260210_162617_071340_36010_81a5/baselines/perp_weekly_profit_chase_20260210_162617_075293_2c53/window_analysis.json`

- Window: `7d`, Step: `7d`
- `trade_window_frac`: `100%`
- Mean weekly return: `0.9106%`
- Median weekly return: `0.7668%`
- Worst week: `-4.7391%`
- Best week: `+3.2880%`
- Beat-SPY weekly fraction: `73.91%`

## Additional Robustness Testing
### Walk-forward cost stress (45d/14d/14d, step 14d)
Run: `outputs/validation/validate_20260210_194030_186715_91252_44e7`

- Cost grid: slippage `1.5,3,5,8` bps x fee `6,10,15,25` bps
- Scenario pass count (using evaluator stress thresholds): `12/16`
- Best scenario mean segment return: `+2.49%` (`1.5/6`)
- Worst scenario mean segment return: `-11.13%` (`8/25`)

### Walk-forward alternate segmentation (60d/21d/21d, step 21d)
Run: `outputs/validation/validate_20260210_195155_584244_93813_c707`

- Cost grid: slippage `1.5,3,5` bps x fee `6,10,15` bps
- Scenario pass count: `9/9`
- Mean segment return range: `+2.35%` to `+4.11%`

### Leverage / notional sensitivity (BTC only, base costs)
Source CSV: `outputs/evaluations/algo_ab_robustness_matrix_20260210.csv`

- `max_notional=1500`: `+22.70%`, DD `-7.25%`
- `max_notional=2500`: `+24.77%`, DD `-7.25%`
- `max_notional=4000`: `+24.77%`, DD `-7.25%`

Interpretation:
- Performance saturates above a threshold because strategy-level risk caps dominate.

### Cross-symbol transfer (same params, base costs)
Source CSV: `outputs/evaluations/algo_ab_robustness_matrix_20260210.csv`

- `ETH-PERP` only: `-32.73%`, DD `-39.25%` (fails risk gate)
- `BTC-PERP,ETH-PERP`: `+23.21%`, DD `-31.32%` (fails risk gate)

Interpretation:
- This parameterization is BTC-specific and does not generalize safely to ETH/multi-symbol.

### Time-split stability (BTC only)
Source CSV: `outputs/evaluations/algo_ab_time_split_20260210.csv`

- Early (`2025-07-18` to `2025-10-15`): `+6.90%`, alpha `+1.38%`
- Late (`2025-10-15` to `2026-01-01`): `-2.64%`, alpha `-5.16%`

Interpretation:
- Performance is not uniform across sub-periods.

### Harsh-cost failure boundary
Source CSV: `outputs/evaluations/algo_ab_harsh_cost_backtest_20260210.csv`

- At slippage `8` bps + fee `25` bps:
  - Return `-90.42%`
  - Max drawdown `-90.80%`

Interpretation:
- Edge is highly sensitive to adverse execution costs.

## Deployment Guidance
- Keep symbol scope to `BTC-PERP` only unless re-tuned per symbol.
- Keep cost assumptions close to validated regime (`1.5-5` slippage, `6-15` fee bps).
- Re-run rolling validation weekly; stop deployment if:
  - 4-week rolling alpha vs SPY turns negative
  - 4-week weekly-positive fraction drops below `0.70`
  - Realized costs move materially above validated band

## Conclusion
- This is a tuned and validated candidate with strong in-sample and walk-forward results on available BTC-PERP data.
- It is not universal: it fails under harsh fees/slippage and does not transfer safely to ETH/multi-symbol without re-tuning.
