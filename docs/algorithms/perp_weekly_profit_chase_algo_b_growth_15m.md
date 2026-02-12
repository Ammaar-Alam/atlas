# Algorithm B: `perp_weekly_profit_chase_algo_b_growth_15m`

## Status
- Strategy family: `perp_weekly_profit_chase`
- File: `strategy_params/perp_weekly_profit_chase_algo_b_growth_15m.json`
- Current role: secondary portfolio-growth candidate (fallback to Algorithm A)

## Was This Tuned?
Yes.

Tuning/selection path:
1. Included in the same curated parameter sweep used to discover Algorithm A.
2. Passed all hard gates in the curated selection run.
3. Promoted to stable filename and re-confirmed in a final two-candidate run.

Key runs:
- Curated sweep: `outputs/evaluations/evaluate_all_20260210_160518_327211_29307_31b3`
- Final confirmation: `outputs/evaluations/evaluate_all_20260210_162617_071340_36010_81a5`

## Backtest Data Coverage
- Market: derivatives
- Symbol: `BTC-PERP`
- Data source: `coinbase`
- Bar timeframe: `15Min`
- Actual bars available: `2025-07-18 22:00:00+00:00` to `2026-01-01 00:00:00+00:00`

## Parameter Set
```json
{
  "perp_weekly_profit_chase": {
    "rebalance_weekday_utc": 0,
    "rebalance_hour_utc": 0,
    "rebalance_minute_utc": 5,
    "atr_window": 14,
    "opening_range_minutes": 30,
    "breakout_buffer_bps": 7.0,
    "lookback_short_days": 1.0,
    "lookback_long_days": 7.0,
    "momentum_threshold_bps": 0.0,
    "min_atr_bps": 3.0,
    "sizing_mode": "leverage",
    "risk_per_trade": 0.01,
    "base_leverage": 2.6,
    "max_leverage": 4.0,
    "max_margin_utilization": 0.45,
    "maintenance_margin_rate": 0.05,
    "stop_atr_mult": 1.8,
    "min_liq_buffer_atr": 4.0,
    "min_trade_notional_usd": 10.0,
    "weekly_heartbeat_exposure": 0.004,
    "weekly_heartbeat_hold_bars": 1,
    "weekly_nudge_exposure": 0.0,
    "max_flips_per_day": 1,
    "weekly_profit_target": 0.0065,
    "weekly_chase_k": 0.2
  }
}
```

## Final Gate-Passing Metrics
Source run: `outputs/evaluations/evaluate_all_20260210_162617_071340_36010_81a5`

- `total_return`: `+19.91%`
- `spy_total_return`: `+8.66%`
- `alpha_vs_spy`: `+11.25%`
- `max_drawdown`: `-12.11%`
- `weekly_positive_frac`: `86.96%` (20/23 windows)
- `stress_pass_frac`: `1.00` (9/9 scenarios passed)
- `sharpe_daily`: `1.7187`
- `trades`: `132`

## Weekly Window Diagnostics
Source: `outputs/evaluations/evaluate_all_20260210_162617_071340_36010_81a5/baselines/perp_weekly_profit_chase_20260210_162636_113410_7f16/window_analysis.json`

- Window: `7d`, Step: `7d`
- `trade_window_frac`: `100%`
- Mean weekly return: `0.7396%`
- Median weekly return: `0.7148%`
- Worst week: `-7.5281%`
- Best week: `+11.1326%`
- Beat-SPY weekly fraction: `73.91%`

## Additional Robustness Testing
### Walk-forward cost stress (45d/14d/14d, step 14d)
Run: `outputs/validation/validate_20260210_194030_186865_91251_039b`

- Cost grid: slippage `1.5,3,5,8` bps x fee `6,10,15,25` bps
- Scenario pass count (using evaluator stress thresholds): `13/16`
- Best scenario mean segment return: `+1.71%` (`1.5/6`)
- Worst scenario mean segment return: `-6.47%` (`8/25`)

### Walk-forward alternate segmentation (60d/21d/21d, step 21d)
Run: `outputs/validation/validate_20260210_195155_584370_93812_84dc`

- Cost grid: slippage `1.5,3,5` bps x fee `6,10,15` bps
- Scenario pass count: `9/9`
- Mean segment return range: `+2.68%` to `+4.06%`

### Leverage / notional sensitivity (BTC only, base costs)
Source CSV: `outputs/evaluations/algo_ab_robustness_matrix_20260210.csv`

- `max_notional=1500`: `+19.53%`, DD `-9.80%`
- `max_notional=2500`: `+19.91%`, DD `-12.11%`
- `max_notional=4000`: `+19.88%`, DD `-12.11%`

Interpretation:
- Returns are fairly stable across tested caps, with DD rising modestly at higher notional.

### Cross-symbol transfer (same params, base costs)
Source CSV: `outputs/evaluations/algo_ab_robustness_matrix_20260210.csv`

- `ETH-PERP` only: `-18.26%`, DD `-32.48%` (fails risk gate)
- `BTC-PERP,ETH-PERP`: `-24.38%`, DD `-40.97%` (fails risk gate)

Interpretation:
- This setting does not generalize safely beyond BTC.

### Time-split stability (BTC only)
Source CSV: `outputs/evaluations/algo_ab_time_split_20260210.csv`

- Early (`2025-07-18` to `2025-10-15`): `+3.31%`, alpha `-2.22%`
- Late (`2025-10-15` to `2026-01-01`): `+16.65%`, alpha `+14.14%`

Interpretation:
- More back-loaded than Algorithm A; stronger in later sub-period.

### Harsh-cost failure boundary
Source CSV: `outputs/evaluations/algo_ab_harsh_cost_backtest_20260210.csv`

- At slippage `8` bps + fee `25` bps:
  - Return `-68.86%`
  - Max drawdown `-71.82%`

## Deployment Guidance
- Use only on `BTC-PERP` under validated cost regime.
- Treat this as secondary to Algorithm A due lower alpha and higher drawdown.
- Keep ongoing checks identical to Algorithm A:
  - rolling weekly-positive fraction
  - rolling alpha vs SPY
  - realized cost drift

## Conclusion
- Algorithm B is tuned and passes all hard gates in current validated setup.
- It is less profitable and less robust than Algorithm A under baseline settings, but still deployable as a fallback profile.
