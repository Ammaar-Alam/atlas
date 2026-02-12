# Perp Weekly Profit Chase A/B: Random Multi-Year Window Test (2026-02-11)

## Purpose
Re-test promoted Algorithm A and B on random windows across many years, to avoid over-trusting one recent regime.

## Setup
- Algorithms:
  - `strategy_params/perp_weekly_profit_chase_algo_a_intraday_winner_15m.json`
  - `strategy_params/perp_weekly_profit_chase_algo_b_growth_15m.json`
- Engine mode: `derivatives`
- Symbol used for long history: `BTC/USD` (Coinbase spot bars, traded by derivatives engine logic)
- Timeframe: `15Min`
- Cost model: `slippage_bps=1.5`, `taker_fee_bps=6`
- Risk config: `initial_cash=500`, `max_position_notional_usd=2500`, `allow_short=true`
- Prewarm: `90d`
- Weekly scoring: non-overlapping `7d` windows (`step=7d`)
- Random seed: `20260211`

Artifacts:
- Run root: `outputs/evaluations/ab_random_year_windows_20260211_020934`
- Windows spec: `outputs/evaluations/ab_random_year_windows_20260211_020934/windows.json`
- Per-run rows: `outputs/evaluations/ab_random_year_windows_20260211_020934/ab_random_year_windows_results.csv`
- Aggregate summary: `outputs/evaluations/ab_random_year_windows_20260211_020934/ab_random_year_windows_summary.json`

## Randomized Windows Used
One random 180-day window per year, 2016 through 2025.

| Year | Window |
|---|---|
| 2016 | 2016-02-03 to 2016-08-01 |
| 2017 | 2017-03-05 to 2017-09-01 |
| 2018 | 2018-06-08 to 2018-12-05 |
| 2019 | 2019-06-16 to 2019-12-13 |
| 2020 | 2020-06-19 to 2020-12-16 |
| 2021 | 2021-06-15 to 2021-12-12 |
| 2022 | 2022-01-13 to 2022-07-12 |
| 2023 | 2023-04-14 to 2023-10-11 |
| 2024 | 2024-02-03 to 2024-08-01 |
| 2025 | 2025-06-02 to 2025-11-29 |

## Weekly Accuracy (Profit Weeks / Total Weeks)
Each 180-day run yields 25 weekly windows.

### Algorithm A
| Year | Positive Weeks | Losing/Flat Weeks | Weekly Positive % | Total Return |
|---|---:|---:|---:|---:|
| 2016 | 20/25 | 5 | 80.00% | -33.08% |
| 2017 | 23/25 | 2 | 92.00% | +30.18% |
| 2018 | 20/25 | 5 | 80.00% | -7.65% |
| 2019 | 23/25 | 2 | 92.00% | -5.78% |
| 2020 | 16/25 | 9 | 64.00% | -23.21% |
| 2021 | 12/25 | 13 | 48.00% | -54.79% |
| 2022 | 20/25 | 5 | 80.00% | -28.77% |
| 2023 | 16/25 | 9 | 64.00% | -40.60% |
| 2024 | 22/25 | 3 | 88.00% | -15.06% |
| 2025 | 22/25 | 3 | 88.00% | -19.31% |

Aggregate:
- Positive weeks: `194 / 250` = `77.60%`
- Runs meeting `>=70%` weekly-positive gate: `7 / 10`
- Profitable runs (`total_return > 0`): `1 / 10`

### Algorithm B
| Year | Positive Weeks | Losing/Flat Weeks | Weekly Positive % | Total Return |
|---|---:|---:|---:|---:|
| 2016 | 18/25 | 7 | 72.00% | -9.53% |
| 2017 | 22/25 | 3 | 88.00% | -26.58% |
| 2018 | 22/25 | 3 | 88.00% | +27.14% |
| 2019 | 24/25 | 1 | 96.00% | +24.65% |
| 2020 | 20/25 | 5 | 80.00% | +5.21% |
| 2021 | 13/25 | 12 | 52.00% | -42.61% |
| 2022 | 21/25 | 4 | 84.00% | -29.59% |
| 2023 | 16/25 | 9 | 64.00% | -45.34% |
| 2024 | 21/25 | 4 | 84.00% | -20.49% |
| 2025 | 20/25 | 5 | 80.00% | -42.12% |

Aggregate:
- Positive weeks: `197 / 250` = `78.80%`
- Runs meeting `>=70%` weekly-positive gate: `8 / 10`
- Profitable runs (`total_return > 0`): `3 / 10`

## Key Interpretation
- Both A and B clear the `70% weekly-positive` threshold in aggregate (`77.6%` and `78.8%`).
- That weekly hit-rate does not imply profitable 180-day windows:
  - A profitable in `10%` of runs.
  - B profitable in `30%` of runs.
- Failure mode is tail-loss concentration: many small winning weeks plus fewer very large losing weeks.

## Practical Takeaway
- If your hard requirement is weekly hit-rate `>=70%`, both A/B are often acceptable.
- If your hard requirement is robust positive 6-month return across random historical windows, these current A/B settings are not sufficient.
