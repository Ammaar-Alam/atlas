# Coinbase CB10 Frontier (2026-02-15)

## Test Profile
- Venue/cost model: Coinbase derivatives with taker `10.0 bps` + fixed `$0.15/contract` + slippage `1.5 bps`
- Account assumptions: `$500` initial cash, `$5000` max notional
- Symbol/data: `BTC/USD`, `coinbase`, `1H`
- Window sets:
  - Random-year robustness: `outputs/evaluations/ab_random_year_windows_20260211_020934/windows.json` (`10` windows, 2016-2025)
  - Launch-era validation: `outputs/evaluations/coinbase_perp_rolling_180d_20260213/windows.json` (`12` rolling 180d windows)

## Promoted Candidates

### 1) Carry Shield
- Params: `strategy_params/perp_weekly_carry_shield_cb10_rnd10_launch12_best_cand103.json`
- Random windows source:
  - `outputs/evaluations/strategy_strict_gate_search/carry_1h_random10_cb10fix015_s9301_20260214_231050_seed9301`
  - mean return: `-9.21%`
  - mean alpha vs SPY: `-16.97%`
  - mean weekly positive frac: `0.18`
  - weekly gate count (`>=0.70`): `0/10`
- Launch windows source:
  - `outputs/evaluations/strategy_eval/carry_1h_randomsearch_top20_launch12_cb10fix015_20260215_071419`
  - mean return: `+12.27%`
  - profitable runs: `12/12`
  - aggregate weekly positive frac: `0.1867`
  - weekly gate runs (`>=0.70`): `0/12`

### 2) Trend Guard
- Params: `strategy_params/perp_trend_vol_guard_cb10_rnd10_launch12_best_cand063.json`
- Random windows source:
  - `outputs/evaluations/strategy_strict_gate_search/trend_guard_1h_random10_cb10fix015_s9303_20260214_231050_seed9303`
  - mean return: `-8.67%`
  - mean alpha vs SPY: `-16.43%`
  - mean weekly positive frac: `0.124`
  - weekly gate count (`>=0.70`): `0/10`
- Launch windows source:
  - `outputs/evaluations/strategy_eval/trend_guard_1h_randomsearch_top20_launch12_cb10fix015_20260215_071419`
  - mean return: `-0.31%`
  - profitable runs: `5/12`
  - aggregate weekly positive frac: `0.04`
  - weekly gate runs (`>=0.70`): `0/12`

### 3) Weekly Trend Reset (post-fix + hard-score)
- Params: `strategy_params/perp_weekly_trend_reset_cb10_rnd10_hardscore_best_cand013.json`
- Random windows source:
  - `outputs/evaluations/strategy_strict_gate_search/weekly_trend_reset_1h_random10_cb10fix015_longonly_hardscore_s9501_20260215_055302_seed9501`
  - mean return: `-2.76%`
  - mean alpha vs SPY: `-10.52%`
  - mean weekly positive frac: `0.232`
  - weekly gate count (`>=0.70`): `0/10`
- Launch windows source:
  - `outputs/evaluations/strategy_eval/weekly_trend_reset_1h_hardscore_top10_launch12_cb10fix015_20260215_070326`
  - mean return: `-2.01%`
  - profitable runs: `1/12`
  - aggregate weekly positive frac: `0.1433`
  - weekly gate runs (`>=0.70`): `0/12`

## Frontier Status
- No candidate currently meets:
  - weekly positive gate `>= 0.70`
  - robust SPY outperformance on random-year windows
  - simultaneous profitability on both random-year and launch-era sets

## Notes
- `perp_weekly_trend_reset` was patched for correct non-5m behavior:
  - dynamic bars/day inference
  - weekly schedule trigger for coarse bars
- strict-gate scorer in `scripts/optimize_strategy_weekly_gates.py` was hardened to penalize outlier/lottery profiles.
