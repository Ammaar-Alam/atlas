# Atlas Strategy Search Execution State

## Objective
Find the single strongest strategy under pessimistic costs for a $500 account, with PDT-safe constraints and SPY outperformance.

## Hard Gates
- max_drawdown >= -20%
- weekly_positive_frac >= 70%
- beat SPY over matched dates
- stress_pass_frac >= configured threshold

## Latest Broad Run
- generated_at: 2026-02-13T00:25:34.196325
- run_dir: `outputs/evaluations/evaluate_all_20260212_215902_162445_47390_d450`
- total_candidates: 207
- validated_candidates: 0
- passed_candidates: 0
- algorithm_a_candidate_id: `perp_weekly_carry_shield|derivatives|BTC-PERP,ETH-PERP|1H|coinbase|perp_weekly_carry_shield_tuned_v3_1h.json`
- algorithm_b_candidate_id: `None`
- winner_candidate_id: `perp_weekly_carry_shield|derivatives|BTC-PERP,ETH-PERP|1H|coinbase|perp_weekly_carry_shield_tuned_v3_1h.json`

## Latest Focused Run (BTC-PERP Launch-Era Rolling 180d)
- generated_at: 2026-02-13
- windows file: `outputs/evaluations/coinbase_perp_rolling_180d_20260213/windows.json`
- strict realism update: contract-lot quantization enforced in backtest (`0.01 BTC` lot, integer contracts)
- key result: prior `perp_weekly_profit_chase` winners do **not** survive strict lot-model validation

### Current best launch-era candidates (strict lot model)
- `perp_weekly_carry_shield_cb10_lot_best_roll180d_v1`
  - params file: `strategy_params/perp_weekly_carry_shield_cb10_lot_best_roll180d_v1.json`
  - eval run: `outputs/evaluations/strategy_eval/carry_shield_best_roll180d_v1_eval_20260213_211533`
  - symbol/data: `BTC-PERP` on `coinbase` (`1H`)
  - costs: taker `10 bps` + fixed `$0.15/contract` + slippage `1.5 bps`
  - account/notional: `$500` initial cash, `$5000` max notional
  - runs: `12/12` profitable
  - mean 180d return: `+3.85%`
  - worst 180d return: `+2.57%`
  - aggregate weekly positive frac: `0.14`
  - weekly gate runs (`>=0.70`): `0/12`
- `perp_trend_vol_guard_cb10_lot_tuned_return_1h_longonly`
  - params file: `strategy_params/perp_trend_vol_guard_cb10_lot_tuned_return_1h_longonly.json`
  - eval run: `outputs/evaluations/strategy_eval/trend_guard_cb10_lot_roll180d_1h_longonly_tuned_20260213_180208`
  - runs: `12/12` profitable
  - mean 180d return: `+4.03%`
  - aggregate weekly positive frac: `0.0767`
  - weekly gate runs (`>=0.70`): `0/12`

### Overfit check (multi-year BTC/USD proxy windows, strict lot model)
- trend-guard candidate:
  - run: `outputs/evaluations/strategy_eval/trend_guard_cb10_longonly_randomyears_proxy_20260213_194712`
  - runs: `30`
  - profitable runs: `4/30`
  - mean return: `-9.93%`
  - weekly gate runs (`>=0.70`): `0/30`
- carry-shield candidate:
  - run: `outputs/evaluations/strategy_eval/carry_shield_best_roll180d_v1_randomyears_proxy_20260213_211703`
  - runs: `30`
  - profitable runs: `8/30`
  - mean return: `-8.33%`
  - weekly gate runs (`>=0.70`): `0/30`

## Latest Frontier Update (2026-02-14)
- strategy family focus: `perp_trend_vol_guard` (strict Coinbase taker+fixed fee, lot quantization)
- code updates:
  - `src/atlas/strategies/perp_trend_vol_guard.py`
    - added fallback-floor controls
    - added bounded weekly-chase controls
  - `src/atlas/ml/tune.py`
    - added tune space + validation for new trend-guard controls
  - `scripts/optimize_strategy_weekly_gates.py` / `scripts/evaluate_strategy_windows.py`
    - added multi-symbol support and optional per-symbol fee/contract maps
  - `src/atlas/backtest/engine.py` / `src/atlas/backtest/derivatives_engine.py`
    - added per-symbol contract size and fixed-fee support in backtest config/engine

### New best launch-era candidate (profit + consistency balance)
- params file: `strategy_params/trend_guard_manual_chase_grid2/cand08.json`
- eval run: `outputs/evaluations/strategy_eval/trend_guard_manual_chase_grid2_full12_20260214_071137`
- symbol/data: `BTC-PERP` on `coinbase` (`1H`)
- costs: taker `10 bps` + fixed `$0.15/contract` + slippage `1.5 bps`
- account/notional: `$500` initial cash, `$5000` max notional
- 12x rolling 180d windows:
  - profitable runs: `12/12`
  - profitable+beat-SPY runs: `9/12` (from strict-gate search summary)
  - mean 180d return: `+20.34%`
  - worst 180d return: `+3.69%`
  - aggregate weekly positive frac: `0.6467`
  - weekly gate runs (`>=0.70`): `4/12`

### Best weekly-consistency candidate found so far
- params file: `strategy_params/trend_guard_subset_promoted_topweekly_synth/cand01_cd685c3c43.json`
- eval run: `outputs/evaluations/strategy_eval/trend_guard_candidate066_vs_others_full12_20260214_042132`
- aggregate weekly positive frac: `0.6233`
- mean 180d return: `+1.73%`
- weekly gate runs (`>=0.70`): `3/12`

### Random-year proxy recheck (BTC/USD, 2016-2025 windows)
- run: `outputs/evaluations/strategy_eval/trend_guard_cand08_family_randomyears_20260214_075823`
- key result: launch-era winners degrade materially on proxy windows
  - `cand08`: mean return `-27.99%`, weekly positive frac `0.036`
  - `cand02`: mean return `-28.10%`, weekly positive frac `0.044`
  - legacy `cand011`: mean return `-6.79%`, weekly positive frac `0.104`

### Current status vs hard gates
- Weekly gate (`>=0.70`): best observed aggregate is `0.6467` (not met)
- SPY outperformance: met on launch-era for top candidate in `9/12` runs; not robust on random-year proxy
- Stress robustness (multi-year proxy): not met

## Latest Frontier Update (2026-02-15)
- core issue confirmed: strong launch-era candidates remain overfit on random-year windows under exact Coinbase fee model (`10 bps` + `$0.15/contract` + `1.5 bps` slippage).

### Code updates (2026-02-15)
- `src/atlas/strategies/perp_weekly_trend_reset.py`
  - fixed timeframe portability bug:
    - trend lookback now infers bars/day from actual bar spacing (not hardcoded to 5-minute bars)
    - weekly rebalance schedule now triggers at the first bar after scheduled time for coarse bars (`1H`, `4H`)
- `scripts/optimize_strategy_weekly_gates.py`
  - hardened strict-gate score against outlier-driven overfit:
    - winsorized mean return/alpha contribution
    - added median return/alpha contribution
    - stronger downside penalties
    - explicit penalties for lottery-like best-run outliers

### Major new runs
- Weekly trend reset (post-fix, 1H, random windows):
  - run: `outputs/evaluations/strategy_eval/weekly_trend_reset_v2_1h_random10_cb10fix015_afterfix_20260215_034610`
  - result: mean return `-9.61%`, aggregate weekly positive frac `0.348`, weekly gate runs `0/10`
- Weekly trend reset strict random search (short + long-only):
  - runs:
    - `outputs/evaluations/strategy_strict_gate_search/weekly_trend_reset_1h_random10_cb10fix015_short_s9401_20260215_034715_seed9401`
    - `outputs/evaluations/strategy_strict_gate_search/weekly_trend_reset_1h_random10_cb10fix015_longonly_s9402_20260215_034715_seed9402`
  - result: no candidate met weekly gate; top candidates showed large downside / instability
- Weekly trend reset hard-score search (long-only):
  - run: `outputs/evaluations/strategy_strict_gate_search/weekly_trend_reset_1h_random10_cb10fix015_longonly_hardscore_s9501_20260215_055302_seed9501`
  - top random-window weekly-positive candidates remained far below gate (`~0.18-0.24`)
  - launch validation of top-10:
    - run: `outputs/evaluations/strategy_eval/weekly_trend_reset_1h_hardscore_top10_launch12_cb10fix015_20260215_070326`
    - best candidate mean return `-2.01%`, weekly gate runs `0/12`
- Carry shield cross-validation:
  - random-search top-20 launch validation:
    - run: `outputs/evaluations/strategy_eval/carry_1h_randomsearch_top20_launch12_cb10fix015_20260215_071419`
    - best launch candidate: `cand_103` with mean return `+12.27%`, profitable `12/12`, aggregate weekly positive frac `0.1867`
    - same candidate on random windows (source leaderboard): mean return `-9.21%`, weekly gate `0/10`
- Carry shield hard-score search + launch validation:
  - random search: `outputs/evaluations/strategy_strict_gate_search/carry_1h_random10_cb10fix015_hardscore_s9601_20260215_075227_seed9601`
  - launch top-20 validation: `outputs/evaluations/strategy_eval/carry_1h_hardscore_top20_launch12_cb10fix015_20260215_094044`
  - best launch candidate: `cand_039` mean return `+5.95%`, profitable `12/12`, aggregate weekly positive frac `0.1533`
  - random-window robustness still failed (all weekly gates `0/10`, mean returns negative)
- Trend guard long-only hard-score search + launch validation:
  - random search: `outputs/evaluations/strategy_strict_gate_search/trend_guard_1h_random10_cb10fix015_longonly_hardscore_s9701_20260215_101116_seed9701`
  - launch top-20 validation: `outputs/evaluations/strategy_eval/trend_guard_1h_longonly_hardscore_top20_launch12_cb10fix015_20260215_115518`
  - best launch candidate: `cand_000` mean return `+0.95%`, profitable `12/12`, aggregate weekly positive frac `0.0767`
  - random-window robustness still failed (weekly gate `0/10`, mean returns negative)

### Status against gates after 2026-02-15 runs
- Weekly gate (`>=0.70`): still not met; best random-window aggregate weekly positive seen in robust runs remains `<< 0.70`.
- Beat-SPY + profitability across random windows: not met.
- Launch-era profitability can be achieved, but random-year robustness fails under realistic Coinbase costs.

## Latest Frontier Update (2026-02-16, tri-source proxy transfer)
- protocol tightened to avoid launch-era overfit:
  - optimize on Deribit full perp 1H CSV (`outputs/evaluations/external_perp_probe/20260215_165955/deribit_btc_perpetual_full_1h.csv`)
  - transfer on Coinbase launch rolling windows (`outputs/evaluations/coinbase_perp_rolling_180d_20260213/windows.json`)
  - independent proxy check on OKX full perp 1H CSV (`outputs/evaluations/external_perp_probe/20260215_205800/okx_btc_usdt_swap_full_1h.csv`)

### Completed runs (2026-02-16)
- RATC search from prior robust seed:
  - `outputs/evaluations/strategy_strict_gate_search/ratc_from_c000_deribit22_s3811_20260216_065736_seed3811`
  - Coinbase transfer: `outputs/evaluations/strategy_eval/ratc_from_c000_s3811_transfer_cb_launch12_20260216_071208`
  - OKX probe (top set): `outputs/evaluations/strategy_eval/ratc_from_c000_s3811_okx20_all_20260216_072038`
- RATC neighborhood search around tri-source-positive seed:
  - `outputs/evaluations/strategy_strict_gate_search/ratc_from_c026_deribit22_s3826_20260216_072950_seed3826`
  - Coinbase transfer: `outputs/evaluations/strategy_eval/ratc_from_c026_s3826_transfer_cb_launch12_20260216_075053`
  - OKX transfer: `outputs/evaluations/strategy_eval/ratc_from_c026_s3826_okx20_20260216_075737`
  - merged cross-source summary: `outputs/evaluations/strategy_eval/ratc_from_c026_s3826_crossproxy_summary.csv`
- RATC long-only branch (rejected for transfer failure):
  - `outputs/evaluations/strategy_strict_gate_search/ratc_from_c026cand008_deribit22_longonly_s3827_20260216_080909_seed3827`
  - Coinbase transfer: `outputs/evaluations/strategy_eval/ratc_from_c026cand008_longonly_s3827_transfer_cb_launch12_20260216_082406` (negative mean)
- RATC short-enabled refinement around `cand_008`:
  - `outputs/evaluations/strategy_strict_gate_search/ratc_from_c026cand008_deribit22_s3828_20260216_090616_seed3828`
  - Coinbase transfer: `outputs/evaluations/strategy_eval/ratc_from_c026cand008_s3828_transfer_cb_launch12_20260216_091747`
  - OKX transfer: `outputs/evaluations/strategy_eval/ratc_from_c026cand008_s3828_okx20_20260216_091747`
  - merged cross-source summary: `outputs/evaluations/strategy_eval/ratc_from_c026cand008_s3828_crossproxy_summary.csv`
- RATC focused refinement around the new frontier winner:
  - `outputs/evaluations/strategy_strict_gate_search/ratc_from_c023_deribit22_s3829_20260216_092636_seed3829`
  - Coinbase transfer: `outputs/evaluations/strategy_eval/ratc_from_c023_s3829_transfer_cb_launch12_20260216_093631`
  - OKX transfer: `outputs/evaluations/strategy_eval/ratc_from_c023_s3829_okx20_20260216_093631`
  - merged cross-source summary: `outputs/evaluations/strategy_eval/ratc_from_c023_s3829_crossproxy_summary.csv`
- Additional continuation refinements (no frontier improvement):
  - `outputs/evaluations/strategy_strict_gate_search/ratc_from_c023_deribit22_s3830_20260216_142215_seed3830`
    - Coinbase transfer: `outputs/evaluations/strategy_eval/ratc_from_c023_s3830_transfer_cb_launch12_20260216_143117`
    - OKX transfer: `outputs/evaluations/strategy_eval/ratc_from_c023_s3830_okx20_20260216_143730`
    - summary: `outputs/evaluations/strategy_eval/ratc_s3830_crossproxy_summary.csv`
  - `outputs/evaluations/strategy_strict_gate_search/ratc_from_c020_deribit22_s3831_20260216_142215_seed3831`
    - Coinbase transfer: `outputs/evaluations/strategy_eval/ratc_from_c020_s3831_transfer_cb_launch12_20260216_143117`
    - OKX transfer: `outputs/evaluations/strategy_eval/ratc_from_c020_s3831_okx20_20260216_143730`
    - summary: `outputs/evaluations/strategy_eval/ratc_s3831_crossproxy_summary.csv`

### Best robust candidates currently
- Current robust frontier winner (short-enabled, tri-source-positive):
  - params: `outputs/evaluations/strategy_strict_gate_search/ratc_from_c026cand008_deribit22_s3828_20260216_090616_seed3828/candidates/cand_023.json`
  - Deribit22 mean return: `+18.91%`, profitable runs: `10/22`, weekly positive frac: `0.231`
  - Coinbase launch12 mean return: `+15.41%`, profitable runs: `9/12`, weekly positive frac: `0.337`
  - OKX20 mean return: `+21.94%`, profitable runs: `8/20`, weekly positive frac: `0.228`
  - tri-source min mean return: `+15.41%` (best seen so far in robust tri-source checks)
- Higher hit-rate alternative (lower min-mean):
  - params: `outputs/evaluations/strategy_strict_gate_search/ratc_from_c023_deribit22_s3829_20260216_092636_seed3829/candidates/cand_018.json`
  - Deribit22 / Coinbase12 / OKX20 profitable-run fractions: `0.50 / 0.75 / 0.55`
  - mean returns: `+15.37% / +9.57% / +18.82%`

### Gate status (still unmet)
- Weekly-positive gate (`>=0.70`): still `0` qualifying runs in robust tri-source checks.
- Tri-source returns can be positive, but weekly consistency remains far below target (`~0.23-0.34`).
- Trend-guard high-weekly branch remains negative mean on proxy; RATC short-enabled remains the only family currently showing positive transfer across Deribit+Coinbase+OKX.
- Continuation note: additional `s3830/s3831` searches increased some Coinbase weekly-positive values but did not beat current tri-source frontier (`s3828/cand_023`) on robust minimum-return criteria.

## Artifacts
- leaderboard_csv: `outputs/evaluations/evaluate_all_20260212_215902_162445_47390_d450/leaderboard.csv`
- leaderboard_json: `outputs/evaluations/evaluate_all_20260212_215902_162445_47390_d450/leaderboard.json`
- full_json: `outputs/evaluations/evaluate_all_20260212_215902_162445_47390_d450/evaluation_result.json`
- latest_state_json: `outputs/evaluations/latest_state.json`

## Resume Checklist
1. Open `outputs/evaluations/latest_state.json` and identify the current winner and failed gates.
2. Re-run `atlas evaluate-all` with tighter strategy filters or revised costs if needed.
3. Re-check deployment feasibility for the winner before moving to live paper/live execution.
4. For live Coinbase perp readiness, prefer rolling-window validation on `BTC-PERP` launch-era data using `outputs/evaluations/coinbase_perp_rolling_180d_20260213/windows.json`.
5. Current strict lot-model launch-era leader is `strategy_params/trend_guard_manual_chase_grid2/cand08.json`; continue strict-gate search because weekly gate and random-year robustness remain unmet.

## Snapshot
```json
{
  "run_dir": "outputs/evaluations/evaluate_all_20260212_215902_162445_47390_d450",
  "winner_candidate_id": "perp_weekly_carry_shield|derivatives|BTC-PERP,ETH-PERP|1H|coinbase|perp_weekly_carry_shield_tuned_v3_1h.json",
  "algorithm_a_candidate_id": "perp_weekly_carry_shield|derivatives|BTC-PERP,ETH-PERP|1H|coinbase|perp_weekly_carry_shield_tuned_v3_1h.json",
  "algorithm_b_candidate_id": null,
  "passed_candidates": 0,
  "total_candidates": 207,
  "leaderboard_csv": "outputs/evaluations/evaluate_all_20260212_215902_162445_47390_d450/leaderboard.csv",
  "leaderboard_json": "outputs/evaluations/evaluate_all_20260212_215902_162445_47390_d450/leaderboard.json",
  "full_json": "outputs/evaluations/evaluate_all_20260212_215902_162445_47390_d450/evaluation_result.json",
  "strict_lot_eval_carry_shield": "outputs/evaluations/strategy_eval/carry_shield_best_roll180d_v1_eval_20260213_211533",
  "strict_lot_eval_trend_guard": "outputs/evaluations/strategy_eval/trend_guard_cb10_lot_roll180d_1h_longonly_tuned_20260213_180208",
  "strict_lot_proxy_eval_carry_shield": "outputs/evaluations/strategy_eval/carry_shield_best_roll180d_v1_randomyears_proxy_20260213_211703",
  "strict_lot_proxy_eval_trend_guard": "outputs/evaluations/strategy_eval/trend_guard_cb10_longonly_randomyears_proxy_20260213_194712",
  "strict_best_launch_params": "strategy_params/perp_weekly_carry_shield_cb10_lot_best_roll180d_v1.json"
}
```

## Latest Frontier Update (2026-02-16 evening, reverse-transfer success)
- New robust frontier found by Coinbase-first search + Deribit/OKX transfer:
  - search run: `outputs/evaluations/strategy_strict_gate_search/ratc_from_s3835cand005_coinbase12_s3861_20260216_203856_seed3861`
  - best candidate: `cand_034`
  - promoted params: `strategy_params/perp_regime_adaptive_trend_capture_trisource_best_s3861_c034.json`

### New frontier metrics (`cand_034`)
- Deribit22:
  - run: `outputs/evaluations/strategy_eval/ratc_s3861_transfer_deribit22_20260216_205651`
  - mean return: `+21.51%`
  - profitable frac: `0.3636`
  - weekly positive frac: `0.2109`
- Coinbase launch12:
  - source leaderboard: `outputs/evaluations/strategy_strict_gate_search/ratc_from_s3835cand005_coinbase12_s3861_20260216_203856_seed3861/leaderboard.csv`
  - mean return: `+25.47%`
  - profitable frac: `1.0000`
  - weekly positive frac: `0.4000`
- OKX20:
  - run: `outputs/evaluations/strategy_eval/ratc_s3861_transfer_okx20_20260216_205651`
  - mean return: `+23.75%`
  - profitable frac: `0.3000`
  - weekly positive frac: `0.2120`

Cross-source summary:
- `outputs/evaluations/strategy_eval/ratc_s3861_crossproxy_summary.csv`
- `all_three_mean_pos=true`
- `min_mean=+21.51%` (improved from prior frontier `+15.41%`)
- `avg_mean=+23.58%`
- `min_prof_frac=0.30`
- `min_weekly=0.2109`

### Pessimistic stress checks (new vs old frontier)
- comparison table:
  - `outputs/evaluations/strategy_eval/ratc_c034_vs_c000_base_stress_20260216.csv`
- stress A (`12.5 bps`, `2.0 bps slippage`): `cand_034` remains tri-source positive and above prior frontier on robust floor.
- stress B (`15.0 bps`, `3.0 bps slippage`): `cand_034` still tri-source positive; stronger Coinbase/OKX means than prior frontier.

### Additional refinement branches (rejected)
- from `cand_034` on Deribit and OKX objective surfaces:
  - `outputs/evaluations/strategy_strict_gate_search/ratc_from_c034_deribit22_s3870_20260216_211528_seed3870`
  - `outputs/evaluations/strategy_strict_gate_search/ratc_from_c034_okx20_s3871_20260216_211528_seed3871`
- cross merges:
  - `outputs/evaluations/strategy_eval/ratc_s3870_crossproxy_summary.csv`
  - `outputs/evaluations/strategy_eval/ratc_s3871_crossproxy_summary.csv`
- result: no improvement over `s3861/cand_034`.

### Status vs gate
- Weekly-positive gate (`>=0.70`) still unmet.
- Robust return frontier improved materially.

### Additional 2026-02-16 late checks after c034 promotion
- c034 stress checks completed:
  - `outputs/evaluations/strategy_eval/ratc_c034_stress125_deribit22_20260216_211306`
  - `outputs/evaluations/strategy_eval/ratc_c034_stress125_cb12_20260216_211306`
  - `outputs/evaluations/strategy_eval/ratc_c034_stress125_okx20_20260216_211306`
  - `outputs/evaluations/strategy_eval/ratc_c034_stress15_deribit22_20260216_211413`
  - `outputs/evaluations/strategy_eval/ratc_c034_stress15_cb12_20260216_211413`
  - `outputs/evaluations/strategy_eval/ratc_c034_stress15_okx20_20260216_211413`
- baseline comparison vs prior frontier (`cand_000`) saved to:
  - `outputs/evaluations/strategy_eval/ratc_c034_vs_c000_base_stress_20260216.csv`
- post-promotion retune branches (rejected):
  - `ratc_from_c034_deribit22_s3870` + transfers (`ratc_s3870_*`)
  - `ratc_from_c034_okx20_s3871` + transfers (`ratc_s3871_*`)
- local manual c034 variants (rejected as new frontier):
  - params: `strategy_params/perp_regime_adaptive_trend_capture_experiments/s3861_c034_refine/*.json`
  - merged summary: `outputs/evaluations/strategy_eval/ratc_c034_refine_crossproxy_summary_20260216_2200.csv`
