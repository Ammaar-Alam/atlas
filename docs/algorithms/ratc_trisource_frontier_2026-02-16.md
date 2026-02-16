# RATC Tri-Source Frontier (Updated 2026-02-16)

## Objective
Track the strongest non-launch-overfit `perp_regime_adaptive_trend_capture` candidate under realistic Coinbase-style costs for a `$500` account.

## Current Best Candidate
- Strategy: `perp_regime_adaptive_trend_capture`
- Frontier params (promoted):
  - `strategy_params/perp_regime_adaptive_trend_capture_trisource_best_s3861_c034.json`
- Original source file:
  - `outputs/evaluations/strategy_strict_gate_search/ratc_from_s3835cand005_coinbase12_s3861_20260216_203856_seed3861/candidates/cand_034.json`

### Exact Params (`cand_034`)
```json
{
  "perp_regime_adaptive_trend_capture": {
    "mom_horizon_a": 255,
    "mom_horizon_b": 638,
    "mom_horizon_c": 753,
    "ema_fast_regime": 71,
    "ema_slow_regime": 426,
    "bear_exit_bps": 86.638,
    "short_entry_bps": 352.853,
    "cooldown_bars": 79,
    "long_base_exposure": 0.3951,
    "short_base_exposure": 0.32479919999999995,
    "extreme_vol_scale": 0.29568000000000005,
    "high_vol_scale": 0.5726448000000002,
    "extreme_vol_rank": 0.861168,
    "high_vol_rank": 0.6812960000000001,
    "vol_lookback_bars": 56,
    "vol_regime_window": 799,
    "crash_threshold_bps": 520.7319040000001,
    "max_hold_bars": 861,
    "rebalance_exposure_threshold": 0.033264,
    "daily_loss_limit": 0.0707,
    "weekly_loss_limit": 0.148176,
    "kill_switch": 0.3802
  }
}
```

## Cost Model (all critical runs)
- `initial_cash=500`
- `max_notional=5000`
- `slippage_bps=1.5`
- `taker_fee_bps=10`
- `coinbase_fee_model=true`
- `fixed_fee_per_contract_usd=0.15`
- `contract_size_units=0.01`
- `allow_short=true`

## Validation Protocol Used
1. Optimize on Coinbase launch rolling windows (12 x 180d):
- Search run:
  - `outputs/evaluations/strategy_strict_gate_search/ratc_from_s3835cand005_coinbase12_s3861_20260216_203856_seed3861`
2. Transfer to Deribit proxy random windows (22 x 180d):
- Eval run:
  - `outputs/evaluations/strategy_eval/ratc_s3861_transfer_deribit22_20260216_205651`
3. Transfer to OKX proxy random windows (20 x 180d):
- Eval run:
  - `outputs/evaluations/strategy_eval/ratc_s3861_transfer_okx20_20260216_205651`
4. Cross-source merge:
- `outputs/evaluations/strategy_eval/ratc_s3861_crossproxy_summary.csv`

## Base Results (`cand_034`)
| Dataset | Runs | Profitable Runs | Profitable Frac | Mean Return | Weekly Positive Frac | Worst Return | Worst Max DD |
|---|---:|---:|---:|---:|---:|---:|---:|
| Deribit proxy | 22 | 8 | 0.364 | +21.51% | 0.211 | -40.24% | -43.46% |
| Coinbase launch | 12 | 12 | 1.000 | +25.47% | 0.400 | +7.12% | -29.02% |
| OKX proxy | 20 | 6 | 0.300 | +23.75% | 0.212 | -41.45% | -43.00% |

Cross-source summary:
- `all_three_mean_pos = true`
- `min_mean = +21.51%`
- `avg_mean = +23.58%`
- `min_prof_frac = 0.30`
- `min_weekly = 0.2109`

## Comparison vs prior robust frontier (`cand_000` lineage)
- Prior robust floor (`cand_000`): `min_mean = +15.41%`
- New robust floor (`cand_034`): `min_mean = +21.51%`
- Improvement in robust floor mean: `+6.10 pts`

Tradeoff:
- `cand_034` raises robust mean strongly.
- `cand_034` has lower worst-source profitable-run fraction than `cand_000` (`0.30` vs `0.40`).

## Pessimistic Stress Checks
Comparison file:
- `outputs/evaluations/strategy_eval/ratc_c034_vs_c000_base_stress_20260216.csv`

### Stress A: `taker_fee_bps=12.5`, `slippage_bps=2.0`
- `cand_034` means: Deribit `+21.15%`, Coinbase `+23.93%`, OKX `+23.69%`
- `cand_000` means: Deribit `+18.80%`, Coinbase `+14.04%`, OKX `+21.17%`
- `cand_034` still retains higher tri-source min-mean.

### Stress B: `taker_fee_bps=15.0`, `slippage_bps=3.0`
- `cand_034` means: Deribit `+14.34%`, Coinbase `+22.34%`, OKX `+22.40%`
- `cand_000` means: Deribit `+16.37%`, Coinbase `+12.43%`, OKX `+20.77%`
- Under this harsher stress, `cand_034` still dominates on Coinbase/OKX and remains tri-source positive.

## Reproduction Commands
### 1) Generate `cand_034` (Coinbase-first search)
```bash
python3 scripts/optimize_strategy_weekly_gates.py \
  --strategy perp_regime_adaptive_trend_capture \
  --base-params outputs/evaluations/strategy_strict_gate_search/ratc_from_stable_c000_deribit22_s3835_20260216_175917_seed3835/candidates/cand_005.json \
  --windows-json outputs/evaluations/coinbase_perp_rolling_180d_20260213/windows.json \
  --seed 3861 --trials 50 --keep-top 25 \
  --label ratc_from_s3835cand005_coinbase12_s3861 \
  --market derivatives --data-source coinbase --bar-timeframe 1H --symbols BTC-PERP \
  --initial-cash 500 --max-notional 5000 --slippage-bps 1.5 --taker-fee-bps 10 \
  --coinbase-fee-model --fixed-fee-per-contract-usd 0.15 --contract-size-units 0.01 \
  --allow-short --weekly-positive-gate 0.70 --weekly-beat-spy-gate 0.60 --drift-frac 0.12
```

### 2) Transfer to Deribit22
```bash
python3 scripts/evaluate_strategy_windows.py \
  --strategy perp_regime_adaptive_trend_capture \
  --windows-json outputs/evaluations/ab_random30_regime_windows_20260215_191958/windows.json \
  --window-indices 1,3,4,5,6,7,8,9,10,11,12,13,17,19,20,21,22,23,24,26,27,30 \
  --params-glob 'outputs/evaluations/strategy_strict_gate_search/ratc_from_s3835cand005_coinbase12_s3861_20260216_203856_seed3861/candidates/*.json' \
  --label ratc_s3861_transfer_deribit22 \
  --market derivatives --data-source csv \
  --csv-path outputs/evaluations/external_perp_probe/20260215_165955/deribit_btc_perpetual_full_1h.csv \
  --bar-timeframe 1H --symbols BTC-PERP \
  --initial-cash 500 --max-notional 5000 --slippage-bps 1.5 --taker-fee-bps 10 \
  --coinbase-fee-model --fixed-fee-per-contract-usd 0.15 --contract-size-units 0.01 \
  --allow-short --min-weekly-gate 0.70
```

### 3) Transfer to OKX20
```bash
python3 scripts/evaluate_strategy_windows.py \
  --strategy perp_regime_adaptive_trend_capture \
  --windows-json outputs/evaluations/ab_random30_regime_windows_20260215_191958/windows.json \
  --window-indices 1,3,4,5,6,7,8,9,10,12,13,17,19,20,21,22,23,24,26,30 \
  --params-glob 'outputs/evaluations/strategy_strict_gate_search/ratc_from_s3835cand005_coinbase12_s3861_20260216_203856_seed3861/candidates/*.json' \
  --label ratc_s3861_transfer_okx20 \
  --market derivatives --data-source csv \
  --csv-path outputs/evaluations/external_perp_probe/20260215_205800/okx_btc_usdt_swap_full_1h.csv \
  --bar-timeframe 1H --symbols BTC-PERP \
  --initial-cash 500 --max-notional 5000 --slippage-bps 1.5 --taker-fee-bps 10 \
  --coinbase-fee-model --fixed-fee-per-contract-usd 0.15 --contract-size-units 0.01 \
  --allow-short --min-weekly-gate 0.70
```

## Gate Status (as of update)
- Weekly-positive hard gate (`>= 0.70`): still not met.
- Best observed robust weekly-positive remains around `~0.21-0.40` depending on source.
- Robust profitability improved materially vs prior frontier by tri-source min-mean.

## Practical status
- Best robust candidate is now `s3861/cand_034`.
- Not gate-complete for the weekly-consistency requirement.
- Use this as the active optimization anchor for next iterations.

## Post-Promotion Refinement Sweep (2026-02-16 late)
Manual local variants around `cand_034` were tested:
- variant set: `strategy_params/perp_regime_adaptive_trend_capture_experiments/s3861_c034_refine/*.json`
- evaluation runs:
  - Deribit22: `outputs/evaluations/strategy_eval/ratc_c034_refine_deribit22_20260216_215040`
  - Coinbase12: `outputs/evaluations/strategy_eval/ratc_c034_refine_cb12_20260216_215040`
  - OKX20: `outputs/evaluations/strategy_eval/ratc_c034_refine_okx20_20260216_215040`
- merged summary:
  - `outputs/evaluations/strategy_eval/ratc_c034_refine_crossproxy_summary_20260216_2200.csv`

Result:
- No variant beat `v00_base` (`cand_034`) on robust floor return (`min_mean`).
- Best alternative by robustness tradeoff was `v01_less_short` (higher `min_prof_frac`), but at materially lower `min_mean`.
- `cand_034` remains promoted best-return tri-source frontier.
