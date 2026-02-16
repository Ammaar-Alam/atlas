# Perp Weekly Profit Chase: Fee 12.5 bps Multiyear Evaluation (2026-02-12)

## Scope
- Strategy family: `perp_weekly_profit_chase`
- Symbol: `BTC/USD`
- Market/data: `derivatives` + `coinbase`
- Bar timeframe: `15Min`
- Windows: `outputs/evaluations/ab_random_year_windows_20260211_020934/windows.json` (10 random 180-day windows from 2016 to 2025)
- Costs: `slippage_bps=1.5`, `taker_fee_bps=12.5`
- Portfolio assumptions used in these runs:
  - `initial_cash=500`
  - `max_notional=2500`
  - `allow_short=true`

## Key finding
No tested candidate in this run set is net profitable across the 10 multiyear windows under `12.5 bps` taker fee.

Best tradeoff found so far:
- Preset file: `strategy_params/perp_weekly_profit_chase_fee12p5_multiyear_tuned_20260212.json`
- Full-10-window stats:
  - profitable runs: `1/10`
  - aggregate weekly positive fraction: `0.676`
  - mean 180d return: `-0.1018`
  - worst max drawdown: `-0.2363`

## Candidate leaderboard snapshot (fee 12.5)
Evaluated from `outputs/evaluations/perp_weekly_profit_chase_candidate_eval/fee12p5_multiyear_*/leaderboard.csv`.

1. `fee12p5_multiyear_tuned5201_r1_20260212_032703`
   - mean return `-10.18%`
   - profitable runs `1/10`
   - aggregate weekly positive `0.676`
   - worst max drawdown `-23.63%`
2. `fee12p5_multiyear_riskrefine_lowdd_r3_20260212_022154`
   - mean return `-11.68%`
   - profitable runs `1/10`
   - aggregate weekly positive `0.628`
   - worst max drawdown `-25.02%`
3. `fee12p5_multiyear_tuned5201_r4_20260212_032703`
   - mean return `-12.36%`
   - profitable runs `0/10`
   - aggregate weekly positive `0.704`
   - worst max drawdown `-35.49%`
4. `fee12p5_multiyear_profitplus_20260212_021334`
   - mean return `-16.31%`
   - profitable runs `4/10`
   - aggregate weekly positive `0.808`
   - worst max drawdown `-80.59%`

## Simulation guardrail update
- Backtest engine update applied in `src/atlas/backtest/derivatives_engine.py`:
  - liquidation now requires open positions + positive maintenance requirement;
  - bankruptcy lockout prevents post-wipeout re-entry spam (caps pathological tails at `-100%` instead of repeatedly compounding after insolvency).
- Smoke validation:
  - `outputs/evaluations/perp_weekly_profit_chase_candidate_eval/smoke_liqlog_fix_20260212_035047`
  - `outputs/evaluations/perp_weekly_profit_chase_candidate_eval/smoke_liqlog_fix_v2_20260212_035209`

## Why `BTC-PERP` history starts in 2025-07 in this environment
- Local Coinbase API candles check shows first daily candle for `BTC-PERP` at `2025-07-18`.
- `BTC/USD` candle history in the same API extends back to `2015-07-20`.
- Operational implication:
  - decade-scale random-window tuning cannot be done on Coinbase perp candles yet;
  - multiyear robustness runs in this repo therefore use `BTC/USD` as the long-history proxy for feature/risk tuning and then re-check on available perp history.

## Commands used (representative)
```bash
PYTHONPATH=src python3 scripts/evaluate_perp_weekly_profit_chase_candidates.py \
  --windows-json outputs/evaluations/ab_random_year_windows_20260211_020934/windows.json \
  --params-file strategy_params/perp_weekly_profit_chase_algo_profit_plus_15m_coinbase_profile.json \
  --label fee12p5_multiyear_profitplus \
  --symbol BTC/USD --market derivatives --data-source coinbase \
  --bar-timeframe 15Min --prewarm-days 90 \
  --initial-cash 500 --max-notional 2500 \
  --slippage-bps 1.5 --taker-fee-bps 12.5 --allow-short
```

```bash
PYTHONPATH=src python3 scripts/optimize_perp_weekly_profit_chase_random_windows.py \
  --base-params outputs/evaluations/perp_weekly_profit_chase_search/riskrefine_full10_lowdd_20260211_200755_seed4301/candidates/riskrefine_full10_lowdd_seed4301_rank03_prof4of10_wgate3of10.json \
  --windows-json outputs/evaluations/ab_random_year_windows_20260211_020934/windows.json \
  --window-indices 1,3,5,7,9 \
  --seed 5201 --trials 8 --keep-top 4 \
  --label fee12p5_lowdd_r3_w13579_s5201 \
  --symbol BTC/USD --market derivatives --data-source coinbase \
  --bar-timeframe 15Min --prewarm-days 90 \
  --initial-cash 500 --max-notional 2500 \
  --slippage-bps 1.5 --taker-fee-bps 12.5 --allow-short
```
