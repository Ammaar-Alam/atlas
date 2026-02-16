# Algorithm Record: `perp_trend_vol_guard` (cand08)

## Purpose
Current strongest launch-era BTC-PERP candidate under strict Coinbase fee + lot assumptions for a `$500` account.

## Strategy + Params
- strategy: `perp_trend_vol_guard`
- params file: `strategy_params/trend_guard_manual_chase_grid2/cand08.json`

```json
{
  "perp_trend_vol_guard": {
    "ema_fast": 5,
    "ema_slow": 65,
    "momentum_window_bars": 7,
    "breakout_window": 22,
    "breakout_buffer_bps": 11.463,
    "atr_window": 17,
    "trend_strength_min": 0.778,
    "min_atr_bps": 6.796600000000001,
    "edge_floor_bps": 8.428896000000003,
    "k_cost": 1.3702,
    "risk_budget": 0.065,
    "stop_atr_mult": 5.932,
    "target_vol_bps_per_bar": 27.03,
    "max_positions": 3,
    "max_gross_exposure": 1.5,
    "max_per_symbol_exposure": 0.9,
    "rebalance_interval_bars": 3,
    "rebalance_exposure_threshold": 0.025500000000000005,
    "min_hold_bars": 17,
    "flip_confirm_bars": 7,
    "market_vol_reduce_bps": 30.059,
    "market_vol_off_bps": 298.3908,
    "weekly_loss_limit": 0.0419,
    "weekly_profit_target": 0.009519999999999999,
    "weekly_lock_risk_scale": 0.18734,
    "fallback_floor_exposure": 0.1239,
    "fallback_trend_strength_min": 0.0,
    "fallback_min_momentum_bps": 0.0,
    "fallback_min_atr_bps": 6.8238,
    "daily_loss_limit": 0.0394,
    "kill_switch": 0.2648,
    "weekly_chase_target": 0.012,
    "weekly_chase_k": 9.0,
    "weekly_chase_max_extra_exposure": 0.45,
    "weekly_chase_start_weekday_utc": 3
  }
}
```

## Backtest Configuration
- market/data: `derivatives` / `coinbase`
- symbol: `BTC-PERP`
- bar timeframe: `1H`
- windows: `outputs/evaluations/coinbase_perp_rolling_180d_20260213/windows.json`
- initial cash: `500`
- max notional: `5000`
- costs:
  - slippage: `1.5 bps`
  - taker fee: `10.0 bps`
  - fixed fee: `$0.15/contract/side`
  - contract size: `0.01 BTC`

## Primary Validation Result
- eval run: `outputs/evaluations/strategy_eval/trend_guard_manual_chase_grid2_full12_20260214_071137`
- candidate id: `trend_guard_manual_chase_grid2__cand08`
- runs: `12`
- profitable runs: `12/12`
- mean 180d return: `+20.3438%`
- median 180d return: `+26.2177%`
- worst 180d return: `+3.6889%`
- aggregate weekly positive fraction: `0.6467`
- weekly gate runs (`>=0.70`): `4/12`
- mean max drawdown: `-18.2257%`

## Additional Strict Search Signal
- run: `outputs/evaluations/strategy_strict_gate_search/trend_guard_local_from_cand08_s8101_20260214_072759_seed8101`
- `cand_000` (same params) summary:
  - mean alpha vs SPY: `+11.6062%`
  - profitable+beat-SPY runs: `9/12`
  - weekly_gate_and_beat_count: `4/12`

## Random-Year Proxy Stress (BTC/USD)
- run: `outputs/evaluations/strategy_eval/trend_guard_cand08_family_randomyears_20260214_075823`
- candidate id: `trend_guard_manual_chase_grid2__cand08`
- runs: `10`
- profitable runs: `0/10`
- mean return: `-27.9902%`
- aggregate weekly positive fraction: `0.036`

## Reproduce Command
```bash
PYTHONPATH=src python3 scripts/evaluate_strategy_windows.py \
  --strategy perp_trend_vol_guard \
  --params-file strategy_params/trend_guard_manual_chase_grid2/cand08.json \
  --windows-json outputs/evaluations/coinbase_perp_rolling_180d_20260213/windows.json \
  --label trend_guard_cand08_repro \
  --symbol BTC-PERP \
  --market derivatives \
  --data-source coinbase \
  --bar-timeframe 1H \
  --prewarm-days 45 \
  --initial-cash 500 \
  --max-notional 5000 \
  --slippage-bps 1.5 \
  --taker-fee-bps 10.0 \
  --coinbase-fee-model \
  --fixed-fee-per-contract-usd 0.15 \
  --contract-size-units 0.01 \
  --allow-short \
  --min-weekly-gate 0.70
```
