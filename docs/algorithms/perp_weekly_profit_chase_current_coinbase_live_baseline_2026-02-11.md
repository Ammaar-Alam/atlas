# `perp_weekly_profit_chase` Coinbase Live Baseline (2026-02-11)

## Purpose
Pin the exact current baseline configuration and measured results before further tuning.

## Canonical Preset
- Strategy params file:
  - `strategy_params/perp_weekly_profit_chase_algo_profit_plus_15m.json`
- TUI session profile file:
  - `strategy_params/perp_weekly_profit_chase_algo_profit_plus_15m_coinbase_profile.json`

## Current TUI Baseline Settings
- `market=derivatives`
- `data_source=coinbase`
- `bar_timeframe=15Min`
- `timeframe=180d`
- `initial_cash=500`
- `max_position_notional_usd=2500`
- `slippage_bps=1.5`
- `allow_short=true`
- `strategy=perp_weekly_profit_chase`

## Fee Assumption
- From Coinbase order preview screenshot:
  - `notional_value ≈ 673.40`
  - `fee ≈ 0.84`
- Implied taker fee:
  - `taker_fee_bps = 0.84 / 673.40 * 10000 = 12.474012`
- Operational setting:
  - use `taker_fee_bps=12.5` (rounded conservative setting)

## Data Coverage Constraint (Critical)
- Coinbase `BTC-PERP` daily history currently observed in Atlas:
  - start: `2025-07-18`
  - end: `2026-02-12`
  - rows: `210` daily candles
- Coinbase `BTC/USD` daily history currently observed in Atlas:
  - start: `2015-07-20`
  - end: `2026-02-12`
  - rows: `3861` daily candles

This means decade-scale validation on true `BTC-PERP` candles is not possible with currently available Coinbase perp history.

## Reproduced 180d Results (same window from TUI screenshot)
- Window:
  - `2025-08-15T19:23:22.372335-04:00` to `2026-02-11T19:23:22.372335-05:00`

### A) Derivatives engine + `BTC/USD` + real fee (12.474012)
- Command run output:
  - `outputs/backtests/backtest_20260211_192507_808021_53560_b2e7`
- Summary:
  - `total_return=+2.9889%`
  - `max_drawdown=-21.8801%`
  - `sharpe=0.38`
  - `fills=136`

### B) Derivatives engine + `BTC/USD` + old fee assumption (6.0)
- Command run output:
  - `outputs/backtests/backtest_20260211_192617_547807_55537_3c14`
- Summary:
  - `total_return=+10.2621%`
  - `max_drawdown=-17.4202%`
  - `sharpe=1.15`
  - `fills=132`

### C) Derivatives engine + `BTC-PERP` + real fee (12.474012)
- Command run output:
  - `outputs/backtests/backtest_20260211_192507_807595_53561_4201`
- Summary:
  - `total_return=-5.4480%`
  - `max_drawdown=-27.5823%`
  - `sharpe=-0.41`
  - `fills=134`

## Immediate Next Actions
1. Keep this baseline pinned as rollback/reference.
2. Tune against `BTC-PERP` with `taker_fee_bps=12.5`, `slippage_bps=1.5`.
3. Validate on all feasible random windows within available perp history.
4. In parallel, add Coinbase execution path (broker + order submit/polling + dry-run safety).
