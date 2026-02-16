# RATC Implementation + Validation (2026-02-15)

## Scope
Implemented a new strategy candidate from the regime-adaptive brief as a separate strategy:
- `perp_regime_adaptive_trend_capture`
- file: `src/atlas/strategies/perp_regime_adaptive_trend_capture.py`
- registry wired: `src/atlas/strategies/registry.py`
- TUI param/default wiring: `src/atlas/tui/app.py`
- presets:
  - `strategy_params/perp_regime_adaptive_trend_capture_cb10_conservative_1h.json`
  - `strategy_params/perp_regime_adaptive_trend_capture_cb10_balanced_1h.json`
  - `strategy_params/perp_regime_adaptive_trend_capture_cb10_aggressive_1h.json`

## Key implementation details
- Long-biased state machine with cooldown and crash override.
- Contract-size aware sizing for Coinbase nano-perp assumptions.
- Event-driven hold fix to avoid over-trading from one-lot quantization at next-bar-open execution.
- Risk controls: daily/weekly loss lockouts + kill switch.

## Exact CLI commands used

### Single-window smoke (180d, Coinbase fee model)
```bash
python3 -m atlas.cli backtest \
  --market derivatives \
  --symbols BTC-PERP \
  --data-source coinbase \
  --bar-timeframe 1H \
  --start 2025-08-19T00:00:00Z \
  --end 2026-02-15T18:00:00Z \
  --strategy perp_regime_adaptive_trend_capture \
  --strategy-params strategy_params/perp_regime_adaptive_trend_capture_cb10_balanced_1h.json \
  --initial-cash 500 \
  --max-position-notional-usd 5000 \
  --slippage-bps 1.5 \
  --taker-fee-bps 10 \
  --coinbase-fee-model \
  --fixed-fee-per-contract-usd 0.15 \
  --contract-size-units 0.01 \
  --allow-short
```

### Launch-era 12-window evaluation
```bash
python3 scripts/evaluate_strategy_windows.py \
  --strategy perp_regime_adaptive_trend_capture \
  --params-file strategy_params/perp_regime_adaptive_trend_capture_cb10_balanced_1h.json \
  --windows-json outputs/evaluations/coinbase_perp_rolling_180d_20260213/windows.json \
  --label ratc_balanced_launch12_cb10 \
  --market derivatives \
  --data-source coinbase \
  --bar-timeframe 1H \
  --symbols BTC-PERP \
  --initial-cash 500 \
  --max-notional 5000 \
  --slippage-bps 1.5 \
  --taker-fee-bps 10 \
  --coinbase-fee-model \
  --fixed-fee-per-contract-usd 0.15 \
  --contract-size-units 0.01 \
  --allow-short \
  --min-weekly-gate 0.7
```

### External multi-year reality probe
```bash
python3 scripts/probe_external_perp_reality.py \
  --strategy perp_regime_adaptive_trend_capture \
  --params-file strategy_params/perp_regime_adaptive_trend_capture_cb10_balanced_1h.json \
  --symbol BTC-PERP \
  --initial-cash 500 \
  --max-notional 5000 \
  --slippage-bps 1.5 \
  --taker-fee-bps 10 \
  --fixed-fee-per-contract-usd 0.15 \
  --contract-size-units 0.01 \
  --allow-short
```

## Results summary

### Single 180d smoke runs
- Conservative run dir: `outputs/backtests/backtest_20260215_155559_239112_88424_0c34`
  - return: `+10.3255%`
  - fills: `6`
  - max drawdown: `-15.5628%`
  - beat SPY: `yes` on this window
- Balanced run dir: `outputs/backtests/backtest_20260215_155624_330556_88629_65f4`
  - return: `+51.6158%`
  - fills: `16`
  - max drawdown: `-22.1668%`
  - beat SPY: `yes` on this window
- Aggressive run dir: `outputs/backtests/backtest_20260215_155624_330871_88630_5529`
  - return: `-30.8870%`
  - fills: `26`
  - max drawdown: `-31.5860%`
  - beat SPY: `no`

### Launch12 (balanced)
- out dir: `outputs/evaluations/strategy_eval/ratc_balanced_launch12_cb10_20260215_205714`
- profitable runs: `12/12`
- mean 180d return: `+42.9859%`
- aggregate weekly positive fraction: `0.3267`
- weekly gate (>=0.70): `0/12`
- conclusion: strong launch-era profitability, fails weekly-positive consistency gate.

### Launch12 (conservative)
- out dir: `outputs/evaluations/strategy_eval/ratc_conservative_launch12_cb10_20260215_205933`
- profitable runs: `0/12`
- mean 180d return: `-15.8941%`
- aggregate weekly positive fraction: `0.1067`

### Launch12 (aggressive)
- out dir: `outputs/evaluations/strategy_eval/ratc_aggressive_launch12_cb10_20260215_205933`
- profitable runs: `0/12`
- mean 180d return: `-24.2568%`
- aggregate weekly positive fraction: `0.0367`

### External reality probe (balanced)
- out dir: `outputs/evaluations/external_perp_probe/20260215_205800`
- best overlap source: `okx_btc_usdt_swap`
- launch summary:
  - profitable runs: `0/12`
  - mean return: `-16.1683%`
  - aggregate weekly positive fraction: `0.0367`
- random summary:
  - profitable runs: `1/6`
  - mean return: `-4.4609%`
  - aggregate weekly positive fraction: `0.16`
- conclusion: fails external multi-year robustness.

## Gate assessment vs hard target
- Weekly-positive fraction >= 0.70: **FAIL**
- Beat SPY in >= 60% of windows: **FAIL** (outside launch in external probe)
- Mean and median return > 0 across robust multi-window protocol: **FAIL**

## Current status
- Strategy is fully wired and runnable in TUI/CLI.
- Strategy is **not deployment-ready** under the hard robustness gates.
- Balanced profile is the strongest launch-era candidate; external-reality robustness remains insufficient.

## Post-implementation iteration (external proxy optimization)

Additional tuning was run on OKX 1H proxy bars with Coinbase fee model active, using combined windows:
- launch12 + random6 (2020-2025)
- windows file: `outputs/evaluations/windows_okx_launch12_plus_random6_20260215.json`

Search runs:
- Short-enabled search (81 candidates):
  - out dir: `outputs/evaluations/strategy_strict_gate_search/ratc_okx_launch12_random6_cb10_20260215_214245_seed21`
  - best candidate: `cand_055`
  - saved params: `strategy_params/perp_regime_adaptive_trend_capture_okxmix_bestscore_seed21_cand055.json`
  - metrics (18 windows):
    - profitable runs: `14/18`
    - mean total return: `+24.07%`
    - mean alpha vs SPY: `+15.52%`
    - worst 180d return: `-37.43%`
    - worst max drawdown: `-46.31%`
    - mean weekly-positive fraction: `0.3333`

- Long-only search (61 candidates):
  - out dir: `outputs/evaluations/strategy_strict_gate_search/ratc_okx_launch12_random6_cb10_longonly_20260215_221055_seed33`
  - best candidate: `cand_016`
  - saved params: `strategy_params/perp_regime_adaptive_trend_capture_okxmix_longonly_bestseed33_cand016.json`
  - metrics (18 windows):
    - profitable runs: `15/18`
    - mean total return: `+8.14%`
    - mean alpha vs SPY: `-0.41%`
    - worst 180d return: `-24.85%`
    - worst max drawdown: `-26.48%`
    - mean weekly-positive fraction: `0.1489`

- Hardening search around `cand_055` with launch12 + bear5 objective:
  - out dir: `outputs/evaluations/strategy_strict_gate_search/ratc_hardened_from_cand55_launch12_bear5_cb10_20260215_223740_seed44`
  - no candidate improved the base `cand_055` trade-off (higher downside persisted).

Conclusion from the additional iteration:
- A materially profitable cross-window RATC variant exists (`cand_055`) under proxy testing.
- It still fails the weekly-positive gate and has unacceptable downside tails for strict deployment.

Cross-source sanity check (important):
- Running `cand_055` on actual Coinbase launch-era bars for `2025-08-19` to `2026-02-15` produced:
  - return: `-25.66%`
  - run: `outputs/backtests/backtest_20260215_180451_139125_33938_d8bc`
- This confirms source-fragility: proxy-optimized parameters do not transfer cleanly to Coinbase bars.
