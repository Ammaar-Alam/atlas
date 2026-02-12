# Strategy plug-in guide

This repo is a scaffold. The idea is:

- backtests and paper trading both call a `Strategy` to get a target exposure
- the broker/execution layer turns that target exposure into orders

## Where to implement your novel algorithm

Add a new strategy module under:

- `src/atlas/strategies/`

Implement the `Strategy` interface:

- `src/atlas/strategies/base.py:1`

Then register it so the CLI can build it:

- `src/atlas/strategies/registry.py:1`

## How the strategy is called

Backtest:

- `src/atlas/backtest/engine.py:1` calls `strategy.target_exposure(...)` once per bar
- orders fill at the next bar open (simple, avoids lookahead)
- the engine can model per-side slippage and taker fees (see `atlas backtest --help`)

Paper:

- `src/atlas/paper/runner.py:1` fetches the latest minute bars
- it calls `strategy.target_exposure(...)` and places Alpaca paper orders if the target changes

## Validation (recommended)

Run walk-forward validation (fixed params, cost stress testing supported) via:

- `atlas validate --help`
- `atlas validate --overfit-report` (writes `overfit_report.json` with bootstrap confidence intervals)

## Param presets

This repo includes a few strategy param presets under:

- `strategy_params/`

You can load these in the TUI via:

- `/preset list`
- `/preset <name>` (e.g. `/preset crypto_tsm_stable_6h_coinbase`)

## Strategy spec placeholder

Use `strategy_spec.json` as a scratchpad for the next model’s spec, parameters, and notes.
