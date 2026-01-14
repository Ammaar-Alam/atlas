# atlas

Research scaffold for intraday US-equity strategies:

- Backtest intraday strategies on minute bars (CSV or Alpaca Market Data)
- Paper trade intraday in real time using Alpaca paper trading (safe by default)

This repository includes a minimal baseline strategy (`ma_crossover`) as an executable example and sanity-check for the data → backtest → ledger/equity outputs and paper-execution plumbing. It is not intended to be the research contribution.

Strategy extension points and a spec placeholder for research algorithms:

- `STRATEGY.md:1`
- `strategy_spec.json:1`

Disclaimer: this is research/educational software and is not financial advice.

## Quickstart

### 1) Create and activate a virtualenv

```bash
cd atlas
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -e .
```

### 2) Run a local backtest (no API keys required)

Uses bundled sample minute bars in `data/sample/`.

```bash
atlas backtest --symbol SPY --data-source sample
```

At the end of the run, the CLI prints a short backtest summary (window, bar count, strategy, and core metrics).

Outputs land in `outputs/backtests/{run_id}/`:

- `trades.csv` and `trades.json`
- `equity_curve.csv`
- `metrics.json`
- `run.log`

### 3) Alpaca paper trading setup

1. Create an Alpaca account and enable paper trading
2. Create paper API keys in the Alpaca dashboard
3. Copy `.env.example` to `.env` and fill the keys

`.env` must include:

- `ALPACA_API_KEY` and `ALPACA_SECRET_KEY`
- `ALPACA_PAPER=true` (default)

### 4) Run the paper trading loop

Runs the paper trading loop, submits paper orders, and logs fills.

```bash
atlas paper --symbols SPY --strategy ma_crossover --fast-window 10 --slow-window 30
```

Outputs land in `outputs/paper/{run_id}/`:

- `orders.csv`
- `orders.jsonl`
- `fills.csv`
- `fills.jsonl`
- `equity_curve.csv`
- `run.log`

## Safety: paper by default

- Paper trading is the default (`ALPACA_PAPER=true`)
- Live trading is blocked unless both are set:
  - `ALPACA_PAPER=false`
  - `ATLAS_ALLOW_LIVE=true`

If live endpoints are attempted without the explicit allow flag, the CLI exits with an error before placing orders.

## Environment variables

Copy `.env.example` to `.env`.

- `ALPACA_API_KEY` / `ALPACA_SECRET_KEY`: required for Alpaca data + paper loop
- `ALPACA_PAPER`: `true` (default) uses paper trading endpoints
- `ATLAS_ALLOW_LIVE`: must be `true` to allow `ALPACA_PAPER=false`
- `ATLAS_LOG_LEVEL`: `INFO` by default
- `ATLAS_BACKTEST_MAX_POSITION_NOTIONAL_USD`: default max notional per symbol in backtests
- `ATLAS_PAPER_MAX_POSITION_NOTIONAL_USD`: default max notional per symbol in paper loop

## CLI

```bash
atlas --help
atlas backtest --help
atlas tune --help
atlas paper --help
atlas download-bars --help
atlas tui
```

`atlas tui` launches a full-screen terminal UI with live settings/results panels. Use slash commands such as:

- `/backtest`
- `/tune` (walk-forward parameter optimization)
- `/timeframe 7d` (also supports `6h`, `1m`, `1y`, `clear`)
- `/algorithm ma_crossover`
- `/paper start` / `/paper stop`
- `/save` / `/load`

## Data sources

Backtests can use:

- `sample` (bundled CSV under `data/sample/`)
- `csv` (a user-supplied CSV file)
- `alpaca` (downloads minute bars via Alpaca Market Data into `data/alpaca/` and uses them)
- `coinbase` (public Coinbase market-data for spot + futures/perps)

Example (Alpaca):

```bash
atlas download-bars --symbol SPY --start 2024-01-02T09:30:00-05:00 --end 2024-01-02T16:00:00-05:00
atlas backtest --symbol SPY --data-source alpaca --start 2024-01-02T09:30:00-05:00 --end 2024-01-02T16:00:00-05:00
```

Example (Coinbase futures/perps):

```bash
# Market mode enables the derivatives backtest engine.
# For Coinbase US futures/perps, Atlas accepts "BTC-PERP"/"ETH-PERP" and resolves to the
# current Coinbase contract-style product_id (e.g. "BIP-20DEC30-CDE").
atlas backtest --market derivatives --symbol BTC-PERP --data-source coinbase --bar-timeframe 5Min --start 2026-01-01T00:00:00Z --end 2026-01-02T00:00:00Z
```

## Walk-forward parameter tuning (Option A)

Atlas includes a walk-forward hyperparameter optimizer to tune strategy knobs on rolling train/validate/test windows.

Example (PerpFlare on Coinbase BTC-PERP):

```bash
atlas tune --market derivatives --symbol BTC-PERP --data-source coinbase --bar-timeframe 5Min --timeframe 60d --trials-per-segment 60 --train 30d --validate 7d --test 7d --step 7d
```

The run directory includes:

- `best_params.json` (strategy-keyed params usable via `--strategy-params`)
- `best_params_stable.json` (strategy-keyed, stability-biased params; median of recent segments)
- `selections.json` (chosen params per segment)
- `stability.json` (stability summary across segments)

## Strategy wiring

The dummy strategy lives here:

- `src/atlas/strategies/ma_crossover.py`

To add a new strategy:

1. Write a new file under `src/atlas/strategies/` implementing `Strategy`
2. Register it in `src/atlas/strategies/registry.py`

More details: `STRATEGY.md:1`

## Backtest model (current assumptions)

- Strategy sees bars up to the current bar close
- Orders (when the target exposure changes) fill at the next bar open
- Equities/spot: no commissions/fees (today)
- Derivatives: taker fees and liquidation/funding adjustments are modeled (funding requires `funding_rate` data)
- Optional slippage via `--slippage-bps`
