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
atlas paper --help
atlas download-bars --help
```

## Data sources

Backtests can use:

- `sample` (bundled CSV under `data/sample/`)
- `csv` (a user-supplied CSV file)
- `alpaca` (downloads minute bars via Alpaca Market Data into `data/alpaca/` and uses them)

Example (Alpaca):

```bash
atlas download-bars --symbol SPY --start 2024-01-02T09:30:00-05:00 --end 2024-01-02T16:00:00-05:00
atlas backtest --symbol SPY --data-source alpaca --start 2024-01-02T09:30:00-05:00 --end 2024-01-02T16:00:00-05:00
```

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
- No commissions/fees
- Optional slippage via `--slippage-bps`
