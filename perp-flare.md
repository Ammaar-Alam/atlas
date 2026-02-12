Below is a code-first integration roadmap for adding **`perp_flare`** into your existing Atlas-style repository, with a focus on *where* the code goes, *what* each module should do, and *how* the components should interact end-to-end.

I’m basing this on:

* The repo’s current architecture: strategy interface + registry + universe loader + spot backtest engine + Alpaca paper runner. 
* The PerpFlare spec: features, signal timing (signal on close → fill next open), admission controller, position sizing, liquidation-buffer gate, and the repo-level extension plan (parallel derivatives engine, Coinbase loaders, Coinbase runner). 

---

## Architecture target

You want PerpFlare to be a first-class strategy that can run in two modes:

1. **Derivatives backtest mode**
   `atlas.cli backtest --market derivatives --data-source coinbase ...` should:

* load Coinbase candles (+ funding),
* run a dedicated **derivatives backtest engine** that models funding + fees + margin + liquidation,
* write outputs with extra fields (fees, spread, slippage, funding, liquidations). 

2. **Derivatives paper/live loop**
   `atlas.cli paper --market derivatives --strategy perp_flare ...` should:

* fetch Coinbase candles live,
* compute exposures,
* place Coinbase perp orders (or dry-run log them),
* enforce safety gates and flatten-on-error. 

This matches the “parallel engine” approach (keep the existing spot backtester untouched) described in the markdown. 

---

## Phase 1 Add the market mode and symbol plumbing

### 1. Add `Market.DERIVATIVES` and symbol defaults

**File:** `src/atlas/market.py` (modify) 

**Tasks**

* Extend the `Market` enum to include `DERIVATIVES`.
* Extend `parse_market()` to accept `"derivatives"` (and optionally synonyms like `"perps"`).
* Extend `default_symbols(mkt)`:

  * `Market.DERIVATIVES` should default to `["BTC-PERP", "ETH-PERP"]` per the paper spec. 
* Extend `coerce_symbols_for_market()`:

  * For derivatives, normalize symbols to uppercase and ensure they are in the expected Coinbase-perp format.
  * Decide a single canonical format early (recommended: `BTC-PERP`, `ETH-PERP`) and stick to it across: CLI, universe loader, strategy, broker.

**Acceptance check**

* `parse_market("derivatives")` works.
* CLI can accept `--market derivatives` without failing before it even reaches loaders/engines.

---

## Phase 2 Add PerpFlare strategy module

### 2. Implement `src/atlas/strategies/perp_flare.py`

**Files:**

* New: `src/atlas/strategies/perp_flare.py` 

**Goal**
Implement `Strategy` with `warmup_bars()` and `target_exposures(...)` using the spec’s mechanics:

* features (ATR, EMAs, trend score, ER, Donchian HH/LL, BTC/ETH relative strength),
* directional setup rules,
* admission controller (edge-over-friction with funding penalty),
* liquidation-buffer gate,
* exits (initial stop, trailing stop, time stop). 

The repo’s strategy interface is:

* `warmup_bars() -> int`
* `target_exposures(bars_by_symbol, state) -> StrategyDecision` returning `target_exposures: dict[str, float]`. 

#### 2.1 Decide how PerpFlare receives funding

You need funding `f_t` for the admission controller and for logging. The strategy signature doesn’t include funding directly, so you have two clean options:

**Option A (recommended): funding column inside bars**

* Universe loader attaches a `funding_rate` column (aligned to bar timestamps) to each symbol DataFrame.
* Strategy reads `bars["funding_rate"].iloc[-1]`.

**Option B: separate `funding_by_symbol` side channel**

* Extend `StrategyState` or pass via `debug`/global… but this ripples across all strategies/engines. Not recommended.

The markdown’s approach assumes the loader supports funding. 

#### 2.2 Implement feature functions in-strategy

Implement small internal helpers, mirroring existing strategy patterns (OrbTrend already computes ER and uses a cost gate). 

* `_true_range(df)` and `_atr(df, w_atr)` per spec. 
* `_ema(series, n)` per spec. 
* `_efficiency_ratio(close, w_er)` per spec. 
* `_donchian(df, w_bk)` returning `HH_t`, `LL_t` excluding current bar. 
* `_relative_strength_selector(btc_close, eth_close, fast, slow)` using EMA of log ratio. 

#### 2.3 Implement signal rules and score

For each symbol:

* Determine long/short candidacy using Donchian + trend score + ER gate. 
* Compute breakout strength `B_t` in bps and Edge proxy in bps. 
* Compute friction estimate:

  * `Fric_t = 2*(halfSpread_bps + baseSlip_bps + takerFee_bps) + λ_f * |f_t|*1e4` (as specified). 
* Apply admission threshold:

  * Trade only if `Edge_t >= EdgeFloor + k_cost * Fric_t`. 
* Score candidates and pick the highest. 

**Important integration choice:**
The *strategy* should not “simulate fills”; it just outputs exposures and reasons. Your engines are already responsible for “fill at next open” mechanics. (That’s consistent with the repo’s current engine and the PerpFlare spec.) 

#### 2.4 Implement sizing as exposure

The spec defines:

* risk budget `R = ρ * Equity_t`
* stop distance `D = m_stop * ATR`
* qty `q = R/D`
* notional `N = q*C_t`
* exposure `x = sign * min( N/(Equity_t*L_max), x_cap )` 

In this repo, “exposure” is what the engine consumes. So you implement:

* compute `x` per above,
* return `{chosen_symbol: x, others: 0.0}`.

#### 2.5 Implement liquidation-buffer gate and exit logic

Strategy-level liquidation buffer gate:

* compute `P_liq` approximations from equity, qty, entry, mmr,
* require distance-to-liq in ATR units ≥ `m_liq`. 

Exits:

* initial stop, trailing stop updates, time stop `H_max`. 

**Where to store entry price / stop state**

* Keep internal mutable fields in the strategy object:

  * `_entry_price: dict[symbol, float]`
  * `_stop_price: dict[symbol, float]`
  * `_holding_bars_internal` if needed (but `state.holding_bars` already exists) 
* Because the engine is stateless re: “entry” right now, strategy should update these on “position changed” events (see Phase 4 where the derivatives engine can also emit fills to the strategy via debug; if you keep interface unchanged, strategy infers entry from fills stored in engine outputs only, which is too late; better: strategy tracks entry when it commands a new exposure and sees position qty flip sign or go from 0 to non-zero).

**Acceptance checks**

* `perp_flare` imports cleanly.
* Calling `target_exposures` on dummy bars returns a `StrategyDecision` with:

  * deterministic keys for debug (edge_bps, friction_bps, chosen, funding_rate, ATR, ER, stop, etc.),
  * no crashes when funding column missing (should degrade safely to `f_t=0` but log `reason="no_funding"` or include debug flag).

---

## Phase 3 Register the strategy so CLI and TUI can build it

### 3. Update `src/atlas/strategies/registry.py`

**File:** `src/atlas/strategies/registry.py` (modify) 

**Tasks**

* Add `"perp_flare"` to `list_strategy_names()` (current list doesn’t include it). 
* Update `build_strategy(...)`:

  * import `PerpFlare` and create it when `name == "perp_flare"` (or canonicalized form).
  * Pass in parameters via the repo’s existing `params` dictionary mechanism.

**Acceptance checks**

* `python -m atlas.cli strategies` (or equivalent) shows `perp_flare` if you have a listing command.
* Backtest and paper commands can select `--strategy perp_flare` without failing at build time.

---

## Phase 4 Add Coinbase data source and funding support in universe loader

### 4. Add Coinbase candle loader

**Files**

* Modify: `src/atlas/data/universe.py` 
* New: `src/atlas/data/coinbase_data.py` 

**Tasks**

* Extend `load_universe_bars(...)` to accept `data_source="coinbase"` (currently it’s `sample|csv|alpaca`). 
* Implement `coinbase_data.load_coinbase_candles_cached(...)`:

  * Input: symbol (`BTC-PERP`), timeframe (`1Min`), start/end
  * Output: DataFrame with `open, high, low, close, volume`, indexed by tz-aware timestamps
  * Cache locally (parquet/csv) so repeated backtests don’t hammer the API
* Ensure bars are aligned across symbols (your backtest engine intersects common indices). That’s already how spot backtests work. 

### 5. Add funding loader and bar alignment

**Files**

* New: `src/atlas/data/coinbase_funding.py` 
* Modify: `src/atlas/data/universe.py` (to attach funding) 

**Tasks**

* Implement a funding loader that can:

  * load from an API (optional credentials), or
  * load from a CSV directory fallback (explicitly called out in the runbook). 
* Convert funding events to a bar-aligned series:

  * For each bar timestamp `t`, attach the funding rate that will apply at/after `t` (choose and document alignment rule).
* Attach to each symbol’s bars:

  * Add column: `funding_rate` as decimal rate (e.g., 0.0001).
  * Strategy will use it in admission controller; engine will use it for funding cashflows. 

**Acceptance checks**

* `load_universe_bars(..., data_source="coinbase")` returns bars with required columns plus `funding_rate`.
* Missing funding gracefully becomes 0 (but you log that funding is absent, since ignoring funding is explicitly called out as a way backtests lie). 

---

## Phase 5 Build the derivatives backtest engine

The markdown is explicit that the existing backtester is “not suitable” for perps because it lacks margin/leverage/funding/liquidation and explicit fee/spread models. So implement a dedicated engine. 

### 6. Create `src/atlas/backtest/derivatives_engine.py`

**Files**

* New: `src/atlas/backtest/derivatives_engine.py` 
* Optionally reuse: `src/atlas/backtest/metrics.py` for baseline metrics

**Core requirements**
You should preserve the repo’s conservative timing model:

* strategy sees history up to `t`,
* fills occur at `t+1 open`. 

**Engine responsibilities**
Implement the portfolio mechanics the strategy assumes exist:

#### 6.1 Config surface

Create a config dataclass similar to `BacktestConfig`, but with derivatives knobs:

* `initial_equity`
* `venue_max_leverage` (L_max)
* `max_margin_utilization` (x_cap / utilization cap)
* `maintenance_margin_rate` (mmr)
* `taker_fee_bps`, `half_spread_bps`, `base_slippage_bps`
* `cooldown_bars`, `daily_loss_limit`, `weekly_loss_limit`, `max_drawdown`, maintenance window definition, etc. 

You’ll also want to include `funding_csv_dir` in the orchestration layer (CLI), not inside the engine config directly.

#### 6.2 Fill model and cost decomposition

Per fill:

* Determine delta qty needed based on target exposure.
* Fill price should reflect:

  * half-spread,
  * slippage,
  * taker fee.
* Record costs separately:

  * `fees_usd`, `spread_usd`, `slippage_usd` so your outputs match the runbook expectations. 

#### 6.3 Funding cashflows

At funding timestamps τ:

* apply `Δcash = - q * P * f` (linear perp cashflow) per spec. 
* Record `funding_usd` cumulatively.

#### 6.4 Margin and liquidation checks

Implement:

* margin utilization cap (don’t exceed `max_margin_utilization`),
* maintenance margin requirement via `mmr`,
* liquidation behavior and counting `liquidations` (target 0). 

Even if you model liquidation conservatively, keep it deterministic and auditable:

* Document whether you use bar low/high for “worst intrabar” checks.

#### 6.5 Engine-level risk controls

Portfolio-level controls belong in the engine per spec:

* daily loss, weekly loss, max drawdown → flatten + halt,
* cooldown after stop-out or risk breach,
* data gap breaker,
* maintenance window flatten/no-trade. 

**Outputs**
Match the repo’s output pattern:

* `equity_curve.csv`
* `trades.csv`
* `decisions.jsonl`
* `metrics.json`

Add additional fields in `metrics.json`:

* `fees_usd`, `spread_usd`, `slippage_usd`, `funding_usd`, `liquidations` as required by the runbook. 

**Acceptance checks**

* A minimal smoke run produces outputs and includes baseline keys (`total_return`, `max_drawdown`, `sharpe`, `trades`) plus the new derivatives keys (funding/costs/liquidations). Your existing UI expects those baseline metrics fields. 
* Unit test: when funding is positive and you’re long, cash decreases; when short, cash increases (sign correctness). 

---

## Phase 6 Add Coinbase API clients and broker wrapper

### 7. Add Coinbase REST clients

**Files**

* New:

  * `src/atlas/coinbase/advanced_auth.py`
  * `src/atlas/coinbase/advanced_client.py`
  * `src/atlas/coinbase/derivatives_auth.py`
  * `src/atlas/coinbase/derivatives_client.py` 

**Responsibilities**

* Advanced client:

  * list products / resolve product IDs,
  * fetch public candles,
  * place orders (market/limit),
  * fetch fills/order status.
* Derivatives client:

  * fetch funding history (when enabled),
  * otherwise rely on CSV fallback. 

### 8. Add `coinbase_broker.py`

**File:** new `src/atlas/broker/coinbase_broker.py` 

Model it after the Alpaca broker module patterns (submit order, query account/positions). 

**Key design decision**
Keep a small “broker interface” shape (even if informal):

* `get_positions() -> dict[symbol, qty]`
* `get_equity() -> float`
* `submit_market_order(symbol, qty, side)`
* `close_all()`

This makes the paper runner simpler and reduces coupling.

**Acceptance checks**

* You can run a “connectivity test” that only lists products / fetches one candle set.
* Auth failures are explicit and do not silently skip (live trading safety).

---

## Phase 7 Add Coinbase paper runner

### 9. Create `coinbase_perp_runner.py`

**File:** new `src/atlas/paper/coinbase_perp_runner.py` 

The existing paper loop:

* runs on a bar schedule aligned to “next bar open”,
* writes equity_curve rows,
* tracks last target per symbol. 

You want the same operational behavior, but with Coinbase:

* polling and bar alignment,
* load bars via Coinbase candles,
* build a `StrategyState`,
* call `strategy.target_exposures(...)`,
* translate target exposures into order deltas using your derivatives notional rules,
* enforce `--dry-run` and `ATLAS_ALLOW_LIVE` safety gate. 

**Paper runner must log**

* `run.log`
* `decisions.jsonl`
* `orders.jsonl`
  Optionally:
* `fills.jsonl` if you fetch fills back from Coinbase. 

**Acceptance checks**

* Dry-run produces consistent decisions without order placement.
* Candle gap detection triggers “flatten + stop” behavior (as called out in monitoring). 

---

## Phase 8 Wire everything through CLI and optional TUI

### 10. Modify `src/atlas/config.py` and `.env.example`

**File:** `src/atlas/config.py` (modify) 

Add environment getters for:

* Coinbase Advanced Trade JWT fields
* Coinbase Derivatives HMAC fields (optional)
* `ATLAS_ALLOW_LIVE` safety gate 

### 11. Modify `src/atlas/cli.py`

**File:** `src/atlas/cli.py` (modify) 

Add routing:

#### 11.1 Backtest command changes

* Allow `--market derivatives`
* Allow `--data-source coinbase`
* Add derivatives knobs:

  * `--funding-csv-dir`
  * `--taker-fee-bps`
  * `--half-spread-bps`
  * `--base-slippage-bps`
  * `--venue-max-leverage`
  * `--max-margin-utilization`
  * `--maintenance-margin-rate` 

Then route:

* If market is derivatives: call `run_derivatives_backtest(...)`
* Else: call existing `run_backtest(...)` unchanged. 

#### 11.2 Paper command changes

* If market is derivatives: call `run_coinbase_perp_loop(...)`
* Else: call existing Alpaca paper loop.

### 12. Optional TUI updates

Not strictly required for correctness, but if you use the TUI:

* Add `"derivatives"` to market selection
* Add `"coinbase"` to data source selection
* Add PerpFlare parameter schema / defaults in the TUI param editor
  (The TUI reads and displays `decision.debug` fields like “chosen” already, so PerpFlare debug becomes visible immediately if you use those keys.) 

---

## Phase 9 Testing and validation checklist

This is still “code-wise” because it’s about what to test and where.

### 13. Unit tests

Add a small test suite for:

* ER calculation matches spec. 
* Donchian bands exclude current bar. 
* Admission controller thresholding works (edge < floor + k_cost*fric → abstain). 
* Liquidation buffer formula produces sane output and blocks unsafe trades. 

### 14. Engine integration tests

Use a tiny synthetic dataset where you can predict results:

* Fees/spread/slippage are charged correctly.
* Funding is applied at the expected timestamps (sign correct). 
* Liquidation triggers under an engineered crash.
* Output contains `liquidations` and the cost breakdown keys expected in the runbook. 

### 15. Smoke run commands

The markdown already gives canonical smoke commands for:

* derivatives backtest with funding CSV,
* derivatives paper loop dry-run. 

---

## File-by-file implementation checklist

This is the fastest “apply to repo” checklist you can execute:

### Modified files

* `src/atlas/config.py`

  * add Coinbase settings + env var getters + allow-live gate. 
* `src/atlas/market.py`

  * add `Market.DERIVATIVES`, parsing, defaults for BTC-PERP/ETH-PERP. 
* `src/atlas/data/universe.py`

  * add `coinbase` loader path and funding support; attach funding column. 
* `src/atlas/strategies/registry.py`

  * register `perp_flare` and include in list. 
* `src/atlas/cli.py`

  * add derivatives routing and new CLI flags (fee/spread/slip/leverage/mmr/funding dir). 

### New files

* `src/atlas/strategies/perp_flare.py`

  * full strategy logic from spec. 
* `src/atlas/backtest/derivatives_engine.py`

  * margin + funding + liquidation + fee/spread/slippage + risk controls + outputs. 
* `src/atlas/data/coinbase_data.py`

  * candles + caching. 
* `src/atlas/data/coinbase_funding.py`

  * funding API + CSV fallback, bar alignment. 
* `src/atlas/coinbase/advanced_auth.py`, `advanced_client.py`, `derivatives_auth.py`, `derivatives_client.py`

  * API auth + REST wrappers. 
* `src/atlas/broker/coinbase_broker.py`

  * simplified broker wrapper for paper/live. 
* `src/atlas/paper/coinbase_perp_runner.py`

  * dry-run first, allow-live gate, flatten-on-error. 

---

If you want, I can also provide a “definition of done” set of invariants for each module (for example: what exactly must be present in `decisions.jsonl` debug fields; what the `trades.csv` schema should be for derivatives; exact behavior when a candle gap is detected).
