I dug through the tuning code (`src/atlas/ml/tune.py`) plus the perp strategy (`src/atlas/strategies/perp_flare.py`) and the derivatives backtest engine (`src/atlas/backtest/derivatives_engine.py`). Conceptually you’re on the right track (walk‑forward with train/validation/test), but there are a few issues that will make the tuning output hard to trust until you fix them.

## What your “ML” piece is actually doing (and what’s good about it)

This is not “machine learning” in the predictive‑model sense yet; it’s **hyperparameter optimization** (random search) with a **walk‑forward** evaluation loop:

* Split the data into repeated segments: **train → validate → test**, sliding forward by `step`.
* For each segment, try `trials_per_segment` random parameter sets.
* Score each trial on train and validation, pick the best, then evaluate once on the test window.
* Carry the chosen params forward as the next segment’s “incumbent” and optionally limit drift via `drift_frac`.

What’s good here:

* **No lookahead in the backtest loop**: decisions are made using history through bar *t* and applied at bar *t+1* open.
* **Walk‑forward** is the right structure for trading systems (time series, non‑stationary).
* **Hard rejection constraints** (drawdown, worst day, turnover, liquidations) are a good safety rail.
* **Drift limiting** is a practical way to avoid parameters whipsawing every segment.

## The biggest problems to fix before trusting tuned parameters

### 1) PerpFlare is outputting quantities, but the engines treat them like “exposures”

`StrategyDecision.target_exposures` is treated by both engines as a *multiplier* on `cfg.max_position_notional_usd`:

* In `derivatives_engine.py`: `target_notional = target_pct * max_notional_cap`, then `desired_qty = target_notional / price`.

But in `perp_flare.py`, you compute `qty_base` in **units** (BTC/ETH) and then return:

```python
target_qtys[sym] = qty_base * direction
return StrategyDecision(target_qtys)
```

So the engine interprets “BTC quantity” as “fraction of max_notional,” which rescales the position by `max_notional / price`. That means:

* Your tuned `risk_per_trade`, `stop_atr_mult`, leverage, etc. are not behaving the way you think.
* The optimizer is tuning against a sizing bug, so it may converge on nonsense.

**Fix (recommended):** make PerpFlare return *exposures*:

* Compute the notional you want: `desired_notional = qty_base * price`
* Convert to exposure: `exposure = desired_notional / max_position_notional_usd`
* Return `exposure * direction`

To do that cleanly, you need `max_position_notional_usd` inside the strategy. Easiest approach:

* Add it to `StrategyState.extra` inside both backtest engines (and paper runner), e.g. `extra={"max_position_notional_usd": cfg.max_position_notional_usd, ...}`
* In PerpFlare, read it from `state.extra`.

Until this is fixed, I would not “lock in” any tuned PerpFlare params.

---

### 2) You’re tuning “cost” and “margin” parameters that aren’t actually simulated

In your tuning search space you include:

* `taker_fee_bps`, `half_spread_bps`, `base_slippage_bps`
* `maintenance_margin_rate`

But in `derivatives_engine.py`:

* Fees are hardcoded: `TAKER_FEE_BPS = 3.0`
* Maintenance margin used for liquidation is hardcoded: `MAINTENANCE_MARGIN = 0.05`
* Funding is always `0.0` and never debited/credited to PnL

So right now those params only affect **admission gating** inside the strategy (and some liquidation buffer math), not the actual realized trading costs and risk of ruin.

That creates a classic optimizer failure mode: it can “game” the gate by shrinking assumed costs even if real costs are higher, and you’ll never see it in PnL.

**Fix (recommended):**

* Treat environment parameters (fees/spread/slippage/funding/margin rules) as **fixed constants** derived from your venue, not tunables.
* Simulate them in the backtest engine PnL (or at least via `cfg`), then remove them from the tuning space.

---

### 3) Validation/test runs are “cold started” with no warmup carryover

For each trial, you run train and validation as separate backtests with a fresh strategy instance and only that window’s bars. That means:

* Indicators “warm up” again at the start of validation and test
* For PerpFlare, worst case warmup is ~`max(ema_slow, breakout_window, …)+10`, which can be about a day on 5‑minute bars
* That can materially change performance on short 7‑day validation/test windows

**Fix (recommended):**

* When evaluating validation and test, prepend enough prior bars to warm indicators, but **exclude the warmup region from scoring**.
* If you don’t want to implement “exclude warmup from scoring” yet, a simpler compromise is to make validation/test longer so warmup is a smaller fraction.

---

### 4) Incumbent comparison is inconsistent

You compute `chosen_score` as:

* `selection_score = 0.25 * train_score + 0.75 * val_score`

But you compute the incumbent score using **validation only**, then compare:

* `chosen_score < incumbent_val_score + improvement_margin`

That’s mixing apples and oranges.

**Fix (recommended):**

* Either compare validation-to-validation,
* Or compute incumbent `selection_score` the same way (train+val) and compare like‑to‑like.

---

## Your current walk‑forward settings: the core issue is too few segments

With a ~60‑day dataset and windows like:

* train 30d, validate 7d, test 7d, step 7d

each full segment consumes 44 days. With a 7‑day step, that yields only about **3 segments** across the entire run.

Three test windows is nowhere near enough to judge:

* parameter stability,
* performance consistency across regimes,
* whether your tuner is actually generalizing.

So even if everything else were perfect, that lookback is mainly useful as a quick smoke test, not something you’d “save as the final tuning setup.”

## What tuning settings you should save

Because you’re trading perps (24/7) and want walk‑forward behavior, I’d pick settings based on two goals:

1. **How often you want to retune in “live mode”** (that’s basically your `step`)
2. **How many walk‑forward segments you want in your evaluation run** (ideally 10–30+)

### Recommendation A: Keep your weekly cadence, but run it on much more history

If you want a weekly retune cadence (a reasonable starting point):

* **Bar timeframe:** 5 minutes (fine to start)
* **Step:** 7 days
* **Train:** 30–45 days
* **Validate:** 14 days (I’d increase this from 7 to reduce noise)
* **Test:** 7 days
* **Overall lookback for the tuning run:** **180–365 days**
* **Trials per segment:** **100–200**
* **Drift fraction:** **0.25–0.50** (start at 0.50; once stable, tighten to 0.25)

Why this is the best “saveable” setup:

* It matches how you’d actually retune in production (weekly),
* But it gives you enough segments to evaluate whether weekly retuning is consistently beneficial.

If you absolutely must keep validate at 7 days, then I’d push even harder on having at least ~180 days of total history so you get many segments and can average out noise.

---

### Recommendation B: More stability, fewer retunes

If you prefer slower parameter churn:

* **Step:** 14 days
* **Train:** 60 days
* **Validate:** 14 days
* **Test:** 14 days
* **Overall lookback:** **365 days**
* **Trials per segment:** **150–300**
* **Drift fraction:** **0.25**

This tends to produce parameters that are less “twitchy,” at the cost of adapting more slowly.

---

### What I would not save as “final”

* A ~60‑day tuning run as your primary config (it’s fine for iteration, not for deciding what’s robust).
* 7‑day validation + 7‑day test if you’re going to make conclusions from only a handful of segments.

## What parameters should you actually “save” after tuning

Right now `best_params.json` stores **the latest segment’s** chosen params. That’s a reasonable “deploy now” choice *once the sizing and cost-model issues are fixed*.

But I’d save **two** parameter sets:

1. **Latest params** (for deployment now)
2. **A stability-biased set**, for example:

   * median of each numeric parameter over the last N segments (say last 5–10),
   * or the params from the segment with the best test score *among the last N segments* (so you’re not cherry-picking far-back history).

This is especially important for perps because conditions shift fast and single-window optima can be fragile.

## Two small objective tweaks that will help immediately

1. **Set a nonzero `min_trades`**
   Without it, the optimizer can prefer “barely trade” regimes because they avoid drawdown and worst-day penalties.

A decent starting point:

* For 7–14 day validation windows: `min_trades = 5` or `10`

2. **Downweight raw total return a bit, upweight risk**
   Right now total return dominates the score. If you don’t have realistic fees/funding modeled, this will bias heavily toward overactive strategies.

## Bottom line

* The walk‑forward structure is a good foundation.
* But **PerpFlare’s sizing/output mismatch** plus **incomplete derivatives cost/funding simulation** means the tuner is currently optimizing a distorted version of the strategy.
* After fixing those, your *best “saveable” tuning setup* is:

  * keep the weekly step if that’s how you plan to operate,
  * but run the tuner across **at least 180 days** (preferably **365**) so you get enough segments,
  * and use **longer validation** (14 days) or more segments to reduce noise.

If you want one concrete config to lock in after fixes:

* **Lookback:** 365d
* **Bars:** 5Min
* **Trials/segment:** 150
* **Train:** 45d
* **Validate:** 14d
* **Test:** 7d
* **Step:** 7d
* **Drift:** 0.25–0.50 (start 0.50, then tighten)

That gives you a realistic weekly retune simulation with enough repetitions to judge whether it’s working.
