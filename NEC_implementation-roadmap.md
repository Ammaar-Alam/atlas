## Evaluation of the current repository (what you have so far)

I reviewed the repo structure and the key modules (data download/caching, backtest engine, paper loop, strategy plug-in, CLI/TUI). 

### What’s already strong (good foundations)

* **Clean separation of concerns**:

  * `atlas/data/*` handles ingestion (CSV + Alpaca caching).
  * `atlas/strategies/*` defines a simple strategy interface and a registry.
  * `atlas/backtest/engine.py` is a minimal event loop with **next-bar-open fills** (avoids same-bar lookahead).
  * `atlas/paper/runner.py` is an operational paper-trading loop with order/fill logging.
* **Safety defaults**:

  * Live trading is blocked unless explicitly enabled (good guardrail).
* **Usable tooling**:

  * CLI and TUI provide a quick feedback loop, which is helpful for research iteration.

### What’s missing or mismatched vs the NEC‑X method requirements

The NEC‑X method I proposed is **portfolio-level** (needs SPY and QQQ together every decision) and **cost-aware** (spread floor + slippage + fees) with **intraday session gating**. Your current scaffold is mostly **single-symbol** and **slippage-only**.

Key gaps:

1. **Single-symbol Strategy interface / execution loop**

   * Current `Strategy.target_exposure(bars: DataFrame) -> float` is called once per symbol (paper loop) or only for one symbol (backtest).
   * NEC‑X needs both SPY and QQQ simultaneously to compute:

     * cross-ETF correlation,
     * agreement gate,
     * instrument selection (choose SPY vs QQQ).
   * Result: you need a **portfolio strategy interface** and matching backtest/paper runners.

2. **Cost model is too optimistic**

   * Backtest only models `slippage_bps`, no explicit spread, no regulatory fees.
   * NEC‑X explicitly requires conservative costs (spread floor + slippage per side + fees).

3. **Intraday constraints not enforced**

   * No “no new entries after 15:30 ET” and “flat by 15:55 ET” enforcement.
   * No day boundary resets (for daily risk limits, VWAP reset, etc.).

4. **Paper trading doesn’t support shorting**

   * The paper loop logic only buys to enter and sells to exit; it cannot open shorts.
   * NEC‑X requires “shorting optional” mode, so paper trading should support:

     * long/short mode when enabled,
     * long-only fallback when disabled.

5. **Decision logging is incomplete for research**

   * Orders/fills are logged, but the research method needs:

     * feature snapshots,
     * gate pass/fail reasons,
     * estimated costs and netEdge,
     * “why abstain” events.

---

## Roadmap to implement NEC‑X (and replace `STRATEGY.md`)

Below is a practical plan that keeps your existing MA crossover scaffold intact, while adding a **portfolio-capable research strategy** (SPY+QQQ only) with conservative costs and intraday gating.

### Phase 1 — Add portfolio strategy support (minimal but correct)

**Goal:** Call strategy once per decision time with both symbols and return targets for both symbols.

**Add new base class (do not break existing Strategy):**

* `src/atlas/strategies/portfolio_base.py`

  * `PortfolioStrategy.warmup_bars() -> int`
  * `PortfolioStrategy.target_exposures(bars_by_symbol: dict[str, pd.DataFrame]) -> PortfolioDecision`
  * `PortfolioDecision.targets: dict[str, float]` (e.g., `{"SPY": 1.0, "QQQ": 0.0}`)
  * Include `reason` and optional `debug` dict for logging.

**Update strategy registry to build portfolio strategies:**

* `src/atlas/strategies/registry.py`

  * Add builder for `nec_x`.
  * Keep `ma_crossover` unchanged.

Acceptance criteria:

* You can instantiate `nec_x` from the registry.
* Existing MA crossover backtests still run unchanged.

---

### Phase 2 — Resampling + session filtering utilities (data correctness)

**Goal:** NEC‑X uses 5‑minute bars; your data layer currently supports 1‑minute bars. Keep 1‑minute as source-of-truth and resample.

Add:

* `src/atlas/data/resample.py`

  * `resample_ohlcv(df_1m, rule="5min") -> df_5m`
  * OHLCV aggregation:

    * open = first
    * high = max
    * low = min
    * close = last
    * volume = sum
  * Drop incomplete last bin in live/paper mode (avoid using partial bars).

* `src/atlas/utils/session.py`

  * `is_regular_hours(ts)`, `filter_regular_hours(df)`
  * `is_entry_allowed(next_bar_ts)` enforcing **no entry fills after 15:30**
  * `is_forced_flat(next_bar_ts)` enforcing **flat by 15:55**

Acceptance criteria:

* Unit test resampling behavior on sample data.
* Regular-hours filter removes pre/post-market bars.

---

### Phase 3 — Portfolio backtest engine (multi-symbol + conservative costs)

**Goal:** Backtest SPY & QQQ together with a single cash account and portfolio positions.

Add:

* `src/atlas/backtest/portfolio_engine.py`

  * Inputs:

    * `bars_by_symbol: dict[str, pd.DataFrame]` (already resampled to 5m + filtered)
    * portfolio strategy instance
    * portfolio config with:

      * initial_cash,
      * max_position_notional_usd (for 1.0 gross),
      * allow_short,
      * cost params: spread_floor_bps, slippage_bps, k_cost multiplier (for gating), fee config
  * Align timestamps using intersection of indices (inner join).
  * Enforce **next-bar-open** fills:

    * decision on bar close at t
    * execute at open of t+1
  * Apply conservative costs at fills:

    * buy fills worse by (spread+slip)
    * sell fills worse by (spread+slip)
    * deduct fees where applicable

Add cost model:

* `src/atlas/backtest/cost_model.py`

  * `spread_side_bps = max(tick_bps, spread_floor_bps)`
  * `slip_side_bps = slippage_bps`
  * (Optional) fee function (can start with 0 and add later, but structure it now)

Acceptance criteria:

* Backtest produces:

  * trades with per-fill cost breakdown fields,
  * portfolio equity curve,
  * metrics.
* Switching between SPY and QQQ is supported (if strategy requests it).

---

### Phase 4 — Implement NEC‑X strategy module

**Goal:** Full NEC‑X (cross-ETF confirmation + cost-aware netEdge + abstention + intraday gates).

Add:

* `src/atlas/strategies/nec_x.py`

Implementation details (match the earlier spec):

* Input bars: 5‑minute regular-hours bars for SPY and QQQ.
* Compute features:

  * log returns per symbol
  * EMA drift `m` and EMA abs-return `v`
  * score = m / (v + eps)
  * rolling correlation of returns
  * agreement + strength (min of abs scores)
  * volume ratio proxy (short/long EMA)
* Gates:

  * correlation gate
  * agreement gate
  * strength gate
  * time gate (no new entries after 15:30 fill-time)
  * forced flat by 15:55
* Cost-aware controller:

  * `expMove = |m| * Hmax * 10000`
  * `costRT = 2*(spread+slip) + fees`
  * `netEdge = expMove - k_cost*costRT`
  * choose symbol with max netEdge if > 0; else abstain
* Outputs: `{"SPY": -1/0/1, "QQQ": 0}` or vice versa (single position constraint)

Shorting-optional mode:

* If `allow_short=False`, negative targets must be clamped to 0.

Acceptance criteria:

* Strategy abstains most of the time (by design) unless gates pass and netEdge > 0.
* Strategy always returns targets that satisfy:

  * only SPY and QQQ keys,
  * sum abs(targets) ≤ 1.0,
  * forced flat rule triggers by 15:55.

---

### Phase 5 — Portfolio paper trading runner (real-time, SPY+QQQ together)

**Goal:** Run NEC‑X paper trading without accidental multi-position exposure.

Add:

* `src/atlas/paper/portfolio_runner.py`

  * Fetch 1‑minute bars for both symbols each loop.
  * Filter regular hours.
  * Resample to 5‑minute bars.
  * Call portfolio strategy once with `{SPY: df, QQQ: df}`.
  * Enforce single-position exposure:

    * If switching, decide whether to:

      * (simple) exit old and enter new in the same loop (two orders), or
      * (more conservative) exit first, then only enter next cycle.
  * Implement shorting if enabled:

    * if target < 0 and current_qty == 0: submit SELL to open short (if broker/account supports)
    * confirm behavior in Alpaca paper account settings.
  * Add end-of-day flatten:

    * if now >= 15:55 ET, force target=0 for both and submit closing orders.

Logging upgrades:

* In addition to orders/fills, log:

  * decision JSONL per loop (features summary, gates, selected symbol, netEdge, cost estimates)
  * current positions (qty, notional, direction)
  * realized vs modeled slippage (post-hoc diagnostics)

Acceptance criteria:

* You can run: “paper NEC‑X on SPY+QQQ” and it never ends up long both simultaneously.
* At 15:55 it is flat.

---

### Phase 6 — Replace/add CLI entry points (keep existing commands stable)

**Goal:** Don’t break the current CLI; add explicit portfolio commands.

Add to `src/atlas/cli.py`:

* `atlas portfolio-backtest --start ... --end ... --strategy nec_x ...`
* `atlas portfolio-paper --symbols SPY --symbols QQQ --strategy nec_x ...`

TUI:

* Optional. You can keep TUI as single-symbol for now and run NEC‑X via CLI first.

Acceptance criteria:

* MA crossover continues to work via existing CLI/TUI.
* NEC‑X works via new portfolio commands.

---

### Phase 7 — Evaluation harness (research-grade, but still lightweight)

**Goal:** Make it paper-worthy by proving the **cost-aware abstention/controller** adds value.

Add:

* walk-forward runner with fixed windows
* cost inflation stress tests
* ablations:

  1. remove cost gate (always trade when gates pass)
  2. remove cross-ETF agreement gate (single-ETF signal only)
  3. remove correlation gate

This can live under:

* `src/atlas/eval/walk_forward.py`
* `src/atlas/eval/ablations.py`
* Or start as scripts under `scripts/` and later promote.

Acceptance criteria:

* Reports show results separately for SPY and QQQ.
* Clear deltas from ablations (even if negative; that’s still a valid result).

---

# Proposed replacement `STRATEGY.md` (copy-paste as the new file)

```markdown
# NEC-X Strategy (SPY + QQQ): Net-Edge Controller with Cross-ETF Confirmation

This repository is a research scaffold for intraday strategies. The baseline `ma_crossover` exists only to validate the plumbing (data → backtest → paper loop).

This document specifies and guides implementation of the v1 research strategy:

**NEC-X (Net-Edge Controller with Cross-ETF Confirmation)**

- Universe is fixed: **SPY and QQQ only**
- Data: **1-minute OHLCV** as source, resampled to **5-minute OHLCV**
- Session: **Regular hours only (09:30–16:00 ET)**
- Decision timing: decisions made using **completed bars only**
- Execution baseline: **signal at bar close → earliest eligible fill is next bar open**
- Hard constraints:
  - **No entry fills after 15:30 ET**
  - **Must be flat by 15:55 ET** (time-based exit)
- Long/short allowed, but must support **long-only fallback**
- Must include **conservative, explicit costs** in backtest and in the decision gate

This is educational research content and not financial advice. No profit is promised or implied.

---

## 1. Why a portfolio strategy interface is required

NEC-X uses SPY↔QQQ relative information (agreement and correlation gates) and selects which instrument to trade. Therefore, it cannot be implemented correctly as a per-symbol `Strategy.target_exposure(bars)` function.

We keep the existing single-symbol `Strategy` interface for baseline strategies and add a separate portfolio strategy interface.

---

## 2. New strategy interface (portfolio)

Create: `src/atlas/strategies/portfolio_base.py`

Recommended shape:

- `PortfolioDecision.targets: dict[str, float]`
  - keys must include **exactly**: `"SPY"` and `"QQQ"`
  - values are target exposures in `[-1.0, 1.0]`
  - NEC-X must satisfy: `abs(SPY) + abs(QQQ) <= 1.0` and (usually) only one nonzero at a time
- `PortfolioStrategy.target_exposures(bars_by_symbol: dict[str, pd.DataFrame]) -> PortfolioDecision`

A portfolio runner (backtest + paper) calls this once per decision time using aligned bars for both symbols.

---

## 3. Data requirements and preprocessing

### 3.1 Source bars
- Fetch/ingest 1-minute OHLCV for SPY and QQQ.

### 3.2 Regular-hours filtering
- Filter to timestamps between 09:30 and 16:00 ET.
- No extended hours in v1.

### 3.3 Resampling to 5-minute bars
Resample 1-minute bars to 5-minute OHLCV:

- open = first open
- high = max high
- low = min low
- close = last close
- volume = sum volume

Drop incomplete final bins in live mode.

Add helper: `src/atlas/data/resample.py`

---

## 4. NEC-X features (<= 25 total)

All features use only SPY and QQQ completed bars up to time t.

Per symbol s ∈ {SPY, QQQ}:

1) log return:
   r_s,t = ln(C_s,t / C_s,t-1)

2) EMA drift:
   m_s,t = EMA(r_s, span=M)

3) EMA abs-return:
   v_s,t = EMA(|r_s|, span=V)

4) normalized score:
   score_s,t = m_s,t / (v_s,t + 1e-8)

5) volume ratio proxy:
   vol_ratio_s,t = EMA(volume, span=4) / EMA(volume, span=78)

Cross-ETF:

6) rolling correlation of returns:
   rho_t = Corr(r_SPY window Wcorr, r_QQQ window Wcorr)

7) agreement:
   agree_t = 1[sign(score_SPY,t) == sign(score_QQQ,t)]

8) strength:
   strength_t = min(|score_SPY,t|, |score_QQQ,t|)

---

## 5. NEC-X decision logic (gates + cost-aware controller)

### 5.1 Time rules (hard)
- No entry fills after **15:30 ET**
- Forced flat: if next-bar open is **>= 15:55 ET**, target must be 0 for both symbols

Implement gates using the next bar timestamp (bar schedule is known).

### 5.2 Regime gates (abstain unless all pass)
At each 5-minute bar close t, allow entries only if:

- rho_t >= rho_min
- agree_t == 1
- strength_t >= strength_entry
- max(vol_ratio_SPY,t, vol_ratio_QQQ,t) >= 0.6 (fixed constant in v1)

If any gate fails: **abstain** (target exposures = 0 unless already holding and exit is not triggered).

### 5.3 Conservative cost-aware decision
For each symbol s:

- expMove_s,t (bps) = |m_s,t| * Hmax * 10000
- costRT_s,t (bps) = 2*(spread_side_bps + slippage_side_bps) + fees_bps
- netEdge_s,t = expMove_s,t - k_cost * costRT_s,t

Choose s* = argmax netEdge_s,t.
Enter only if netEdge_s*,t > 0.

Direction:
- dir = sign(score_SPY,t) = sign(score_QQQ,t)

Targets:
- if enter: {s*: dir, other: 0}
- else: {SPY: 0, QQQ: 0}

Long-only mode:
- if dir < 0, abstain.

---

## 6. Tunable parameters (v1 must remain small)

Recommended defaults (with small sensitivity ranges; do not do large grid searches):

- M = 6 (range 4–10)
- V = 12 (range 8–20)
- Wcorr = 12 (range 8–24)
- rho_min = 0.60 (range 0.4–0.8)
- strength_entry = 0.80 (range 0.6–1.2)
- strength_exit = 0.20 (range 0.1–0.5)
- Hmax = 6 bars (range 4–12)
- k_cost = 1.25 (range 1.0–2.0)
- spread_floor_bps_per_side = 0.50 (range 0.3–1.0)
- slippage_bps_per_side = 0.75 (range 0.5–2.0)
- daily_loss_limit_pct = 1.0 (range 0.5–2.0)
- kill_switch_drawdown_pct = 2.5 (range 1.5–5.0)

Total knobs: 12.

---

## 7. Backtest requirements (portfolio engine)

Create: `src/atlas/backtest/portfolio_engine.py`

Must enforce:
- next-bar-open fills (1-bar latency baseline)
- conservative spread + slippage at fills
- forced flat by 15:55 ET
- no entry fills after 15:30 ET
- single cash account + portfolio positions (SPY and/or QQQ)

Outputs:
- trades.csv/json with per-fill cost details
- equity_curve.csv
- metrics.json (expand metrics later)

---

## 8. Paper trading requirements (portfolio runner)

Create: `src/atlas/paper/portfolio_runner.py`

Must:
- fetch both symbols each loop
- resample to 5-minute and call portfolio strategy once
- enforce single-position rule (preferably one symbol at a time)
- support long-only fallback
- support optional shorting (if enabled and account supports)
- enforce forced flat by 15:55 ET

Logging must include:
- decision snapshots (features summary + gates + netEdge + chosen action)
- orders and fills
- end-of-day flat confirmation

---

## 9. CLI integration

Add commands (recommended, to avoid breaking existing single-symbol CLI):

- `atlas portfolio-backtest --start ... --end ... --strategy nec_x --strategy-params params/nec_x.json`
- `atlas portfolio-paper --strategy nec_x --strategy-params params/nec_x.json`

TUI integration is optional; implement CLI first.

---

## 10. Research evaluation checklist (minimum)

Run:
- walk-forward OOS windows
- cost inflation stress tests (1.0x / 1.5x / 2.0x)
- ablations:
  1) remove cost gate
  2) remove agreement gate
  3) remove correlation gate
- report SPY and QQQ separately

If NEC-X only works with optimistic fills or fails under cost inflation, treat that as a falsification result and revise the method.

---
```

---

## Practical implementation notes (to keep you from stepping on landmines)

* **Do not retrofit NEC‑X into the existing per-symbol paper loop** by “calling it twice”; you’ll get inconsistent gates and accidental dual exposure. Implement the portfolio runner.
* **Be strict about completed bars** in paper mode:

  * If you resample 1m → 5m, drop the last partial 5m bin.
* **Enforce time gating twice**:

  * in strategy logic (primary),
  * and as a safety clamp in the runner (secondary), so a bug cannot hold overnight.
* **Shorting in Alpaca paper** can behave differently depending on account configuration. Treat short support as optional and always keep long-only mode operational.
