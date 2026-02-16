# Research Brief: `perp_research_vol_momentum` (Design Pre-Evaluation)

## 1) Objective and Constraints

This document defines a research-driven derivatives strategy intended for rigorous review by external researchers (including GPT‑5 Pro), before any additional tuning/evaluation runs.

Primary goal:
- maximize robust net profitability under realistic Coinbase nano BTC perp costs.

Hard practical constraints from this project:
- account size target: `$500` initial cash.
- realistic costs:
  - taker fee: `10 bps` per side
  - slippage proxy: `1.5 bps` per side
  - fixed fee: `$0.15` per contract per side
  - contract size: `0.01 BTC`
- derivatives venue assumptions: Coinbase BTC perpetual (`BTC-PERP` / resolved product id).
- low-turnover preference to reduce fee drag.

Code location:
- strategy implementation: `src/atlas/strategies/perp_research_vol_momentum.py`

## 2) Project Context (What Has Already Failed)

Existing strategy families tested in this repo include:
- `perp_trend_vol_guard`
- `perp_weekly_carry_shield`
- `perp_weekly_trend_reset`
- `perp_quant_fusion`
- `perp_weekly_profit_chase`

Observed pattern:
- many candidates look strong on Coinbase launch-era windows (2025+),
- but fail random-year robustness / weekly-consistency gates under realistic Coinbase costs.

Key references in repo documenting this:
- `docs/EXECUTION_STATE_STRATEGY_SEARCH.md`
- `docs/algorithms/coinbase_cb10_frontier_2026-02-15.md`
- `docs/algorithms/external_perp_reality_probe_2026-02-15.md`

Critical data limitation:
- Coinbase BTC perp history is short (launch-era only), so robustness requires cross-source proxy testing.

## 3) Research Basis (Why This Strategy Form)

This strategy intentionally combines three literature-backed concepts:

1. **Time-series momentum directionality**  
   - use medium-horizon own-asset trend signal as the primary directional driver.  
   - Reference: Moskowitz, Ooi, Pedersen (2012), *Time series momentum*.  
   - URL: https://www.sciencedirect.com/science/article/pii/S0304405X11002613

2. **Volatility-managed position sizing**  
   - scale risk inversely with realized volatility, targeting a stable per-bar risk budget instead of constant notional.  
   - Reference: Moreira, Muir (NBER WP 22208), *Volatility-Managed Portfolios*.  
   - URL: https://www.nber.org/papers/w22208

3. **Crash-state de-risking for momentum**  
   - momentum strategies can crash during violent reversals; add explicit reversal+vol-spike risk scaling.  
   - Reference: Daniel, Moskowitz (2016), *Momentum Crashes*.  
   - URL: https://www.sciencedirect.com/science/article/pii/S0304405X1600183X

Recent workflow inspiration (methodological, not blindly copied):
- ArXiv: AutoQuant (2025) for robust strategy lifecycle + overfit controls.  
  URL: https://arxiv.org/abs/2508.06788
- ArXiv: AdaptiveTrend (2026) for adaptive trend logic in changing regimes.  
  URL: https://arxiv.org/abs/2610.13072

## 4) Strategy Definition (Exact)

### 4.1 Core Features

For each symbol at bar `t`:
- log return: `r_t = ln(P_t / P_{t-1})`
- long momentum (bps):  
  `mom_long_bps = 10000 * ln(P_t / P_{t-L})`
- short momentum (bps):  
  `mom_short_bps = 10000 * ln(P_t / P_{t-S})`
- trend strength:  
  `trend_strength = (EMA_fast(P)_t - EMA_slow(P)_t) / ATR_t`
- realized volatility (per bar): std of recent log returns over `vol_lookback_bars`
- volatility regime z-score:
  `vol_z = (vol_now - median(vol_hist_tail)) / std(vol_hist_tail)`

### 4.2 Entry Qualification

Directional side candidate:
- `side = sign(mom_long_bps)`

Entry requires all of:
- `|mom_long_bps| >= min_abs_long_momentum_bps`
- `sign(trend_strength) == side`
- `|trend_strength| >= trend_strength_min`
- `ATR_bps >= min_atr_bps`
- cost-adjusted edge passes:
  - round-trip cost proxy:
    `cost_rt_bps = 2 * (slippage_bps + taker_fee_bps)`
  - edge estimate:
    `edge_bps = |trend_strength| * ATR_bps + 0.35 * |mom_long_bps|`
  - required:
    `edge_bps >= edge_floor_bps + k_cost * cost_rt_bps`

### 4.3 Volatility-Managed Sizing

Base leverage:
- `lev_raw = target_vol_per_bar / max(vol_floor, vol_now)`
- `lev_cap = min(max_leverage, max_margin_utilization / maintenance_margin_rate)`
- `lev = clip(lev_raw, 0, lev_cap)`

Confidence scaling:
- `confidence = clip((edge_bps - required_bps) / required_bps, 0, 1)`

Initial notional:
- `notional_target = lev * equity * max(confidence, 0.25)`

Exposure caps:
- per-symbol and gross exposure caps
- min trade notional floor

### 4.4 Crash-State De-Risking

If short-horizon reversal conflicts with long momentum in a volatility spike:
- long signal but `mom_short_bps <= -crash_reversal_bps` and `vol_z >= crash_vol_z`
- or short signal but opposite reversal condition

Then:
- multiply risk by `crash_risk_scale` (default strong de-risk).

Hard volatility off-switch:
- if `vol_z >= vol_off_z`, block new entries.

### 4.5 Execution Cadence

- Weekly rebalance at configured UTC weekday/hour/minute.
- Off-cycle risk exits still allowed (stop/trailing/max-hold/risk guards).
- This intentionally targets low turnover and fee control.

### 4.6 Risk Controls

- daily loss lockout: flat for rest of day if breached.
- weekly loss lockout: flat for rest of week if breached.
- global kill switch on deep drawdown.
- ATR hard stop + trailing stop.
- max hold bars exit.

## 5) Default Parameters (Current v1)

From `src/atlas/strategies/perp_research_vol_momentum.py`:

- schedule:
  - `rebalance_weekday_utc=0`
  - `rebalance_hour_utc=0`
  - `rebalance_minute_utc=5`
- signal:
  - `long_momentum_bars=336`
  - `short_momentum_bars=48`
  - `ema_fast=24`
  - `ema_slow=168`
  - `atr_window=48`
  - `vol_lookback_bars=120`
  - `vol_regime_window=720`
  - `min_abs_long_momentum_bps=45.0`
  - `min_atr_bps=8.0`
  - `trend_strength_min=0.10`
  - `edge_floor_bps=8.0`
  - `k_cost=2.6`
- sizing:
  - `target_vol_per_bar=0.0065`
  - `vol_floor=0.0020`
  - `max_leverage=4.0`
  - `max_margin_utilization=0.40`
  - `max_gross_exposure=0.95`
  - `max_per_symbol_exposure=0.95`
  - `max_positions=1`
  - `min_trade_notional_usd=25.0`
  - `rebalance_exposure_threshold=0.04`
- crash controls:
  - `crash_vol_z=1.25`
  - `crash_reversal_bps=55.0`
  - `crash_risk_scale=0.30`
  - `vol_off_z=2.4`
- risk:
  - `stop_atr_mult=3.2`
  - `trail_atr_mult=4.2`
  - `min_hold_bars=24`
  - `max_hold_bars=240`
  - `weekly_loss_limit=0.03`
  - `daily_loss_limit=0.02`
  - `kill_switch=0.20`

## 6) Why This Is Different from Prior Repo Candidates

- Prior candidates were mostly tuned within existing family structures.
- This one is constructed from explicit research priors first:
  - momentum direction,
  - volatility normalization,
  - crash-aware risk suppression,
  - strict weekly cadence to reduce fee churn.
- It is intentionally designed before new tuning sweeps.

## 7) Known Risks / Assumptions

- The edge proxy (`edge_bps`) is heuristic, not a calibrated expected return model.
- Weekly cadence may miss profitable intraday reversals.
- Volatility z-score depends on historical window quality and stationarity.
- Cross-source historical perp data may differ in microstructure/funding from Coinbase.
- With small capital (`$500`), lot size + fixed fees can dominate realized PnL.

## 8) External Reviewer Request (What to Critique)

Please evaluate:

1. Signal validity:
- Is long-momentum + EMA-trend + ATR gate the right triad for BTC perp?
- Should momentum horizon be longer/shorter for 1H bars?

2. Volatility sizing:
- Is `target_vol_per_bar` sensible for small-cap leveraged accounts?
- Better estimator: EWMA/GARCH/realized kernel?

3. Crash control:
- Is reversal+vol-z sufficient, or should we use broader market state (e.g., BTC term structure/funding/basis)?

4. Execution/cost model:
- Should admission include explicit fixed fee conversion into effective bps at current price?
- Should we enforce minimum expected holding-time edge net of financing/funding?

5. Robustness protocol:
- Better walk-forward protocol across mixed venues and launch-era reconciliation.
- Recommended holdout structure to reduce regime overfit.

6. Alternative mathematically superior formulation:
- If a better model class is justified (state-space regime switching, Bayesian trend filter, convex risk budgeting, etc.), provide a concrete replacement spec.

## 9) Planned Evaluation Protocol (After Review)

Post-review, run:
- Launch-era Coinbase windows (existing rolling 180d set).
- Random-year windows (multi-year proxy set).
- External-perp reality probes (OKX/Deribit alignment + backtests).
- Same strict cost model and lot constraints.

Decision standard:
- prioritize robustness and repeatability under realistic costs over launch-era headline returns.

## 10) External Handoff Package (Ready-to-Send)

Primary reviewer prompt (send this first):
- `docs/algorithms/perp_external_reviewer_prompt_2026-02-15.md`

Supporting context bundle:
- `docs/algorithms/perp_research_vol_momentum_research_brief_2026-02-15.md`
- `docs/algorithms/researcher1_implementation_report_2026-02-15.md`
- `src/atlas/strategies/perp_research_vol_momentum.py`
- `strategy_params/perp_research_vol_momentum_v1_1h_coinbase_profile.json`
- `outputs/evaluations/strategy_eval/research_vm_launch12_20260215_172307/evaluation_result.json`
- `outputs/evaluations/strategy_eval/research_vm_random10_20260215_172307/evaluation_result.json`
- `outputs/evaluations/external_perp_probe/20260215_165618/result.json`
- `outputs/evaluations/external_perp_probe/20260215_165955/result.json`
- `outputs/evaluations/external_perp_probe/20260215_170534/result.json`
