# Joint Researcher Brief + Hard Prompt (Researcher 1 + Researcher 2)

## Audience
This brief is intended to be sent to **both researchers** so each has complete context of:
- what the other researcher proposed,
- what was actually implemented,
- what was tested,
- why current outputs are still rejected,
- and what the next deliverable must be.

## Program Objective (Non-Negotiable)
Produce a deployable algorithm for Coinbase derivatives (small account constraints) that is:
1. Weekly-profitable with high consistency.
2. Better than passive SPY on average over evaluation windows.
3. Robust across launch-era and multi-year/random regimes.
4. Realistic under full cost model (slippage + taker fee + fixed fee/contract + contract granularity).

The sponsor requests a **guaranteed weekly-profit algorithm that beats SPY on average**. Treat this as the hard target and optimize as if no compromise is acceptable.

---

## Shared Environment and Constraints

### Trading venue and account assumptions
- Venue: Coinbase derivatives (nano BTC perp style mechanics)
- Starting equity: `$500`
- Symbol focus: `BTC-PERP` (and `BTC/USD` as long-horizon proxy where needed)
- Allow short: `true`

### Cost model used in backtests
- `slippage_bps = 1.5`
- `taker_fee_bps = 10.0`
- `coinbase_fee_model = true`
- `fixed_fee_per_contract_usd = 0.15`
- `contract_size_units = 0.01`

### Evaluation harness references
- Window evaluator: `scripts/evaluate_strategy_windows.py`
- External source probe: `scripts/probe_external_perp_reality.py`
- Strategy under study: `src/atlas/strategies/perp_research_vol_momentum.py`
- Param wiring: `src/atlas/strategies/registry.py`

---

## Researcher 1: Approach, Implementation, and Outcomes

### Researcher 1 approach (high-level)
Researcher 1 proposed improving the original momentum strategy with:
- cost-aware expected-return admission gates,
- trend consistency filters,
- volatility percentile regime filtering,
- multi-day rebalance support,
- equity-aware stop sizing,
- lot-size-aware minimum tradable notional.

### Researcher 1 implementation artifacts
- Report: `docs/algorithms/researcher1_implementation_report_2026-02-15.md`
- Response: `docs/algorithms/researcher1_response_2026-02-15.md`
- Params:
  - `strategy_params/perp_research_vol_momentum_reviewer1/conservative.json`
  - `strategy_params/perp_research_vol_momentum_reviewer1/balanced.json`
  - `strategy_params/perp_research_vol_momentum_reviewer1/aggressive.json`

### Researcher 1 key results
- Launch12: no meaningful pass (`0/12` weekly-gate for all; aggressive negative).
- Random10: best mean improved vs older baseline but still negative/weak and failed weekly gates.

Conclusion: structural improvements, but still not deployable and too restrictive in practice.

---

## Researcher 2: Approach, Implementation, and Outcomes

### Researcher 2 approach (high-level)
Researcher 2 proposed a deeper redesign:
- multi-horizon standardized TSMOM ensemble,
- regression t-stat trend significance filter,
- efficiency-ratio trendiness filter,
- smooth vol-ratio de-risking,
- cost-complete hurdle including fixed fee bps,
- contract-quantized sizing,
- signal-break exits + cooldown,
- optional (disabled) day/week equity lockouts.

### Researcher 2 implementation artifacts
- Report: `docs/algorithms/researcher2_implementation_report_2026-02-15.md`
- Response: `docs/algorithms/researcher2_response_2026-02-15.md`
- Full protocol analysis: `docs/algorithms/researcher2_full_protocol_analysis_2026-02-15.md`
- Raw protocol metrics: `docs/algorithms/researcher2_protocol_metrics_2026-02-15.json`
- Params:
  - `strategy_params/perp_research_vol_momentum_reviewer2/conservative.json`
  - `strategy_params/perp_research_vol_momentum_reviewer2/balanced.json`
  - `strategy_params/perp_research_vol_momentum_reviewer2/aggressive.json`

### Researcher 2 key results
- Launch12 suite:
  - `outputs/evaluations/strategy_eval/research_vm_r2_launch12_cb10fix015_full_20260215_192008`
  - All profiles effectively no-trade.
- Stratified Random30 suite:
  - `outputs/evaluations/strategy_eval/research_vm_r2_random30_cb10fix015_full_20260215_192335`
  - Best (aggressive): mean ~`+0.12699%`, median `0.0%`, weekly-positive fraction ~`0.00667`, beat-SPY fraction `0.10`.
- Sensitivity (8 perturbations):
  - `outputs/evaluations/strategy_eval/research_vm_r2_random30_sensA_cb10fix015_20260215_200537`
  - `outputs/evaluations/strategy_eval/research_vm_r2_random30_sensB_cb10fix015_20260215_200537`
  - Mean > 0 survived, but weekly gate still catastrophically failed.
- Holdout test:
  - `outputs/evaluations/strategy_eval/research_vm_r2_random30_holdout40_cb10fix015_20260215_202427`
  - Holdout mean negative.
- External probes (cross-venue):
  - `outputs/evaluations/external_perp_probe_r2_aggressive/20260215_193441`
  - `outputs/evaluations/external_perp_probe_r2_balanced/20260215_193925`
  - `outputs/evaluations/external_perp_probe_r2_conservative/20260215_193925`
  - Source alignment with Coinbase was strong, but strategy edge still weak.

Conclusion: robust engineering and better realism, but still non-deployable and far from weekly/SPY requirements.

---

## Cross-Researcher Synthesis (What each should understand about the other)

1. Researcher 1 correctly identified participation vs cost friction, but profiles remained either too restrictive (no-trade) or still net-unprofitable.
2. Researcher 2 correctly fixed microstructure realism (fixed-fee bps, contract quantization, cooldown/exit logic), but current profile set remains under-participative and fails weekly edge criteria.
3. Combined lesson: this is no longer a basic coding bug. The bottleneck is **economic edge under realistic fee + lot-size constraints at low capital**, not missing indicators.
4. Any next design must explicitly solve **edge density vs cost drag** and **regime-specific participation** while preserving robustness.

---

## Post-R1/R2 Follow-on: RATC Candidate (Implemented)

A follow-on candidate was implemented from a long-horizon regime-adaptive brief:
- Strategy: `perp_regime_adaptive_trend_capture`
- Implementation + validation report: `docs/algorithms/ratc_implementation_and_validation_2026-02-15.md`
- Code: `src/atlas/strategies/perp_regime_adaptive_trend_capture.py`
- Presets:
  - `strategy_params/perp_regime_adaptive_trend_capture_cb10_conservative_1h.json`
  - `strategy_params/perp_regime_adaptive_trend_capture_cb10_balanced_1h.json`
  - `strategy_params/perp_regime_adaptive_trend_capture_cb10_aggressive_1h.json`

Summary:
- Launch12 (balanced) showed strong profitability, but failed weekly-positive >=0.70.
- External multi-year probe (OKX closest overlap) failed robustness gates.
- Result: still not deployment-ready under hard criteria.

---

## Hard Prompt to Both Researchers (Copy/Paste)

You are receiving this as a hard directive. You must use both researcher tracks (R1 and R2) as prior art and produce a new final strategy that satisfies the following:

### Non-negotiable target
Design and deliver a trading algorithm that **guarantees weekly profit** and **beats SPY on average** across the evaluation framework below, even if this requires a radically different architecture.

### Constraints and realism
- Must include full Coinbase fee realism:
  - slippage bps
  - taker bps
  - fixed fee per contract
  - contract-size quantization
- Must work for small account constraints (`$500` start, lot-size aware).
- Must not rely on one cherry-picked launch period.

### Required evaluation protocol
1. Launch-era rolling windows (same 12-window set).
2. Stratified random multi-year windows (minimum 30 windows: 10 high-vol, 10 low/mod-vol, 10 random).
3. External cross-venue probe validation (OKX/Deribit alignment and replay).
4. Sensitivity perturbation test (at least 8 perturbations on core params).
5. Selection/holdout split with one-shot holdout reporting.

### Hard acceptance criteria
- Weekly-positive fraction >= `0.70` overall.
- At least `60%` of windows with weekly-positive fraction >= `0.65`.
- Mean and median 180d return > 0.
- Beats SPY in >= `60%` of windows.
- Mean excess return over SPY >= `+2%` per 180d window.
- No overfitting signatures in holdout and perturbation results.

### Deliverables you must return
1. Exact algorithm description (math + rules + state machine).
2. Full code patches and parameter files.
3. Reproducible command list for all tests.
4. Full results tables with paths to raw artifacts.
5. A concise “why this is guaranteed” section with explicit assumptions and proof-style argument.

### Additional instruction
Do not return an incremental tweak of existing profiles if it still fails gates. If needed, switch strategy class entirely (ensemble of orthogonal edges, regime-conditioned policy, or market-making/latency-agnostic design) as long as it is implementable in this repo and testable with the same harness.

### Quality bar
Treat this as a production research mandate with zero tolerance for hand-wavy claims. Every claim must be tied to run artifacts and reproducible paths.

---

## Notes to researchers
- You are allowed to critique the feasibility assumptions, but you are still required to produce the strongest possible construction aimed at the hard target.
- If you change assumptions, state them explicitly and re-run the protocol under both original and revised assumptions.
