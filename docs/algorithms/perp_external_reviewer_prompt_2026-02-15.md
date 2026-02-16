# External Reviewer Prompt: Derivatives Strategy Rescue (Coinbase Realistic Fees)

## Purpose
You are reviewing and improving an existing quantitative trading strategy implementation in a Python repository (`atlas`).
Your output must be actionable at code level: concrete algorithm changes, specific parameter values, and reproducible validation steps.

This is not a generic essay request.

## Current State (Important)
- Venue/market target: Coinbase Advanced derivatives, nano BTC PERP.
- Account size target: `$500`.
- Cost model (must be modeled in all recommendations):
  - taker fee: `10 bps` per side
  - slippage: `1.5 bps` per side
  - fixed fee: `$0.15` per contract per side
  - contract size: `0.01 BTC`
- Strategy currently under review:
  - code: `src/atlas/strategies/perp_research_vol_momentum.py`
  - params/profile: `strategy_params/perp_research_vol_momentum_v1_1h_coinbase_profile.json`
  - design brief: `docs/algorithms/perp_research_vol_momentum_research_brief_2026-02-15.md`

## Data Reality (Critical)
- Coinbase BTC-PERP history in this environment effectively starts around `2025-07-18`.
- A requested `5y` backtest can still produce an observed data window around `~211` days.
- The TUI now shows both `requested_window` and `observed_data_window` for clarity.

## Robustness Targets
Propose improvements intended to satisfy ALL of the following on realistic-cost evaluation:
1. Positive mean return across random 180d windows.
2. Weekly positive fraction target >= `0.70`.
3. Beat SPY in a meaningful fraction of windows (not just launch-era).
4. Avoid apparent launch-era-only overfit behavior.

## Existing Evidence of Failure
Current research strategy does not meet targets:
- `outputs/evaluations/strategy_eval/research_vm_launch12_20260215_172307/evaluation_result.json`
- `outputs/evaluations/strategy_eval/research_vm_random10_20260215_172307/evaluation_result.json`

External-proxy reality probes (cand08 family) show instability across regimes/sources:
- `outputs/evaluations/external_perp_probe/20260215_165618/result.json`
- `outputs/evaluations/external_perp_probe/20260215_165955/result.json`
- `outputs/evaluations/external_perp_probe/20260215_170534/result.json`

## What You Must Deliver
Return your answer in five sections, in this exact order.

### 1) Diagnosis
- Identify the top 3-7 root causes for failure under the current constraints.
- Distinguish structural issues (model design) from parameter issues (thresholds/windows).

### 2) Algorithm Design Changes (Required)
- Provide a revised algorithm design at implementation level.
- Include exact formulas and decision logic.
- If you recommend replacing the strategy entirely, provide replacement architecture and rationale.

### 3) Code-Level Patch Plan (Required)
- Specify exact files to edit.
- Provide patch-style pseudo-diffs or explicit code blocks for the changed methods.
- Include any new strategy params with defaults.
- Ensure compatibility with existing registry and TUI parameter workflows.

### 4) Concrete Parameter Sets (Required)
- Provide at least 3 candidate parameter profiles:
  - conservative
  - balanced
  - aggressive
- Output them as JSON objects compatible with `strategy_params/*.json` style.
- Include intended market/data profile values (bar timeframe, fee model toggles, shorting mode).

### 5) Validation Protocol and Acceptance Logic (Required)
- Provide explicit backtest/evaluation commands and ordering.
- Define pass/fail metrics and minimum sample sizes.
- Include anti-overfit checks and how to reject false positives.

## Additional Constraints
- No hand-wavy language like "likely profitable" without explicit test criteria.
- Account for fixed per-contract fee impact on small accounts.
- Do not ignore turnover/cost drag.
- Assume live deployment should only happen after robustness pass.

## Preferred Response Quality
- Think at quantitative-research depth.
- Prioritize robust repeatability over headline one-window returns.
- Be explicit enough that an engineer can implement your recommendations without guessing.

