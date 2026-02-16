# Execution Continuation TODO (Do Not Stop Until Goal)

## Mission
Find a deployable algorithm for Coinbase derivatives (`BTC-PERP`) under realistic costs that maximizes real-world profitability and robustness.

## Hard constraints to keep enforcing
- Use realistic Coinbase model in all critical tests:
  - `slippage_bps=1.5`
  - `taker_fee_bps=10.0`
  - `coinbase_fee_model=true`
  - `fixed_fee_per_contract_usd=0.15`
  - `contract_size_units=0.01`
- Do not present in-sample or single-window metrics as final claims.
- Prioritize out-of-sample transfer and robustness over headline return.

## Acceptance target (current sponsor requirement)
- Weekly-positive fraction gate target: `>= 0.70`
- Beat SPY requirement across windows
- Positive long-run returns across diverse windows
- No obvious overfit signatures

## Operational rule
- Keep iterating nonstop across strategy families and parameter regions.
- If context compacts or session restarts, resume from this file immediately.

## Current known blockers
- Candidate transfer instability across sources/regimes.
- Weekly-positive fraction far below target in robust evaluations.

## Current frontier (2026-02-16)
- Primary robust family: `perp_regime_adaptive_trend_capture` (short-enabled).
- Tri-source-positive candidates (Deribit22 + Coinbase launch12 + OKX20):
  - return frontier winner: `outputs/evaluations/strategy_strict_gate_search/ratc_from_c026cand008_deribit22_s3828_20260216_090616_seed3828/candidates/cand_023.json`
  - higher hit-rate alternative: `outputs/evaluations/strategy_strict_gate_search/ratc_from_c023_deribit22_s3829_20260216_092636_seed3829/candidates/cand_018.json`
- Current robust weekly-positive range remains low (`~0.23-0.34`, far below `0.70` gate).
- Rejected branch:
  - `ratc_from_c026cand008_deribit22_longonly_s3827` (fails Coinbase transfer).

## Active plan (always continue)
1. Maintain multi-track parallel search:
   - RATC family tuning (short-enabled and long-only variants).
   - Existing families (`perp_research_vol_momentum`, `perp_trend_vol_guard`, `perp_quant_fusion`) under same cost model.
   - Hybrid/ensemble gating if individual families fail robustness.
2. Evaluate candidates on:
   - Coinbase launch rolling windows.
   - External proxy multi-year windows.
   - Transfer checks (proxy-selected candidates replayed on Coinbase launch windows).
3. Rank by robust criteria:
   - Mean/median return, worst-window return, worst drawdown, weekly-positive fraction, SPY-relative metrics.
4. Persist best candidates + full reproduction commands and artifact paths.
5. Repeat until a materially better robust candidate is found or all explored families saturate.

## Immediate continuation loop
1. Tune RATC around `cand_000` and `cand_008` with short enabled, small drift (`0.10-0.20`) and varied crash/cooldown/vol-scaling.
2. For every completed proxy search:
   - Coinbase launch12 transfer check.
   - OKX20 transfer check.
   - tri-source merge (`min_mean`, `min_profitable_frac`, `min_weekly_positive_frac`).
3. Promote only candidates with:
   - positive mean return on all three sources,
   - non-trivial profitable-run fraction on all three,
   - no extreme drawdown blow-up relative to frontier.
4. Keep weekly gate hard target unchanged (`0.70`) and report gap explicitly each cycle.

## Latest executed continuation (2026-02-16)
- Completed:
  - `ratc_from_c023_deribit22_s3830` + Coinbase/OKX transfers
  - `ratc_from_c020_deribit22_s3831` + Coinbase/OKX transfers
- Result:
  - no robust frontier improvement vs `s3828/cand_023`
  - weekly-positive gap to target remains large

## Minimum report standard for any “best candidate”
- Exact params file path.
- Exact commands used.
- Artifact directories (`outputs/...`).
- Transfer-check metrics (must include adverse windows).
- Weekly-positive fraction and SPY-relative performance explicitly stated.

## Storage hygiene (periodic)
- After each major search batch, prune bulky dead-end artifacts while preserving strategy knowledge.
- Preserve:
  - strategy code in `src/atlas/strategies/`
  - candidate params in `strategy_params/`
  - research summaries and protocols in `docs/algorithms/`
  - leaderboard/window summary CSV/JSON outputs
- Prune older heavy run payloads (especially stale):
  - `decisions.jsonl`
  - `equity_curve.csv`
  - debug-only run artifacts no longer needed for active diagnosis
- Standard prune commands:
  - Dry run:
    - `python3 scripts/prune_outputs.py --min-age-hours 0 --keep-recent-dirs 0 --keep-top-runs 5`
  - Apply:
    - `python3 scripts/prune_outputs.py --min-age-hours 0 --keep-recent-dirs 0 --keep-top-runs 5 --apply`
  - Dead-end folder cleanup:
    - `python3 scripts/prune_outputs.py --min-age-hours 2 --keep-recent-dirs 4 --keep-top-runs 3 --drop-deadends --apply`

## Superseding Frontier Update (2026-02-16 evening)
- Replace prior primary frontier candidate with:
  - `strategy_params/perp_regime_adaptive_trend_capture_trisource_best_s3861_c034.json`
- Source run:
  - `outputs/evaluations/strategy_strict_gate_search/ratc_from_s3835cand005_coinbase12_s3861_20260216_203856_seed3861/candidates/cand_034.json`
- Cross-source summary:
  - `outputs/evaluations/strategy_eval/ratc_s3861_crossproxy_summary.csv`
- Current robust metrics for promoted frontier:
  - `min_mean=+21.51%`, `avg_mean=+23.58%`, `min_prof_frac=0.30`, `min_weekly=0.2109`

## Immediate continuation loop (updated)
1. Keep `s3861/cand_034` as anchor base.
2. Run bidirectional transfer searches (coinbase-first and proxy-first), then mandatory tri-source merge.
3. Promote only if candidate beats current frontier on at least one of:
   - higher `min_mean` with non-collapsing `min_prof_frac`, or
   - materially higher `min_prof_frac` while preserving `all_three_mean_pos` and acceptable `min_mean`.
4. Keep weekly gate requirement hard (`0.70`) and report distance every cycle.
5. Continue periodic storage hygiene after each major batch.

## Latest decision lock
- Keep `s3861/cand_034` as return-frontier anchor.
- Keep legacy `cand_000` as stability comparator.
- Reject `s3870`, `s3871`, and local `c034_refine` variants for frontier promotion.
