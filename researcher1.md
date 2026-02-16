Let me start by reading the strategy implementation code.Now I have the full picture. Here's my review across the five sections you've laid out in the reviewer prompt.

---

## 1) Diagnosis: Root Causes for Failure

**Structural issues (model design):**

1. **Edge proxy is circular and inflated.** The formula `edge_bps = |trend_strength| * ATR_bps + 0.35 * |mom_long_bps|` doesn't estimate expected return — it's a composite signal strength metric with arbitrary weights. A trend_strength of 0.5 and ATR_bps of 100 gives 50 bps of "edge" just from the first term, but trend_strength is itself normalized by ATR, so you're partially double-counting volatility. The 0.35 coefficient on momentum is unjustified. This means the cost gate (`edge >= floor + k * cost_rt`) is not actually filtering by expected PnL — it's filtering by signal magnitude, which can be large even when the actual expected holding-period return after costs is negative.

2. **Fixed per-contract fee is ignored in the admission logic.** At BTC ≈ $95k, one contract (0.01 BTC) is ~$950 notional. The $0.15 fixed fee per side is ~1.6 bps on top of the 10+1.5 bps already modeled. But more critically, for a $500 account trading $950 notional round-trip, the fixed fee alone is $0.30, which is 6 bps on account equity per round-trip. The `cost_rt_bps` variable only captures taker + slippage (23 bps on notional), but doesn't convert the fixed fee into effective bps at the actual trade size. Small accounts get crushed here.

3. **Weekly cadence is too slow for the signal horizons used.** Long momentum is 336 bars (14 days) and short momentum is 48 bars (2 days), but you only act once per week. A 2-day reversal signal that fires on Tuesday is stale by Monday rebalance. The mismatch between signal frequency and action frequency means you're systematically late to entries and late to crash exits.

4. **Single-instrument, single-position design has no diversification.** With max_positions=1 on BTC-PERP only, you're making a single binary bet per week. In any regime where BTC chops sideways (which is common), every entry eats a full round-trip cost for zero expected directional alpha.

**Parameter issues:**

5. **Momentum threshold too low for the cost structure.** `min_abs_long_momentum_bps=45` means you'll enter on a 0.45% 14-day move. After 23+ bps round-trip cost on notional (plus fixed fees), this leaves almost no room for the trade to work, especially if the signal is mean-reverting noise rather than genuine trend.

6. **Leverage sizing overshoot for $500.** `target_vol_per_bar=0.0065` with `vol_floor=0.002` can produce `lev_raw = 3.25x` even in calm markets. On $500 equity, that's $1,625 notional — roughly 1.7 contracts. The fixed fee burden per contract makes this corrosive on small size.

7. **Stop/trail ATR multipliers are wide relative to edge.** A 3.2×ATR hard stop and 4.2×ATR trailing stop on 1H bars means you're willing to lose 3-4% of entry price before exiting. With a $500 account at 3x leverage, that's a ~10% equity drawdown per losing trade, vs. the 2% daily loss limit and 3% weekly loss limit. The stops don't protect equity meaningfully before the loss limits already fire.

---

## 2) Algorithm Design Changes

I'd recommend a hybrid approach: keep the momentum+vol-management skeleton but fix the structural flaws.

**A. Replace the edge proxy with a cost-aware expected return estimate:**

```
# Effective round-trip cost in bps on NOTIONAL
n_contracts = ceil(notional_target / contract_value)
fixed_fee_total = n_contracts * fixed_fee_per_contract * 2  # round-trip
fixed_fee_bps_on_notional = (fixed_fee_total / notional_target) * 10000
cost_rt_bps_total = 2 * (taker_fee_bps + slippage_bps) + fixed_fee_bps_on_notional

# Expected holding-period return proxy (bps)
# Use historical realized conditional return for this signal bucket
expected_return_bps = mom_long_bps / long_momentum_bars * expected_hold_bars * decay_factor
# where decay_factor accounts for signal decay (e.g., 0.5-0.7)

# Admission gate
net_edge_bps = expected_return_bps - cost_rt_bps_total
REQUIRE: net_edge_bps >= min_net_edge_bps  # e.g., 15-25 bps
```

This directly estimates whether the trade can pay for itself.

**B. Move to semi-weekly or bi-weekly rebalance with fast risk exits:**

Instead of a single Monday rebalance, allow rebalance on two fixed points (e.g., Monday 00:00 UTC and Thursday 00:00 UTC). Keep off-cycle exits for stops/crashes, but also allow off-cycle *entry reduction* if the crash detector fires. This halves your average signal latency without going to daily cadence.

**C. Add a regime filter using realized vol rank:**

```
vol_percentile = rank(vol_now, vol_history_720_bars)  # 0-1 scale
# Only enter if vol is in the 20th-80th percentile band
# Very low vol = no opportunity; very high vol = crash risk
REQUIRE: vol_pctl_low <= vol_percentile <= vol_pctl_high
```

This replaces the binary vol_off_z switch with a smoother filter.

**D. Add minimum expected holding period gate:**

```
# Don't enter if the maximum hold (max_hold_bars) is too short to overcome costs
min_profitable_bars = cost_rt_bps_total / (target_vol_per_bar * 10000 * sharpe_assumption_per_bar)
REQUIRE: max_hold_bars >= min_profitable_bars * 1.5
```

**E. Trend confirmation via return dispersion:**

Instead of requiring `sign(trend_strength) == sign(mom_long)`, add a requirement that the trend has been consistent, not a single-bar spike:

```
# Fraction of sub-windows where momentum direction agrees
n_subwindows = 4
consistency = mean([sign(mom(P, long_momentum_bars//n_subwindows, offset=i*(long_momentum_bars//n_subwindows))) == side 
                    for i in range(n_subwindows)])
REQUIRE: consistency >= 0.75
```

---

## 3) Code-Level Patch Plan

**File: `src/atlas/strategies/perp_research_vol_momentum.py`**

**3a. New parameters to add to the dataclass:**

```python
# Rebalance frequency
rebalance_days_utc: tuple[int, ...] = (0, 3)  # Monday and Thursday

# Cost-aware admission
expected_hold_bars: int = 120  # 5 days
signal_decay_factor: float = 0.55
min_net_edge_bps: float = 18.0

# Regime filter
vol_pctl_low: float = 0.15
vol_pctl_high: float = 0.82

# Trend consistency
trend_consistency_min: float = 0.75
trend_consistency_subwindows: int = 4
```

**3b. Patch `_is_rebalance_time`:**

```python
def _is_rebalance_time(self, ts: pd.Timestamp) -> bool:
    ts = pd.Timestamp(ts)
    if ts.tz is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    if int(ts.dayofweek) not in self.rebalance_days_utc:
        return False
    if int(ts.hour) != int(self.rebalance_hour_utc):
        return False
    return int(ts.minute) >= int(self.rebalance_minute_utc)
```

**3c. Patch `_features` to add consistency and vol percentile:**

After computing `vol_z`, add:

```python
# Vol percentile
vol_percentile = float((vol_hist_tail < vol_now).mean()) if len(vol_hist_tail) > 10 else 0.5

# Trend consistency
n_sub = max(2, int(self.trend_consistency_subwindows))
sub_len = max(2, lb_long // n_sub)
consistency_votes = 0
for i in range(n_sub):
    offset = i * sub_len
    if len(close) > sub_len + offset:
        c_sub_start = float(close.iloc[-(sub_len + offset) - 1])
        c_sub_end = float(close.iloc[-offset - 1]) if offset > 0 else c
        if c_sub_start > 0:
            sub_mom = math.log(c_sub_end / c_sub_start)
            if _sign(sub_mom) == _sign(mom_long_bps):
                consistency_votes += 1
trend_consistency = consistency_votes / n_sub
```

Return these in the features dict.

**3d. Patch the entry logic in `target_exposures`:**

Replace the current edge calculation block with:

```python
# Cost-aware expected return
contract_value = close * 0.01  # contract_size_units from profile
n_contracts = max(1, math.ceil(notional_target / contract_value))
fixed_fee_rt = n_contracts * 0.15 * 2  # $0.15 per contract per side
fixed_fee_bps = (fixed_fee_rt / max(notional_target, 1.0)) * 10000
cost_rt_bps_full = cost_rt_bps + fixed_fee_bps

expected_ret_bps = (mom_long_bps / max(1, self.long_momentum_bars)) * self.expected_hold_bars * self.signal_decay_factor
net_edge = abs(expected_ret_bps) - cost_rt_bps_full

if net_edge < self.min_net_edge_bps:
    continue

# Vol percentile filter
vol_pctl = f["vol_percentile"]
if vol_pctl < self.vol_pctl_low or vol_pctl > self.vol_pctl_high:
    continue

# Trend consistency filter
if f["trend_consistency"] < self.trend_consistency_min:
    continue
```

**3e. Adjust stops to be tighter and equity-aware:**

```python
# Scale stop width by account fraction at risk
max_loss_per_trade_pct = 0.015  # 1.5% of equity per trade
stop_atr_mult_adj = min(
    self.stop_atr_mult,
    (max_loss_per_trade_pct * equity) / (abs(current_exp[s]) * max_notional * atr / close)
)
```

---

## 4) Concrete Parameter Sets

**Conservative:**
```json
{
  "perp_research_vol_momentum": {
    "rebalance_days_utc": [0, 3],
    "rebalance_hour_utc": 0,
    "rebalance_minute_utc": 5,
    "long_momentum_bars": 504,
    "short_momentum_bars": 72,
    "ema_fast": 48,
    "ema_slow": 336,
    "atr_window": 72,
    "vol_lookback_bars": 168,
    "vol_regime_window": 720,
    "min_abs_long_momentum_bps": 80.0,
    "min_atr_bps": 10.0,
    "trend_strength_min": 0.15,
    "min_net_edge_bps": 25.0,
    "signal_decay_factor": 0.45,
    "expected_hold_bars": 144,
    "trend_consistency_min": 0.75,
    "trend_consistency_subwindows": 4,
    "vol_pctl_low": 0.20,
    "vol_pctl_high": 0.75,
    "target_vol_per_bar": 0.0045,
    "vol_floor": 0.0025,
    "max_leverage": 2.5,
    "max_margin_utilization": 0.30,
    "max_gross_exposure": 0.80,
    "max_per_symbol_exposure": 0.80,
    "max_positions": 1,
    "min_trade_notional_usd": 50.0,
    "rebalance_exposure_threshold": 0.05,
    "crash_vol_z": 1.0,
    "crash_reversal_bps": 45.0,
    "crash_risk_scale": 0.20,
    "vol_off_z": 2.0,
    "stop_atr_mult": 2.5,
    "trail_atr_mult": 3.0,
    "min_hold_bars": 36,
    "max_hold_bars": 288,
    "weekly_loss_limit": 0.02,
    "daily_loss_limit": 0.015,
    "kill_switch": 0.15
  },
  "atlas_profile": {
    "market": "derivatives",
    "data_source": "coinbase",
    "symbols": "BTC-PERP",
    "bar_timeframe": "1H",
    "timeframe": "180d",
    "initial_cash": 500.0,
    "max_position_notional_usd": 5000.0,
    "slippage_bps": 1.5,
    "taker_fee_bps": 10.0,
    "coinbase_fee_model": true,
    "fixed_fee_per_contract_usd": 0.15,
    "contract_size_units": 0.01,
    "allow_short": true
  }
}
```

**Balanced:**
```json
{
  "perp_research_vol_momentum": {
    "rebalance_days_utc": [0, 3],
    "rebalance_hour_utc": 0,
    "rebalance_minute_utc": 5,
    "long_momentum_bars": 336,
    "short_momentum_bars": 48,
    "ema_fast": 36,
    "ema_slow": 240,
    "atr_window": 48,
    "vol_lookback_bars": 120,
    "vol_regime_window": 720,
    "min_abs_long_momentum_bps": 60.0,
    "min_atr_bps": 8.0,
    "trend_strength_min": 0.12,
    "min_net_edge_bps": 18.0,
    "signal_decay_factor": 0.55,
    "expected_hold_bars": 120,
    "trend_consistency_min": 0.75,
    "trend_consistency_subwindows": 4,
    "vol_pctl_low": 0.15,
    "vol_pctl_high": 0.80,
    "target_vol_per_bar": 0.0055,
    "vol_floor": 0.0020,
    "max_leverage": 3.0,
    "max_margin_utilization": 0.35,
    "max_gross_exposure": 0.90,
    "max_per_symbol_exposure": 0.90,
    "max_positions": 1,
    "min_trade_notional_usd": 40.0,
    "rebalance_exposure_threshold": 0.04,
    "crash_vol_z": 1.15,
    "crash_reversal_bps": 50.0,
    "crash_risk_scale": 0.25,
    "vol_off_z": 2.2,
    "stop_atr_mult": 2.8,
    "trail_atr_mult": 3.5,
    "min_hold_bars": 24,
    "max_hold_bars": 240,
    "weekly_loss_limit": 0.025,
    "daily_loss_limit": 0.018,
    "kill_switch": 0.18
  },
  "atlas_profile": {
    "market": "derivatives",
    "data_source": "coinbase",
    "symbols": "BTC-PERP",
    "bar_timeframe": "1H",
    "timeframe": "180d",
    "initial_cash": 500.0,
    "max_position_notional_usd": 5000.0,
    "slippage_bps": 1.5,
    "taker_fee_bps": 10.0,
    "coinbase_fee_model": true,
    "fixed_fee_per_contract_usd": 0.15,
    "contract_size_units": 0.01,
    "allow_short": true
  }
}
```

**Aggressive:**
```json
{
  "perp_research_vol_momentum": {
    "rebalance_days_utc": [0, 2, 4],
    "rebalance_hour_utc": 0,
    "rebalance_minute_utc": 5,
    "long_momentum_bars": 240,
    "short_momentum_bars": 36,
    "ema_fast": 24,
    "ema_slow": 168,
    "atr_window": 36,
    "vol_lookback_bars": 96,
    "vol_regime_window": 600,
    "min_abs_long_momentum_bps": 50.0,
    "min_atr_bps": 8.0,
    "trend_strength_min": 0.10,
    "min_net_edge_bps": 12.0,
    "signal_decay_factor": 0.60,
    "expected_hold_bars": 96,
    "trend_consistency_min": 0.50,
    "trend_consistency_subwindows": 4,
    "vol_pctl_low": 0.10,
    "vol_pctl_high": 0.85,
    "target_vol_per_bar": 0.0070,
    "vol_floor": 0.0018,
    "max_leverage": 4.0,
    "max_margin_utilization": 0.40,
    "max_gross_exposure": 0.95,
    "max_per_symbol_exposure": 0.95,
    "max_positions": 1,
    "min_trade_notional_usd": 25.0,
    "rebalance_exposure_threshold": 0.03,
    "crash_vol_z": 1.30,
    "crash_reversal_bps": 60.0,
    "crash_risk_scale": 0.30,
    "vol_off_z": 2.5,
    "stop_atr_mult": 3.0,
    "trail_atr_mult": 4.0,
    "min_hold_bars": 18,
    "max_hold_bars": 192,
    "weekly_loss_limit": 0.035,
    "daily_loss_limit": 0.022,
    "kill_switch": 0.22
  },
  "atlas_profile": {
    "market": "derivatives",
    "data_source": "coinbase",
    "symbols": "BTC-PERP",
    "bar_timeframe": "1H",
    "timeframe": "180d",
    "initial_cash": 500.0,
    "max_position_notional_usd": 5000.0,
    "slippage_bps": 1.5,
    "taker_fee_bps": 10.0,
    "coinbase_fee_model": true,
    "fixed_fee_per_contract_usd": 0.15,
    "contract_size_units": 0.01,
    "allow_short": true
  }
}
```

---

## 5) Validation Protocol and Acceptance Logic

**Phase 1: Sanity (run first, reject fast)**

Run each of the 3 parameter profiles on the launch-era Coinbase window. Reject any profile where:
- Total return is negative.
- Max drawdown exceeds the kill_switch threshold.
- Turnover produces > 15 round-trips in 180 days (cost drag check).
- Sharpe < 0.3 annualized.

**Phase 2: Random-window robustness (must pass)**

Run 20+ random 180-day windows from multi-year BTC proxy data (OKX/Deribit hourly if available). For each surviving profile:
- Mean return across windows must be positive.
- Weekly positive fraction ≥ 0.65 (relaxed from 0.70 initially; raise threshold only after the algorithm changes are validated).
- Median return across windows must be positive (not just the mean — this catches right-skewed distributions propped up by one lucky window).
- No more than 30% of windows should have max drawdown exceeding 10%.

**Phase 3: Anti-overfit checks**

- Compare in-sample vs. out-of-sample Sharpe ratio. If in-sample Sharpe is more than 2× out-of-sample, flag as overfit.
- Compute the Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014) adjusting for the number of parameter sets tried. With 3 profiles and say 5 prior failed strategies, that's ~8 trials — the DSR penalty should be computable.
- Run a permutation test: shuffle the signal (randomize entry timing while keeping the same number of trades and holding periods). The real strategy should beat 95% of permuted variants.

**Phase 4: Cross-venue consistency**

Run the surviving profile(s) on at least one non-Coinbase BTC perp data source. The sign of mean return should be the same. Sharpe shouldn't collapse by more than 50%.

**Acceptance decision:**

A profile passes if and only if it clears all four phases. If multiple profiles pass, prefer the conservative one unless the balanced profile materially dominates on both Sharpe and weekly consistency.

**Kill criteria (hard reject the entire strategy):**

If zero profiles pass Phase 2, the strategy architecture itself is likely insufficient for this cost structure at $500. In that case, the honest recommendation is: either increase account size to $2,000+ (where fixed fees matter less), switch to a maker-fee execution model (if available), or move to a longer timeframe (4H/daily bars) where per-trade edge has more room to exceed costs.

---

**Bottom line:** The biggest single fix is replacing the edge proxy with a cost-aware expected return gate that incorporates the fixed per-contract fee at the actual trade size. That alone would prevent most of the losing trades that are currently passing admission. The second most impactful change is the trend consistency filter, which should eliminate whipsaw entries in choppy regimes. Everything else is refinement.
