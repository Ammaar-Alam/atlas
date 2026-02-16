## 1) Diagnosis

### Root causes (structural)

1. **Entry/edge logic is not actually cost-complete for nano-perp microstructure**

* The strategy’s admission gate uses `cost_rt_bps = 2 * (slippage_bps + taker_fee_bps)` but does **not** incorporate the **fixed per-contract fee** into the effective bps hurdle at the current BTC price, even though the project cost model explicitly includes it.
* Fixed fee in bps is **price-dependent**:

  * per-side fixed bps ≈ `10000 * fixed_fee_usd / (contract_size_btc * price_usd)`
  * with `fixed_fee_usd=0.15` and `contract_size_btc=0.01`, per-side fixed bps ≈ `150000 / price`
  * e.g., at `price=50,000`, fixed ≈ `3 bps/side`, so roundtrip adds ≈ `6 bps` on top of the existing 23 bps proxy.
* Net effect: the admission hurdle is systematically understated, especially damaging on a small account where contract granularity makes “true” effective costs non-linear.

2. **Sizing is not contract-granularity-aware**

* The strategy sizes in continuous notional fractions and uses `min_trade_notional_usd=25`, but the venue contract is `0.01 BTC`. If the executor rounds to contract increments (very likely given the explicit fixed per-contract fee modeling), the strategy can unintentionally “snap” to **1 contract** even when its internal target is far smaller.
* For a $500 account, 1 contract often represents a large discrete step in exposure (and therefore PnL variance), which interacts badly with tight lockouts and stop logic.

3. **Equity-level lockouts are too tight for BTC 1H noise and create path-dependent “sell low, sit out rebound” behavior**

* `daily_loss_limit=2%` and `weekly_loss_limit=3%` are extremely tight relative to typical BTC hourly/2-day ATR regimes.
* These are **circuit breakers**, not position-level risk sizing tools: they force a flat target regardless of whether the trend signal remains valid, then prevent re-engagement until the next day/week.
* This tends to:

  * crystallize routine noise into realized losses,
  * suppress participation in subsequent recovery,
  * reduce weekly consistency (because you lock in negative weeks and then produce flat/low-capture weeks).

4. **Signal design is under-filtered for chop and the “edge proxy” is not forward-return calibrated**

* The current triad is essentially “trend-following twice”: sign of 14d momentum plus sign of EMA spread; these are highly correlated and do not add independent confirmation.
* There is no explicit “trendiness”/chop metric (efficiency ratio, ADX, regression t-stat, etc.). In BTC, this typically means too many entries during sideways regimes where fees dominate.

5. **Regime logic is discontinuous and can be unstable**

* `vol_z` uses a volatility z-score computed from the std of rolling vol. In quiet regimes, that std can compress, making `vol_z` jumpy. Hard cutoffs (`vol_off_z`, `crash_vol_z`) then create brittle on/off behavior across random windows.

6. **Forced minimum sizing via `max(confidence, 0.25)`**

* Even marginal edge gets at least 25% of base risk budget, which is a structural way to over-allocate to weak signals and then get hit by lockouts/stops.

7. **Operational gotcha: rebalance minute must match bar timestamps**

* On strict 1H bars where `ts.minute == 0`, any `rebalance_minute_utc > 0` would prevent rebalances entirely. Your current JSON uses `0` (good), but the design brief mentions `5`, which would be fatal if used unchanged.

(References: design brief, reviewer requirements, current code + profile.)    

---

## 2) Algorithm Design Changes (Required)

### Replacement spec: **Contract-quantized, cost-aware TSMOM ensemble with robust “trendiness” filter and smooth regime de-risking**

This keeps your original intent (TSMOM + vol targeting + crash de-risk + low turnover) but fixes the two largest mismatches: **contract mechanics** and **noise-driven lockouts**.

### 2.1 Features (computed each bar, used weekly for entry/resize; used off-cycle for exits)

Let `c_t` be close, `r_t = ln(c_t / c_{t-1})`.

**(A) Multi-horizon standardized time-series momentum (direction + strength)**
Choose horizons `h ∈ {h1, h2, h3, h4}` (bars).

* raw momentum:

  * `mom_h = ln(c_t / c_{t-h})`
  * `mom_h_bps = 10000 * mom_h`
* long-run vol estimate (per bar):

  * `σ_long = sqrt(EWMA(r^2, span=vol_long_span))`
* standardized momentum:

  * `z_h = mom_h / (σ_long * sqrt(h))`
* bounded contribution:

  * `m_h = tanh(z_h / mom_z_scale)`
* ensemble signal:

  * `m = Σ w_h * m_h` where `Σ w_h = 1`
* direction:

  * `side = sign(m)`
* strength:

  * `mom_score = |m|` (0..1-ish)

**(B) Trend significance via regression t-stat (anti-chop filter)**
On `L = trend_regression_bars`:

* regress `y_i = ln(c_{t-L+i})` on `x_i = i`
* compute slope t-stat `t_trend`
* require `sign(t_trend) == side` and `|t_trend| >= trend_tstat_entry`

**(C) “Trendiness” via Efficiency Ratio (Kaufman ER)**
On `N = er_window_bars`:

* `ER = |c_t - c_{t-N}| / Σ_{i=1..N} |c_{t-i+1} - c_{t-i}|`
* require `ER >= er_min`

**(D) Volatility regime via short/long EWMA ratio**

* `σ_short = sqrt(EWMA(r^2, span=vol_short_span))`
* `vol_ratio = σ_short / max(σ_long, eps)`
* entry block if `vol_ratio >= vol_ratio_off`
* continuous de-risk if `vol_ratio > vol_ratio_delever`:

  * `lev *= (vol_ratio_delever / vol_ratio) ^ vol_ratio_power`

**(E) ATR for stops/trailing and minimum movement gate**

* compute ATR on `atr_window`, convert to bps:

  * `ATR_bps = 10000 * (ATR / c_t)`
* require `ATR_bps >= min_atr_bps`

### 2.2 Cost-complete admission hurdle (includes fixed per-contract fee)

Read from `state.extra`:

* `slippage_bps`, `taker_fee_bps`
* `fixed_fee_per_contract_usd`
* `contract_size_units` (BTC per contract, e.g. 0.01)

Per-side fixed fee in bps (price dependent):

* `fixed_bps_side = 10000 * fixed_fee_per_contract_usd / (contract_size_units * c_t)`

Total per-side cost in bps:

* `cost_side_bps = slippage_bps + taker_fee_bps + fixed_bps_side`

Roundtrip estimate:

* `cost_rt_bps = 2 * cost_side_bps`

Admission requires **both**:

1. regime and trend filters (mom_score, t_trend, ER, vol_ratio, ATR_bps), and
2. a minimum medium-horizon move large enough to plausibly amortize costs:

   * define primary horizon momentum `mom_primary_bps = mom_h3_bps`
   * require:

     * `|mom_primary_bps| >= min_abs_long_momentum_bps + k_cost * cost_rt_bps`

This is simple, stable, and explicitly cost-aware (including fixed fee).

### 2.3 Sizing (vol-targeted, contract-quantized)

Raw leverage:

* `lev_raw = target_vol_per_bar / max(vol_floor, σ_short)`
* cap by margin:

  * `lev_cap = min(max_leverage, max_margin_utilization / maintenance_margin_rate)`
* de-risk for regime:

  * if `vol_ratio > vol_ratio_delever`, `lev = lev_raw * (vol_ratio_delever/vol_ratio)^vol_ratio_power`
  * else `lev = lev_raw`
* clamp: `lev = clip(lev, 0, lev_cap)`

Confidence scaling:

* `mom_conf = clip((mom_score - mom_score_min) / (1 - mom_score_min), 0, 1)`
* `trend_conf = clip((|t_trend| - trend_tstat_entry) / (trend_tstat_full - trend_tstat_entry), 0, 1)`
* `er_conf = clip((ER - er_min) / (er_full - er_min), 0, 1)`
* `confidence = clip( (0.5*mom_conf + 0.5*trend_conf) * (0.5 + 0.5*er_conf), 0, 1)`

Target notional:

* `notional_target = equity * lev * confidence`
* cap per symbol: `<= max_notional * max_per_symbol_exposure`

**Contract quantization**

* contract notional: `contract_notional = c_t * contract_size_units`
* minimum tradable notional: `min_notional = min_contracts * contract_notional`
* if `notional_target < min_notional`, do not enter (0 exposure)
* otherwise:

  * `qty_target_btc = (notional_target / c_t) * side`
  * quantize to contracts (floor/round/ceil):

    * `contracts = quantize(|qty_target_btc| / contract_size_units)`
    * `qty_quant_btc = contracts * contract_size_units * side`
  * translate back to exposure fraction:

    * `exp = (qty_quant_btc * c_t) / max_notional`

### 2.4 Exits (reduce churn, remove noise lockouts)

Keep ATR hard stop and trailing stop as a catastrophe guard, but add **signal-based trend break exit**:

If in position and `held >= min_hold_bars`, exit when either:

* **direction flips strongly**:

  * `sign(m) != current_side` and `|m| >= flip_exit_mom_score`
* or **trend collapses**:

  * `|m| < mom_exit_score` or `|t_trend| < trend_tstat_exit`

Add **cooldown** after a stop/flip exit:

* `cooldown_until_bar = now + cooldown_bars`
* block new entries while in cooldown

Critically: disable (or greatly widen) equity-level daily/weekly lockouts; they are the wrong tool for BTC 1H noise.

---

## 3) Code-Level Patch Plan (Required)

### Files to edit

1. `src/atlas/strategies/perp_research_vol_momentum.py` (primary changes)
2. `strategy_params/*.json` (new v2 profiles)

(Your uploaded version corresponds to the same strategy file content.) 

---

### 3.1 Patch: add helper functions (near the top, after `_atr`)

```python
def _ewma_rms(rets: pd.Series, span: int) -> Optional[float]:
    x = pd.to_numeric(rets, errors="coerce").dropna()
    if len(x) < 2:
        return None
    span = max(2, int(span))
    v = float((x * x).ewm(span=span, adjust=False).mean().iloc[-1])
    if not np.isfinite(v) or v < 0.0:
        return None
    return float(math.sqrt(v))


def _efficiency_ratio(close: pd.Series, window: int) -> Optional[float]:
    c = pd.to_numeric(close, errors="coerce").dropna()
    window = int(max(2, window))
    if len(c) < window + 2:
        return None
    c_tail = c.iloc[-(window + 1):]
    net = float(abs(c_tail.iloc[-1] - c_tail.iloc[0]))
    diffs = c_tail.diff().abs().dropna()
    denom = float(diffs.sum())
    if not np.isfinite(denom) or denom <= 0.0:
        return 0.0
    return float(_clamp(net / denom, 0.0, 1.0))


def _linreg_tstat(y: np.ndarray) -> Optional[float]:
    # OLS y = a + b x; return t-stat of slope b
    if y is None:
        return None
    y = np.asarray(y, dtype=float)
    n = int(y.shape[0])
    if n < 8:
        return None
    x = np.arange(n, dtype=float)
    x_mean = float(x.mean())
    y_mean = float(y.mean())
    xx = x - x_mean
    yy = y - y_mean
    sxx = float(np.dot(xx, xx))
    if not np.isfinite(sxx) or sxx <= 1e-12:
        return None
    b = float(np.dot(xx, yy) / sxx)
    a = float(y_mean - b * x_mean)
    resid = y - (a + b * x)
    sse = float(np.dot(resid, resid))
    dof = n - 2
    if not np.isfinite(sse) or dof <= 2:
        return None
    s2 = sse / float(dof)
    se_b = math.sqrt(max(s2 / sxx, 1e-18))
    t = float(b / se_b) if se_b > 0 else 0.0
    if not np.isfinite(t):
        return None
    return float(t)


def _fixed_fee_bps_side(price: float, contract_size_units: float, fixed_fee_per_contract_usd: float) -> float:
    # per-side fixed fee expressed in bps of notional; price-dependent
    p = float(price)
    cs = float(contract_size_units)
    ff = float(fixed_fee_per_contract_usd)
    if p <= 0.0 or cs <= 0.0 or ff <= 0.0:
        return 0.0
    notional_per_contract = p * cs
    if notional_per_contract <= 0.0:
        return 0.0
    return float((ff / notional_per_contract) * 10_000.0)


def _quantize_qty_to_contracts(qty_btc: float, contract_size_units: float, mode: str) -> float:
    cs = float(contract_size_units)
    if cs <= 0.0:
        return float(qty_btc)
    q = float(qty_btc)
    sign = 1.0 if q >= 0.0 else -1.0
    n = abs(q) / cs
    if mode == "ceil":
        k = int(math.ceil(n - 1e-12))
    elif mode == "round":
        k = int(round(n))
    else:  # "floor" default
        k = int(math.floor(n + 1e-12))
    return float(sign * k * cs)
```

---

### 3.2 Patch: add new parameters + internal state to the dataclass

Add these fields to `PerpResearchVolMomentum` (keep existing ones, do not delete):

```python
    # --- New: robust signal stack (TSMOM ensemble + trendiness) ---
    mom_h1_bars: int = 48
    mom_h2_bars: int = 168
    mom_h3_bars: int = 504
    mom_h4_bars: int = 1512
    mom_w1: float = 0.15
    mom_w2: float = 0.25
    mom_w3: float = 0.30
    mom_w4: float = 0.30
    mom_z_scale: float = 2.0
    mom_score_min: float = 0.20

    trend_regression_bars: int = 504
    trend_tstat_entry: float = 2.2
    trend_tstat_full: float = 4.0
    trend_tstat_exit: float = 1.0

    er_window_bars: int = 168
    er_min: float = 0.28
    er_full: float = 0.45

    vol_short_span: int = 48
    vol_long_span: int = 336
    vol_ratio_delever: float = 1.25
    vol_ratio_off: float = 1.80
    vol_ratio_power: float = 1.5

    # --- New: contract + cost mechanics ---
    min_contracts: int = 1
    qty_rounding: str = "floor"  # floor | round | ceil
    include_fixed_fee_in_cost: bool = True

    # --- New: exits/cooldown ---
    mom_exit_score: float = 0.12
    flip_exit_mom_score: float = 0.22
    cooldown_bars: int = 24

    # --- New: make tight equity lockouts optional ---
    use_daily_loss_lockout: bool = False
    use_weekly_loss_lockout: bool = False

    # Internal state
    _cooldown_until_bar: dict[str, int] = field(default_factory=dict, init=False, repr=False)
```

Update `warmup_bars()` to include the new windows:

```python
    def warmup_bars(self) -> int:
        return int(
            max(
                int(self.mom_h1_bars) + 3,
                int(self.mom_h2_bars) + 3,
                int(self.mom_h3_bars) + 3,
                int(self.mom_h4_bars) + 3,
                int(self.trend_regression_bars) + 3,
                int(self.er_window_bars) + 3,
                int(self.atr_window) + 3,
                int(self.vol_long_span) + 3,
                int(self.vol_short_span) + 3,
            )
            + 8
        )
```

---

### 3.3 Patch: replace `_features()` to compute the new stack

Replace your current `_features` entirely with:

```python
    def _features(self, df: pd.DataFrame) -> Optional[dict[str, float]]:
        if df is None or df.empty:
            return None
        if not df.index.is_monotonic_increasing:
            df = df.sort_index()
        if len(df) < self.warmup_bars():
            return None

        close = pd.to_numeric(df.get("close"), errors="coerce").dropna()
        if len(close) < self.warmup_bars():
            return None
        c = float(close.iloc[-1])
        if not np.isfinite(c) or c <= 0.0:
            return None

        # ATR
        atr = _atr(df, int(self.atr_window))
        if atr is None or atr <= 0.0:
            return None
        atr_bps = float((atr / c) * 10_000.0)

        # Returns + vols (EWMA RMS)
        rets = np.log(close / close.shift(1)).dropna()
        if len(rets) < max(int(self.vol_long_span), int(self.vol_short_span), 20):
            return None

        vol_short = _ewma_rms(rets, int(self.vol_short_span))
        vol_long = _ewma_rms(rets, int(self.vol_long_span))
        if vol_short is None or vol_long is None or vol_short <= 0.0 or vol_long <= 0.0:
            return None
        vol_ratio = float(vol_short / max(vol_long, 1e-12))

        # Efficiency ratio
        er = _efficiency_ratio(close, int(self.er_window_bars))
        if er is None:
            return None

        # Trend t-stat on log prices
        L = int(max(8, self.trend_regression_bars))
        if len(close) < L + 2:
            return None
        y = np.log(close.tail(L).to_numpy(dtype=float))
        trend_tstat = _linreg_tstat(y)
        if trend_tstat is None:
            return None

        # Multi-horizon momentum ensemble (standardized)
        horizons = [int(self.mom_h1_bars), int(self.mom_h2_bars), int(self.mom_h3_bars), int(self.mom_h4_bars)]
        weights = [float(self.mom_w1), float(self.mom_w2), float(self.mom_w3), float(self.mom_w4)]
        w_sum = float(sum(weights)) if sum(weights) != 0 else 1.0
        weights = [w / w_sum for w in weights]

        def _mom_bps(h: int) -> Optional[float]:
            h = int(max(2, h))
            if len(close) <= h:
                return None
            c_prev = float(close.iloc[-h - 1])
            if not np.isfinite(c_prev) or c_prev <= 0.0:
                return None
            return float(math.log(c / c_prev) * 10_000.0)

        mom_bps_list: list[float] = []
        mom_signal = 0.0
        for h, w in zip(horizons, weights):
            mb = _mom_bps(h)
            if mb is None:
                return None
            mom_bps_list.append(float(mb))
            # standardized z on log-return scale
            z = (mb / 10_000.0) / (float(vol_long) * math.sqrt(float(h)))
            contrib = math.tanh(float(z) / float(self.mom_z_scale))
            mom_signal += float(w) * float(contrib)

        mom_score = float(abs(mom_signal))
        mom_h1_bps, mom_h2_bps, mom_h3_bps, mom_h4_bps = [float(x) for x in mom_bps_list]

        return {
            "close": float(c),
            "atr": float(atr),
            "atr_bps": float(atr_bps),
            "vol_short": float(vol_short),
            "vol_long": float(vol_long),
            "vol_ratio": float(vol_ratio),
            "er": float(er),
            "trend_tstat": float(trend_tstat),
            "mom_signal": float(mom_signal),
            "mom_score": float(mom_score),
            "mom_h1_bps": float(mom_h1_bps),
            "mom_h2_bps": float(mom_h2_bps),
            "mom_h3_bps": float(mom_h3_bps),  # primary hurdle
            "mom_h4_bps": float(mom_h4_bps),
        }
```

---

### 3.4 Patch: modify `target_exposures()` in three places

#### (A) Make daily/weekly lockouts optional, keep kill switch

Replace the lockout checks with:

```python
        if self._risk_disabled_forever:
            return self._risk_off(symbols, reason="kill_switch", debug=debug)
        if drawdown <= -abs(float(self.kill_switch)):
            self._risk_disabled_forever = True
            return self._risk_off(symbols, reason="kill_switch", debug=debug)

        if self.use_daily_loss_lockout:
            if float(state.day_return) <= -abs(float(self.daily_loss_limit)):
                self._risk_disabled_day = today
                return self._risk_off(symbols, reason="daily_loss_limit", debug=debug)
            if self._risk_disabled_day == today:
                return self._risk_off(symbols, reason="risk_disabled_day", debug=debug)

        if self.use_weekly_loss_lockout:
            if week_ret <= -abs(float(self.weekly_loss_limit)):
                return self._risk_off(symbols, reason="weekly_loss_limit", debug=debug)
```

#### (B) Compute **cost_rt_bps including fixed fee**, and compute a **min exposure equal to 1+ contracts**

Right after you read `extra`:

```python
        fixed_fee_usd = float(extra.get("fixed_fee_per_contract_usd", 0.0) or 0.0)
        contract_size_units = float(extra.get("contract_size_units", 0.0) or 0.0)
        if contract_size_units <= 0.0:
            contract_size_units = 0.01  # safe default for nano BTC

        # NOTE: fixed fee in bps needs a price; we’ll compute it per-symbol once we know last price.
```

Then, after you have `px` per symbol inside the loop, compute per-symbol costs and keep the **max** (conservative) for gating, or just store per-symbol:

```python
            px = float(last_prices[s])
            fixed_bps_side = 0.0
            if self.include_fixed_fee_in_cost and px > 0.0:
                fixed_bps_side = _fixed_fee_bps_side(px, contract_size_units, fixed_fee_usd)

            cost_side_bps = float(abs(slippage_bps) + abs(taker_fee_bps) + abs(fixed_bps_side))
            cost_rt_bps_sym = float(2.0 * cost_side_bps)

            # dynamic 1-contract minimum exposure (prevents sub-contract rounding blowups)
            contract_notional = float(px * contract_size_units) if px > 0 else 0.0
            min_contracts = int(max(1, int(self.min_contracts)))
            min_notional = float(min_contracts * contract_notional)
            min_exp_sym = float(_clamp(min_notional / float(max_notional), 0.0, 1.0)) if min_notional > 0 else 0.0
```

You can store `cost_rt_bps_sym` and `min_exp_sym` in dicts for later use in entry sizing.

#### (C) Replace the weekly entry scoring block to use the new features + cooldown + contract quantization

Inside the weekly rebalance section, replace the per-symbol scoring logic with:

```python
            f = features.get(s)
            if f is None:
                continue

            # Cooldown after exits
            cd_until = int(self._cooldown_until_bar.get(s, 0) or 0)
            if int(self._bars_seen) < cd_until:
                continue

            mom_signal = float(f["mom_signal"])
            mom_score = float(f["mom_score"])
            trend_t = float(f["trend_tstat"])
            er = float(f["er"])
            atr_bps = float(f["atr_bps"])
            vol_short = float(f["vol_short"])
            vol_ratio = float(f["vol_ratio"])
            mom_primary_bps = float(f["mom_h3_bps"])

            if mom_score < float(self.mom_score_min):
                continue

            side = _sign(mom_signal)
            if side == 0:
                continue

            if _sign(trend_t) != side:
                continue
            if abs(trend_t) < float(self.trend_tstat_entry):
                continue
            if er < float(self.er_min):
                continue
            if atr_bps < float(self.min_atr_bps):
                continue
            if vol_ratio >= float(self.vol_ratio_off):
                continue
            if (not state.allow_short) and side < 0:
                continue

            # Per-symbol cost (include fixed fee bps at current price)
            px = float(f["close"])
            fixed_bps_side = _fixed_fee_bps_side(px, contract_size_units, fixed_fee_usd) if self.include_fixed_fee_in_cost else 0.0
            cost_side_bps = float(abs(slippage_bps) + abs(taker_fee_bps) + abs(fixed_bps_side))
            cost_rt_bps_sym = float(2.0 * cost_side_bps)

            required_mom_bps = float(self.min_abs_long_momentum_bps) + float(self.k_cost) * float(cost_rt_bps_sym)
            if abs(mom_primary_bps) < required_mom_bps:
                continue

            # Vol targeting with smooth deleveraging
            lev = float(self.target_vol_per_bar) / float(max(float(self.vol_floor), vol_short))
            if vol_ratio > float(self.vol_ratio_delever):
                lev *= float((float(self.vol_ratio_delever) / float(max(vol_ratio, 1e-12))) ** float(self.vol_ratio_power))
            lev = float(_clamp(lev, 0.0, lev_cap))

            # Confidence scaling
            mom_conf = float(_clamp((mom_score - float(self.mom_score_min)) / max(1.0 - float(self.mom_score_min), 1e-9), 0.0, 1.0))
            trend_conf = float(_clamp((abs(trend_t) - float(self.trend_tstat_entry)) / max(float(self.trend_tstat_full) - float(self.trend_tstat_entry), 1e-9), 0.0, 1.0))
            er_conf = float(_clamp((er - float(self.er_min)) / max(float(self.er_full) - float(self.er_min), 1e-9), 0.0, 1.0))
            confidence = float(_clamp((0.5 * mom_conf + 0.5 * trend_conf) * (0.5 + 0.5 * er_conf), 0.0, 1.0))
            if confidence <= 0.0 or lev <= 0.0:
                continue

            notional_target = float(lev) * float(equity) * float(confidence)
            notional_target = float(min(notional_target, float(max_notional) * float(self.max_per_symbol_exposure)))

            # Contract quantization + minimum contracts
            contract_notional = float(px * contract_size_units)
            min_contracts = int(max(1, int(self.min_contracts)))
            min_notional = float(min_contracts * contract_notional)
            if notional_target < min_notional:
                continue

            qty_target = float((notional_target / px) * float(side))
            qty_quant = _quantize_qty_to_contracts(qty_target, contract_size_units, str(self.qty_rounding or "floor"))
            if abs(qty_quant) < float(min_contracts) * contract_size_units:
                qty_quant = float(side) * float(min_contracts) * contract_size_units

            notional_quant = float(abs(qty_quant) * px)
            exp = float(_clamp(notional_quant / float(max_notional), 0.0, float(self.max_per_symbol_exposure)))

            score = float(abs(trend_t) * mom_score * confidence)
            scored.append((s, float(exp * side), score, f))
```

#### (D) Add trend-break exit + cooldown to the off-cycle exit loop

Inside the existing “Off-cycle risk exits” loop, after stop logic, add:

```python
            # Signal-based trend-break exits (after minimum hold)
            if held >= int(max(1, self.min_hold_bars)):
                mom_signal = float(f.get("mom_signal", 0.0))
                mom_score = float(abs(mom_signal))
                trend_t = float(f.get("trend_tstat", 0.0))

                flip = (_sign(mom_signal) != side) and (mom_score >= float(self.flip_exit_mom_score))
                collapse = (mom_score < float(self.mom_exit_score)) or (abs(trend_t) < float(self.trend_tstat_exit))

                if flip or collapse:
                    targets[s] = 0.0
                    self._cooldown_until_bar[s] = int(self._bars_seen + int(max(0, self.cooldown_bars)))
```

Also, when a stop triggers, set cooldown the same way.

---

## 4) Concrete Parameter Sets (Required)

Each JSON below is drop-in compatible with your existing `strategy_params/*.json` structure. Update filenames as you prefer. (These include the Coinbase nano-perp fee fields and contract specs.) 

### A) Conservative (low variance, higher selectivity, strong regime avoidance)

```json
{
  "perp_research_vol_momentum": {
    "rebalance_weekday_utc": 0,
    "rebalance_hour_utc": 0,
    "rebalance_minute_utc": 0,

    "mom_h1_bars": 48,
    "mom_h2_bars": 168,
    "mom_h3_bars": 504,
    "mom_h4_bars": 1512,
    "mom_w1": 0.10,
    "mom_w2": 0.20,
    "mom_w3": 0.30,
    "mom_w4": 0.40,
    "mom_z_scale": 2.2,
    "mom_score_min": 0.25,

    "trend_regression_bars": 1008,
    "trend_tstat_entry": 2.8,
    "trend_tstat_full": 5.0,
    "trend_tstat_exit": 1.2,

    "er_window_bars": 168,
    "er_min": 0.32,
    "er_full": 0.50,

    "vol_short_span": 48,
    "vol_long_span": 336,
    "vol_ratio_delever": 1.15,
    "vol_ratio_off": 1.60,
    "vol_ratio_power": 2.0,

    "min_abs_long_momentum_bps": 300.0,
    "min_atr_bps": 10.0,
    "k_cost": 3.0,

    "target_vol_per_bar": 0.0045,
    "vol_floor": 0.0025,
    "max_leverage": 2.5,
    "max_margin_utilization": 0.35,
    "max_gross_exposure": 0.80,
    "max_per_symbol_exposure": 0.80,
    "max_positions": 1,

    "min_contracts": 1,
    "qty_rounding": "floor",
    "include_fixed_fee_in_cost": true,

    "stop_atr_mult": 5.0,
    "trail_atr_mult": 7.0,
    "min_hold_bars": 48,
    "max_hold_bars": 1080,

    "mom_exit_score": 0.12,
    "flip_exit_mom_score": 0.24,
    "cooldown_bars": 48,

    "use_daily_loss_lockout": false,
    "use_weekly_loss_lockout": false,
    "daily_loss_limit": 0.06,
    "weekly_loss_limit": 0.10,
    "kill_switch": 0.20,

    "rebalance_exposure_threshold": 0.03,
    "min_trade_notional_usd": 25.0
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
    "allow_short": true,
    "paper_lookback_bars": 400,
    "paper_poll_seconds": 60,
    "paper_max_position_notional_usd": 5000.0,
    "paper_regular_hours_only": false,
    "paper_allow_trading_when_closed": true,
    "paper_dry_run": true
  }
}
```

---

### B) Balanced (target profile for robustness)

```json
{
  "perp_research_vol_momentum": {
    "rebalance_weekday_utc": 0,
    "rebalance_hour_utc": 0,
    "rebalance_minute_utc": 0,

    "mom_h1_bars": 48,
    "mom_h2_bars": 168,
    "mom_h3_bars": 504,
    "mom_h4_bars": 1512,
    "mom_w1": 0.15,
    "mom_w2": 0.25,
    "mom_w3": 0.30,
    "mom_w4": 0.30,
    "mom_z_scale": 2.0,
    "mom_score_min": 0.20,

    "trend_regression_bars": 504,
    "trend_tstat_entry": 2.2,
    "trend_tstat_full": 4.0,
    "trend_tstat_exit": 1.0,

    "er_window_bars": 168,
    "er_min": 0.28,
    "er_full": 0.45,

    "vol_short_span": 48,
    "vol_long_span": 336,
    "vol_ratio_delever": 1.25,
    "vol_ratio_off": 1.80,
    "vol_ratio_power": 1.5,

    "min_abs_long_momentum_bps": 250.0,
    "min_atr_bps": 8.0,
    "k_cost": 2.6,

    "target_vol_per_bar": 0.0055,
    "vol_floor": 0.0020,
    "max_leverage": 3.0,
    "max_margin_utilization": 0.40,
    "max_gross_exposure": 0.90,
    "max_per_symbol_exposure": 0.90,
    "max_positions": 1,

    "min_contracts": 1,
    "qty_rounding": "floor",
    "include_fixed_fee_in_cost": true,

    "stop_atr_mult": 4.5,
    "trail_atr_mult": 6.5,
    "min_hold_bars": 24,
    "max_hold_bars": 720,

    "mom_exit_score": 0.12,
    "flip_exit_mom_score": 0.22,
    "cooldown_bars": 24,

    "use_daily_loss_lockout": false,
    "use_weekly_loss_lockout": false,
    "daily_loss_limit": 0.06,
    "weekly_loss_limit": 0.10,
    "kill_switch": 0.20,

    "rebalance_exposure_threshold": 0.03,
    "min_trade_notional_usd": 25.0
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
    "allow_short": true,
    "paper_lookback_bars": 400,
    "paper_poll_seconds": 60,
    "paper_max_position_notional_usd": 5000.0,
    "paper_regular_hours_only": false,
    "paper_allow_trading_when_closed": true,
    "paper_dry_run": true
  }
}
```

---

### C) Aggressive (higher participation, higher risk budget)

```json
{
  "perp_research_vol_momentum": {
    "rebalance_weekday_utc": 0,
    "rebalance_hour_utc": 0,
    "rebalance_minute_utc": 0,

    "mom_h1_bars": 48,
    "mom_h2_bars": 168,
    "mom_h3_bars": 504,
    "mom_h4_bars": 1512,
    "mom_w1": 0.20,
    "mom_w2": 0.30,
    "mom_w3": 0.30,
    "mom_w4": 0.20,
    "mom_z_scale": 1.8,
    "mom_score_min": 0.18,

    "trend_regression_bars": 336,
    "trend_tstat_entry": 1.8,
    "trend_tstat_full": 3.5,
    "trend_tstat_exit": 0.8,

    "er_window_bars": 168,
    "er_min": 0.24,
    "er_full": 0.40,

    "vol_short_span": 48,
    "vol_long_span": 336,
    "vol_ratio_delever": 1.35,
    "vol_ratio_off": 2.10,
    "vol_ratio_power": 1.2,

    "min_abs_long_momentum_bps": 200.0,
    "min_atr_bps": 8.0,
    "k_cost": 2.3,

    "target_vol_per_bar": 0.0068,
    "vol_floor": 0.0020,
    "max_leverage": 4.0,
    "max_margin_utilization": 0.40,
    "max_gross_exposure": 0.95,
    "max_per_symbol_exposure": 0.95,
    "max_positions": 1,

    "min_contracts": 1,
    "qty_rounding": "floor",
    "include_fixed_fee_in_cost": true,

    "stop_atr_mult": 4.0,
    "trail_atr_mult": 6.0,
    "min_hold_bars": 24,
    "max_hold_bars": 504,

    "mom_exit_score": 0.10,
    "flip_exit_mom_score": 0.20,
    "cooldown_bars": 24,

    "use_daily_loss_lockout": false,
    "use_weekly_loss_lockout": false,
    "daily_loss_limit": 0.08,
    "weekly_loss_limit": 0.12,
    "kill_switch": 0.20,

    "rebalance_exposure_threshold": 0.03,
    "min_trade_notional_usd": 25.0
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
    "allow_short": true,
    "paper_lookback_bars": 400,
    "paper_poll_seconds": 60,
    "paper_max_position_notional_usd": 5000.0,
    "paper_regular_hours_only": false,
    "paper_allow_trading_when_closed": true,
    "paper_dry_run": true
  }
}
```

---

## 5) Validation Protocol and Acceptance Logic (Required)

### 5.1 Pre-flight correctness checks (must pass before any “performance” interpretation)

1. **Rebalance actually triggers on 1H bars**

* Confirm `rebalance_minute_utc == 0` for 1H timestamps.

2. **Fixed fee is included in debug + gating**

* Add debug fields:

  * `fixed_bps_side`, `cost_side_bps`, `cost_rt_bps`
* Sanity check: at BTC 50k, fixed_bps_side should print near 3.

3. **Contract quantization**

* Log `qty_quant_btc` and `contracts` (derived) in debug for a few decisions.
* Verify targets correspond to integer multiples of `contract_size_units` (or at least are consistent with the executor’s rounding rules).

### 5.2 Evaluation ordering (match your existing suites, just swap param file)

Run in this order for each of the three profiles (conservative, balanced, aggressive):

1. **Coinbase launch-era rolling suite**

* Re-run the same suite you already use (e.g., `research_vm_launch12_*`), with realistic fees enabled.
* Goal: verify the strategy is not “launch-only” and does not rely on a single window.

2. **Random 180d windows (increase sample size)**

* Your current evidence references `random10`; increase to **random30** minimum.
* If you have a proxy multi-year dataset feeding these windows, stratify by regime:

  * at least 10 windows with high realized vol,
  * at least 10 windows with low/moderate vol,
  * remainder random.

3. **External perp reality probes**

* Re-run the same three external probes you already have in `external_perp_probe/*`.
* Goal: signal stability (direction + trade frequency) across sources.

### 5.3 Pass/fail metrics (explicit)

**Primary robustness gates (must all pass on realistic cost model):**

1. **Random windows mean return > 0**

* Require:

  * `mean(net_return_180d) > 0`
  * and `median(net_return_180d) > 0`

2. **Weekly positive fraction**

* Require:

  * `overall_weekly_positive_fraction >= 0.70`
  * and at least **60% of windows** have weekly positive fraction `>= 0.65`

3. **SPY comparison**

* Require:

  * strategy beats SPY in **>= 60%** of windows
  * and mean excess return over SPY `>= +2%` per 180d window (net of all costs)

**Risk/operational sanity gates (reject “fragile winners”):**
4) **Drawdown**

* Require:

  * median max drawdown `< 15%`
  * worst-window max drawdown `< 25%`

5. **Turnover / cost drag**

* Require:

  * median roundtrips per 180d window `<= 12`
  * median average holding time `>= 5 days`
  * median fraction of gross PnL paid as fees `< 35%`

6. **Contract realism**

* Reject if a material fraction of trades are at sub-contract sizing (should be impossible after patch) or if “1-contract snap” dominates outcomes (indicates sizing still mismatched).

### 5.4 Anti-overfit checks (concrete rejection rules)

1. **Parameter sensitivity test (local robustness)**

* For the chosen best profile (likely Balanced), run 8 perturbations:

  * `±20%` on: `min_abs_long_momentum_bps`, `mom_score_min`, `trend_tstat_entry`, `vol_ratio_off`
* Accept only if:

  * mean return stays positive in at least **6/8** perturbations
  * weekly positive fraction stays `>= 0.65` in at least **6/8**

2. **One-shot holdout**

* Split random windows into:

  * 60% “selection” (pick best profile)
  * 40% “holdout” (evaluate once, no tuning)
* Reject if holdout mean return flips negative or weekly positive fraction drops below 0.65.

3. **Cross-venue direction stability**

* On external probes, require:

  * the sign of the mean return is consistent (not necessarily equal magnitude),
  * and the distribution of `side` decisions does not invert relative to the proxy source.

---

If you want one “default” to start with, use **Balanced** as the baseline for the next evaluation sweep and treat **Conservative** as the robustness backstop.
