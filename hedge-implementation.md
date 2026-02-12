## 1) Definitions and setup

### Prices, funding, basis

Let (t\in{0,\Delta,2\Delta,\dots}) be a discrete decision grid (bar size (\Delta)). Let ({\tau_k}*{k\ge 1}\subset{t}) be the set of funding timestamps, with funding interval (\Delta_f := \tau*{k+1}-\tau_k) (kept symbolic).

* Spot mid price: (S_t) (quote currency per 1 unit base).
* Perpetual mark price: (P_t) (quote per 1 unit base).
* Funding rate at funding timestamp (\tau_k): (r_{\tau_k}).

**Funding sign convention (explicit):** (r_{\tau_k}>0) means **perp longs pay perp shorts** at (\tau_k), proportional to notional.

Define basis:
[
b_t := \frac{P_t - S_t}{S_t}\quad\Longleftrightarrow\quad P_t = S_t(1+b_t).
]

### Positions and hedge condition

Let:

* Spot position (base units): (q_s(t)\in\mathbb{R}) (positive = long spot, negative = short spot).
* Perp position (base units): (q_p(t)\in\mathbb{R}) (positive = long perp, negative = short perp).

Spot notional and perp notional at time (t):
[
N_s(t) := q_s(t) S_t,\qquad N_p(t) := q_p(t) P_t.
]

**Notional-matched hedge (market-neutral target):**
[
q_s(t) S_t \approx - q_p(t) P_t
\quad\Longleftrightarrow\quad
q_s(t)\approx -q_p(t)\frac{P_t}{S_t}= -q_p(t)(1+b_t).
]

Define hedge error (net directional notional at time (t)):
[
\varepsilon_t := q_s(t)S_t + q_p(t)P_t,
\qquad
\delta_t := \frac{\varepsilon_t}{N_t}\ \ \text{(dimensionless mismatch ratio)},
]
where (N_t) is a gross notional scale, e.g.
[
N_t := \frac{|q_s(t)|S_t + |q_p(t)|P_t}{2}.
]

A convenient parameterization for a **perfect** notional hedge at target gross notional (N_t>0) is:
[
q_p(t)= s_t,\frac{N_t}{P_t},\qquad
q_s(t)= -s_t,\frac{N_t}{S_t},
\qquad s_t\in{+1,-1},
]
where (s_t=+1) means **long perp / short spot** (reverse carry) and (s_t=-1) means **short perp / long spot** (cash-and-carry).

### Transaction costs and execution frictions (symbolic variables)

Let effective proportional costs (fees + expected slippage/half-spread + impact proxy) be:

* Spot: (c_s(t)) per unit spot notional traded.
* Perp: (c_p(t)) per unit perp notional traded.

If you trade (\Delta q_s(t)=q_s(t)-q_s(t^-)) and (\Delta q_p(t)=q_p(t)-q_p(t^-)) at time (t), define trading cost:
[
C_t = c_s(t),|\Delta q_s(t)|,S_t;+; c_p(t),|\Delta q_p(t)|,P_t.
]

If shorting spot requires borrowing base (margin spot), include a **spot financing** term (kept generic):
[
\text{Fin}_t = \rho_t^{(s)} ,|q_s(t)|S_t\cdot \Delta
]
where (\rho_t^{(s)}) is the net financing rate per unit time for the spot leg (could be (0) if fully-funded long spot with no borrow; positive if paying borrow).

Funding cashflow at each (\tau_k) is treated as an instantaneous transfer:
[
\text{Funding CF at }\tau_k:\quad \mathrm{FCF}*{\tau_k} = -,q_p(\tau_k^-),P*{\tau_k},r_{\tau_k}.
]

---

## 2) Exact PnL decomposition (discrete-time, one holding step)

Consider holding positions ((q_s(t),q_p(t))) constant over ([t,t+\Delta]), with possible funding events (\tau\in(t,t+\Delta]).

### Total one-step PnL

Define (\Delta S := S_{t+\Delta}-S_t), (\Delta P := P_{t+\Delta}-P_t). Then the **total** PnL over the step is:
[
\Pi_{t\to t+\Delta}
===================

\underbrace{q_s(t)\Delta S}*{\text{Spot price PnL}}
+\underbrace{q_p(t)\Delta P}*{\text{Perp MTM PnL}}
+\underbrace{\sum_{\tau\in(t,t+\Delta]}\left(-q_p(\tau^-),P_\tau,r_\tau\right)}*{\text{Funding cashflow}}
-\underbrace{C_t}*{\text{Fees+slippage+impact}}
-\underbrace{\text{Fin}*t}*{\text{Spot financing (if any)}}.
]

### Separating basis dynamics and hedge mismatch (exact identity)

Using (P_t=S_t(1+b_t)), we have the **exact** decomposition:
[
\Delta P
= P_{t+\Delta}-P_t
= (1+b_t)\Delta S + S_{t+\Delta}\Delta b,
\quad \Delta b:=b_{t+\Delta}-b_t.
]

Substitute into spot+perp mark-to-market:
[
q_s\Delta S + q_p\Delta P
=========================

\underbrace{\left(q_s + q_p(1+b_t)\right)\Delta S}*{\text{Directional (hedge mismatch) term}}
+\underbrace{q_p,S*{t+\Delta},\Delta b}_{\text{Pure basis term}}.
]

Define the **directional mismatch exposure** in base units:
[
m_t := q_s(t) + q_p(t)(1+b_t) = q_s(t)+q_p(t)\frac{P_t}{S_t}.
]
Then:
[
q_s\Delta S + q_p\Delta P = m_t,\Delta S + q_p,S_{t+\Delta},\Delta b.
]

### Final one-step PnL decomposition

[
\boxed{
\Pi_{t\to t+\Delta}
===================

\underbrace{m_t,\Delta S}*{\text{Residual spot-directional PnL}}
+
\underbrace{q_p(t),S*{t+\Delta},\Delta b}*{\text{Basis PnL}}
+
\underbrace{\sum*{\tau\in(t,t+\Delta]}\left(-q_p(\tau^-),P_\tau,r_\tau\right)}_{\text{Funding PnL}}
---------------------------------------------------------------------------------------------------

## \underbrace{C_t}_{\text{Trading costs}}

\underbrace{\text{Fin}*t}*{\text{Spot financing}}
}
]

### Perfect hedge simplification and remaining risks

If the hedge is perfect at the start of the interval:
[
q_s(t)S_t + q_p(t)P_t=0
\quad\Longleftrightarrow\quad
m_t=0,
]
then:
[
\boxed{
\Pi_{t\to t+\Delta}
===================

q_p(t),S_{t+\Delta},\Delta b
+
\sum_{\tau\in(t,t+\Delta]}\left(-q_p(\tau^-),P_\tau,r_\tau\right)

* C_t - \text{Fin}_t
  }
  ]

If additionally basis is (approximately) constant over the step ((\Delta b\approx 0)), then:
[
\Pi_{t\to t+\Delta}\approx
\sum_{\tau\in(t,t+\Delta]}\left(-q_p(\tau^-),P_\tau,r_\tau\right)

* C_t - \text{Fin}_t.
  ]

**What risks remain even under perfect hedge:**

* **Basis risk:** (\Delta b) can move sharply (perp dislocation), producing MTM losses/gains (q_p S \Delta b).
* **Funding regime risk:** (r_\tau) can flip sign or spike.
* **Costs/turnover:** (C_t) can dominate small edges; adverse selection in fast markets.
* **Liquidation/margin risk:** if the perp margin wallet cannot absorb mark moves before the hedge offsets are realizable/credited.

---

## 3) The edge hypothesis and decision rule

### What is being forecasted / estimated

Fix a holding horizon (h) (multiple of (\Delta)) and let (\mathcal{T}(t,t+h]) be the set of funding timestamps in ((t,t+h]). Define:

* Cumulative funding over (h):
  [
  R_{t,h} := \sum_{\tau\in\mathcal{T}(t,t+h]} r_\tau.
  ]

* Basis change over (h):
  [
  \Delta b_{t,h} := b_{t+h}-b_t.
  ]

Under a perfectly hedged construction with gross notional (N_t) and sign (s_t) (perp sign), the **approximate** (spot-return-neutralized) expected PnL per unit notional is:
[
\mu_{t,h}^{(per;notional)}(s_t)
\approx
s_t\left(\frac{\mathbb{E}*t[\Delta b*{t,h}]}{1+b_t} - \mathbb{E}*t[R*{t,h}]\right)
-\underbrace{\kappa_{t,h}}*{\substack{\text{expected costs per notional}\\text{(entry+rebalance+exit)}}}
-\underbrace{\phi*{t,h}}*{\text{expected financing per notional}},
]
where (\kappa*{t,h}) and (\phi_{t,h}) are modelable buffers using the cost model and expected turnover (Section 5).

Define the **core edge statistic**:
[
X_{t,h} := \frac{\widehat{\Delta b}*{t,h}}{1+b_t} - \widehat{R}*{t,h}.
]

Then the direction that maximizes expected gross edge is:
[
s_t^\star = \mathrm{sign}(X_{t,h}),
\qquad \text{expected gross edge magnitude } = |X_{t,h}|.
]

Interpretation:

* If (X_{t,h}<0): prefer (s_t=-1) (**short perp / long spot**) to benefit from (i) positive funding and/or (ii) basis mean reversion downward.
* If (X_{t,h}>0): prefer (s_t=+1) (**long perp / short spot**) to benefit from negative funding and/or basis mean reversion upward.

### Estimator A: robust non-ML baseline (OU/ARX + EMA)

A strong baseline is to model basis as a mean-reverting process with a funding-dependent equilibrium:

1. **Basis OU with time-varying mean**:
   [
   b_{t+\Delta}-b_t = -\kappa_b\left(b_t-\theta_t\right)\Delta + \sigma_b\sqrt{\Delta},\varepsilon_t,
   \qquad \varepsilon_t\sim \mathcal{N}(0,1).
   ]

2. **Equilibrium basis (\theta_t) from funding (ARX)**:
   [
   \theta_t = \beta_0 + \beta_1 \widehat{r}_t,
   \qquad
   \widehat{r}*t = (1-\alpha)\widehat{r}*{t-\Delta} + \alpha r_t
   ]
   (EMA on realized funding (r_t) or on the exchange’s announced “next funding” if available).

Then the baseline forecast for basis change over (h) is:
[
\widehat{\Delta b}_{t,h}
========================

# \mathbb{E}*t[b*{t+h}-b_t]

\left(\theta_t-b_t\right)\left(1-e^{-\kappa_b h}\right).
]

For funding, an EMA or AR(1) on funding itself (per funding interval) yields:
[
r_{\tau_{k+1}} = a_r + \varphi_r r_{\tau_k} + \eta_k,
\quad
\Rightarrow\quad
\widehat{R}*{t,h} = \sum*{\tau\in\mathcal{T}(t,t+h]} \widehat{r}_\tau.
]

This produces (X_{t,h}) directly.

### Estimator B: ML option (ridge or GBDT) with targets and features

Two practical ML formulations:

**(B1) Direct net-edge regression**
Target:
[
y_t := \frac{b_{t+h}-b_t}{1+b_t} - \sum_{\tau\in\mathcal{T}(t,t+h]} r_\tau.
]
Predict (\widehat{y}*t\approx X*{t,h}) using ridge regression or GBDT.

**(B2) Two-head forecasting (funding and basis separately)**
Targets:
[
y_t^{(b)} := b_{t+h}-b_t,
\qquad
y_t^{(r)} := \sum_{\tau\in\mathcal{T}(t,t+h]} r_\tau,
\qquad
X_{t,h} = \frac{\widehat{y}_t^{(b)}}{1+b_t}-\widehat{y}_t^{(r)}.
]

Feature set (all observable at decision time (t), with lags):

* Basis features: (b_t), (\Delta b_t), rolling z-score (z_b(t)=\frac{b_t-\bar b}{\hat\sigma_b}), basis term structure if multiple perps (optional).
* Funding features: last (k) realized fundings ({r_{\tau_{k-i}}}), EMA funding (\widehat r_t), change (\Delta r), sign persistence.
* Microstructure/liquidity: spot/perp volume, bid-ask spreads, depth proxy, order book imbalance.
* Derivatives positioning: open interest, OI change, liquidation volume proxy (if available), perp volume/spot volume ratio.
* Volatility regime: realized vol (\sigma_t), intraday range, jump indicators.
* Momentum/flow: short-term returns, funding-basis interaction terms (b_t r_t).

### Entry/exit rule as explicit inequality

Let:

* (\widehat{X}_{t,h}) be the predicted core edge.
* (\widehat{\kappa}_{t,h}) be predicted all-in cost per notional over (h) (entry + expected rebal + exit).
* (\widehat{\phi}_{t,h}) be predicted financing drag per notional over (h).
* (\widehat{\sigma}_{\text{edge},t,h}) be forecasted standard deviation of hedged MTM PnL per notional over (h) (from basis/mismatch; Section 4).
* Choose a risk buffer multiple (z_{\text{risk}}>0).

Define expected **net** edge magnitude:
[
\widehat{E}*{t,h} := |\widehat{X}*{t,h}| - \widehat{\kappa}*{t,h} - \widehat{\phi}*{t,h}.
]

**Decision inequality (trade if and only if):**
[
\boxed{
\widehat{E}*{t,h} ;>; z*{\text{risk}};\widehat{\sigma}*{\text{edge},t,h}
}
]
and set direction:
[
\boxed{
s_t = \mathrm{sign}(\widehat{X}*{t,h}).
}
]

**Exit rule:** close (or reduce to zero) when the inequality fails or when (s_t) flips and the flipped edge exceeds the same threshold after costs (hysteresis to avoid churn).

---

## 4) Risk control math (liquidation/margin + sizing)

### Margin/liquidation approximation for the perp leg

Let (N_t := |q_p(t)|P_t) be perp notional. Let:

* Initial margin rate: (\mathrm{IMR}) (so max leverage (\approx 1/\mathrm{IMR})).
* Maintenance margin rate: (\mathrm{MMR}).
* Perp margin equity available at time (t): (E_t) (in quote currency).

Define **effective leverage**:
[
L_t := \frac{N_t}{E_t}.
]

For a linear (quote-margined) perp, if the mark price moves by simple return (R_{t,h}:=\frac{P_{t+h}-P_t}{P_t}), then directional MTM PnL is approximately (\pm N_t R_{t,h}) (sign depends on long/short).

A standard liquidation barrier approximation (one-sided adverse move) is:
[
\text{liquidation if}\quad E_t + \mathrm{PNL}*{t,h} \le \mathrm{MMR},N_t.
]
If directional risk is dominant and hedges do **not** contribute to perp equity, worst-case adverse PnL over horizon is (\approx -N_t |R*{t,h}|). Then liquidation occurs if:
[
E_t - N_t |R_{t,h}| \le \mathrm{MMR} N_t
\quad\Longleftrightarrow\quad
|R_{t,h}| \ge d_t,
]
where the **distance to liquidation in return space** is:
[
d_t := \frac{E_t}{N_t} - \mathrm{MMR} = \frac{1}{L_t} - \mathrm{MMR}.
]

> If spot collateral is cross-margined with the perp, replace (E_t) with an “effective equity” (E_t^{\mathrm{eff}} = E_t^{cash} + \kappa,q_s S_t) with haircut (\kappa\in[0,1]); the same algebra holds with (E_t^{\mathrm{eff}}).

### Volatility scaling

Let (\sigma_t) be an estimate of spot (or perp) return volatility per unit time (in the same time units as (h)). Under a diffusive approximation:
[
\sigma_{t,h} \approx \sigma_t\sqrt{h}.
]

For basis risk, let (\sigma_{b,t}) be a volatility estimate for (\Delta b) per (\sqrt{\text{time}}), so:
[
\sigma_{b,t,h} \approx \sigma_{b,t}\sqrt{h}.
]

### Liquidation probability constraint (usable closed form)

Assume (R_{t,h}\sim \mathcal{N}(0,\sigma_{t,h}^2)) (or use a heavier-tail quantile; the form is the same with a different (z)). For one-sided liquidation (adverse direction), enforce:
[
\mathbb{P}(|R_{t,h}|\ge d_t)\le \alpha
\quad\Rightarrow\quad
d_t \ge z_{1-\alpha},\sigma_{t,h}.
]
Thus:
[
\frac{1}{L_t}-\mathrm{MMR}\ \ge\ z_{1-\alpha}\sigma_{t,h}
\quad\Longleftrightarrow\quad
\boxed{
L_t\ \le\ \frac{1}{\mathrm{MMR}+z_{1-\alpha}\sigma_{t,h}}
}
]
and equivalently:
[
\boxed{
N_t \le \frac{E_t}{\mathrm{MMR}+z_{1-\alpha}\sigma_{t,h}}.
}
]

### Basis-driven equity risk in a perfectly hedged (cross-margined) setup

Under perfect notional hedge, the step MTM is dominated by basis change:
[
\Delta E \approx q_p S,\Delta b
\approx s_t\frac{N_t}{1+b_t},\Delta b.
]
Adverse basis move magnitude (|\Delta b|) causes loss (\approx \frac{N_t}{1+b_t}|\Delta b|). Impose:
[
\mathbb{P}!\left(\frac{N_t}{1+b_t}|\Delta b_{t,h}|\ge E_t-\mathrm{MMR}N_t\right)\le \alpha.
]
With (\Delta b_{t,h}\sim \mathcal{N}(0,\sigma_{b,t,h}^2)), a sufficient condition is:
[
\frac{E_t-\mathrm{MMR}N_t}{N_t} \ge \frac{z_{1-\alpha}\sigma_{b,t,h}}{1+b_t}
]
which yields:
[
\boxed{
N_t \le \frac{E_t}{\mathrm{MMR}+\frac{z_{1-\alpha}\sigma_{b,t,h}}{1+b_t}}
}.
]
In practice, use the **minimum** of directional and basis-based constraints depending on whether spot collateral offsets perp MTM.

### Hedged PnL risk model and sizing objective

Define a per-notional hedged MTM random variable over horizon (h):
[
\mathrm{MTM}*{t,h}^{(per;notional)}
\approx
\underbrace{\delta_t R*{t,h}}*{\text{residual delta mismatch}}
+
\underbrace{\frac{\Delta b*{t,h}}{1+b_t}}_{\text{basis risk}},
]
where (\delta_t := \frac{\varepsilon_t}{N_t}) is the mismatch ratio (targeted to be small via rebalancing).

Assume the vector ((R_{t,h},\Delta b_{t,h})) has covariance matrix (\Sigma_{t,h}). Then:
[
\widehat{\sigma}_{\text{edge},t,h}^2
====================================

# \mathrm{Var}!\left(\delta_t R_{t,h} + \frac{\Delta b_{t,h}}{1+b_t}\right)

w_t^\top \Sigma_{t,h} w_t,
\quad
w_t :=
\begin{pmatrix}
\delta_t[3pt]
\frac{1}{1+b_t}
\end{pmatrix}.
]

Let (\widehat{E}*{t,h}) be expected net edge per notional (Section 3). A mean-variance sizing objective is:
[
\max*{N_t\ge 0}\ \ \widehat{E}*{t,h},N_t ;-;\frac{\lambda}{2},\widehat{\sigma}*{\text{edge},t,h}^2,N_t^2
]
subject to constraints:
[
\begin{aligned}
&N_t \le N^{(\mathrm{lev})}_t &&\text{(liquidation/leverage bound above)}\
&N_t \le N^{(\mathrm{cap})}_t &&\text{(exchange leverage / risk limits)}\
&N_t \le N^{(\mathrm{liq})}_t &&\text{(liquidity/impact cap; e.g. fraction of ADV and depth)}\
&N_t \le N^{(\mathrm{dd})}_t &&\text{(drawdown/margin utilization cap)}.
\end{aligned}
]

**Closed-form unconstrained optimum** (if (\widehat{E}*{t,h}>0)):
[
N_t^{\star} = \frac{\widehat{E}*{t,h}}{\lambda,\widehat{\sigma}_{\text{edge},t,h}^2}.
]
Final sizing:
[
\boxed{
N_t = \min\Big{N_t^\star,\ N^{(\mathrm{lev})}_t,\ N^{(\mathrm{cap})}*t,\ N^{(\mathrm{liq})}*t,\ N^{(\mathrm{dd})}*t\Big},
\qquad
\text{trade only if }\widehat{E}*{t,h} > z*{\text{risk}}\widehat{\sigma}*{\text{edge},t,h}.
}
]

---

## 5) Rebalancing and turnover control

### Hedge drift and rebalance trigger

Define the hedge mismatch ratio at time (t):
[
\delta_t = \frac{q_s(t)S_t + q_p(t)P_t}{N_t}.
]

**Rebalance band rule:** choose a tolerance (\delta_{\max}>0). Rebalance to restore (q_s S \approx -q_p P) when:
[
\boxed{
|\delta_t| > \delta_{\max}.
}
]

Rebalancing targets (given current desired notional (N_t) and sign (s_t)):
[
q_p^{\text{new}}(t)= s_t\frac{N_t}{P_t},
\qquad
q_s^{\text{new}}(t)= -s_t\frac{N_t}{S_t}.
]

### Turnover enters costs

Define notional turnover in the rebalance:
[
\mathrm{TO}_t := |\Delta q_s(t)|S_t + |\Delta q_p(t)|P_t.
]
Then cost is:
[
C_t = c_s(t),|\Delta q_s(t)|S_t + c_p(t),|\Delta q_p(t)|P_t
\le \max{c_s(t),c_p(t)},\mathrm{TO}_t.
]

Under a band policy, smaller (\delta_{\max}) reduces mismatch risk but increases expected turnover, raising (\widehat{\kappa}_{t,h}) and potentially eliminating the edge.

### Choosing (\delta_{\max}) (cost–risk tradeoff)

A usable analytic heuristic comes from balancing:

* (i) expected **rebalance cost rate** (\propto \delta_{\max}^{-2}), and
* (ii) expected **residual directional risk** (\propto \delta_{\max}^2).

Model (\delta_t) as a diffusion driven mainly by basis moves (and microstructure), with instantaneous variance rate (\nu_\delta) (estimated empirically). Then the expected time to hit (\pm\delta_{\max}) scales like (\mathbb{E}[\tau_{\text{hit}}]\propto \delta_{\max}^2/\nu_\delta), so rebalance frequency (\propto \nu_\delta/\delta_{\max}^2).

Let per-rebalance proportional cost be approximately (k N_t) (where (k) aggregates (c_s,c_p) and typical adjustment fraction). Then expected cost rate:
[
\text{CostRate}(\delta_{\max}) \approx kN_t \frac{\nu_\delta}{\delta_{\max}^2}.
]

Residual mismatch (|\delta_t|\le \delta_{\max}) induces directional variance (\approx (\delta_{\max}N_t)^2\sigma_t^2) per unit time, so a risk penalty rate (\approx \lambda (\delta_{\max}N_t)^2\sigma_t^2).

Minimize:
[
kN_t \frac{\nu_\delta}{\delta_{\max}^2} + \lambda (\delta_{\max}N_t)^2\sigma_t^2
]
yields:
[
\boxed{
\delta_{\max}^{\star} \propto \left(\frac{k\nu_\delta}{\lambda N_t\sigma_t^2}\right)^{1/4}.
}
]
Operationally: increase (\delta_{\max}) when costs/spreads widen; decrease (\delta_{\max}) when volatility rises or when you run larger (N_t).

---

## 6) Evaluation protocol (fully specified, no code)

### Walk-forward design to avoid leakage

Use a walk-forward procedure with strict information timing:

1. Choose horizons:

* Decision grid (\Delta) (e.g., 1–15 min bars).
* Holding horizon (h) (e.g., 1–3 funding intervals or 1 day; symbolic in research).

2. Splits:

* Train window length (T_{\text{train}}).
* Validation window (T_{\text{val}}).
* Test window (T_{\text{test}}).
  Use either:
* **Rolling window:** train on ([t-T_{\text{train}},t)), validate on ([t,t+T_{\text{val}})), test next; then roll forward.
* **Expanding window:** train from start to (t), validate then roll.

3. No leakage rules:

* Only use features observable at time (t).
* Funding rates (r_{\tau}) are only usable after they are published/known; if using “next funding prediction” from the venue, treat it as a feature with timestamp when published.
* Execution prices should be simulated using realistic fills (mid ± spread, or mark-to-trade mapping), and costs (c_s(t),c_p(t)) must be applied at each trade.

### Metrics (include decomposition)

Compute:

* Net return series (after all costs and financing).
* Annualized Sharpe (and/or per-day Sharpe) of net returns.
* Sortino, max drawdown, Calmar.
* Tail risk: 95%/99% VaR and CVaR of daily PnL.
* Turnover: (\sum_t \mathrm{TO}_t) and turnover-to-AUM ratio.
* Margin utilization statistics: (L_t), proximity to liquidation (d_t), and frequency of near-liquidation events.

**PnL attribution using the exact decomposition:**
For each step, store:
[
\Pi^{(\text{fund})}*{t} = \sum*{\tau\in(t,t+\Delta]}(-q_pP_\tau r_\tau),
\quad
\Pi^{(\text{basis})}*{t} = q_p S*{t+\Delta}\Delta b,
\quad
\Pi^{(\text{dir})}_{t} = m_t\Delta S,
\quad
\Pi^{(\text{cost})}_t = -C_t,
\quad
\Pi^{(\text{fin})}_t = -\text{Fin}_t.
]
Then verify:
[
\Pi_t = \Pi^{(\text{fund})}_t+\Pi^{(\text{basis})}_t+\Pi^{(\text{dir})}_t+\Pi^{(\text{cost})}_t+\Pi^{(\text{fin})}_t.
]

Report:

* fraction of total PnL from funding vs basis vs residual mismatch,
* regimes where basis dominates and funding fails (and vice versa).

### Stress tests (explicit)

Re-run evaluation under perturbations:

1. **Higher fees/slippage:** multiply (c_s(t),c_p(t)) by factors ({1.5,2,3}).
2. **Basis shocks:** inject jumps (b_t \mapsto b_t + J) with (J\in{\pm 1%,\pm 3%,\pm 5%}) at random times; evaluate drawdown and liquidation constraints.
3. **Funding regime flips:** flip sign of (r_\tau) in blocks (or bootstrap blocks of historically negative/positive regimes) to test robustness.
4. **Volatility spikes:** scale (\sigma_t) by ({2,3,5}) for stress periods; confirm the leverage bound reduces (N_t) accordingly.
5. **Liquidity drought:** widen spreads (increase (c_s,c_p)) and cap fill size; measure churn and missed rebalances.

---

## 7) Algorithm (implementable step-by-step specification)

### Inputs / fixed parameters

* Horizon (h), decision grid (\Delta), funding times ({\tau_k}).
* Risk parameters: (\alpha) (liquidation tail prob), (z_{\text{risk}}), (\lambda).
* Rebalance tolerance (\delta_{\max}) (or dynamic (\delta_{\max}^\star)).
* Constraints: max leverage / margin utilization; liquidity cap; max turnover.

### State at time (t)

Observe ((S_t,P_t,b_t)), recent funding history ({r_{\tau}}), spreads/liquidity proxies, vol estimates (\sigma_t,\sigma_{b,t}), current positions ((q_s(t^-),q_p(t^-))), margin equity (E_t).

### Step 1: Forecast edge over horizon (h)

Compute:

* (\widehat{\Delta b}_{t,h}) via baseline OU/ARX or ML.
* (\widehat{R}*{t,h}=\sum \widehat r*\tau) via EMA/AR(1) or ML.
* Core edge:
  [
  \widehat{X}*{t,h}=\frac{\widehat{\Delta b}*{t,h}}{1+b_t}-\widehat{R}*{t,h}.
  ]
  Set direction:
  [
  s_t=\mathrm{sign}(\widehat{X}*{t,h}).
  ]

### Step 2: Estimate costs and hedged risk

* Estimate expected total cost per notional over horizon:
  [
  \widehat{\kappa}_{t,h} = \text{(entry cost + expected rebalance cost + exit cost)}/N.
  ]
* Estimate financing drag (\widehat{\phi}_{t,h}).
* Estimate hedged MTM volatility per notional:
  [
  \widehat{\sigma}*{\text{edge},t,h} = \sqrt{w_t^\top \Sigma*{t,h} w_t},\quad w_t=(\delta_t,\frac{1}{1+b_t})^\top,
  ]
  with (\Sigma_{t,h}) built from empirical covariances of ((R_{t,h},\Delta b_{t,h})).

Compute net edge magnitude:
[
\widehat{E}*{t,h}=|\widehat{X}*{t,h}|-\widehat{\kappa}*{t,h}-\widehat{\phi}*{t,h}.
]

### Step 3: Entry/hold/exit decision

Trade only if:
[
\widehat{E}*{t,h} > z*{\text{risk}}\widehat{\sigma}_{\text{edge},t,h}.
]
Otherwise target (N_t=0) (flat) or reduce exposure.

### Step 4: Size notional (N_t) subject to liquidation and other caps

Compute liquidation/leverage bound (choose appropriate version depending on collateral netting):
[
N_t \le N^{(\mathrm{lev})}*t :=
\frac{E_t}{\mathrm{MMR}+z*{1-\alpha}\sigma_{t,h}}
\quad \text{(directional)}\qquad\text{and/or}\qquad
N_t \le
\frac{E_t}{\mathrm{MMR}+\frac{z_{1-\alpha}\sigma_{b,t,h}}{1+b_t}}
\quad \text{(basis)}.
]

Compute mean-variance optimum:
[
N_t^\star = \frac{\widehat{E}*{t,h}}{\lambda,\widehat{\sigma}*{\text{edge},t,h}^2}\ \ \ \text{if }\widehat{E}_{t,h}>0,\quad \text{else }0.
]

Set:
[
N_t = \min{N_t^\star,\ N^{(\mathrm{lev})}_t,\ N^{(\mathrm{cap})}_t,\ N^{(\mathrm{liq})}_t,\ N^{(\mathrm{dd})}_t}.
]

### Step 5: Set hedged target positions

If (N_t>0), set:
[
q_p^{\text{tar}}(t)= s_t\frac{N_t}{P_t},\qquad
q_s^{\text{tar}}(t)= -s_t\frac{N_t}{S_t}.
]
Otherwise (q_p^{\text{tar}}=q_s^{\text{tar}}=0).

### Step 6: Rebalance rule (turnover control)

Compute current mismatch:
[
\delta_t = \frac{q_s(t^-)S_t+q_p(t^-)P_t}{N_t\vee \epsilon}.
]
If (|\delta_t|>\delta_{\max}) (or if signal flips and exceeds threshold), trade toward ((q_s^{\text{tar}},q_p^{\text{tar}})). Otherwise keep positions unchanged.

### Step 7: Accounting

At each step, realize PnL using:
[
\Pi_{t\to t+\Delta}
===================

m_t\Delta S + q_p S_{t+\Delta}\Delta b + \sum_{\tau\in(t,t+\Delta]}\left(-q_pP_\tau r_\tau\right) - C_t - \text{Fin}_t,
]
with (m_t=q_s+q_p(1+b_t)). Track PnL attribution components.

---

## Data requirements (exact time series needed)

For a single asset and one exchange:

1. **Spot market**

* Spot OHLCV on grid (\Delta), or at minimum mid/mark (S_t).
* Bid/ask or spread proxy to model (c_s(t)).
* If margin shorting spot: borrow rates or funding/interest rates relevant to spot borrow/lend, to model (\rho_t^{(s)}).

2. **Perpetual futures**

* Perp mark price (P_t) (and ideally best bid/ask, last trade).
* Perp OHLCV on grid (\Delta).
* Funding history ({(\tau_k,r_{\tau_k})}) with exact timestamps.
* Any published “next funding” estimate with its publication timestamp (optional feature).
* Contract specs: multiplier (linear vs inverse), fee schedule, margin rules ((\mathrm{IMR},\mathrm{MMR})), liquidation mechanics approximations.

3. **Costs and constraints**

* Fee rates (maker/taker) and any rebates.
* Slippage/impact model inputs: spread, depth, volume, ADV.
* Exchange risk limits: max leverage, position limits, margin utilization constraints.

4. **Optional but valuable**

* Open interest and its changes.
* Liquidation volume proxy (if available).
* Order book imbalance and depth snapshots (to improve cost/turnover estimates and ML features).

This specification is self-contained: the hedge is explicit ((q_s S \approx -q_p P)), the exact discrete-time PnL is decomposed into directional mismatch, basis, funding, and costs, and the strategy’s decision and sizing are derived from an explicit predicted net edge exceeding cost and risk buffers under liquidation-safe constraints.
