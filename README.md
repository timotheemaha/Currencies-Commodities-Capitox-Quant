# Capitox Quant Macro

## Project 1: # HMM Asset Allocation Model

A concise, reproducible research repo for **macro regime detection** using a **Hidden Markov Model (HMM)** and its application to **risk-parity / All-Weather** style portfolio allocation.

The repo ships a starter notebook with engineered features (volatility, growth, inflation, rates/credit). Your job as a teammate is to **extend features**, **shape regimes**, and **evaluate portfolio value**.

---

## Goals

1. **Enhance Feature Engineering**  
   - Add higher-signal macro & market-implied series.  
   - Build a robust feature selection / dimensionality pipeline (e.g., PCA by blocks, stability tests).

2. **HMM Model Building**  
   - Explore alternative **state definitions** (not just high/low vol): growth × inflation quadrants, liquidity/crisis, policy shocks, etc.  
   - Tune number of states, covariance type, and frequency (daily vs. weekly).

3. **Evaluation & Backtest**  
   - Measure statistical fit, classification/calibration, and—most importantly—**economic value** vs. baselines.  
   - Demonstrate how regime probabilities drive **risk-parity** allocations and tilts.


## 1. Features (current & to extend)

### Volatility block
* VIX/VXN/VXD O/H/L/C, VIX3M (VXV), VVIX, term-structure (`VIX3M - VIX`), contango/backwardation, realized/impl. ratios, CESI z-scores (optional).

### Growth block
* ISM/PMI levels & surprises, Industrial Production YoY, Initial Claims (4-wk avg YoY), equity cyclicals/defensives ratios, global PMIs (EU/China) as proxies.

### Inflation block
* US Core CPI/PCE YoY (choose one as anchor), 5y/10y breakevens, WTI/BCOM YoY.

### Rates / Financial Conditions
* 3m/2y/10y UST, 10y-3m spread, 10y TIPS (real yield), HY-IG OAS (or CDX IG/HY), USD broad index momentum.

---
Keep total inputs **8-15**; consider **PCA (1-2) / Lasso Pick** per block to reduce collinearity.
---

## 2. How the HMM fits into allocation

1.  Fit K-state Gaussian HMM to the feature matrix $X_t$.
2.  Obtain filtered $P(S_t \mid x_{1:t})$ and one-step-ahead $P(S_{t+1} \mid x_{1:t})$.
3.  Compute state-conditional moments of asset returns $(\mu_k, \Sigma_k)$.
4.  Blend covariances for tomorrow:
   ```math
   \Sigma_{t+1|t} = \sum_{k=1}^{K} p_{t+1|t, k} \Sigma_k
   ```
5.  Build risk-parity core (min-var or ERC) from $\Sigma_{t+1|t}$.
6.  Apply **state tilts** (e.g., +equity in ↑Growth/↓Inflation, +duration/+gold in ↓Growth/↑Inflation).
7.  Vol target, bounds, and **turnover/TC** penalties.

## 3. Modeling: what to explore

* **K (states):** try 3-6; pick via *OOS log-likelihood* + portfolio utility + interpretability (state durations should be weeks-months).
* **Covariance type:** `full` vs `diag`; try Student-t mixtures if tails matter.
* **Frequency:** weekly often yields cleaner regimes with macro.
* **Sticky transitions:** initialize high diagonals for realistic persistence.
* **Labeling:** name states by conditional asset behavior (equity up/down, duration up/down) to avoid label switching.

## 4. Evaluation framework

### 1) Statistical fit / parsimony
* OOS predictive log-likelihood per step
* BIC/AIC on train windows
* Expected state duration $d_k=1/(1-A_{kk})$
* Emission checks (Gaussianity; switch to `diag` / t-mixture if needed)

### 2) Classification / calibration
* One-step regime accuracy vs. ex-post Viterbi (proxy)
* Brier score & log loss for events ("Risk-Off tomorrow?")
* Reliability curve (probability calibration)

### 3) Economic value (primary)
* Portfolio OOS: Sharpe, Sortino, max DD, turnover, capacity proxies
* Decision hit-rate: if $p_{t|\text{off}} > \tau$ choose Risk-Off sleeve (e.g., TLT+Gold) vs. Risk-On sleeve (SPY+HYG); a **win** = chosen sleeve outperforms next day/week
* Incremental contribution vs. static RP and naive vol-target

---
Also test **Markov(1) adequacy** by comparing to a 2nd-order HMM (or duration models / HSMM). If OOS gains are negligible, first-order is fine.
---

## 5. Deliverables by workstream

### A) Feature Engineering
* PR with new data loaders + documentation & provenance
* Rolling z-score transforms and **PCA-by-block** option
* Stability analysis: feature importance under resampling; VIF/correlation heatmaps

### B) HMM & States
* Walk-forward training utility (expanding or rolling)
* Model selection (grid over K, cov types); report durations, entropies
* Clear state naming and summary plots (feature means, asset conditional returns)

### C) Evaluation & Portfolio
* Backtest notebook: RP core + regime tilts + vol targeting
* Metrics panel and attribution vs. baseline
* Sensitivity to thresholds (e.g., $\tau$ for Risk-Off), transaction costs, rebalance frequency


## Project 2: Commodity FX Basket

A focused research repo around **commodity-linked currencies** and cross-asset features for building a robust **FX basket strategy**. 

This project complements our macro-HMM work and aims to (i) **expand features**, (ii) **upgrade portfolio construction**, and (iii) **diagnose & fix the 2019–2022 underperformance**.


---

## Goals

1) **Enhance Feature Engineering**
- Add advanced data (rates, commodities, positioning, macro surprises).
- Build a **feature picking** pipeline (regularization + stability).
- Broaden the **commodity-currency universe**.

2) **Strategy / Portfolio Optimization**
- Move beyond equal notional / simple vol-targeting.
- Explore **risk budgeting, robust optimization, Black–Litterman, HRP**, and **TC-aware** rebalancing.

3) **2019–2022 Post-Mortem**
- Understand macro regimes & events behind the drawdown.
- Design **pairing / hedging / long-vol overlays** and regime gating.


## 1. Universe (expandable)

### Developed "commodity" FX (base in USD unless noted)
* AUDUSD, NZDUSD, USDCAD, USDNOK, USDSEK* (semi-commodity), NOKSEK (hedges USD factor)
* Crosses to reduce USD beta: **AUDCAD, AUDNZD, CADNOK**

### Liquid EM "commodity" FX (use with liquidity/risk limits)
* USDMXN, USDZAR, USDBRL, USDCLP, USDIDR

> Always apply per-pair TC, slippage & carry/roll conventions.

---

## 2. Features (additive, pick a parsimonious set)

### 1) FX Core Factors
* **Carry:** short-term rate differential (OIS/3M) or forward points.
* **Momentum/Trend:** $r_{t,63}$, $r_{t,252}$; average of multiple lookbacks.
* **Value:** PPP or **REER** deviation (normalize by z-score).
* **Seasoned Vol:** realized vol (20/63d) & skew for convexity targeting.

### 2) Commodity Linkages
Map currencies to relevant commodities and build **beta/ratio** features:

| Currency | Primary links |
| :--- | :--- |
| CAD | WTI/Brent, Gas |
| NOK | Brent, European gas |
| AUD | Iron ore, Coal, Copper, Gold |
| NZD | Dairy index, Softs |
| MXN | Oil, US cycle (autos/maquiladoras) |
| ZAR | Gold, Platinum, Palladium |
| BRL | Soy, Sugar, Iron ore, Oil |
| CLP | Copper |

**Ideas:**
* **Beta-to-commodity** via rolling regression of FX returns on commodity returns.
* **Commodity momentum** (63/252d) & **term structure** (front-deferred) as features.
* **Terms-of-trade proxy:** export-weighted commodity basket YoY.

### 3) Macro / Rates / Risk
* **Short-rate & slope** differentials (OIS 1m-1y, 2s10s) vs USD.
* **Inflation/breakevens** differentials (where available).
* **PMI surprises**, industrial production YoY (US vs local).
* **Risk appetite:** VIX, MOVE, HY-IG OAS, USD broad index momentum.
* **Positioning:** CFTC futures net specs (scaled), ETF flows (if reliable).

### 4) Transform & Sanity
* Rolling z-scores, winsorize 1-99% (per pair).
* Cross-sectional standardization for signals meant to rank within the basket.
* Orthogonalize overlapping features (e.g., **PCA by blocks**).

---

## 3. Feature Selection Pipeline (menu)

* **Filter layer:** data availability, stability, VIF/correlation threshold (e.g., >0.85 drop one).
* **Wrapper/embedded:**
    * **Lasso / Elastic-Net** with **purged, embargoed CV** (time-series split).
    * **Tree models (XGB/LightGBM)** → SHAP importance; **Boruta-SHAP** for stability.
    * **Mutual information & distance correlation** for non-linear links.
* **Stability selection:** bootstrap subsamples; keep features picked $\ge$60%.
* **Dimensionality:** PCA(1-2) per block (FX core / commodities / macro).

---

## 4. Signals & Labeling

* Combine standardized features into sleeves: **Carry, Trend, Commodity, Macro**.
* **Per sleeve:**
    * Score = weighted sum of block PCs or top features.
* **Meta-labeling:** learn when a sleeve is likely to work (logit on market/risk vars).
* **Final signal:** weighted ensemble of sleeves with **rolling weights** (expanding regression of next-period returns on sleeve scores, ridge-regularized).

---

## 5. Portfolio Construction (beyond equal weight)

### Risk budgeting
* **ERC (Equal Risk Contribution)** with constraints (gross/net, per-pair caps).
* **Hierarchical Risk Parity (HRP)** for correlation-aware allocation.
* **Block budgets** (e.g., 40% trend, 30% carry, 30% commodity).

### Mean-variance & robust
* **Shrinkage** (Ledoit-Wolf, OAS) + **box constraints**.
* **Min-CVaR / min-Expected Shortfall** (CVaR optimization via `cvxpy`).
* **Black-Litterman:** convert signals into Bayesian "views" with uncertainty.

### Costs & turnover
* Objective: maximize $E[R] - \lambda \cdot \text{risk} - \kappa \cdot \text{TC} - \eta \cdot \Delta w^2$
* Use **L1/L2 penalties** for turnover, model $TC = a \cdot |\Delta \text{notional}| + b \cdot |\Delta \text{notional}| \cdot \sigma_{mkt}$ per pair.
* **Rebalance frequency** search (daily/weekly) with drift bands.

### Leverage & controls
* **Vol targeting** to $\sigma^*$ (e.g., 10% ann) using $\hat{\sigma}_{t}$ from HAR/RV.
* **Drawdown stop** & regime kill-switch (e.g., HMM "risk-off" probability > 0.6).

### Example sizing (sketch)
```math
w_i \propto \frac{\text{signal}_i}{\hat{\sigma}_i}, \quad \text{then solve ERC/HRP with} \sum |w_i| \le G, |w_i| \le c
```
---

## 6. 2019–2022 Underperformance: What to Test

### Macro timeline (high-level)
* **2019:** Fed pivot; trade-war & late-cycle slowdown; compressed carry.
* **2020:** COVID shock → FX whipsaw, USD squeeze, unprecedented policy.
* **2021:** Re-opening reflation; China slowdown; commodity rally, uneven FX response.
* **2022:** Energy shock & **rapid Fed hiking** → broad USD bull market; EM stress.

### Hypotheses
* USD factor dominated; basket insufficiently USD-neutral.
* **Trend regime shifts** → momentum whipsaws; vol-target oversold bottoms.
* Commodity beta mis-specified (e.g., NOK diverging from Brent during Europe risk).
* Carry sleeve damaged by zero-rate era then abrupt hikes (timing mismatch).

### Diagnostics
* Break PnL by **currency, sleeve, and regime** (risk-on/off; commodity up/down).
* **Attribution:** USD beta vs idiosyncratic; rolling regression of basket on DXY & commodities.
* **Event studies:** COVID crash, vaccine week, Russia-Ukraine invasion, key Fed days.
* **Turnover/TC impact** vs gross alpha.
* **Horizon mismatch:** test weekly vs daily signals; reduce noise.

---
