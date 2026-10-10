# Scientific Research Gap Assessment: Retail Demand Forecasting vs. Inventory Replenishment Decision Support

**Author:** Literature Research Working Group (Milestone M1)  
**Date:** October 2026  
**Document Purpose:** Formal identification and mathematical specification of the scientific research gaps addressed by the empirical study.

---

## 1. Executive Summary

Despite significant advances in supervised machine learning (ML) and gradient boosted decision tree (GBDT) architectures for time-series forecasting, modern retail supply chains continue to experience severe operational failures, characterized by simultaneous stockouts and bloated warehouse holding costs. 

A thorough systematic review of the literature (*International Journal of Forecasting*, *European Journal of Operational Research*, *Management Science*, *Interfaces*, *NeurIPS*, *KDD*) reveals a fundamental root cause: **the forecasting and inventory control disciplines operate within disconnected silos.** Machine learning benchmarks evaluate models strictly by surrogate statistical loss functions (MAE, RMSE, WAPE), under the unverified assumption that superior statistical accuracy automatically translates into superior inventory performance. Conversely, classical inventory theory formulates replenishment policies assuming known parametric demand distributions, ignoring the high-dimensional non-linear interactions captured by modern ML ensembles.

This assessment identifies and formalizes **four critical scientific research gaps** at the intersection of retail demand forecasting and stochastic inventory replenishment.

---

## 2. Formal Specification of Research Gaps

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        FOUR SCIENTIFIC RESEARCH GAPS ADDRESSED                         │
├──────────────────────────┬──────────────────────────┬──────────────────────────────────┤
│ Research Gap             │ Current Literature State │ Our Empirical Contribution       │
├──────────────────────────┼──────────────────────────┼──────────────────────────────────┤
│ Gap 1: Downstream Cost   │ Terminates at MAE/RMSE   │ Full closed-loop discrete-time   │
│ Replenishment Omission   │ without operational KPIs │ inventory cost accounting        │
├──────────────────────────┼──────────────────────────┼──────────────────────────────────┤
│ Gap 2: Fallacy of Static │ Static newsvendor or     │ Chronological event simulation   │
│ Inventory Snapshots      │ single-period safety stock│ with pipeline transit delay      │
├──────────────────────────┼──────────────────────────┼──────────────────────────────────┤
│ Gap 3: Unconstrained ML  │ Selects lowest MAE/RMSE  │ Tri-criterion operational filter │
│ Accuracy vs Bias Hazard  │ despite negative bias    │ with |Bias%| <= 5% safeguard     │
├──────────────────────────┼──────────────────────────┼──────────────────────────────────┤
│ Gap 4: Cross-Regime      │ Isolated to single data  │ Unified cross-regime benchmark   │
│ Transformation Distortion│ regime; ignores Jensen's │ (Smooth, Lumpy, Intermittent)    │
└──────────────────────────┴──────────────────────────┴──────────────────────────────────┘
```

---

### Gap 1: Downstream Replenishment Omission in Empirical Machine Learning Benchmarks

#### Current State of the Literature
The overwhelming majority of modern machine learning forecasting research terminates evaluation at point or probabilistic accuracy metrics. In premier competitions and benchmark studies (e.g., Makridakis et al., 2022 [M5]; Lim et al., 2021 [TFT]; Salinas et al., 2020 [DeepAR]), models are ranked strictly by:
- Weighted Root Mean Squared Scaled Error (WRMSSE)
- Mean Absolute Scaled Error (MASE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- Quantile / Pinball Loss ($QL_q$)

These studies conclude that the model achieving the lowest validation error is the "optimal" forecasting system for enterprise deployment. However, as noted in theoretical critiques by Syntetos, Boylan, and Disney (2009) and Syntetos et al. (2016), **statistical loss functions do not correspond to the economic objective function of a supply chain**.

#### The Scientific Defect
In enterprise distribution networks, customer orders are filled from inventory buffers. The cost structure governing these buffers is highly asymmetric ($p_{shortage} \gg h_{period}$) and subject to batching constraints ($Q = EOQ$) and non-linear reorder thresholds ($ROP$). A model that achieves a 5% reduction in MAE by shaving off peak demand forecasts will score higher on Kaggle or in an academic ML paper, but when deployed downstream in an ERP system, that exact same model causes frequent inventory exhaustion, driving stockout penalties that dwarf any theoretical accuracy gains.

#### What Is Missing
An empirical evaluation framework that takes candidate forecasting models across an entire ML pool (Baselines, Random Forests, LightGBM, XGBoost) and translates their out-of-sample predictions directly into physical inventory replenishment decisions, measuring real business metrics:
- Cycle Service Level ($\alpha$) and Customer Fill Rate ($\beta$)
- Total Stockout Days and Units Lost
- Cumulative Holding Cost ($C_{holding}$), Ordering Setup Cost ($C_{ordering}$), and Shortage Penalties ($C_{shortage}$)
- Total Replenishment Cost ($C_{total}$)

---

### Gap 2: The Fallacy of Static Inventory Snapshots (Need for Dynamic Pipeline Simulation)

#### Current State of the Literature
Where operations research literature does attempt to evaluate forecasting models within an inventory context (e.g., Babai et al., 2010; Kourentzes, 2013; Petropoulos & Kourentzes, 2015), the inventory models evaluated are almost exclusively **static single-period models** (the classical Newsvendor problem) or **myopic order-up-to periodic-review models without physical pipeline delays**.

#### The Scientific Defect
Real-world enterprise supply chains operate under continuous-review $(s, Q)$ policies with **multi-period replenishment lead times ($L > 0$)**. Orders placed at simulation time step $t$ do not arrive until step $t + L$. 

Static and single-period models commit the **fallacy of static inventory snapshots**:
1. **Omission of In-Transit Pipeline Stock:** In reality, the decision to place a purchase order depends on the *Inventory Position* ($IP_t = I_t^{on\_hand} + I_t^{on\_order}$). If outstanding orders in transit are omitted, the inventory policy places redundant, duplicate orders, creating massive inventory whipsawing and the Bullwhip Effect.
2. **Ignoring Multi-Period Error Compounding:** Forecast errors do not occur in isolated single periods; they accumulate over the replenishment lead time $L$. When demand exhibits serial autocorrelation or systematic directional drift, residual errors compound across the $L$ periods, driving stock levels far below theoretical expectations (Snyder et al., 2004; Eppen & Martin, 1988).
3. **Absence of Multi-Batch Ordering Logic:** Under intermittent and lumpy demand surges, a single order of size $Q$ is frequently insufficient to restore the inventory position above $ROP$. Without integer multi-batch replenishment logic ($\lceil \frac{ROP - IP}{Q} \rceil Q$), continuous-review policies suffer chronic repeat stockouts.

#### What Is Missing
A rigorous, chronological discrete-time event simulation that:
- Maintains distinct physical on-hand ($I^{on\_hand}$) and pipeline on-order ($I^{on\_order}$) state vectors.
- Strictly enforces non-anticipative delivery transit delays ($t + L$).
- Models lost sales ($Lost_t = \max(0, d_t - I_t)$) with full cost accounting.
- Evaluates multi-period dynamic continuous-review $(s, Q)$ performance on locked out-of-sample chronological holdouts.

---

### Gap 3: Unconstrained Statistical Accuracy Selection vs. Operational Bias Safeguards

#### Current State of the Literature
Standard machine learning pipelines universally follow the **Accuracy-First Selection Principle** (Policy B):
$$M_B^* = \arg\min_{M \in \mathcal{M}} \text{MAE}_{val}(M) \quad \text{or} \quad \arg\min_{M \in \mathcal{M}} \text{RMSE}_{val}(M)$$
The model exhibiting the lowest point error on validation folds is selected for production deployment without secondary operational constraints.

#### The Scientific Defect: The Negative Bias Starvation Hazard
Under non-linear transformations or non-Gaussian objective functions, statistical error minimization actively favors models that induce severe **negative under-forecasting bias**.

1. **Log1p Transformation Bias (Jensen's Inequality):**
   When GBDT models are trained on $z = \ln(1 + y)$ and inverted via $\hat{y} = \exp(\hat{z}) - 1$, Jensen's Inequality guarantees that:
   $$\mathbb{E}[\ln(1 + Y)] < \ln(1 + \mathbb{E}[Y]) \implies \hat{y}_{raw} = \exp(\mathbb{E}[\ln(1 + Y)]) - 1 < \mathbb{E}[Y]$$
   The model estimates the conditional geometric median rather than the mean, inducing an audited **$-43.14\%$ negative percentage bias** on high-variance wholesale demand (Online Retail). Because the log transform squashes large errors, the model achieves an attractively low validation MAE ($15.13$), deceiving Policy B into selecting it. However, in inventory simulation, under-forecasting demand by $43\%$ depresses the reorder point by nearly half, causing complete inventory starvation, chronic stockouts, and catastrophic lost-sales penalties.
2. **Tweedie Compound Poisson-Gamma Distortion:**
   On zero-inflated series (M5 Grocery, 63.28% zeros), XGBoost with Tweedie deviance ($p=1.5$) achieves the lowest statistical MAE ($0.9146$) by heavily shrinking positive predictions to minimize deviance across the majority zero observations. This optimization landscape introduces an audited **$-21.01\%$ negative percentage bias**, severely under-predicting positive grocery demand surges.

#### What Is Missing
A formalized **Tri-Criterion Model Selection Framework** that couples statistical error minimization with an explicit operational safeguard:
- **Policy C (Bias-Constrained Selection):**
  $$\mathcal{M}_{safe} = \left\{ M \in \mathcal{M} \;\middle|\; |\text{Bias\%}_{val}(M)| \le 5.0\% \right\}$$
  $$M_C^* = \arg\min_{M \in \mathcal{M}_{safe}} \text{MAE}_{val}(M)$$
  *(Fallback: if $\mathcal{M}_{safe} = \emptyset$, $M^* = \arg\min_{M \in \mathcal{M}} |\text{Bias\%}_{val}(M)|$)*
- **Policy D (Cost-Aware Selection):**
  $$M_D^* = \arg\min_{M \in \mathcal{M}} C_{total}^{val}(M)$$
  Directly evaluates total replenishment cost ($C_{holding} + C_{ordering} + C_{shortage}$) on validation folds, bypassing surrogate loss functions entirely.

No prior study has systematically benchmarked Policy B (Accuracy-First), Policy C (Bias-Constrained), and Policy D (Cost-Aware) against baseline heuristics (Policy A) across identical locked holdouts.

---

### Gap 4: Loss Function & Transformation Distortion Across Heterogeneous Retail Demand Regimes

#### Current State of the Literature
Prior empirical studies typically investigate a single demand regime in isolation:
- Macro aggregate sales (e.g., Walmart weekly sales) evaluated purely on linear/tree regression.
- Spare-parts intermittent series (e.g., RAF automotive panels) evaluated purely on Croston/SBA.
- E-commerce transaction panels evaluated purely on deep learning or gradient boosting.

Furthermore, literature often assumes that algorithmic findings in one regime generalise uniformly to others.

#### The Scientific Defect
Retail enterprises do not operate within a single demand regime; their product catalogs span multiple heterogeneous regimes simultaneously:
1. **Macro Smooth Demand (Walmart Weekly Store Sales):** High volume, continuous transactions, low variance ($\text{ADI} < 1.32, CV^2 < 0.49$).
2. **Erratic / Lumpy Wholesale Demand (Online Retail Daily SKU Panel):** Continuous or semi-continuous transactions with extreme order quantity skewness ($\text{ADI} < 1.32, CV^2 \ge 0.49$).
3. **Zero-Inflated Intermittent Food Retail (M5 Grocery Sales):** Severe intermittency with 63.28% zero transactions, bounded small positive counts ($\text{ADI} \ge 1.32, CV^2 < 0.49$).

The failure modes of machine learning models depend critically on the interaction between the **demand regime**, the **loss function**, and the **target transformation**:
- In smooth macro sales, XGBoost MSE with second-order gradients excels with negligible bias ($+0.15\%$).
- In erratic wholesale orders, Log1p transformations cause catastrophic under-replenishment ($-43.14\%$), whereas Standard MSE LightGBM achieves a 26.58% RMSE reduction with safe operational bias ($+3.92\%$).
- In zero-inflated grocery retail, Tweedie deviance introduces severe under-forecasting ($-21.01\%$), whereas Random Forest variance reduction achieves near-unbiased predictions ($-3.03\%$) and the lowest operational inventory cost.

#### What Is Missing
A unified cross-regime empirical benchmark evaluating identical supervised architectures (Random Forest, XGBoost, LightGBM) under identical locked chronological validation splits across all three representative commercial retail regimes, documenting the precise failure boundaries of log-transforms and compound loss objectives in continuous-review inventory control.

---

## 3. Comparative Gap Matrix

The following table contrasts the existing literature against our empirical study:

| Capability / Investigation Dimension | Classical Forecasting Literature (Croston, 1972; Syntetos & Boylan, 2005) | Classical Inventory Literature (Silver et al., 1998; Eppen & Martin, 1988) | Contemporary ML Benchmarks (Makridakis et al., 2022; Lim et al., 2021) | Our Decision Support Framework (Milestone M1–M4) |
|---|---|---|---|---|
| **High-Dimensional ML Ensembles** | ❌ (Univariate only) | ❌ (Univariate / Parametric) | ✅ (XGBoost, LightGBM, TFT) | ✅ (RF, XGBoost, LightGBM, Tweedie) |
| **Leak-Free Chronological Cross-Validation** | ⚠️ (Often static split) | ⚠️ (Analytical / Static) | ⚠️ (Frequent inadvertent lookahead) | ✅ (Strict expanding-window walk-forward; no bfill) |
| **Discrete-Time Dynamic Replenishment Simulation** | ❌ (No inventory simulation) | ⚠️ (Static / Single-echelon) | ❌ (Omitted entirely) | ✅ (Chronological multi-period event simulation) |
| **Pipeline In-Transit Delay ($t+L$) & Multi-Batch Ordering** | ❌ (Omitted) | ⚠️ (Theoretical convolution only) | ❌ (Omitted) | ✅ (Exact pipeline tracking; deficit multi-batching) |
| **Operational Bias Safeguard ($|\text{Bias\%}| \le 5\%$)** | ⚠️ (Analytical SBA only) | ⚠️ (Heuristic tracking signal) | ❌ (Unconstrained accuracy selection) | ✅ (Explicit Policy C constraint + Trigg's tracking signal) |
| **Direct Cost-Aware Model Selection (Policy D)** | ❌ (Optimizes MSE/RAE) | ⚠️ (Newsvendor critical fractile) | ❌ (Optimizes MAE/WRMSSE) | ✅ (Simulated total replenishment cost minimization) |
| **Unified Cross-Regime Empirical Benchmark** | ⚠️ (Spare parts only) | ⚠️ (Theoretical distributions) | ⚠️ (M5 grocery only) | ✅ (Walmart Smooth, Online Retail Lumpy, M5 Intermittent) |
| **Jensen's Inequality / Transformation Audit** | ❌ (Not studied in ML) | ❌ (Not studied in ML) | ❌ (Overlooked; reported as low MAE) | ✅ (Audited -43.14% Log1p bias and -21.01% Tweedie bias) |

---

## 4. Formal Research Questions (RQs)

Guided by the four identified research gaps, our investigation addresses four formal research questions:

- **RQ1 (Accuracy vs. Cost Disconnect):** Does selecting demand forecasting models strictly by lowest validation MAE or RMSE (Policy B) consistently yield the lowest inventory replenishment costs and highest fill rates in multi-period discrete-time simulations across distinct retail demand regimes?
- **RQ2 (Operational Bias Safeguards):** Does enforcing an explicit operational bias safeguard ($|\text{Bias\%}| \le 5.0\%$, Policy C) prevent the severe inventory starvation and stockout penalties induced by unconstrained statistical accuracy selection?
- **RQ3 (Cost-Aware Optimization):** How does direct validation-fold replenishment cost minimization (Policy D) compare against statistical accuracy selection (Policy B) and naive baselines (Policy A) in total operational cost and service level trade-offs?
- **RQ4 (Cross-Regime Robustness & Transformation Hazards):** How do non-linear target transformations (Log1p) and compound probability objectives (Tweedie deviance) alter forecasting bias and downstream continuous-review $(s, Q)$ replenishment across smooth, erratic/lumpy, and zero-inflated demand regimes?

---

## 5. Summary & Alignment with Project Milestones

The formalization of these four research gaps establishes the scientific necessity of our study:
- **Milestone M2:** Implements the zero-leakage feature pipeline (`features.py`), expanding-window walk-forward cross-validation (`validation.py`), and metric integrity engine (`metrics.py`) required to address Gap 3 and Gap 4.
- **Milestone M3:** Implements the discrete-time stochastic inventory simulation (`inventory.py`) and 4-policy comparative benchmark (`policy_comparison.py`) addressing Gap 1, Gap 2, and Gap 3 across all three retail regimes.
- **Milestone M4:** Synthesizes the empirical findings into publication-grade Markdown and LaTeX manuscripts (`paper/manuscript.md`, `paper/manuscript.tex`) and critical reviewer reports.
