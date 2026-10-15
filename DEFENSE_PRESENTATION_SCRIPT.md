# Oral Defense & Conference Presentation Script

**Project Title:** An Empirically Audited Machine Learning and Stochastic Inventory Control Framework for Enterprise Replenishment Decision Support  
**Authors:** Dola Mounika  
**Deck File:** `Research_Presentation_Deck.pptx` (16:9 Widescreen)  
**Estimated Duration:** 12–15 Minutes  

---

## Slide 1: Title & Introduction (1 Minute)
* **Visual:** Slide 1 Title Banner with Author Affiliations.
* **Speaker Script:**  
  > *"Good morning, respected members of the evaluation committee and colleagues. Today, I am presenting our empirical research study entitled: **'An Empirically Audited Machine Learning and Stochastic Inventory Control Framework for Enterprise Replenishment Decision Support'**.*  
  > *In this work, we tackle a long-standing paradox at the intersection of machine learning and supply chain operations: Why do forecasting algorithms that win public accuracy competitions frequently trigger catastrophic stockouts when deployed inside automated enterprise replenishment systems?*  
  > *Over the next twelve minutes, I will walk you through the theoretical proofs of this failure mode, our 4-tier decision support architecture, and our empirical findings across 541,000 wholesale transactions and 143 retail store-weeks."*

---

## Slide 2: The Asymmetric Supply Chain Dilemma (2 Minutes)
* **Visual:** Slide 2 Problem Statement & Penalty Asymmetry Breakdown.
* **Speaker Script:**  
  > *"To understand the core problem, consider the economic penalty structure of real-world retail supply chains. In inventory operations, penalties are inherently asymmetric. If a retailer stocks out, the damage is severe: immediate lost gross margin, contractual service penalties, and permanent customer churn. In contrast, holding surplus buffer inventory incurs only a marginal capital-carrying charge, typically around 20% annually.*  
  > *Despite this fundamental asymmetry, standard machine learning pipelines train and select models using symmetric point loss metrics—namely Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE). These metrics treat a -10 unit under-forecast (a guaranteed stockout disaster) identically to a +10 unit over-forecast (a safe temporary buffer).*  
  > *Furthermore, popular techniques in ML competitions—such as logarithmic Log1p transformations and Tweedie compound Poisson-Gamma objectives—induce deep mathematical distortions that systematically pull predictions downward, starving continuous-review replenishment buffers."*

---

## Slide 3: 4-Tier System Architecture (2 Minutes)
* **Visual:** Slide 3 Horizontal Workflow Diagram (Figure 1).
* **Speaker Script:**  
  > *"To resolve this disconnect, we engineered an end-to-end, leak-free decision support system organized into four decoupled tiers:*  
  > *1. **The Presentation & Telemetry Layer:** Provides real-time exception streaming and statistical process control via Trigg's tracking signal.*  
  > *2. **The Ingestion & Feature Engineering Engine:** Implements continuous Cartesian SKU-date panel alignment with strictly chronological forward-filled prices, eliminating the lookahead bias prevalent in earlier studies.*  
  > *3. **The Predictive Modeling Tier:** Couples supervised gradient boosting regression ensembles with an out-of-sample Platt-calibrated demand surge classifier.*  
  > *4. **The Stochastic Replenishment Decision Policy:** Houses a dynamic Continuous-Review (s, Q) control engine that sizes safety buffers directly from residual forecast error variance via Eve's Law of Total Variance."*

---

## Slide 4: Theoretical Formulations & Mathematical Proofs (2.5 Minutes)
* **Visual:** Slide 4 Native Mathematical Formulations (Theorems 1, 2, and 3).
* **Speaker Script:**  
  > *"Our study establishes three core theoretical contributions:*  
  > *First, in **Theorem 1**, we prove via Jensen's Inequality that when models optimize MSE on logarithmic targets $z = \ln(1+y)$, naive exponentiation $\hat{y} = \exp(\hat{z}) - 1$ estimates the conditional geometric median rather than the conditional mean. In zero-heavy retail panels, this mathematical deficit induces a catastrophic -55.9% to -81.3% negative bias.*  
  > *Second, in **Theorem 2**, we prove that Tweedie deviance for power parameter $p = 1.5$ exhibits an explosive zero-gradient divergence: as positive predictions approach zero, the unit deviance gradient approaches positive infinity ($\partial d/\partial \mu \to +\infty$). This creates an overpowering downward pull that forces negative bias on sporadic grocery items.*  
  > *Third, in **Theorem 3**, we apply Eve's Law of Total Variance across stochastic supplier lead times $L$ and forecast errors $e$. We prove that safety buffers must be sized strictly from residual forecast error variance ($\sigma_e = \sqrt{\text{MSE}}$) rather than total demand variance ($\sigma_d$), eliminating redundant buffer capital without sacrificing customer service availability."*

---

## Slide 5: Multi-Regime Empirical Benchmark (2 Minutes)
* **Visual:** Slide 5 Empirical Benchmark Table & Cross-Regime Holdouts.
* **Speaker Script:**  
  > *"We benchmarked our system across three contrasting commercial datasets:*  
  > *1. **Walmart Weekly Sales:** Macro-aggregate monetary demand across 45 regional stores. Here, Random Forest achieved an out-of-sample MAE of \$35,624.48 with $R^2 = 0.9905$ and an audited operational bias of -0.28%.*  
  > *2. **UCI Online Retail:** 541,909 wholesale transactions aggregated into continuous daily SKU panels. Here, Log1p XGBoost achieved the lowest point MAE of 72.2 units, but suffered a -57.18% negative bias disaster. In contrast, Random Forest achieved MAE = 90.57 units with a tight +3.14% bias, successfully passing our Tri-Criterion operational safeguard.*  
  > *3. **M5 Intermittent Grocery:** Characterized by 63.28% zero-demand days. Classical Croston SBA under-forecasted by -58.97%, while Random Forest provided a robust anti-stockout buffer."*

---

## Slide 6: Policy Benchmarking across 8,000 Simulation Steps (2.5 Minutes)
* **Visual:** Slide 6 Policy Comparison Distributions & ROC Calibration Curves (Figures 7 & 8).
* **Speaker Script:**  
  > *"To test the real-world operational impact, we executed discrete-time replenishment simulations across 8,000 demand interaction steps comparing four policies:*  
  > *• **Policy B (Statistical Min MAE):** Collapsed to an 81.40% cycle fill rate, suffering 24 stockout days and racking up \$128,540 in operating penalties due to chronic under-forecasting.*  
  > *• **Policy C (Our Proposed Bias-Constrained Framework):** By enforcing $|\text{Bias}\%| \le 5.0\%$ before ranking models, Policy C achieved a **flawless 100.0% cycle fill rate**, **0 stockout days**, a **92.9% reduction in total operating costs**, and a **49.9% reduction in safety stock capital** (\$135.7k vs. \$270.8k).*  
  > *Additionally, our Platt-calibrated LightGBM Demand Surge Classifier achieved an out-of-sample **ROC AUC of 0.9278** and **83.21% classification accuracy** with a Brier score of 0.084, enabling proactive reorder point escalation before promotional stockouts occur."*

---

## Slide 7: Real-Time Telemetry & Enterprise Recommendations (1.5 Minutes)
* **Visual:** Slide 7 Closed-Loop Telemetry Trajectory & Takeaways (Figure 9).
* **Speaker Script:**  
  > *"As shown in Figure 9, our real-time closed-loop telemetry dynamically escalates the Reorder Point band from 240 to 360 units during promotional demand surges, completely preventing inventory depletion, and smoothly contracts post-promotion to avoid dead stock.*  
  > *In conclusion, our research delivers three strategic rules for enterprise supply chains:*  
  > *1. Decouple statistical error minimization from operational evaluation.*  
  > *2. Enforce strict operational percentage bias bounds ($|\text{Bias}\%| \le 5\%$) prior to error ranking.*  
  > *3. Size stochastic safety buffers strictly from residual forecast error variance via Eve's Law.*  
  > *All code, data pipelines, and the complete 18-section research notebook are fully reproducible on Google Colab with 90 passing unit and regression tests.*  
  > *Thank you, and I look forward to your questions."*
