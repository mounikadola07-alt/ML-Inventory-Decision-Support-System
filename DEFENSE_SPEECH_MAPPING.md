# 15-Minute Thesis Defense & Conference Speech Alignment Guide

**Project Title:** An Empirically Audited Machine Learning and Stochastic Inventory Control Framework for Enterprise Replenishment Decision Support  
**Authors:** Dola Mounika  
**Department:** Computer Science and Engineering, Koneru Lakshmaiah Education Foundation  
**Target Duration:** Exactly 15 Minutes (12 Minutes Presentation + 3 Minutes Transition / Opening Q&A)

---

## Slide-by-Slide Timing & Verbal Cue Sheet

| Slide # | Slide Title | Benchmark Time | Running Elapsed | Primary Verbal Message & Focus | Anticipated Committee Question / Trap |
| :---: | :--- | :---: | :---: | :--- | :--- |
| **1** | **Title Slide & Executive Thesis** | 1.5 min | `00:00 - 01:30` | Introduce the team and state the core thesis: *Statistical accuracy does not equal operational profitability in retail inventory.* Introduce the 5-stage closed-loop framework. | *"Why is traditional forecasting research insufficient for retail supply chains?"* |
| **2** | **The Operational Dilemma: Asymmetric Loss** | 2.5 min | `01:30 - 04:00` | Explain why MAE/RMSE fail under asymmetric inventory loss ($p \gg h$). Detail the Jensen's Inequality Retransformation Deficit ($-57.18\%$ bias under naive $\log(1+y)$ exponentiation). | *"Why not just train the model directly on asymmetric pinball loss?"* |
| **3** | **System Architecture & 5-Stage Framework** | 2.0 min | `04:00 - 06:00` | Walk through Figure 1: rolling-origin lag aggregates, leakage elimination, calibrated classification, $(s, Q)$ simulation, and ERP telemetry. | *"How did you prevent temporal lookahead leakage during feature engineering?"* |
| **4** | **Formal Mathematical Foundations** | 2.5 min | `06:00 - 08:30` | Present the three core equations: Theorem 1 (Jensen dual-moment smearing), Eve's Law ($\sigma_{LTD}^2 = L\sigma_D^2 + \mu_D^2\sigma_L^2$), and continuous-review expected cost objective $C(s, Q)$. | *"How much does the $(\mu_D)^2\sigma_L^2$ term contribute to total variance in practice?"* |
| **5** | **Empirical Benchmark & Holdout Results** | 2.0 min | `08:30 - 10:30` | Highlight Table 1 results: Platt-calibrated LightGBM achieving ROC AUC = 0.9278, Accuracy = 83.21%, and Brier score = 0.084. Contrast against baseline OLS and Tweedie. | *"Did you test for overfitting on the promotional surge class?"* |
| **6** | **8,000-Step Simulation & Policy Comparison** | 2.5 min | `10:30 - 13:00` | Break down Table 2: Policy B (Min MAE) suffering 24 stockout days; Policy D (Dynamic Eve's Law) achieving 100% fill rate, 0 stockouts, 92.9% cost reduction, and 49.9% working capital liberation. | *"Is 100% service level realistic, or does it require excessive safety buffer?"* |
| **7** | **ERP Telemetry, Strategic Takeaways & Q&A** | 2.0 min | `13:00 - 15:00` | Show Figure 9 real-time telemetry; deliver 4 managerial ERP implementation rules; invite committee examination and open the floor for Q&A. | *"How easily can an enterprise ERP like SAP or Oracle implement your framework?"* |

---

## Detailed Speaking Script Benchmarks

### Slide 1: Introduction (`00:00 - 01:30`)
> *"Good morning, respected committee members and fellow researchers. Today, I am presenting our work on resolving a critical breakdown at the intersection of machine learning and supply chain operations: **Why the most statistically accurate demand forecast frequently leads to disastrous inventory replenishment decisions.** Across retail merchandising, enterprise models are traditionally selected based on symmetrical statistical errors like MAE or RMSE. We show both mathematically and empirically that this practice creates severe operational bias, and we present a closed-loop framework that unifies leakage-safe forecasting with stochastic continuous-review $(s, Q)$ control."*

### Slide 2: The Operational Dilemma (`01:30 - 04:00`)
> *"The root problem lies in cost asymmetry. In retail operations, over-forecasting by 10 units costs modest holding interest and storage fees, but under-forecasting by 10 units triggers stockouts, lost margins, penalty fees, and customer churn ($p \gg h$). Furthermore, when practitioners apply standard $\log(1+y)$ transforms to handle skewed counts, Jensen's inequality dictates that $\mathbb{E}[\exp(Z)] > \exp(\mathbb{E}[Z])$. Naively exponentiating log-predictions creates a systematic under-forecasting bias of up to $-57.18\%$. This statistical artifact leads to persistent inventory starvation."*

### Slide 3: System Architecture (`04:00 - 06:00`)
> *"To resolve this, our system introduces a 5-stage closed-loop architecture. Stage 1 executes rolling-origin temporal feature extraction strictly bounded by historical horizons. Stage 2 trains competitive tree ensembles and Tweedie count models. Stage 3 introduces a Platt-calibrated early-warning surge classifier. Stage 4 conducts event-driven $(s, Q)$ replenishment accounting under stochastic lead times, and Stage 5 provides closed-loop ERP telemetry for live inventory adjustments."*

### Slide 4: Mathematical Foundations (`06:00 - 08:30`)
> *"Our theoretical framework rests on three equations. First, our dual-moment smearing correction resolves Jensen's deficit: $\hat{Y} = \exp(\hat{\mu}_Z + \tfrac{1}{2}\hat{\sigma}_Z^2) - 1$. Second, we size safety stock using Eve's Law: $\sigma_{LTD}^2 = \mathbb{E}[L]\sigma_D^2 + (\mathbb{E}[D])^2\sigma_L^2$. Traditional formulas ignore the second term, creating a buffer deficit of over 60% during delivery delays. Third, our continuous-review cost function balances ordering costs, cycle holding, safety holding, and expected backorder penalties."*

### Slide 5: Empirical Benchmarks (`08:30 - 10:30`)
> *"We validated across two benchmark datasets: Walmart weekly sales and an Online Retail continuous panel. Our Platt-calibrated gradient booster detects tail-risk demand surges ($>95\text{th}$ percentile) with an ROC AUC of 0.9278, Accuracy of 83.21%, and a Brier score of 0.084. Crucially, Tweedie and smearing-corrected models maintained operational bias within $\pm 3.14\%$, whereas unconstrained MSE models exhibited double-digit negative bias."*

### Slide 6: Simulation Results (`10:30 - 13:00`)
> *"In our 8,000-step continuous simulation, Policy B (the min-MAE model) suffered 24 stockout days because its $-18.37\%$ bias depressed reorder points, inflating costs to \$128.5k. In contrast, our proposed Policy D (Dynamic Eve's Law Control) eliminated stockouts entirely, achieved a 100% fill rate, slashed operating costs by 92.9% to \$8.8k, and liberated 49.9% of idle buffer working capital compared to static baseline policies."*

### Slide 7: Managerial Takeaways & Conclusion (`13:00 - 15:00`)
> *"For enterprise ERP deployment, we offer four rules: (1) Never select models solely on MAE—always enforce an operational bias constraint $|\text{Bias}| \le 5\%$; (2) Always apply dual-moment smearing to log-transformed count models; (3) Size safety stock using Eve's Law to account for supplier lead-time variance; and (4) Decouple baseline replenishment from tail-risk surge protection using calibrated classification. Our codebase is 100% reproducible with 90 passing unit tests and public Colab replication. Thank you, and I look forward to your questions."*
