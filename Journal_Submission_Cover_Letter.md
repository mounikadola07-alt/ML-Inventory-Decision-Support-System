# Journal Submission Cover Letter

**Department of Computer Science and Engineering**  
Koneru Lakshmaiah Education Foundation, Hyderabad, Telangana, India  
Email: `mounika.dola07@gmail.com`  

**Date:** 10 October 2026  

**To:**  
The Editor-in-Chief  
*International Journal of Forecasting*  
Elsevier Science B.V.  

**Subject:** Submission of Original Empirical Research Article – *"An Empirically Audited Machine Learning and Stochastic Inventory Control Framework for Enterprise Replenishment Decision Support"*  

Dear Editor-in-Chief and Editorial Board Members,

We are pleased to submit our original research manuscript entitled **"An Empirically Audited Machine Learning and Stochastic Inventory Control Framework for Enterprise Replenishment Decision Support"** for publication consideration in the *International Journal of Forecasting* as a Regular Research Paper.

Modern enterprise supply chains operate under an asymmetric cost structure where customer stockouts inflict catastrophic lost margins, contractual service-level penalties, and permanent customer defection, whereas holding excess safety buffer stock incurs marginal carrying charges (~20%/year). Despite this fundamental economic asymmetry, prevailing machine learning forecasting deployments routinely select algorithms using symmetric point loss objectives (MAE and RMSE). We investigate the critical operational question: *Why do models that win statistical accuracy competitions frequently trigger catastrophic stockouts when deployed inside automated enterprise replenishment systems?*

Our investigation resolves this disconnect through three core scientific contributions:
1. **Analytical Proofs of Negative Bias Traps:** We prove via Jensen's Inequality that fitting log-transformed targets (Log1p) forces predictions to the conditional geometric median rather than the conditional mean, inducing an extreme -55.9% to -81.3% under-forecasting bias. Furthermore, we prove that Tweedie compound Poisson-Gamma deviance exhibits an explosive gradient divergence at zero demand ($\lim_{\mu \to 0^+} \frac{\partial d(0, \mu)}{\partial \mu} = +\infty$), exerting an overpowering downward force that drives chronic replenishment starvation.
2. **Tri-Criterion Operational Model Selection Rule:** We establish a robust selection protocol requiring models to pass a strict operational bias constraint ($|\text{Bias}\%| \le 5.0\%$) prior to error ranking, preventing downstream inventory starvation without sacrificing point forecasting fidelity.
3. **Capital Optimization via Eve's Law of Total Variance:** Grounded in stochastic inventory theory, we decouple deterministic predictable variance from stochastic risk, sizing safety stocks directly from forecast residual standard errors ($\sigma_e$) across lead-time demand convolution. In closed-loop simulations across 8,000 demand interactions, our proposed framework eliminated 24 stockout days, delivered a 92.9% reduction in total inventory operating costs, and liberated 49.9% of safety buffer working capital.

Furthermore, we formulate an integrated Platt-calibrated demand surge early-warning classifier, achieving an out-of-sample **ROC AUC of 0.9278** with **83.21% classification accuracy** across heterogeneous retail regimes, empowering supply chain controllers to execute proactive buffer replenishment before promotional surges deplete on-hand stock.

To ensure uncompromising scientific transparency and reproducibility, all benchmark datasets, feature engineering pipelines, walk-forward cross-validation routines, and simulation engines have been audited, packaged with 90 passing automated unit/regression tests, and made accessible as an interactive cloud notebook runnable in Google Colab.

We confirm that this manuscript represents original work that has not been published previously and is not currently under consideration for publication elsewhere. All authors have approved the manuscript and agreed to its submission.

Thank you for your time and editorial consideration. We look forward to hearing from you.

Sincerely,  

**Dola Mounika** *(Corresponding Author)*  
  
Department of Computer Science and Engineering  
Koneru Lakshmaiah Education Foundation, Hyderabad, Telangana, India  
