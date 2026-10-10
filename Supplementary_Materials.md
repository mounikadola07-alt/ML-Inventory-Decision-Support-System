# Online Supplementary Materials & Technical Appendix

**An Empirically Audited Machine Learning and Stochastic Inventory Control Framework for Enterprise Replenishment Decision Support**  
*Sashank P., Abhinay Sai, Poli Naidu, and Sushama Rani Dutta*  
Department of Computer Science and Engineering, Koneru Lakshmaiah Education Foundation, Hyderabad, India  

---

## Appendix A: Extended Mathematical Derivations and Analytical Proofs

### A.1 Proof of Jensen's Inequality Retransformation Deficit (Theorem 1)
Let $Y$ be a non-negative continuous demand random variable with probability density function $f_Y(y)$ on $[0, \infty)$. Consider the strictly concave transformation $g(y) = \ln(1 + y)$. By Jensen's Inequality for strictly concave functions:
$$\mathbb{E}[\ln(1 + Y)] < \ln(1 + \mathbb{E}[Y])$$

When an estimator minimizes Mean Squared Error in logarithmic space, it converges to the conditional expectation: $\hat{z} = \mathbb{E}[\ln(1 + Y) \mid X]$. Applying naive exponential back-transformation yields $\hat{y}_{\text{naive}} = \exp(\hat{z}) - 1$. Substituting Jensen's inequality establishes:
$$\hat{y}_{\text{naive}} = \exp(\mathbb{E}[\ln(1 + Y)]) - 1 < \exp(\ln(1 + \mathbb{E}[Y])) - 1 = \mathbb{E}[Y]$$

The deficit $\Delta = \mathbb{E}[Y] - \hat{y}_{\text{naive}}$ is strictly positive and scales monotonically with the conditional variance $\text{Var}(\ln(1+Y))$. In zero-inflated retail datasets where variance is elevated, this deficit introduces persistent negative bias exceeding $-50\%$, systematically starving downstream replenishment buffers. $\blacksquare$

### A.2 Proof of Tweedie Zero-Gradient Divergence (Theorem 2)
The unit Tweedie deviance for power parameter $p \in (1, 2)$ is defined as:
$$d(y, \mu) = 2 \left[ \frac{y^{2-p}}{(1-p)(2-p)} - \frac{y \mu^{1-p}}{1-p} + \frac{\mu^{2-p}}{2-p} \right]$$
Evaluating at $y = 0$ simplifies to $d(0, \mu) = \frac{2 \mu^{2-p}}{2-p}$. Differentiating with respect to $\mu$ yields:
$$\frac{\partial d(0, \mu)}{\partial \mu} = \frac{2}{\mu^{p-1}}$$
For compound Poisson-Gamma modeling ($p = 1.5$), $p - 1 = 0.5$. Hence $\frac{\partial d(0, \mu)}{\partial \mu} = \frac{2}{\sqrt{\mu}}$. Taking the limit as positive predictions approach zero:
$$\lim_{\mu \to 0^+} \frac{\partial d(0, \mu)}{\partial \mu} = \lim_{\mu \to 0^+} \frac{2}{\sqrt{\mu}} = +\infty$$
Because gradient boosting steps move in the direction of the negative gradient ($-\frac{\partial L}{\partial \mu}$), this infinite positive derivative creates an overpowering negative force whenever actual demand is zero, driving the conditional mean below true retail requirements ($-11.87\%$ to $-18.04\%$ bias). $\blacksquare$

### A.3 Proof of Strict Boundedness of Trigg's Tracking Signal (Theorem 4)
Let $\{e_t\}$ be an arbitrary sequence of forecast errors, smoothed error $E_t = \alpha_s e_t + (1 - \alpha_s) E_{t-1}$, and smoothed mean absolute deviation $\text{MAD}_t = \alpha_s |e_t| + (1 - \alpha_s) \text{MAD}_{t-1}$ with smoothing parameter $\alpha_s \in (0, 1)$. Expanding recursively from $t = 0$ to $t = n$ yields:
$$E_t = \sum_{k=0}^t \alpha_s (1 - \alpha_s)^k e_{t-k}, \quad \text{MAD}_t = \sum_{k=0}^t \alpha_s (1 - \alpha_s)^k |e_{t-k}|$$
Applying the triangle inequality: $|E_t| \le \sum_{k=0}^t \alpha_s (1 - \alpha_s)^k |e_{t-k}| = \text{MAD}_t$. Dividing by $\text{MAD}_t > 0$ establishes $|TS_t| = |E_t| / \text{MAD}_t \le 1$. Hence $-1 \le TS_t \le 1$ for all $t$. $\blacksquare$

---

## Appendix B: Preprocessing Protocols and Continuous Cartesian Panel Assembly
* **B.1 Online Retail Dataset Cleaning:** Filtered 9,288 cancelled invoices (`InvoiceNo` prefix `'C'`), removed non-positive quantities and zero unit prices, aggregated line items by `StockCode` and calendar date, and assembled a continuous Cartesian SKU-date grid imputing explicit `0.0` values on non-transaction days.
* **B.2 Forward-Fill Price Tracking:** Shelf prices were forward-filled strictly using chronological past values (`Price_t = ffill(Price_t).fillna(0.0)`), completely eliminating lookahead bias from backward filling (`bfill`).

---

## Appendix C: Model Architecture and Hyperparameter Configuration

| Model Architecture | Key Hyperparameters | Loss Objective | Stopping Criteria | Regularization |
| :--- | :--- | :--- | :--- | :--- |
| Random Forest | n_estimators=100, max_depth=12, min_samples_leaf=3 | MSE (Variance Reduction) | Full Tree Ensemble | Feature sub-sampling (sqrt) |
| LightGBM MSE | n_estimators=150, lr=0.03, max_depth=6, num_leaves=31 | L2 Regression | Early stopping (20 rounds) | L1=0.05, L2=0.10 |
| XGBoost MSE | n_estimators=150, lr=0.03, max_depth=6, subsample=0.8 | reg:squarederror | Early stopping (20 rounds) | alpha=0.05, lambda=0.10 |
| LightGBM Tweedie | n_estimators=150, lr=0.03, tweedie_variance_power=1.5 | Tweedie deviance | Early stopping (20 rounds) | L1=0.05, L2=0.10 |
| Calibrated LGB Surge | n_estimators=100, max_depth=5, lr=0.05, isotonic=True | Binary Logistic + Platt | Early stopping (15 rounds) | Platt Sigmoid Calibration |

---

## Appendix D: Computational Environment and Automated Verification
All models, walk-forward validation routines, inventory simulations, and statistical tracking signals were implemented in Python 3.10+ using scikit-learn 1.4+, LightGBM 4.3+, XGBoost 2.0+, NumPy 1.26+, and pandas 2.2+. Full automated verification executes across 11 modular test suites (**90 unit and regression tests**) in 21.76 seconds, confirming complete absence of data leakage, parameter stability, and mathematical correctness.
