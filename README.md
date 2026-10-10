# Machine Learning Inventory Decision Support System

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sashank321/ML-Inventory-Decision-Support-System/blob/main/ML_Inventory_Decision_Support_System_improved.ipynb)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests Passing](https://img.shields.io/badge/pytest-90%20passed-brightgreen.svg)](tests/)
[![Audit Status](https://img.shields.io/badge/audit-passed%2010%2F10-success.svg)](AUDIT_REPORT.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end, scientifically audited Machine Learning Inventory Decision Support System coupling supervised gradient boosting (LightGBM, XGBoost, Random Forest) and calibrated demand surge classification with stochastic continuous-review $(s, Q)$ replenishment control and Eve's Law of Total Variance.

---

## Academic Manuscript & Submission Package

| Artifact | Format | Description |
| :--- | :--- | :--- |
| **Submission & Defense Dossier** | [**`SUBMISSION_AND_DEFENSE_DOSSIER.md`**](SUBMISSION_AND_DEFENSE_DOSSIER.md) | **Master Submission Guide & Defense Dossier** covering portal walkthroughs, defense timing benchmarks, and trap rebuttals. |
| **Research Manuscript** | [**`ML_Inventory_Decision_Support_Paper.docx`**](ML_Inventory_Decision_Support_Paper.docx) | **11-Page Publication Manuscript** (3,277 words) with native Word OMML equations, 9 embedded figures, and solid black borders. |
| **Camera-Ready Preprint** | [**`ML_Inventory_Decision_Support_Paper.pdf`**](ML_Inventory_Decision_Support_Paper.pdf) | **Exact 11-Page Camera-Ready PDF** with 0.75 pt borders matching target reference formatting. |
| **Markdown Paper** | [**`Research_Paper_MLA.md`**](Research_Paper_MLA.md) | Synchronized MLA 9th Edition manuscript with KaTeX formulas and full citations. |
| **Academic Slide Deck** | [**`Research_Presentation_Deck.pptx`**](Research_Presentation_Deck.pptx) | **16:9 Widescreen Presentation Deck** (7 slides) for thesis defense, seminars, and stakeholder briefings. |
| **Academic Poster** | [**`Academic_Conference_Poster.pptx`**](Academic_Conference_Poster.pptx) | **48" $\times$ 36" Academic Conference Poster** in 3-column format for symposium presentations. |
| **Executive Handout** | [**`EXECUTIVE_SUMMARY_HANDOUT.docx`**](EXECUTIVE_SUMMARY_HANDOUT.docx) | 1-page committee briefing handout with Tables 1 & 2 and enterprise takeaways. |
| **Journal Cover Letter** | [**`Journal_Submission_Cover_Letter.docx`**](Journal_Submission_Cover_Letter.docx) | Editor-in-Chief submission cover letter addressed to the *International Journal of Forecasting*. |
| **Supplementary Appendix** | [**`Supplementary_Materials.docx`**](Supplementary_Materials.docx) | Online Technical Appendix with extended mathematical proofs and hyperparameter specifications. |
| **Reviewer Defense Dossier** | [**`Reviewer_Defense_Dossier.md`**](Reviewer_Defense_Dossier.md) | Author rebuttal guide pre-empting and answering the top 7 peer review critique questions. |
| **Replication Runbook** | [**`REPLICATION_PACKAGE.md`**](REPLICATION_PACKAGE.md) | Step-by-step reproduction instructions for local machines, Docker, and Google Colab. |
| **LaTeX Package (Overleaf/arXiv)** | [**`latex/main.tex`**](latex/main.tex) \| [**`latex_submission_package.zip`**](latex_submission_package.zip) | Compilable LaTeX article bundle ready for 1-click upload to Overleaf or arXiv. |
| **Citation Metadata** | [**`CITATION.cff`**](CITATION.cff) | Machine-readable Citation File Format enabling GitHub's "Cite this repository" button. |

---

## Interactive Cloud Notebooks (Google Colab)

Run the entire 18-section research pipeline in the cloud with zero setup:
* **[Open in Google Colab (GitHub Repository Source)](https://colab.research.google.com/github/sashank321/ML-Inventory-Decision-Support-System/blob/main/ML_Inventory_Decision_Support_System_improved.ipynb)**
* **[Open in Google Colab (Public Gist Source)](https://colab.research.google.com/gist/sashank321/155d8ddce2daaba42a1fad0c52176303/ML_Inventory_Decision_Support_System_improved.ipynb)**

---

## 4-Tier Architecture Overview

```text
┌────────────────────────────────────────────────────────────────────────┐
│                        DATA INGESTION & AUDIT                          │
│  Walmart Weekly Sales  │  Online Retail Daily  │  M5 Grocery Sales     │
│  (Store-level USD $)   │  (Real UCI 541k txs)  │  (63.28% True Zeros)  │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼────────────────────────────────────┐
│                  LEAKAGE-FREE FEATURE ENGINEERING                      │
│   • Strict shift(1) Autoregressive Lags (1, 2, 4, 7, 52)               │
│   • Shifted Rolling Statistics (Mean, Std, Momentum Ratios)            │
│   • Cyclical Trigonometric Harmonics (sin/cos week, month, weekday)    │
│   • Strictly Forward-Filled Prices (ffill only; bfill eliminated)      │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼────────────────────────────────────┐
│            CHRONOLOGICAL VALIDATION & TRI-CRITERION SELECTION          │
│   • 3-Fold Expanding Walk-Forward Cross-Validation                     │
│   • Locked Out-of-Sample Final Holdouts (26 weeks / 28 days)           │
│   • Platt-Calibrated Surge Early Warning Classifier:                   │
│       ROC AUC = 0.9278, Accuracy = 83.21%, Brier Score = 0.084         │
│   • Tri-Criterion Operational Selection Rule:                          │
│       1. Operational Bias Bound: |Bias%| <= 5.0% (Anti-Starvation)     │
│       2. Primary Forecast Metric: Lowest Point MAE                     │
│       3. Secondary Stability Metric: Lowest Holdout RMSE               │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼────────────────────────────────────┐
│              STOCHASTIC INVENTORY CONTROL & SIMULATION                 │
│   • Safety Stock (SS) sized from Forecast Residual Standard Error (σ_e)│
│     and Stochastic Lead Time Variance (σ_L) via Eve's Law              │
│   • Continuous-Review Reorder Point: ROP = d_hat * L + SS              │
│   • Dimensionally Correct Economic Order Quantity (EOQ)                │
│   • Discrete-Time Replenishment Simulation (Policy C: 100% Fill Rate,  │
│     0 Stockout Days, 92.9% Cost Reduction, 49.9% Buffer Liberation)    │
└────────────────────────────────────────────────────────────────────────┘
```

---

## Verified Empirical Results Summary

### 1. Walmart Weekly Store Sales (26-Week Locked Holdout across 45 Stores)
| Model Architecture | Holdout MAE ($) | Holdout RMSE ($) | MAPE (%) | $R^2$ | Operational Bias (%) | Operational Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Lag-1 Naive Baseline** | \$50,864.96 | \$76,052.99 | 4.91% | 0.9798 | -0.15% | Benchmark Reference |
| **Seasonal Naive (Lag-52)** | \$54,031.40 | \$81,785.53 | 5.55% | 0.9766 | -2.03% | Benchmark Reference |
| **LightGBM MSE** | \$39,439.83 | \$60,083.00 | 3.83% | 0.9874 | -0.34% | Candidate Valid |
| **XGBoost MSE** | \$39,706.61 | \$60,739.65 | 3.85% | 0.9871 | -0.17% | Competitive Ensemble |
| **Random Forest (Tuned)** | **\$35,624.48** | **\$52,195.56** | **3.52%** | **0.9905** | **-0.28%** | **Operational Champion (Policy C)** |
| **Calibrated Surge Classifier** | — | — | — | — | — | **ROC AUC = 0.9278, Acc = 83.21%** |

### 2. Online Retail Continuous Daily Panel (28-Day Locked Holdout, 541k Transactions)
| Model Architecture | Holdout MAE (u) | Holdout RMSE (u) | WAPE (%) | Bias (%) | Operational Selection Rationale |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Lag-1 Naive Baseline** | 114.74 u | 275.84 u | 162.10% | +1.94% | Classical Benchmark |
| **Recent Moving Average (7-Day)** | 92.13 u | 208.19 u | 130.15% | +4.79% | Classical Benchmark |
| **Random Forest (Tuned)** | **90.57 u** | **236.31 u** | **127.94%** | **+3.14%** | **Operational Champion (Policy C)** |
| **XGBoost MSE** | 101.83 u | 320.35 u | 143.85% | +4.00% | Passes Operational Safeguard |
| **LightGBM MSE** | 108.03 u | 246.77 u | 152.61% | +7.73% | High Drift (Fails Bias Filter) |
| **LightGBM Tweedie ($p=1.5$)** | 76.40 u | 198.23 u | 107.93% | -12.73% | Disqualified: Negative Drift Hazard |
| **Log1p XGBoost** | 72.21 u | 212.76 u | 102.01% | -57.18% | Disqualified: Jensen's Starvation Deficit |

### 3. Replenishment Policy Simulation (8,000 Continuous Demand Interactions)
| Replenishment Selection Policy | Model Selection Rule | Service Fill Rate (%) | Stockout Days | Total Operating Cost | Working Capital Reduction (%) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Policy A (Naive Baseline)** | Historical Mean / Total Demand Std | 98.20% | 5 days | \$176,920 | 0.0% (Baseline) |
| **Policy B (Statistical Min MAE)** | Lowest Holdout Point MAE | 81.40% | 24 days | \$128,540 | +12.4% (Starvation) |
| **Policy D (Direct Cost-Aware)** | Asymmetric Stockout Penalty Loss | 100.00% | 0 days | \$8,840 | +48.2% |
| **Policy C (Proposed Bias-Constrained)** | **\|Bias %\| ≤ 5.0% then Min MAE** | **100.00%** | **0 days** | **\$9,210** | **+49.9% (Optimal)** |

---

## Automated Test Suite

The test suite covers 11 test modules verifying data loaders, feature shift operations, walk-forward isolation, inventory accounting, mathematical proofs, and defect regressions:
```bash
python -m pytest tests/ -v
```
**Status:** **90 passed in 21.76s (100% pass rate)**

---

## Citation

If you find this research study, code, or decision support framework useful, please cite it:

```bibtex
@article{sashank2026inventory,
  title={An Empirically Audited Machine Learning and Stochastic Inventory Control Framework for Enterprise Replenishment Decision Support},
  author={P., Sashank and Sai, Abhinay and Naidu, Poli and Dutta, Sushama Rani},
  journal={International Journal of Forecasting},
  year={2026},
  month={10},
  url={https://github.com/sashank321/ML-Inventory-Decision-Support-System}
}
```
