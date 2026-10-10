# Machine Learning Inventory Decision Support System

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sashank321/ML-Inventory-Decision-Support-System/blob/main/ML_Inventory_Decision_Support_System_improved.ipynb)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests Passing](https://img.shields.io/badge/pytest-71%20passed-brightgreen.svg)](test_report.md)
[![Audit Status](https://img.shields.io/badge/audit-passed%2010%2F10-success.svg)](AUDIT_REPORT.md)

An end-to-end Machine Learning-Based Inventory Decision Support System coupling supervised gradient boosting demand forecasting and surge classification with stochastic continuous-review $(s, Q)$ inventory optimization across diverse retail operational paradigms.

---

## Architecture Overview

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
│   • Demand Surge / Stockout Early Warning Classifier (ROC AUC = 0.8305)│
│   • Tri-Criterion Operational Selection Rule:                          │
│       1. Operational Bias Bound: |Bias%| <= 5.0% (Anti-Starvation)     │
│       2. Primary Forecast Metric: Lowest MAE                           │
│       3. Secondary Stability Metric: Lowest RMSE                       │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼────────────────────────────────────┐
│              STOCHASTIC INVENTORY CONTROL & SIMULATION                 │
│   • Safety Stock (SS) sized from Forecast Residual Standard Error (σ_e)│
│     and Stochastic Lead Time Variance (σ_L) via Eve's Law              │
│   • Continuous-Review Reorder Point: ROP = d_hat * L + SS              │
│   • Dimensionally Correct Economic Order Quantity (EOQ)                │
│   • Discrete-Time Event-Driven Replenishment Simulation (s, Q)         │
│       (Fill Rate, Stockout Days, Holding Cost, Ordering Cost)          │
└────────────────────────────────────────────────────────────────────────┘
```

---

## Key Files & Repository Structure

```text
├── ML_Inventory_Decision_Support_System_original.ipynb # Preserved original notebook backup
├── ML_Inventory_Decision_Support_System_improved.ipynb # Polished, research-grade 18-section notebook
├── Research_Paper_MLA.md                              # Complete MLA 9th Edition academic research paper (~7,500 words)
├── ML_Inventory_Decision_Support_Paper.docx           # Publication-formatted Word document with embedded figures
├── AUDIT_REPORT.md                                    # Comprehensive technical audit report
├── DEFECT_LOG.md                                      # 18 categorized audit findings and fixes
├── test_report.md                                     # Automated test verification report (71/71 tests passing)
├── experiment_results.json                            # Programmatically exported experiment outputs
├── run_experiments.py                                 # Single-command end-to-end experiment pipeline
├── research/                                          # Systematic literature review & BibTeX matrix
│   ├── literature_review.md                           # 49 KB exhaustive academic literature review
│   ├── literature_matrix.csv                          # 30 peer-reviewed papers with verified DOIs
│   ├── research_gap_assessment.md                     # Formal assessment of Gaps 1–4
│   └── references.bib                                 # Complete BibTeX bibliography
├── paper_figures/                                     # 6 publication-grade high-resolution vector diagrams
├── src/                                               # Core production modules
│   ├── data_loader.py                                 # Multi-source ingestion & benchmark generators
│   ├── features.py                                    # Leak-free lags, rolling stats, cyclical harmonics
│   ├── validation.py                                  # Rolling-origin walk-forward CV engine
│   ├── models.py                                      # Baselines (Naive, SBA), Tree ensembles, & Classifiers
│   ├── metrics.py                                     # Hardened metrics (MAE, RMSE, R², WAPE, Bias%, ROC AUC)
│   └── inventory.py                                   # Continuous-review (s, Q) control & event simulation
└── tests/                                             # Pytest test suite (71 passing unit & regression tests)
    ├── test_metrics.py
    ├── test_features.py
    ├── test_validation.py
    ├── test_inventory.py
    ├── test_pipeline_integration.py
    └── test_defect_regressions.py
```

---

## Execution Instructions

### 1. Environment Setup
Install required dependencies:
```bash
pip install numpy pandas scipy scikit-learn xgboost lightgbm matplotlib pytest openpyxl
```

### 2. Run Automated Verification Tests
Execute the full test suite verifying all 24 unit, integration, and defect regression assertions:
```bash
python -m pytest tests/ -v
```

### 3. Run End-to-End Walk-Forward Cross-Validation
Run the complete pipeline across all three datasets and generate `experiment_results.json`:
```bash
python run_experiments.py
```

### 4. Inspect Results
Display a formatted table of all model benchmarks and inventory replenishment policies:
```bash
python print_results.py
```

---

## Reproducible Empirical Results Summary

### Walmart Weekly Store Sales (26-Week Locked Holdout)
| Model | MAE ($) | RMSE ($) | MAPE (%) | $R^2$ | Bias (%) | Operational Status |
|---|---|---|---|---|---|---|
| **Lag-1 Naive Baseline** | $50,864.96 | $76,052.99 | 4.91% | 0.9798 | -0.15% | Benchmark |
| **Seasonal Naive (Lag-52)** | $54,031.40 | $81,785.53 | 5.55% | 0.9766 | -2.03% | Benchmark |
| **Random Forest** | $35,890.53 | $52,643.56 | 3.56% | 0.9903 | +0.01% | **Winner (Lowest MAE & RMSE)** |
| **LightGBM MSE** | $37,913.51 | $56,925.32 | 3.83% | 0.9887 | -0.04% | Competitive Runner-up |
| **XGBoost MSE** | $38,275.36 | $57,741.70 | 3.81% | 0.9884 | -0.26% | Competitive Runner-up |
| **Surge Classifier (RF / LGB)** | — | — | — | — | — | **ROC AUC: 0.7269, Acc: 71.54%** |

### Online Retail Daily Demand (28-Day Locked Holdout, Continuous Panel)
| Model | MAE (u) | RMSE (u) | WAPE (%) | $R^2$ | Bias (%) | Operational Status |
|---|---|---|---|---|---|---|
| **Lag-1 Naive Baseline** | 114.74 | 275.84 | 110.18% | -0.4777 | +1.94% | Benchmark |
| **Recent Moving Average** | 92.13 | 208.19 | 88.47% | 0.1583 | +4.79% | Benchmark |
| **Random Forest** | 90.57 | 236.31 | 86.97% | -0.0845 | +3.14% | Operational Candidate |
| **LightGBM MSE** | 96.22 | 211.54 | 92.40% | 0.1309 | -4.46% | **Anti-Starvation Compliant Winner** |
| **XGBoost MSE (Tuned)** | 98.13 | 209.74 | 94.23% | 0.1457 | +3.94% | **Lowest Holdout Ensemble RMSE** |
| **XGBoost Log1p** | 72.58 | 213.72 | 69.70% | 0.1129 | -57.18% | **Disqualified (Severe Starvation Risk)** |
| **Surge Classifier (LGB / RF)** | — | — | — | — | — | **ROC AUC: 0.8361, Acc: 76.07%** |

### M5 Intermittent Grocery Demand (28-Day Locked Holdout, 63.28% Zeros)
| Model | MAE (u) | RMSE (u) | WAPE (%) | $R^2$ | Bias (%) | Operational Status |
|---|---|---|---|---|---|---|
| **Lag-1 Naive Baseline** | 1.12 | 1.89 | 94.07% | -0.9197 | +2.44% | Benchmark |
| **Recent Moving Average** | 1.01 | 1.41 | 85.04% | -0.0661 | +3.62% | Benchmark |
| **Croston SBA** | 0.85 | 1.43 | 71.86% | -0.1002 | -58.97% | Classical Intermittent Benchmark |
| **Random Forest** | 1.03 | 1.33 | 86.73% | 0.0464 | +14.65% | Operational Candidate |
| **XGBoost MSE (Regularized)** | 1.00 | 1.33 | 84.18% | 0.0549 | +2.97% | **Bias-Controlled Regressor** |
| **XGBoost Tweedie ($p=1.2$)** | 0.99 | 1.32 | 83.15% | 0.0573 | +1.29% | **Lowest RMSE & Near-Zero Bias** |
| **Occurrence Classifier (LGB)** | — | — | — | — | — | **ROC AUC: 0.6206, Acc: 53.21%** |

---

## Methodological Distinctions
* **Implemented vs Tested:** Every implemented feature and model configuration is covered by unit or integration tests.
* **Tested vs Validated:** Model selection is conducted strictly on development walk-forward CV folds; final metrics are evaluated on an untouched chronological holdout.
* **Point Accuracy vs Operational Utility:** Minimizing MAE through log transformations induces negative bias that starves inventory; operational models are chosen via the Tri-Criterion framework.
* **Raw Demand Std vs Residual Error:** Safety stock is sized using forecast residual standard error ($\sigma_e = \sqrt{\text{MSE}}$) rather than total demand variance ($\sigma_d$), eliminating redundant buffer costs.
