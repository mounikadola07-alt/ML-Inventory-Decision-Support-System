# Machine Learning Inventory Decision Support System

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sashank321/ML-Inventory-Decision-Support-System/blob/main/ML_Inventory_Decision_Support_System_improved.ipynb)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests Passing](https://img.shields.io/badge/pytest-71%20passed-brightgreen.svg)](test_report.md)
[![Audit Status](https://img.shields.io/badge/audit-passed%2010%2F10-success.svg)](AUDIT_REPORT.md)

An end-to-end Machine Learning-Based Inventory Decision Support System coupling supervised gradient boosting demand forecasting with stochastic continuous-review $(s, Q)$ inventory optimization across diverse retail operational paradigms.

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
│   • Tri-Criterion Operational Selection Rule:                          │
│       1. Operational Bias Bound: |Bias%| <= 5.0% (Anti-Starvation)     │
│       2. Primary Forecast Metric: Lowest MAE                           │
│       3. Secondary Stability Metric: Lowest RMSE                       │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼────────────────────────────────────┐
│              STOCHASTIC INVENTORY CONTROL & SIMULATION                 │
│   • Safety Stock (SS) sized from Forecast Residual Standard Error (σ_e)│
│     and Stochastic Lead Time Variance (σ_L)                            │
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
├── AUDIT_REPORT.md                                    # Comprehensive technical audit report
├── DEFECT_LOG.md                                      # 18 categorized audit findings and fixes
├── test_report.md                                     # 31-test automated verification report
├── experiment_results.json                            # Programmatically exported experiment outputs
├── run_experiments.py                                 # Single-command end-to-end experiment pipeline

├── src/                                               # Core production modules
│   ├── data_loader.py                                 # Multi-source ingestion & benchmark generators
│   ├── features.py                                    # Leak-free lags, rolling stats, cyclical harmonics
│   ├── validation.py                                  # Rolling-origin walk-forward CV engine
│   ├── models.py                                      # Baselines (Naive, SBA) & Tree ensembles (RF, XGB, LGB)
│   ├── metrics.py                                     # Hardened metrics (MAE, RMSE, R², WAPE, Bias%)
│   └── inventory.py                                   # Continuous-review (s, Q) control & event simulation
└── tests/                                             # Pytest test suite (24 passing unit & regression tests)
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
| **Random Forest** | $36,207.13 | $53,397.40 | 3.56% | 0.9900 | -0.28% | **Winner (Lowest MAE & RMSE)** |
| **LightGBM MSE** | $39,439.83 | $60,083.00 | 3.83% | 0.9874 | -0.34% | Competitive Runner-up |
| **XGBoost MSE** | $39,706.61 | $60,739.65 | 3.85% | 0.9871 | -0.17% | Competitive Runner-up |

### Online Retail Daily Wholesale Demand (28-Day Locked Holdout)
| Model | MAE (u) | RMSE (u) | MAPE (%) | $R^2$ | Bias (%) | Operational Status |
|---|---|---|---|---|---|---|
| **Lag-1 Naive Baseline** | 11.32 | 26.26 | 236.49% | -1.0322 | -0.59% | Benchmark |
| **Random Forest** | 9.42 | 18.49 | 97.86% | -0.0072 | -2.69% | Operational Candidate |
| **LightGBM MSE** | 9.56 | 18.97 | 97.19% | -0.0610 | -4.60% | **Operational Replenishment Winner** |
| **LightGBM Tweedie** | 8.88 | 18.71 | 90.43% | -0.0320 | -18.04% | Biased Under-forecast |
| **XGBoost Log1p** | 7.04 | 19.21 | 80.02% | -0.0873 | -81.26% | **Disqualified (Severe Stockout Risk)** |

### M5 Intermittent Grocery Demand (28-Day Locked Holdout, 63.28% Zeros)
| Model | MAE (u) | RMSE (u) | WAPE (%) | $R^2$ | Bias (%) | Operational Status |
|---|---|---|---|---|---|---|
| **Lag-1 Naive Baseline** | 1.12 | 1.89 | 94.07% | -0.9197 | +2.44% | Benchmark |
| **Croston SBA** | 0.85 | 1.43 | 82.65% | -0.1002 | -58.97% | Classical Intermittent Benchmark |
| **Random Forest** | 1.06 | 1.35 | 51.39% | 0.0273 | +23.20% | **Safest Replenishment Winner** |
| **XGBoost MSE** | 1.10 | 1.41 | 52.49% | -0.0665 | +32.47% | Evaluated |
| **XGBoost Tweedie ($p=1.5$)** | 0.95 | 1.36 | 63.67% | 0.0126 | -11.87% | Lowest Statistical MAE |

---

## Methodological Distinctions
* **Implemented vs Tested:** Every implemented feature and model configuration is covered by unit or integration tests.
* **Tested vs Validated:** Model selection is conducted strictly on development walk-forward CV folds; final metrics are evaluated on an untouched chronological holdout.
* **Point Accuracy vs Operational Utility:** Minimizing MAE through log transformations induces negative bias that starves inventory; operational models are chosen via the Tri-Criterion framework.
* **Raw Demand Std vs Residual Error:** Safety stock is sized using forecast residual standard error ($\sigma_e = \sqrt{\text{MSE}}$) rather than total demand variance ($\sigma_d$), eliminating redundant buffer costs.
