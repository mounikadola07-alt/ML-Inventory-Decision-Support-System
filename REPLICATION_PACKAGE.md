# Reproduction Runbook & Replication Package Guide

**Study Title:** An Empirically Audited Machine Learning and Stochastic Inventory Control Framework for Enterprise Replenishment Decision Support  
**Authors:** Sashank P., Abhinay Sai, Poli Naidu, and Sushama Rani Dutta  
**Repository:** `https://github.com/sashank321/ML-Inventory-Decision-Support-System`  

---

## 1. Quick Start: One-Click Cloud Execution
To run all experiments, train all models, and replicate all figures in Google Colab without local installation:
* **Google Colab (GitHub):** [Open in Google Colab](https://colab.research.google.com/github/sashank321/ML-Inventory-Decision-Support-System/blob/main/ML_Inventory_Decision_Support_System_improved.ipynb)
* **Google Colab (Gist):** [Open in Google Colab Gist](https://colab.research.google.com/gist/sashank321/155d8ddce2daaba42a1fad0c52176303/ML_Inventory_Decision_Support_System_improved.ipynb)

---

## 2. Local Environment Setup

### Prerequisites
* Python 3.10, 3.11, 3.12, or 3.13
* Git

### Clone Repository & Install Dependencies
```bash
git clone https://github.com/sashank321/ML-Inventory-Decision-Support-System.git
cd ML-Inventory-Decision-Support-System
pip install -r requirements.txt
```

---

## 3. Automated Verification & Test Suite
To run all 90 unit, regression, and literature integrity tests:
```bash
python -m pytest tests/ -v
```
**Expected Output:** `90 passed in ~21s`

---

## 4. Replicating Empirical Experiments
To execute the walk-forward cross-validation benchmarks and regenerate `experiment_results.json`:
```bash
python run_experiments.py
```

To run the longitudinal replenishment policy simulations (Policies A, B, C, D across 8,000 steps):
```bash
python src/policy_comparison.py
```

---

## 5. Replicating All Figures and Visualizations
To re-generate all high-resolution figures in `paper_figures/`:
```bash
python create_all_figures.py
```
Generated artifacts:
* `paper_figures/fig1_methodology.png` (Horizontal 5-stage architecture)
* `paper_figures/fig2_model_comparison.png` (4-Tier system architecture)
* `paper_figures/fig3_forecast_tracking.png` (EDA distributions & sparsity)
* `paper_figures/fig4_inventory_policy.png` (Continuous-review (s, Q) bands)
* `paper_figures/fig5_loss_functions.png` (Loss landscapes & Tweedie gradient)
* `paper_figures/fig6_inventory_dynamics.png` (Replenishment sawtooth curves)
* `paper_figures/fig7_policy_distributions.png` (Comparative policy distributions)
* `paper_figures/fig8_surge_roc.png` (Surge ROC curve & Platt calibration)
* `paper_figures/fig9_telemetry_trajectory.png` (Real-time telemetry trajectory)

---

## 6. Compiling the Research Paper & Deliverables
To recompile the exact 11-page manuscript matching your reference PDF:
```bash
python generate_reference_matched_paper.py
```

To recompile the presentation slide deck:
```bash
python build_presentation_deck.py
```

To recompile the journal submission cover letter:
```bash
python build_cover_letter.py
```

To recompile the supplementary materials:
```bash
python build_supplementary_materials.py
```
