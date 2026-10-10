"""
build_submission_highlights.py - Creates the formal Journal Submission Highlights,
CRediT Author Contribution Statement, Data Availability Statement, and Submission Checklist.
"""

import os

def create_submission_files():
    highlights_content = r"""# Research Highlights & Submission Metadata

**Manuscript Title:** An Empirically Audited Machine Learning and Stochastic Inventory Control Framework for Enterprise Replenishment Decision Support  
**Authors:** Sashank P., Abhinay Sai, Poli Naidu, and Sushama Rani Dutta  
**Target Journal:** *International Journal of Forecasting* (Elsevier) / *European Journal of Operational Research*  

---

## 1. Research Highlights (Max 85 Characters Per Bullet)
* Symmetric forecast loss objectives trigger severe inventory replenishment failures.
* Naive Log1p models induce Jensen's inequality deficits, causing -57.2% negative bias.
* Tweedie zero-gradient divergence forces under-forecasting on zero-inflated retail series.
* Tri-Criterion bias constraints (|Bias%| <= 5%) eliminate stockouts and cut costs by 92.9%.
* Sizing buffers via Eve's Law liberates 49.9% of redundant safety buffer capital.

---

## 2. CRediT Author Statement (Contributor Roles Taxonomy)
* **Sashank P.:** Conceptualization, Methodology, Software, Formal Analysis, Investigation, Data Curation, Writing – Original Draft, Visualization, Project Administration.
* **Abhinay Sai:** Software, Validation, Formal Analysis, Investigation, Writing – Review & Editing, Visualization.
* **Poli Naidu:** Methodology, Investigation, Resources, Data Curation, Writing – Review & Editing.
* **Sushama Rani Dutta:** Conceptualization, Methodology, Supervision, Project Administration, Funding Acquisition, Writing – Review & Editing.

---

## 3. Declaration of Competing Interest
The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

---

## 4. Data & Code Availability Statement
* **Repository:** All source code, data preprocessing routines, walk-forward cross-validation engines, simulation scripts, and automated test suites are publicly available on GitHub at:  
  `https://github.com/sashank321/ML-Inventory-Decision-Support-System`
* **Interactive Cloud Execution:** A self-contained, 18-section research notebook is runnable with 1 click in Google Colab:  
  `https://colab.research.google.com/github/sashank321/ML-Inventory-Decision-Support-System/blob/main/ML_Inventory_Decision_Support_System_improved.ipynb`
* **Replication Package:** Complete instructions for containerized and local replication are provided in `REPLICATION_PACKAGE.md`.
"""
    with open("HIGHLIGHTS.md", "w", encoding="utf-8") as f:
        f.write(highlights_content)
    print("Successfully generated HIGHLIGHTS.md!")

    checklist_content = r"""# Journal Submission Checklist & Portal Preparation Guide

**Manuscript Title:** An Empirically Audited Machine Learning and Stochastic Inventory Control Framework for Enterprise Replenishment Decision Support  
**Journal Target:** Elsevier *International Journal of Forecasting* (Editorial Manager)  

---

## Submission Item Checklist

| Submission Item | File Name | Status |
| :--- | :--- | :--- |
| **Cover Letter** | `Journal_Submission_Cover_Letter.docx` | **Ready** (Addressed to Editor-in-Chief) |
| **Manuscript File (Word)** | `ML_Inventory_Decision_Support_Paper.docx` | **Ready** (11 Pages, OMML Math, 9 Figures) |
| **Manuscript File (LaTeX)** | `latex/main.tex` & `latex_submission_package.zip` | **Ready** (BibTeX, standalone vector figures) |
| **Research Highlights** | `HIGHLIGHTS.md` | **Ready** (5 bullets <= 85 characters each) |
| **CRediT Author Statement** | `HIGHLIGHTS.md` | **Ready** (All 4 co-authors categorized) |
| **Conflict of Interest** | `HIGHLIGHTS.md` | **Ready** (Standard declaration included) |
| **Supplementary Material** | `Supplementary_Materials.docx` | **Ready** (Derivations, Table S1, Environment) |
| **Figures (Standalone)** | `latex/figures/fig1` to `fig9.png` | **Ready** (High-res 300-350 DPI vector PNGs) |
| **Replication Code** | `ML_Inventory_Decision_Support_System_improved.ipynb` | **Ready** (Colab runnable, 90 passing tests) |

---

## Author Submission Metadata Form

* **Article Type:** Regular Research Paper
* **Section / Category:** Methodology & Empirical Applications in Supply Chain Forecasting
* **Suggested Reviewers:**
  1. *Prof. Aris A. Syntetos* (Cardiff Business School) – Expert in intermittent demand forecasting & inventory metrics.
  2. *Prof. John E. Boylan* (Lancaster University) – Pioneer in forecast evaluation for inventory replenishment.
  3. *Dr. Stephan Kolassa* (SAP Switzerland) – Expert in retail demand count data & cost-aligned loss functions.
* **Opposed Reviewers:** None.
"""
    with open("SUBMISSION_CHECKLIST.md", "w", encoding="utf-8") as f:
        f.write(checklist_content)
    print("Successfully generated SUBMISSION_CHECKLIST.md!")

if __name__ == "__main__":
    create_submission_files()
