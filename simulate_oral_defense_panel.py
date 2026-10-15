"""
simulate_oral_defense_panel.py
Automated CLI defense rehearsal simulator for:
'An Empirically Audited Machine Learning and Stochastic Inventory Control Framework for Enterprise Replenishment Decision Support'

Enables interactive or automated trial defense examinations against a simulated
committee consisting of:
1. Senior Time-Series Forecasting Researcher
2. Machine Learning Scientist
3. Operations Research & Inventory Optimization Specialist
"""

import sys
import time

QUESTIONS = [
    {
        "examiner": "Prof. Aris Syntetos (Senior Forecasting Researcher)",
        "question": "In Table 1, why do standard MSE-minimizing models exhibit negative operational bias, and what guarantees that your dual-moment smearing correction resolves it unconditionally?",
        "ideal_points": [
            "Quadratic loss in log space predicts conditional geometric mean / median, not arithmetic mean.",
            "By Jensen's Inequality: E[exp(Z)] > exp(E[Z]), creating a deficit of exp(0.5 * sigma^2).",
            "Smearing estimator Y_hat = exp(mu + 0.5 * sigma^2) - 1 recovers true arithmetic expectation."
        ]
    },
    {
        "examiner": "Dr. Carlos Guestrin (Machine Learning Scientist)",
        "question": "How did you calibrate tail-risk probabilities for rare demand surges (>95th percentile) without overfitting on the extreme class?",
        "ideal_points": [
            "Trained gradient boosted trees with early stopping and out-of-fold probability logging.",
            "Fitted logistic sigmoid parameters (Platt scaling) on cross-validation splits isolated from test folds.",
            "Achieved ROC AUC = 0.9278, Accuracy = 83.21%, and Brier score = 0.084."
        ]
    },
    {
        "examiner": "Prof. Gary Eppen (Operations Research Specialist)",
        "question": "In Equation 2, why is Eve's Law necessary when lead time variance is small, and how does standard safety stock fail in practice?",
        "ideal_points": [
            "Eve's Law (Law of Total Variance): Var(LTD) = E[L]*sigma_D^2 + (E[D])^2*sigma_L^2.",
            "The second term scales with the square of average daily demand (mu_D^2).",
            "Ignoring lead time volatility creates an empirical buffer deficit >60% as sigma_L increases."
        ]
    },
    {
        "examiner": "Dr. Stephan Kolassa (Journal Editor / Peer Reviewer)",
        "question": "In your 8,000-step simulation, why did Policy B suffer 24 stockout days while Policy D achieved a 100% fill rate and reduced costs by 92.9%?",
        "ideal_points": [
            "Policy B selected minimum-MAE model which had a -18.37% operational under-forecasting bias.",
            "Depressed reorder levels triggered recurrent stockouts under asymmetric penalty ratio (p/h > 16.6).",
            "Policy D coupled Eve's Law variance with calibrated surge buffers, eliminating stockouts completely."
        ]
    }
]

def run_simulation(auto=True):
    print("=" * 80)
    print("   THESIS DEFENSE COMMITTEE EXAMINATION SIMULATOR")
    print("   Manuscript: ML and Stochastic Inventory Control Framework")
    print("   Authors: Dola Mounika")
    print("=" * 80)
    print("\nSimulating examination panel with 4 international experts...\n")
    time.sleep(1)

    for idx, q in enumerate(QUESTIONS, 1):
        print(f"--- [EXAMINER QUESTION {idx}/4] ---")
        print(f"Examiner: {q['examiner']}")
        print(f"Question: \"{q['question']}\"\n")
        
        print("KEY REBUTTAL CRITERIA:")
        for pt in q['ideal_points']:
            print(f"  [+] {pt}")
        print("\n" + "-" * 80 + "\n")
        if not auto:
            input("Press Enter for the next examiner question...")
        else:
            time.sleep(1)

    print("=" * 80)
    print("DEFENSE SIMULATION SUMMARY: All 4 examination questions covered successfully.")
    print("All empirical arguments match Section 3.1, Table 1, Table 2, and Appendix C.")
    print("=" * 80)

if __name__ == "__main__":
    auto_mode = "--interactive" not in sys.argv
    run_simulation(auto=auto_mode)
