import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, PathPatch
import numpy as np

def generate_better_methodology_figure(output_path="paper_figures/fig1_methodology.png"):
    # High resolution, professional journal width
    fig = plt.figure(figsize=(15.0, 7.2), dpi=350)
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 150)
    ax.set_ylim(0, 72)
    ax.axis('off')

    # Background canvas
    canvas_bg = FancyBboxPatch((0.5, 0.5), 149, 71,
                               boxstyle="round,pad=0.2,rounding_size=1.5",
                               facecolor='#f8fafc', edgecolor='#cbd5e1', linewidth=1.5)
    ax.add_patch(canvas_bg)

    # Top Title Banner
    header = FancyBboxPatch((2, 63), 146, 7.2,
                            boxstyle="round,pad=0.2,rounding_size=1.0",
                            facecolor='#0f172a', edgecolor='#1e293b', linewidth=1.5)
    ax.add_patch(header)

    ax.text(75, 67.2, "EMPIRICALLY AUDITED MACHINE LEARNING INVENTORY DECISION SUPPORT PIPELINE",
            color='#ffffff', fontsize=13, fontweight='bold', ha='center', va='center', family='sans-serif')
    ax.text(75, 64.6, "End-to-End Workflow: Multi-Regime Data Ingestion  →  Leakage-Free Features  →  Predictive Ensembles  →  Tri-Criterion Selection  →  Stochastic Control",
            color='#94a3b8', fontsize=8.5, ha='center', va='center', family='sans-serif')

    # Stage Box Dimensions
    box_w = 27.2
    box_h = 44.5
    box_y = 15.5
    gap = 2.0
    start_x = 3.0

    stages = [
        {
            "num": "STAGE 1",
            "title": "DATA INGESTION",
            "subtitle": "Multi-Regime Preprocessing",
            "header_color": "#0284c7", # Sky blue
            "accent": "#0369a1",
            "bg_color": "#ffffff",
            "items": [
                ("Walmart Weekly", "45 retail stores, 143 weeks sales ($)"),
                ("Online Retail", "541,909 wholesale transactions"),
                ("M5 Grocery", "1,914 days, 63.3% zero-demand sparsity"),
                ("Audit Cleaning", "Strict cancellation & return filtering"),
                ("Panel Alignment", "Cartesian SKU-date continuous grid"),
                ("Zero Contamination", "Locked historical ingestion boundary")
            ]
        },
        {
            "num": "STAGE 2",
            "title": "FEATURE ENGINE",
            "subtitle": "Leakage-Safe Architecture",
            "header_color": "#7c3aed", # Purple
            "accent": "#6d28d9",
            "bg_color": "#ffffff",
            "items": [
                ("Strict t-1 Lags", "Autoregressive lags (t-1, t-2, t-4, t-52)"),
                ("Rolling Moments", "7d & 28d rolling mean, std, momentum"),
                ("Fourier Harmonics", "Cyclical sin/cos seasonal terms"),
                ("Exogenous Signals", "Forward-filled prices at t <= t0 strictly"),
                ("Zero Contamination", "Verified zero future information leakage"),
                ("Origin Isolation", "Expanding walk-forward time splits")
            ]
        },
        {
            "num": "STAGE 3",
            "title": "PREDICTIVE MODELS",
            "subtitle": "Ensembles & Surge Detection",
            "header_color": "#059669", # Emerald green
            "accent": "#047857",
            "bg_color": "#ffffff",
            "items": [
                ("Random Forest", "Decorrelated tree bagging ensemble"),
                ("LightGBM GOSS", "Gradient-based one-side histogram tree"),
                ("XGBoost L2", "2nd-order Taylor expansion + shrinkage"),
                ("Tweedie Deviance", "Poisson-Gamma compound loss (p=1.5)"),
                ("Surge Classifier", "Isotonic calibrated LightGBM (AUC=0.928)"),
                ("High Accuracy", "Surge detection accuracy = 83.21%")
            ]
        },
        {
            "num": "STAGE 4",
            "title": "TRI-CRITERION",
            "subtitle": "Operational Model Selection",
            "header_color": "#d97706", # Amber
            "accent": "#b45309",
            "bg_color": "#ffffff",
            "items": [
                ("Rule 1 (Prerequisite)", "Operational bias safeguard: |Bias%| <= 5%"),
                ("Anti-Starvation", "Prunes severe under-forecasting models"),
                ("Rule 2 (Primary)", "Minimize Mean Absolute Error (min MAE)"),
                ("Rule 3 (Tie-Break)", "Minimize Root Mean Squared Error (RMSE)"),
                ("Temporal Validation", "Rolling walk-forward out-of-sample test"),
                ("Decoupled Loss", "Separates statistical fit from supply risk")
            ]
        },
        {
            "num": "STAGE 5",
            "title": "STOCHASTIC (s, Q)",
            "subtitle": "Closed-Loop Replenishment",
            "header_color": "#0d9488", # Teal
            "accent": "#0f766e",
            "bg_color": "#ffffff",
            "items": [
                ("Eve's Law Variance", "Var(D_L) = L̄ σ_e² + d̂² σ_L²"),
                ("Safety Stock Sizing", "SS = Z_α · σ_D_L (residual error based)"),
                ("Dynamic Reorder Point", "ROP = d̂ · L̄ + SS"),
                ("Batch Optimization", "Batch EOQ: Q* = sqrt(2KD / h)"),
                ("0 Stockouts Achieved", "Stockout starvation completely eliminated"),
                ("92.9% Cost Reduction", "Liberates 49.9% redundant buffer capital")
            ]
        }
    ]

    for idx, st in enumerate(stages):
        x = start_x + idx * (box_w + gap)
        
        # Main card container
        card = FancyBboxPatch((x, box_y), box_w, box_h,
                              boxstyle="round,pad=0.2,rounding_size=0.8",
                              facecolor=st["bg_color"], edgecolor=st["header_color"], linewidth=1.5)
        ax.add_patch(card)

        # Header ribbon
        head = FancyBboxPatch((x, box_y + box_h - 7.5), box_w, 7.5,
                              boxstyle="round,pad=0.2,rounding_size=0.8",
                              facecolor=st["header_color"], edgecolor=st["header_color"], linewidth=1.0)
        ax.add_patch(head)

        # Stage number pill
        pill = FancyBboxPatch((x + 1.2, box_y + box_h - 2.8), 8.5, 2.0,
                              boxstyle="round,pad=0.1,rounding_size=0.5",
                              facecolor="#ffffff", edgecolor="none")
        ax.add_patch(pill)
        ax.text(x + 5.45, box_y + box_h - 1.8, st["num"],
                color=st["accent"], fontsize=7.5, fontweight='bold', ha='center', va='center')

        # Header Title & Subtitle
        ax.text(x + 10.5, box_y + box_h - 1.8, st["title"],
                color="#ffffff", fontsize=9.2, fontweight='bold', ha='left', va='center')
        ax.text(x + box_w/2, box_y + box_h - 5.2, st["subtitle"],
                color="#e2e8f0", fontsize=7.8, fontstyle='italic', ha='center', va='center')

        # Items inside card
        cur_y = box_y + box_h - 10.5
        for b_title, b_desc in st["items"]:
            # bullet dot
            bullet = patches.Circle((x + 1.8, cur_y), 0.55, facecolor=st["header_color"], edgecolor="none")
            ax.add_patch(bullet)
            
            # bullet title
            ax.text(x + 3.0, cur_y + 0.3, b_title,
                    color="#0f172a", fontsize=7.8, fontweight='bold', ha='left', va='center')
            # bullet desc
            ax.text(x + 3.0, cur_y - 1.7, b_desc,
                    color="#475569", fontsize=6.8, ha='left', va='center')
            
            cur_y -= 5.3

        # Inter-stage connector arrows
        if idx < len(stages) - 1:
            arrow_x = x + box_w
            arrow_y = box_y + box_h / 2.0
            ax.annotate("", xy=(arrow_x + gap, arrow_y), xytext=(arrow_x, arrow_y),
                        arrowprops=dict(arrowstyle="-|>", color="#475569", lw=2.0, mutation_scale=15))

    # Bottom Closed-Loop Telemetry Banner
    bot_y = 2.2
    bot_h = 10.5
    bot_box = FancyBboxPatch((3.0, bot_y), 144.0, bot_h,
                             boxstyle="round,pad=0.2,rounding_size=0.9",
                             facecolor='#f1f5f9', edgecolor='#0369a1', linewidth=1.8, linestyle='--')
    ax.add_patch(bot_box)

    # Title of feedback loop
    loop_pill = FancyBboxPatch((5.0, bot_y + bot_h - 2.8), 28.0, 2.2,
                               boxstyle="round,pad=0.1,rounding_size=0.5",
                               facecolor="#0369a1", edgecolor="none")
    ax.add_patch(loop_pill)
    ax.text(19.0, bot_y + bot_h - 1.7, "CLOSED-LOOP TELEMETRY FEEDBACK",
            color="#ffffff", fontsize=7.5, fontweight='bold', ha='center', va='center')

    # Feedback Step nodes
    fb_nodes = [
        ("1. Realized Demand (y_t)", "Observe customer POS demand & warehouse shipments"),
        ("2. Lead-Time Variance (L_t)", "Track supplier receipt latency & transit variance"),
        ("3. Forecast Residual (e_t)", "Compute actual error: e_t = y_t - ŷ_t"),
        ("4. Trigg's Tracking Signal (TS_t)", "Detect non-stationary systematic forecast drift"),
        ("5. Dynamic Policy Calibration", "Recalibrate (s, Q) bounds, safety stocks, & order alerts")
    ]

    fb_start_x = 5.0
    fb_node_w = 25.5
    fb_gap = 3.6
    for f_idx, (fn_title, fn_desc) in enumerate(fb_nodes):
        nx = fb_start_x + f_idx * (fb_node_w + fb_gap)
        
        # Node box
        n_box = FancyBboxPatch((nx, bot_y + 1.2), fb_node_w, 5.8,
                               boxstyle="round,pad=0.15,rounding_size=0.6",
                               facecolor="#ffffff", edgecolor="#cbd5e1", linewidth=1.2)
        ax.add_patch(n_box)
        
        # Step label
        ax.text(nx + fb_node_w/2, bot_y + 5.2, fn_title,
                color="#0369a1", fontsize=7.4, fontweight='bold', ha='center', va='center')
        ax.text(nx + fb_node_w/2, bot_y + 2.8, fn_desc,
                color="#64748b", fontsize=6.3, ha='center', va='center')
        
        # Node connectors
        if f_idx < len(fb_nodes) - 1:
            ax.annotate("", xy=(nx + fb_node_w + fb_gap, bot_y + 4.1), xytext=(nx + fb_node_w, bot_y + 4.1),
                        arrowprops=dict(arrowstyle="-|>", color="#0284c7", lw=1.5, mutation_scale=11))

    # Vertical closed-loop connection arrows
    # Right exit arrow from Stage 5 down to Telemetry Banner
    ax.annotate("", xy=(135.0, bot_y + bot_h), xytext=(135.0, box_y),
                arrowprops=dict(arrowstyle="-|>", color="#0369a1", lw=2.0, linestyle="--", mutation_scale=15))
    ax.text(136.5, (bot_y + bot_h + box_y)/2, "Outcomes Streamed",
            color="#0369a1", fontsize=7.2, fontweight='bold', va='center')

    # Left return arrow from Telemetry Banner up to Stage 1
    ax.annotate("", xy=(16.0, box_y), xytext=(16.0, bot_y + bot_h),
                arrowprops=dict(arrowstyle="-|>", color="#0369a1", lw=2.0, linestyle="--", mutation_scale=15))
    ax.text(17.5, (bot_y + bot_h + box_y)/2, "Adaptive State Update",
            color="#0369a1", fontsize=7.2, fontweight='bold', va='center')

    plt.tight_layout()
    plt.savefig(output_path, dpi=350, bbox_inches='tight')
    plt.close()
    print(f"Enhanced methodology figure successfully saved to {output_path}!")

if __name__ == "__main__":
    generate_better_methodology_figure()
