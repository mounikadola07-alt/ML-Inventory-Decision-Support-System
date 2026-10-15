"""
build_presentation_deck.py - Generates an executive academic presentation deck (PowerPoint)
for the research paper: An Empirically Audited Machine Learning and Stochastic Inventory Control
Framework for Enterprise Replenishment Decision Support.
"""

import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

def create_deck():
    prs = Presentation()
    prs.slide_width = Inches(13.333) # 16:9 Widescreen
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]
    
    # Color palette
    NAVY = RGBColor(16, 44, 87)
    DARK_BLUE = RGBColor(26, 77, 138)
    CHARCOAL = RGBColor(40, 40, 40)
    LIGHT_GREY = RGBColor(245, 247, 250)
    ACCENT_GREEN = RGBColor(40, 167, 69)
    WHITE = RGBColor(255, 255, 255)
    
    def add_header(slide, title_text, category_text="RESEARCH PRESENTATION"):
        # Header banner
        header_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.733), Inches(1.1))
        tf = header_box.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0
        
        p_cat = tf.paragraphs[0]
        p_cat.text = category_text.upper()
        p_cat.font.name = "Arial"
        p_cat.font.size = Pt(10)
        p_cat.font.bold = True
        p_cat.font.color.rgb = DARK_BLUE
        
        p_title = tf.add_paragraph()
        p_title.text = title_text
        p_title.font.name = "Arial"
        p_title.font.size = Pt(22)
        p_title.font.bold = True
        p_title.font.color.rgb = NAVY
        p_title.space_before = Pt(3)

    # ------------------ SLIDE 1: Title Slide ------------------
    slide1 = prs.slides.add_slide(blank_layout)
    bg1 = slide1.shapes.add_shape(1, Inches(0), Inches(0), Inches(13.333), Inches(7.5)) # Rectangle
    bg1.fill.solid()
    bg1.fill.fore_color.rgb = NAVY
    bg1.line.color.rgb = NAVY
    
    tb1 = slide1.shapes.add_textbox(Inches(1.2), Inches(1.5), Inches(11.0), Inches(4.5))
    tf1 = tb1.text_frame
    tf1.word_wrap = True
    
    p = tf1.paragraphs[0]
    p.text = "RESEARCH DEFENSE & EXECUTIVE BRIEFING"
    p.font.name = "Arial"
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = RGBColor(120, 180, 255)
    
    p = tf1.add_paragraph()
    p.text = "An Empirically Audited Machine Learning and Stochastic Inventory Control Framework for Enterprise Replenishment Decision Support"
    p.font.name = "Arial"
    p.font.size = Pt(26)
    p.font.bold = True
    p.font.color.rgb = WHITE
    p.space_before = Pt(14)
    p.space_after = Pt(24)
    
    p = tf1.add_paragraph()
    p.text = "Authors: Dola Mounika\nDepartment of Computer Science and Engineering, Koneru Lakshmaiah Education Foundation\nHyderabad, Telangana, India | Email: mounika.dola07@gmail.com"
    p.font.name = "Arial"
    p.font.size = Pt(13)
    p.font.color.rgb = RGBColor(220, 230, 245)

    # ------------------ SLIDE 2: Problem & Research Gap ------------------
    slide2 = prs.slides.add_slide(blank_layout)
    add_header(slide2, "The Operational Dilemma: Why Statistical Point Accuracy Fails")
    
    tb2 = slide2.shapes.add_textbox(Inches(0.8), Inches(1.8), Inches(11.733), Inches(5.0))
    tf2 = tb2.text_frame
    tf2.word_wrap = True
    
    points2 = [
        ("Fundamental Supply Chain Penalty Asymmetry", 
         "• Stockouts inflict catastrophic lost margins, contract default penalties, and permanent customer churn.\n• Over-buffering incurs marginal capital holding charges (~20%/yr). Shortage costs exceed holding costs by 10x-50x."),
        ("The Conventional ERP Flaw (Point Loss Leaderboards)",
         "• Standard ML workflows evaluate algorithms strictly on symmetric loss functions: MAE, RMSE, and MAPE.\n• Symmetric loss treats a -10 unit under-forecast (stockout disaster) identically to a +10 unit over-forecast (minor buffer)."),
        ("Theoretical Traps in ML Competitions",
         "• Naive Log1p transforms introduce Jensen's Inequality deficits, causing -55.9% to -81.3% under-forecasting bias.\n• Tweedie compound Poisson-Gamma deviance diverges at zero (∂d/∂μ → ∞), driving negative drift (-11.9% to -18.0%)."),
        ("The Proposed Solution (Closed-Loop Inventory Decision Support)",
         "• A Tri-Criterion Operational Selection Rule (|Bias %| ≤ 5.0% constraint before error minimization).\n• Stochastic Continuous-Review (s, Q) replenishment sizing via Eve's Law of Total Variance.")
    ]
    for title, desc in points2:
        p_t = tf2.add_paragraph() if tf2.paragraphs[0].text else tf2.paragraphs[0]
        p_t.text = title
        p_t.font.name = "Arial"
        p_t.font.size = Pt(14)
        p_t.font.bold = True
        p_t.font.color.rgb = DARK_BLUE
        p_t.space_before = Pt(8)
        
        p_d = tf2.add_paragraph()
        p_d.text = desc
        p_d.font.name = "Arial"
        p_d.font.size = Pt(11)
        p_d.font.color.rgb = CHARCOAL
        p_d.space_before = Pt(2)

    # ------------------ SLIDE 3: System Methodology & Architecture ------------------
    slide3 = prs.slides.add_slide(blank_layout)
    add_header(slide3, "System Architecture: End-to-End Closed-Loop Pipeline")
    
    if os.path.exists("paper_figures/fig1_methodology.png"):
        slide3.shapes.add_picture("paper_figures/fig1_methodology.png", Inches(0.8), Inches(1.8), Inches(11.733))
        
    tb3 = slide3.shapes.add_textbox(Inches(0.8), Inches(5.8), Inches(11.733), Inches(1.3))
    tf3 = tb3.text_frame
    tf3.word_wrap = True
    p3 = tf3.paragraphs[0]
    p3.text = "Key Pipeline Innovations: (1) Leakage-free expanding-window walk-forward validation; (2) Forward-filled price tracking eliminating lookahead leaks; (3) Multi-regime tree ensembles (LightGBM, XGBoost, Random Forest); (4) Platt-calibrated demand surge classification; (5) Closed-loop continuous-review (s, Q) replenishment engine governed by Trigg's tracking signal."
    p3.font.name = "Arial"
    p3.font.size = Pt(11)
    p3.font.color.rgb = CHARCOAL

    # ------------------ SLIDE 4: Mathematical Formulations ------------------
    slide4 = prs.slides.add_slide(blank_layout)
    add_header(slide4, "Formal Mathematical Formulations and Theoretical Proofs")
    
    tb4 = slide4.shapes.add_textbox(Inches(0.8), Inches(1.8), Inches(11.733), Inches(5.2))
    tf4 = tb4.text_frame
    tf4.word_wrap = True
    
    theorems = [
        ("Theorem 1: Jensen's Inequality Retransformation Deficit",
         "When models fit MSE on logarithmic targets z = ln(1+y), naive exponentiation ŷ = exp(ẑ) - 1 estimates the conditional geometric median rather than the conditional mean:\nE[ln(1 + Y)] < ln(1 + E[Y])  ⟹  ŷ_naive = exp(ẑ) - 1 < E[Y]\nResult: Systematic negative bias (-55.9% to -81.3%) starving continuous-review replenishment buffers."),
        ("Theorem 2: Tweedie Zero-Gradient Divergence at Zero Demand",
         "For compound Poisson-Gamma deviance with power parameter p ∈ (1, 2) (p = 1.5):\n∂d(0, μ) / ∂μ = 2 / √μ  ⟹  lim_{μ → 0+} ∂d(0, μ) / ∂μ = +∞\nResult: Infinite gradient penalty at zero pushes positive predictions downward, forcing -11.9% under-forecasting."),
        ("Theorem 3: Stochastic Lead-Time Demand Sizing via Eve's Law of Total Variance",
         "Applying the Law of Total Variance across stochastic lead times L and residual forecast error variance σ_e^2:\nσ_{D_L}^2 = L̄ σ_e^2 + d̂^2 σ_L^2,    SS = Z_α σ_{D_L},    ROP = d̂ L̄ + SS,    EOQ = √((2 K D̂) / h)\nResult: Sizing buffers strictly from residual forecast error (σ_e) liberates 49.9% of safety buffer working capital.")
    ]
    for t_title, t_desc in theorems:
        p_t = tf4.add_paragraph() if tf4.paragraphs[0].text else tf4.paragraphs[0]
        p_t.text = t_title
        p_t.font.name = "Arial"
        p_t.font.size = Pt(13)
        p_t.font.bold = True
        p_t.font.color.rgb = DARK_BLUE
        p_t.space_before = Pt(6)
        
        p_d = tf4.add_paragraph()
        p_d.text = t_desc
        p_d.font.name = "Arial"
        p_d.font.size = Pt(10.5)
        p_d.font.color.rgb = CHARCOAL
        p_d.space_before = Pt(2)

    # ------------------ SLIDE 5: Benchmark Results ------------------
    slide5 = prs.slides.add_slide(blank_layout)
    add_header(slide5, "Empirical Results: Multi-Regime Holdout Evaluation")
    
    tb5 = slide5.shapes.add_textbox(Inches(0.8), Inches(1.8), Inches(5.6), Inches(5.0))
    tf5 = tb5.text_frame
    tf5.word_wrap = True
    
    p = tf5.paragraphs[0]
    p.text = "Cross-Regime Benchmark Findings:"
    p.font.name = "Arial"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = DARK_BLUE
    
    b_pts = [
        "Walmart Weekly Store Sales ($):\n• Random Forest achieved lowest holdout MAE of $35,624.48 (R² = 0.9905) with tight bias of -0.28%.\n• Outperformed persistence baselines (MAE $50,865) by 29.9%.",
        "Online Retail Daily Wholesale Units:\n• Log1p XGBoost minimized raw MAE (72.2 u) but failed with -57.18% bias.\n• LightGBM Tweedie had -12.73% bias.\n• Tuned Random Forest (MAE 90.57 u, +3.14% bias) satisfied Policy C safeguard.",
        "M5 Intermittent Grocery Sales (63.3% Zeros):\n• Croston SBA suffered -58.97% bias starvation.\n• XGBoost Tweedie under-forecast by -11.87%.\n• Random Forest provided robust anti-stockout buffer."
    ]
    for pt in b_pts:
        p = tf5.add_paragraph()
        p.text = pt
        p.font.name = "Arial"
        p.font.size = Pt(11)
        p.font.color.rgb = CHARCOAL
        p.space_before = Pt(6)
        
    if os.path.exists("paper_figures/fig7_policy_distributions.png"):
        slide5.shapes.add_picture("paper_figures/fig7_policy_distributions.png", Inches(6.6), Inches(2.0), Inches(5.9))

    # ------------------ SLIDE 6: Policy Benchmarking ------------------
    slide6 = prs.slides.add_slide(blank_layout)
    add_header(slide6, "Replenishment Simulation: Policy Performance Across 8,000 Steps")
    
    if os.path.exists("paper_figures/fig7_policy_distributions.png"):
        slide6.shapes.add_picture("paper_figures/fig7_policy_distributions.png", Inches(0.8), Inches(1.8), Inches(6.0))
    if os.path.exists("paper_figures/fig8_surge_roc.png"):
        slide6.shapes.add_picture("paper_figures/fig8_surge_roc.png", Inches(7.0), Inches(1.8), Inches(5.5))
        
    tb6 = slide6.shapes.add_textbox(Inches(0.8), Inches(5.6), Inches(11.733), Inches(1.5))
    tf6 = tb6.text_frame
    tf6.word_wrap = True
    p6 = tf6.paragraphs[0]
    p6.text = "• Policy C (Bias-Constrained Champion): 100.0% Cycle Fill Rate, 0 Out-of-Stock Days, $9,210 Operating Cost (92.9% reduction vs. Policy B).\n• Policy B (Statistical Min MAE): 81.40% Fill Rate, 24 Stockout Days, $128,540 Operating Cost (Catastrophic Replenishment Starvation).\n• Surge Early Warning Classifier: Calibrated LightGBM achieved ROC AUC = 0.9278, Accuracy = 83.21%, Brier Score = 0.084."
    p6.font.name = "Arial"
    p6.font.size = Pt(11)
    p6.font.bold = True
    p6.font.color.rgb = NAVY

    # ------------------ SLIDE 7: Real-Time Telemetry & Conclusion ------------------
    slide7 = prs.slides.add_slide(blank_layout)
    add_header(slide7, "Real-Time Telemetry, Conclusions & Enterprise Recommendations")
    
    if os.path.exists("paper_figures/fig9_telemetry_trajectory.png"):
        slide7.shapes.add_picture("paper_figures/fig9_telemetry_trajectory.png", Inches(0.8), Inches(1.8), Inches(5.8))
        
    tb7 = slide7.shapes.add_textbox(Inches(6.8), Inches(1.8), Inches(5.7), Inches(5.2))
    tf7 = tb7.text_frame
    tf7.word_wrap = True
    
    p = tf7.paragraphs[0]
    p.text = "Strategic Insights & Key Takeaways:"
    p.font.name = "Arial"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = DARK_BLUE
    
    c_pts = [
        "1. Decouple ML Training from Evaluation:\nNever select enterprise replenishment models purely on symmetric statistical point metrics (MAE/RMSE).",
        "2. Enforce Operational Bias Safeguards:\nMandate a strict percentage bias constraint (|Bias %| ≤ 5.0%) before ranking models by point error.",
        "3. Capital Optimization via Eve's Law:\nDerive safety stock buffers from residual forecast error variance (σ_e), slashing redundant working capital by 49.9%.",
        "4. Telemetry Exception Streaming:\nEmploy Trigg's tracking signal to trigger automated model re-estimation before bias drift causes stockouts.",
        "5. Reproducibility:\nFull suite of 90 unit/regression tests passing, runnable on Google Colab."
    ]
    for pt in c_pts:
        p = tf7.add_paragraph()
        p.text = pt
        p.font.name = "Arial"
        p.font.size = Pt(10.5)
        p.font.color.rgb = CHARCOAL
        p.space_before = Pt(4)

    # Save presentation
    deck_path = "Research_Presentation_Deck.pptx"
    prs.save(deck_path)
    print(f"Successfully generated {deck_path} ({os.path.getsize(deck_path)} bytes)!")

if __name__ == "__main__":
    create_deck()
