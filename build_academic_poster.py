"""
build_academic_poster.py - Generates an Academic Conference Poster (PowerPoint)
formatted in widescreen / A0 landscape (48" x 36" / 4:3 high-impact poster layout)
for conference poster sessions (e.g., INFORMS, International Symposium on Forecasting - ISF).
"""

import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

def create_poster():
    prs = Presentation()
    # 48" x 36" standard academic poster dimensions
    prs.slide_width = Inches(48.0)
    prs.slide_height = Inches(36.0)
    blank_layout = prs.slide_layouts[6]
    
    slide = prs.slides.add_slide(blank_layout)
    
    # Palette
    NAVY = RGBColor(16, 44, 87)
    DARK_BLUE = RGBColor(26, 77, 138)
    CHARCOAL = RGBColor(40, 40, 40)
    LIGHT_BG = RGBColor(245, 248, 252)
    WHITE = RGBColor(255, 255, 255)
    GREEN = RGBColor(40, 167, 69)
    BORDER_GREY = RGBColor(210, 220, 235)
    
    # Background
    bg = slide.shapes.add_shape(1, Inches(0), Inches(0), Inches(48.0), Inches(36.0))
    bg.fill.solid()
    bg.fill.fore_color.rgb = LIGHT_BG
    bg.line.color.rgb = LIGHT_BG
    
    # Header Banner (Navy)
    banner = slide.shapes.add_shape(1, Inches(1.0), Inches(1.0), Inches(46.0), Inches(4.5))
    banner.fill.solid()
    banner.fill.fore_color.rgb = NAVY
    banner.line.color.rgb = DARK_BLUE
    
    tf_b = banner.text_frame
    tf_b.word_wrap = True
    tf_b.margin_left = Inches(1.0)
    tf_b.margin_top = Inches(0.4)
    
    p = tf_b.paragraphs[0]
    p.text = "An Empirically Audited Machine Learning and Stochastic Inventory Control Framework for Enterprise Replenishment Decision Support"
    p.font.name = "Arial"
    p.font.size = Pt(44)
    p.font.bold = True
    p.font.color.rgb = WHITE
    p.alignment = PP_ALIGN.CENTER
    
    p = tf_b.add_paragraph()
    p.text = "Sashank P., Abhinay Sai, Poli Naidu, and Sushama Rani Dutta | Department of Computer Science and Engineering, Koneru Lakshmaiah Education Foundation, Hyderabad, India | Email: sashank.p@klh.edu.in"
    p.font.name = "Arial"
    p.font.size = Pt(22)
    p.font.color.rgb = RGBColor(210, 230, 255)
    p.alignment = PP_ALIGN.CENTER
    p.space_before = Pt(16)
    
    def create_card(left, top, width, height, title):
        card = slide.shapes.add_shape(1, Inches(left), Inches(top), Inches(width), Inches(height))
        card.fill.solid()
        card.fill.fore_color.rgb = WHITE
        card.line.color.rgb = BORDER_GREY
        card.line.width = Pt(2)
        
        # Header inside card
        header = slide.shapes.add_shape(1, Inches(left), Inches(top), Inches(width), Inches(1.1))
        header.fill.solid()
        header.fill.fore_color.rgb = DARK_BLUE
        header.line.color.rgb = DARK_BLUE
        tf_h = header.text_frame
        tf_h.margin_left = Inches(0.4)
        p = tf_h.paragraphs[0]
        p.text = title.upper()
        p.font.name = "Arial"
        p.font.size = Pt(20)
        p.font.bold = True
        p.font.color.rgb = WHITE
        return card

    # COLUMN 1 (left = 1.0, width = 14.5)
    create_card(1.0, 6.0, 14.5, 13.5, "1. Motivation & Problem Statement")
    tb1 = slide.shapes.add_textbox(Inches(1.4), Inches(7.4), Inches(13.7), Inches(11.8))
    tf1 = tb1.text_frame
    tf1.word_wrap = True
    p = tf1.paragraphs[0]
    p.text = "The Supply Chain Penalty Asymmetry:"
    p.font.name = "Arial"
    p.font.size = Pt(22)
    p.font.bold = True
    p.font.color.rgb = NAVY
    
    pts1 = [
        "• Stockouts inflict catastrophic lost margins, SLA breach penalties, and permanent customer churn.",
        "• Excess buffer inventory incurs only marginal holding capital charges (~20%/yr). Shortage costs exceed holding costs by 10x-50x.",
        "• Conventional ERPs evaluate models on symmetric point loss (MAE/RMSE), ignoring error direction and penalizing over- and under-forecasting equally.",
        "• Popular competition transforms (Log1p, Tweedie) induce severe negative bias, starving replenishment buffers."
    ]
    for pt in pts1:
        p = tf1.add_paragraph()
        p.text = pt
        p.font.name = "Arial"
        p.font.size = Pt(17)
        p.font.color.rgb = CHARCOAL
        p.space_before = Pt(12)

    create_card(1.0, 20.0, 14.5, 14.5, "2. Theoretical Proofs & Derivations")
    tb2 = slide.shapes.add_textbox(Inches(1.4), Inches(21.4), Inches(13.7), Inches(12.8))
    tf2 = tb2.text_frame
    tf2.word_wrap = True
    theorems = [
        ("Theorem 1: Jensen's Inequality Deficit",
         "When models fit MSE on z = ln(1+y), naive exponentiation ŷ = exp(ẑ)-1 estimates conditional geometric median rather than conditional mean:\nE[ln(1+Y)] < ln(1+E[Y])  ⟹  ŷ_naive < E[Y]\nCauses -55.9% to -81.3% negative bias starving buffers."),
        ("Theorem 2: Tweedie Zero-Gradient Divergence",
         "For compound Poisson-Gamma deviance (p=1.5):\n∂d(0, μ)/∂μ = 2/√μ  ⟹  lim_{μ→0+} ∂d(0, μ)/∂μ = +∞\nInfinite gradient penalty at zero pulls predictions downward."),
        ("Theorem 3: Eve's Law of Total Variance",
         "Sizing safety stock across stochastic lead times L and residual forecast error variance σ_e^2:\nσ_{D_L}^2 = L̄ σ_e^2 + d̂^2 σ_L^2,    SS = Z_α σ_{D_L}\nLiberates 49.9% of safety buffer working capital.")
    ]
    for t_t, t_d in theorems:
        p = tf2.add_paragraph() if tf2.paragraphs[0].text else tf2.paragraphs[0]
        p.text = t_t
        p.font.name = "Arial"
        p.font.size = Pt(18)
        p.font.bold = True
        p.font.color.rgb = DARK_BLUE
        p.space_before = Pt(8)
        p = tf2.add_paragraph()
        p.text = t_d
        p.font.name = "Arial"
        p.font.size = Pt(15)
        p.font.color.rgb = CHARCOAL
        p.space_before = Pt(4)

    # COLUMN 2 (left = 16.5, width = 15.0)
    create_card(16.5, 6.0, 15.0, 13.5, "3. System Methodology & Workflow")
    if os.path.exists("paper_figures/fig1_methodology.png"):
        slide.shapes.add_picture("paper_figures/fig1_methodology.png", Inches(17.0), Inches(7.4), Inches(14.0))
    tb_m = slide.shapes.add_textbox(Inches(17.0), Inches(13.2), Inches(14.0), Inches(5.8))
    tf_m = tb_m.text_frame
    tf_m.word_wrap = True
    p = tf_m.paragraphs[0]
    p.text = "Key Pipeline Innovations:\n• Strictly chronological forward-filled prices eliminating lookahead leakage.\n• 3-fold expanding-window walk-forward validation on locked holdouts.\n• Platt-calibrated LightGBM Demand Surge Classifier (AUC = 0.928).\n• Closed-loop continuous-review (s, Q) replenishment governed by Trigg's tracking signal."
    p.font.name = "Arial"
    p.font.size = Pt(16)
    p.font.color.rgb = CHARCOAL

    create_card(16.5, 20.0, 15.0, 14.5, "4. Multi-Regime Holdout Evaluation")
    tb_b = slide.shapes.add_textbox(Inches(17.0), Inches(21.4), Inches(14.0), Inches(12.8))
    tf_b = tb_b.text_frame
    tf_b.word_wrap = True
    p = tf_b.paragraphs[0]
    p.text = "Cross-Regime Empirical Verification:"
    p.font.name = "Arial"
    p.font.size = Pt(20)
    p.font.bold = True
    p.font.color.rgb = NAVY
    b_txts = [
        "1. Walmart Weekly Sales (45 Stores):\n• Random Forest Winner: MAE $35,624.48 (R² = 0.9905, Bias = -0.28%).\n• 29.9% error reduction over Naive Lag-1 baseline ($50,865).",
        "2. Online Retail Daily Wholesale (541k Transactions):\n• Log1p XGBoost minimized point MAE (72.2 u) but suffered -57.18% bias.\n• Random Forest Champion: MAE 90.57 u with +3.14% bias, satisfying safeguard.",
        "3. M5 Intermittent Grocery (63.3% Zeros):\n• Croston SBA collapsed into -58.97% bias starvation.\n• XGBoost Tweedie had -11.87% bias.\n• Random Forest delivered robust anti-stockout replenishment buffers."
    ]
    for b in b_txts:
        p = tf_b.add_paragraph()
        p.text = b
        p.font.name = "Arial"
        p.font.size = Pt(16)
        p.font.color.rgb = CHARCOAL
        p.space_before = Pt(8)

    # COLUMN 3 (left = 32.5, width = 14.5)
    create_card(32.5, 6.0, 14.5, 13.5, "5. Longitudinal Policy Simulation")
    if os.path.exists("paper_figures/fig7_policy_distributions.png"):
        slide.shapes.add_picture("paper_figures/fig7_policy_distributions.png", Inches(33.0), Inches(7.4), Inches(13.5))
    tb_p = slide.shapes.add_textbox(Inches(33.0), Inches(12.5), Inches(13.5), Inches(6.5))
    tf_p = tb_p.text_frame
    tf_p.word_wrap = True
    p = tf_p.paragraphs[0]
    p.text = "Benchmark Across 8,000 Demand Steps:\n• Policy C (Proposed Bias-Constrained): 100.0% Fill Rate, 0 Stockout Days, $9,210 Operating Cost (92.9% reduction vs. Policy B).\n• Policy B (Min MAE): 81.40% Fill Rate, 24 Stockouts, $128,540 Cost.\n• Sizing buffers via Eve's Law liberated 49.9% of safety buffer working capital ($135.7k vs. $270.8k)."
    p.font.name = "Arial"
    p.font.size = Pt(16)
    p.font.color.rgb = CHARCOAL

    create_card(32.5, 20.0, 14.5, 14.5, "6. Telemetry & Enterprise Takeaways")
    if os.path.exists("paper_figures/fig9_telemetry_trajectory.png"):
        slide.shapes.add_picture("paper_figures/fig9_telemetry_trajectory.png", Inches(33.0), Inches(21.4), Inches(13.5))
    tb_t = slide.shapes.add_textbox(Inches(33.0), Inches(27.8), Inches(13.5), Inches(6.2))
    tf_t = tb_t.text_frame
    tf_t.word_wrap = True
    p = tf_t.paragraphs[0]
    p.text = "Key Enterprise Recommendations:\n1. Never select replenishment models purely by point error (MAE/RMSE).\n2. Mandate operational percentage bias bounds (|Bias %| ≤ 5.0%).\n3. Size buffers strictly from residual error variance (σ_e) via Eve's Law.\n4. Deploy Trigg's tracking signal for real-time drift detection.\n5. Full code & 18-section notebook runnable on Google Colab (90/90 tests passing)."
    p.font.name = "Arial"
    p.font.size = Pt(15.5)
    p.font.color.rgb = CHARCOAL

    poster_path = "Academic_Conference_Poster.pptx"
    prs.save(poster_path)
    print(f"Successfully generated {poster_path} ({os.path.getsize(poster_path)} bytes)!")

if __name__ == "__main__":
    create_poster()
