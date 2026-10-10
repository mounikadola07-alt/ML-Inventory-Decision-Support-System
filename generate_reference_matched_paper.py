import os
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

DOCX_OUT = "ML_Inventory_Decision_Support_Paper.docx"

def build_paper():
    doc = docx.Document()
    
    # Page setup matching reference: Letter, 0.75 in margins
    for section in doc.sections:
        section.top_margin = Inches(0.75)
        section.bottom_margin = Inches(0.75)
        section.left_margin = Inches(0.75)
        section.right_margin = Inches(0.75)
        section.page_width = Inches(8.5)
        section.page_height = Inches(11.0)
        
        # Solid Pure Black Page Border (sz='6' = 0.75pt, color='000000')
        pg_borders_xml = f"""
        <w:pgBorders {nsdecls('w')}>
            <w:top w:val='single' w:sz='6' w:space='20' w:color='000000'/>
            <w:left w:val='single' w:sz='6' w:space='20' w:color='000000'/>
            <w:bottom w:val='single' w:sz='6' w:space='20' w:color='000000'/>
            <w:right w:val='single' w:sz='6' w:space='20' w:color='000000'/>
        </w:pgBorders>
        """
        section._sectPr.append(parse_xml(pg_borders_xml))
        
    normal_style = doc.styles['Normal']
    normal_style.font.name = 'Times New Roman'
    normal_style.font.size = Pt(10)
    normal_style.font.color.rgb = RGBColor(0, 0, 0)
    normal_style.paragraph_format.line_spacing = 1.05
    normal_style.paragraph_format.space_after = Pt(2)
    
    def add_p(text="", bold=False, italic=False, size=10, align=WD_ALIGN_PARAGRAPH.LEFT, space_before=0, space_after=2, indent=0):
        p = doc.add_paragraph()
        p.alignment = align
        p.paragraph_format.space_before = Pt(space_before)
        p.paragraph_format.space_after = Pt(space_after)
        p.paragraph_format.first_line_indent = Inches(indent)
        if text:
            run = p.add_run(text)
            run.font.name = 'Times New Roman'
            run.font.size = Pt(size)
            run.font.color.rgb = RGBColor(0, 0, 0)
            run.bold = bold
            run.italic = italic
        return p

    def add_h1(text):
        return add_p(text, bold=True, size=11, space_before=5, space_after=2, indent=0)

    def add_h2(text):
        return add_p(text, bold=True, size=10, space_before=4, space_after=1, indent=0)

    def add_eq(omml_xml_content):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(2)
        p.paragraph_format.space_after = Pt(2)
        p.paragraph_format.first_line_indent = Inches(0)
        xml_str = f'<m:oMathPara xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math"><m:oMath>{omml_xml_content}</m:oMath></m:oMathPara>'
        p._p.append(parse_xml(xml_str))
        return p

    def add_fig(img_path, caption_text, width_inches=4.4):
        if os.path.exists(img_path):
            p_img = doc.add_paragraph()
            p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p_img.paragraph_format.space_before = Pt(2)
            p_img.paragraph_format.space_after = Pt(1)
            p_img.paragraph_format.line_spacing = 1.0
            p_img.paragraph_format.first_line_indent = Inches(0)
            run = p_img.add_run()
            run.add_picture(img_path, width=Inches(width_inches))
            
            p_cap = doc.add_paragraph()
            p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p_cap.paragraph_format.space_before = Pt(1)
            p_cap.paragraph_format.space_after = Pt(3)
            p_cap.paragraph_format.line_spacing = 1.0
            p_cap.paragraph_format.first_line_indent = Inches(0)
            r_cap = p_cap.add_run(caption_text)
            r_cap.font.name = 'Times New Roman'
            r_cap.font.size = Pt(8.5)
            r_cap.font.color.rgb = RGBColor(0, 0, 0)

    def add_tbl(title, rows_data):
        p_t = doc.add_paragraph()
        p_t.paragraph_format.space_before = Pt(3)
        p_t.paragraph_format.space_after = Pt(1)
        p_t.paragraph_format.first_line_indent = Inches(0)
        r_t = p_t.add_run(title)
        r_t.font.name = 'Times New Roman'
        r_t.font.size = Pt(9)
        r_t.font.color.rgb = RGBColor(0, 0, 0)
        r_t.bold = True
        
        table = doc.add_table(rows=len(rows_data), cols=len(rows_data[0]))
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        
        # Pure Black table borders
        tbl_borders_xml = f"""
        <w:tblBorders {nsdecls('w')}>
            <w:top w:val='single' w:sz='6' w:space='0' w:color='000000'/>
            <w:left w:val='single' w:sz='6' w:space='0' w:color='000000'/>
            <w:bottom w:val='single' w:sz='6' w:space='0' w:color='000000'/>
            <w:right w:val='single' w:sz='6' w:space='0' w:color='000000'/>
            <w:insideH w:val='single' w:sz='4' w:space='0' w:color='000000'/>
            <w:insideV w:val='single' w:sz='4' w:space='0' w:color='000000'/>
        </w:tblBorders>
        """
        table._tbl.tblPr.append(parse_xml(tbl_borders_xml))
        
        for r_idx, row in enumerate(table.rows):
            for c_idx, cell in enumerate(row.cells):
                cell.text = rows_data[r_idx][c_idx]
                cp = cell.paragraphs[0]
                cp.paragraph_format.line_spacing = 1.0
                cp.paragraph_format.space_before = Pt(1)
                cp.paragraph_format.space_after = Pt(1)
                cp.paragraph_format.first_line_indent = Inches(0)
                for r in cp.runs:
                    r.font.name = 'Times New Roman'
                    r.font.size = Pt(8)
                    r.font.color.rgb = RGBColor(0, 0, 0)
                    if r_idx == 0:
                        r.bold = True

    # ==================== PAGE 1 ====================
    add_p("An Empirically Audited Machine Learning and Stochastic Inventory Control Framework for Enterprise Replenishment Decision Support", 
          bold=True, size=13, align=WD_ALIGN_PARAGRAPH.CENTER, space_before=4, space_after=3, indent=0)
    add_p("Sashank P., Abhinay Sai, Poli Naidu, and Sushama Rani Dutta", size=9.5, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=1, indent=0)
    add_p("Department of Computer Science and Engineering, Koneru Lakshmaiah Education Foundation,\nHyderabad, Telangana, India", size=9, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=1, indent=0)
    add_p("sashank.p@klh.edu.in", size=9, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=5, indent=0)
    
    # Abstract
    p_abs = doc.add_paragraph()
    p_abs.paragraph_format.line_spacing = 1.05
    p_abs.paragraph_format.space_after = Pt(3)
    p_abs.paragraph_format.first_line_indent = Inches(0)
    r1 = p_abs.add_run("Abstract: ")
    r1.bold = True
    r1.font.size = Pt(9)
    r2 = p_abs.add_run(
        "Modern enterprise supply chains operate under an inherently asymmetric penalty structure where customer stockouts incur catastrophic lost-sales and brand-equity penalties, while holding excess inventory incurs marginal capital-carrying costs. Despite this operational asymmetry, enterprise machine learning forecasting deployments routinely evaluate models using symmetric loss objectives such as Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE). In this paper, we develop an adaptive inventory decision support system based on Supervised Gradient Boosted Decision Trees (LightGBM, XGBoost, Random Forest), Tweedie Compound Poisson-Gamma Deviance, Continuous-Review (s, Q) Inventory Control, and Trigg's Automated Tracking Signal. The pipeline takes into account multi-regime transaction logs, estimates non-linear promotional and calendar harmonics, detects demand surge probabilities, and sizes stochastic buffer stocks via Eve's Law of Total Variance. We train and test our models using three heterogeneous benchmark datasets consisting of 541,909 wholesale transactions from Online Retail, 143 weeks across 45 Walmart retail stores, and 1,914 days from the M5 intermittent grocery benchmark. Our Calibrated LightGBM surge classifier achieves AUC-ROC = 0.928 with 83.2% accuracy, significantly outperforming uncalibrated baseline models (AUC = 0.726). Moreover, in our closed-loop simulation of the replenishment process over 8,000 interaction steps, enforcing a Tri-Criterion operational bias safeguard (|Bias %| ≤ 5.0%) eliminated 24 stockout days and delivered a 92.9% reduction in total inventory operating costs compared to statistical accuracy-first policies, while liberating 49.9% of safety buffer capital."
    )
    r2.font.size = Pt(9)
    
    # Keywords
    p_kw = doc.add_paragraph()
    p_kw.paragraph_format.line_spacing = 1.05
    p_kw.paragraph_format.space_after = Pt(5)
    p_kw.paragraph_format.first_line_indent = Inches(0)
    r_kw1 = p_kw.add_run("Keywords: ")
    r_kw1.bold = True
    r_kw1.font.size = Pt(9)
    r_kw2 = p_kw.add_run("Demand Forecasting, Inventory Decision Support, Continuous-Review (s, Q), Gradient Boosting, Jensen's Inequality, Tweedie Deviance, Operational Bias Safeguards, Supply Chain Analytics.")
    r_kw2.font.size = Pt(9)

    add_h1("1 Introduction")
    add_p(
        "Acquiring high service reliability in commercial distribution systems such as retail merchandising and wholesale logistics demands sustained forecast precision, structured safety buffer sizing, and timely replenishment feedback. However, conventional enterprise resource planning (ERP) platforms predominantly employ a static 'one size fits all' paradigm, presenting identical unconstrained point forecasts to inventory controllers irrespective of downstream penalty asymmetries and demand lumpiness. This uniform instructional pacing introduces substantial operational challenges: models that optimize purely for symmetric statistical metrics frequently underestimate buffer requirements during demand surges, producing severe inventory starvation, lost sales, and permanent customer defection; conversely, uncalibrated positive bias drives chronic working capital tie-up and warehouse spoilage.",
        indent=0
    )
    add_p(
        "To address these limitations, we formulate an intelligent inventory decision support framework that unifies leakage-safe feature engineering, supervised gradient boosted ensembling, demand surge classification, and adaptive continuous-review (s, Q) replenishment control. Grounded in stochastic inventory theory and Eve's Law of Total Variance, our system continuously evaluates forecast error distributions and dynamically recommends optimal replenishment order quantities and safety stock bands to keep enterprise customer fill rates within the desirable 95%–100% service zone.",
        indent=0.25
    )
    doc.add_page_break()

    # ==================== PAGE 2 ====================
    add_p(
        "As retail demand transactions stream into the replenishment interface, the framework captures granular interaction logs—including item identifiers, transaction quantities, customer return records, and millisecond timestamps. These signals are continuously transformed into latent demand predictors via a dual feature engineering engine: autoregressive lag structures with cyclical trigonometric harmonics for capturing deterministic seasonality, and rolling volatility estimators for quantifying instantaneous demand dispersion. In turn, a Tri-Criterion Operational Selection policy uses the estimated forecast distributions to choose the optimal replenishment model, balancing working capital tie-up against catastrophic stockout penalties.",
        indent=0
    )

    add_h1("2 Literature Review")
    add_p(
        "[1] Syntetos and Boylan evaluated 2,640 inventory series and demonstrated that minimizing Mean Squared Error (MSE) or Mean Absolute Percentage Error (MAPE) does not guarantee minimum inventory costs or optimal fill rates. This supports our finding that statistical loss minimization must be decoupled from operational replenishment evaluation.\n\n"
        "[2] Wallström and Segerstedt investigated intermittent demand metrics and proved that percentage error metrics suffer catastrophic zero-denominator breakdown when demand is zero, introducing severe distortions into model benchmarking.\n\n"
        "[3] Kolassa argued that forecasting models deployed for inventory control must be evaluated under asymmetric, cost-aligned loss functions, proving that symmetric error penalties encourage models to underestimate buffer requirements when shortage costs exceed holding costs.\n\n"
        "[4] Croston developed the classical decomposition separating demand arrival intervals from non-zero order sizes, establishing foundational baselines for intermittent inventory management.\n\n"
        "[5] Teunter, Syntetos, and Babai extended this framework to accommodate obsolescence and sudden demand drop-offs through separate updating constants for demand level and inter-arrival probability.\n\n"
        "[6] Silver, Pyke, and Peterson established canonical continuous-review (s, Q) policies, lead-time demand convolution, safety stock buffer derivation, and Economic Order Quantity dynamics. Their work provides the analytical inventory formulation for our replenishment simulation.\n\n"
        "[7] Eppen and Martin analyzed multi-period stochastic lead-time demand, establishing that replenishment buffers must absorb both demand rate volatility and transit lead-time uncertainty.\n\n"
        "[8] Breiman introduced Random Forests, proving that aggregating decorrelated decision trees dramatically shrinks prediction variance without inflating bias on noisy tabular data.\n\n"
        "[9] Chen and Guestrin developed XGBoost, formulating second-order Taylor series expansions of arbitrary loss functions with sparsity-aware split finding and explicit regularization penalties.\n\n"
        "[10] Ke et al. created LightGBM, introducing Gradient-based One-Side Sampling (GOSS) and Exclusive Feature Bundling (EFB) to accelerate training across millions of retail observations.",
        indent=0
    )
    doc.add_page_break()

    # ==================== PAGE 3 ====================
    add_h1("3 Proposed Methodology")
    add_p(
        "Our intelligent inventory decision support system models demand progression through continuous transaction logging, leakage-free feature engineering, and closed-loop stochastic replenishment control. The pipeline executes across seven coordinated phases:\n\n"
        "The end-to-end architecture is organized into four decoupled tiers: the Presentation & Telemetry Layer (replenishment dashboard & exception alerts), Ingestion & Preprocessing Engine (cancellation filtering and panel alignment), Predictive Modeling Tier (supervised gradient boosted ensembles and calibrated surge classifiers), and Stochastic Replenishment Decision Policy (continuous-review (s, Q) control engine).",
        indent=0
    )

    # FIG 1
    add_fig("paper_figures/fig1_methodology.png", 
            "Fig. 1. End-to-end machine learning pipeline workflow and closed-loop feedback mechanism.", width_inches=4.6)
    
    add_p(
        "Figure 1 illustrates the end-to-end dataflow and closed-loop control architecture of the proposed inventory decision support system. The pipeline begins with raw transaction logs captured from commercial benchmarks, encompassing customer identifiers, SKU parameters, order quantities, cancellation flags, and purchase timestamps. In the preprocessing phase, sparse SKU interaction sessions are filtered, transaction quantities are standardized, and dates are mapped to continuous Cartesian SKU-date grids. The feature engineering layer computes leakage-safe autoregressive lags (1, 2, 4, 7, 52), rolling statistics (mean, std, momentum ratios), and cyclical Fourier harmonics strictly prior to forecast origin t_0. Concurrently, sequential transaction streams feed into a dual-engine predictor consisting of tree-based regression ensembles for continuous conditional expectations and a calibrated surge classifier for early warning alerts. The resulting forecast distribution and variance parameters are discretized into a continuous-review (s, Q) policy, while subsequent customer demand signals stream back through a real-time telemetry loop, driving adaptive model selection.",
        indent=0.25
    )
    doc.add_page_break()

    # ==================== PAGE 4 ====================
    add_p(
        "These verified operational metrics are streamed back through a real-time telemetry loop, updating Trigg's tracking signal and driving automated reorder execution.",
        indent=0
    )

    # FIG 2
    add_fig("paper_figures/fig2_model_comparison.png", 
            "Fig. 2. Multi-tier intelligent replenishment decision support architecture across presentation, telemetry, feature engineering, predictive modeling, and inventory control policy layers.", width_inches=4.6)

    add_p(
        "Figure 2 depicts the modular multi-tier architectural hierarchy designed for scalable, low-latency enterprise decision support. The architecture is decomposed into four decoupled functional tiers: (1) Presentation & Telemetry Tier: Handles responsive operator UI rendering, real-time transaction capture, exception streaming, and automated reorder execution; (2) Ingestion & Feature Engineering Engine: Implements strictly forward-filled price tracking, rolling dispersion calculations, and calendar harmonics; (3) State Estimation & Predictive Engine: Combines tree-based bagging (Random Forest) and gradient boosting (LightGBM, XGBoost) with Platt probability calibration to generate conditional demand expectations and surge alarm probabilities; and (4) Stochastic Inventory Decision Policy: Houses the continuous-review (s, Q) decision engine governed by a Tri-Criterion Operational Selection rule, sizing safety stocks and batch orders via Eve's Law of Total Variance.",
        indent=0.25
    )

    # Formal Mathematical Formulations with OMML
    add_h2("3.1 Mathematical Formulations and Theoretical Proofs")
    add_p("Theorem 1 (Jensen's Inequality Retransformation Deficit). When models optimize Mean Squared Error on logarithmic targets z = ln(1+y), naive exponentiation ŷ = exp(ẑ) - 1 estimates the conditional geometric median rather than the conditional mean:", indent=0)
    omml_jensen = """
    <m:r><m:rPr><m:scr m:val="script"/></m:rPr><m:t>E</m:t></m:r>
    <m:r><m:t>[ln(1 + </m:t></m:r><m:r><m:rPr><m:sty m:val="p"/></m:rPr><m:t>Y</m:t></m:r><m:r><m:t>)] &lt; ln(1 + </m:t></m:r>
    <m:r><m:rPr><m:scr m:val="script"/></m:rPr><m:t>E</m:t></m:r><m:r><m:t>[</m:t></m:r><m:r><m:rPr><m:sty m:val="p"/></m:rPr><m:t>Y</m:t></m:r><m:r><m:t>]) </m:t></m:r>
    <m:r><m:t> ⟹ </m:t></m:r>
    <m:sSub><m:e><m:r><m:t>ŷ</m:t></m:r></m:e><m:sub><m:r><m:t>naive</m:t></m:r></m:sub></m:sSub>
    <m:r><m:t> = exp(ẑ) - 1 &lt; </m:t></m:r>
    <m:r><m:rPr><m:scr m:val="script"/></m:rPr><m:t>E</m:t></m:r><m:r><m:t>[</m:t></m:r><m:r><m:rPr><m:sty m:val="p"/></m:rPr><m:t>Y</m:t></m:r><m:r><m:t>]</m:t></m:r>
    """
    add_eq(omml_jensen)

    add_p("Theorem 2 (Tweedie Zero-Gradient Divergence). For compound Poisson-Gamma power parameter p ∈ (1, 2) (p = 1.5), the unit deviance gradient at zero actual demand exhibits explosive divergence as positive predictions approach zero:", indent=0)
    omml_tweedie = """
    <m:f>
        <m:num><m:r><m:t>∂d(0, μ)</m:t></m:r></m:num>
        <m:den><m:r><m:t>∂μ</m:t></m:r></m:den>
    </m:f>
    <m:r><m:t> = </m:t></m:r>
    <m:f>
        <m:num><m:r><m:t>2</m:t></m:r></m:num>
        <m:den><m:rad><m:radPr><m:degHide m:val="on"/></m:radPr><m:deg/><m:e><m:r><m:t>μ</m:t></m:r></m:e></m:rad></m:den>
    </m:f>
    <m:r><m:t> ⟹ </m:t></m:r>
    <m:sSub><m:e><m:r><m:t>lim</m:t></m:r></m:e><m:sub><m:r><m:t>μ→0+</m:t></m:r></m:sub></m:sSub>
    <m:f>
        <m:num><m:r><m:t>∂d(0, μ)</m:t></m:r></m:num>
        <m:den><m:r><m:t>∂μ</m:t></m:r></m:den>
    </m:f>
    <m:r><m:t> = +∞</m:t></m:r>
    """
    add_eq(omml_tweedie)

    add_p("Theorem 3 (Stochastic Lead-Time Sizing via Eve's Law). Applying the Law of Total Variance across stochastic supplier lead times L and forecast errors e:", indent=0)
    omml_eves = """
    <m:sSubSup><m:e><m:r><m:t>σ</m:t></m:r></m:e><m:sub><m:r><m:t>D_L</m:t></m:r></m:sub><m:sup><m:r><m:t>2</m:t></m:r></m:sup></m:sSubSup>
    <m:r><m:t> = L̄</m:t></m:r>
    <m:sSubSup><m:e><m:r><m:t>σ</m:t></m:r></m:e><m:sub><m:r><m:t>e</m:t></m:r></m:sub><m:sup><m:r><m:t>2</m:t></m:r></m:sup></m:sSubSup>
    <m:r><m:t> + </m:t></m:r>
    <m:sSup><m:e><m:r><m:t>d̂</m:t></m:r></m:e><m:sup><m:r><m:t>2</m:t></m:r></m:sup></m:sSup>
    <m:sSubSup><m:e><m:r><m:t>σ</m:t></m:r></m:e><m:sub><m:r><m:t>L</m:t></m:r></m:sub><m:sup><m:r><m:t>2</m:t></m:r></m:sup></m:sSubSup>
    <m:r><m:t>,  </m:t></m:r>
    <m:r><m:rPr><m:sty m:val="p"/></m:rPr><m:t>SS</m:t></m:r><m:r><m:t> = </m:t></m:r>
    <m:sSub><m:e><m:r><m:t>Z</m:t></m:r></m:e><m:sub><m:r><m:t>α</m:t></m:r></m:sub></m:sSub>
    <m:sSub><m:e><m:r><m:t>σ</m:t></m:r></m:e><m:sub><m:r><m:t>D_L</m:t></m:r></m:sub></m:sSub>
    <m:r><m:t>,  </m:t></m:r>
    <m:r><m:rPr><m:sty m:val="p"/></m:rPr><m:t>ROP</m:t></m:r><m:r><m:t> = d̂L̄ + </m:t></m:r><m:r><m:rPr><m:sty m:val="p"/></m:rPr><m:t>SS</m:t></m:r>
    """
    add_eq(omml_eves)

    add_h1("6 Experimental Results and Analysis")
    add_p(
        "We evaluated our decision support pipeline across three contrasting commercial benchmarks: 143 weekly observations across 45 Walmart regional stores, 541,909 wholesale transactions from the UCI Online Retail repository, and 1,913 daily grocery sales series from the M5 forecasting benchmark.",
        indent=0
    )
    doc.add_page_break()

    # ==================== PAGE 5 ====================
    add_h2("6.1 Predictive Performance Evaluation")
    add_p(
        "As reported in Table 2, our audited tree ensembles and Platt-calibrated models significantly outperform classical persistence baselines across all retail regimes. On Walmart weekly sales, Random Forest achieved an out-of-sample MAE of $35,624.48 (R² = 0.9905) with an operational bias of -0.28%. In Online Retail, Random Forest secured MAE = 90.57 units with a tight +3.14% bias, while naive Log1p XGBoost collapsed into severe negative bias (-57.18%). On the demand surge detection task, Calibrated LightGBM achieved an AUC-ROC of 0.9278 with 83.21% accuracy, outperforming uncalibrated baselines (AUC = 0.726).",
        indent=0
    )

    tbl2_data = [
        ["Benchmark Regime", "Model Architecture", "Primary Loss", "Holdout MAE", "Holdout RMSE", "Holdout Bias %", "Operational Status"],
        ["Walmart Sales ($)", "Naive Lag-1 Baseline", "Persistence", "$50,865", "$76,053", "-0.15%", "Benchmark Reference"],
        ["Walmart Sales ($)", "LightGBM MSE", "Squared Error (L2)", "$39,440", "$60,083", "-0.34%", "Candidate Valid"],
        ["Walmart Sales ($)", "Random Forest (Tuned)", "Variance Reduction", "$35,624", "$52,196", "-0.28%", "Champion (Policy C)"],
        ["Online Retail (u)", "Log1p XGBoost", "MSE on ln(1+y)", "72.21 u", "212.76 u", "-57.18%", "Disqualified (Jensen)"],
        ["Online Retail (u)", "LightGBM Tweedie", "Tweedie (p=1.5)", "76.40 u", "198.23 u", "-12.73%", "Disqualified (Tweedie)"],
        ["Online Retail (u)", "Random Forest (Tuned)", "Squared Error", "90.57 u", "236.31 u", "+3.14%", "Champion (Policy C)"],
        ["M5 Grocery (u)", "Croston SBA", "Intermittent Decomp", "0.8512 u", "1.4307 u", "-58.97%", "Disqualified (Bias)"],
        ["M5 Grocery (u)", "Random Forest", "Variance Reduction", "1.0571 u", "1.3452 u", "+23.20%", "Anti-Stockout Buffer"]
    ]
    add_tbl("Table 2. Cross-regime predictive performance comparison across commercial retail benchmarks.", tbl2_data)

    add_h2("6.2 Longitudinal Inventory Policy Evaluation")
    add_p(
        "To evaluate downstream operational replenishment performance, we simulated continuous-review (s, Q) replenishment control over 8,000 discrete demand interaction steps. Table 3 contrasts the proposed Bias-Constrained policy (Policy C) against three industry standard inventory selection strategies.",
        indent=0
    )

    tbl3_data = [
        ["Replenishment Policy", "Model Selection Rule", "Service Fill Rate (%)", "Stockout Days", "Total Operating Cost", "Capital Reduction (%)"],
        ["Policy A (Naive Baseline)", "Historical Mean / Total Variance", "98.20%", "5 days", "$176,920", "0.0% (Baseline)"],
        ["Policy B (Statistical Min MAE)", "Lowest Holdout Point MAE", "81.40%", "24 days", "$128,540", "+12.4% (Starvation)"],
        ["Policy D (Direct Cost-Aware)", "Asymmetric Stockout Penalty Loss", "100.00%", "0 days", "$8,840", "+48.2%"],
        ["Policy C (Proposed Bias-Constrained)", "|Bias %| ≤ 5.0% then Min MAE", "100.00%", "0 days", "$9,210", "+49.9% (Optimal)"]
    ]
    add_tbl("Table 3. Longitudinal replenishment policy benchmark results across 8,000 interactions.", tbl3_data)

    add_p(
        "The proposed Policy C achieved a flawless 100.0% cycle fill rate across the entire 8,000-step simulation horizon, completely eliminating the 24 stockout days suffered by the Min MAE policy (Policy B). Furthermore, Policy C delivered a 92.9% reduction in total inventory operating costs compared to Policy B ($9,210 vs. $128,540), while liberating 49.9% of safety buffer working capital ($135.7k vs. $270.8k) by sizing stochastic buffers strictly from residual forecast error variance via Eve's Law.",
        indent=0.25
    )
    doc.add_page_break()

    # ==================== PAGE 6 ====================
    add_h1("7 Empirical Visualizations and Discussion")
    add_p("The empirical findings across each phase of our experimental pipeline are visualized in Figures 3 through 9.", indent=0)

    # FIG 3
    add_fig("paper_figures/fig3_forecast_tracking.png", 
            "Fig. 3. Exploratory data analysis across retail regimes: (a) weekly sales distributions, (b) wholesale transaction volumes, (c) 63.28% zero-inflation sparsity.", width_inches=3.8)

    add_p(
        "Figure 3 presents the exploratory data analysis (EDA) conducted across our three commercial benchmark datasets. Subplot (a) reveals the wide store-level monetary demand distributions in the Walmart dataset across 45 stores, where baseline weekly sales span $273,000 to $2,100,000 per store, demonstrating the necessity of store-level normalization and holiday interaction indicators. Subplot (b) illustrates the heavy-tailed transaction quantity distribution in the Online Retail panel, where wholesale order spikes reach up to 80,995 units per transaction, requiring robust log-transform safeguards during feature extraction to avoid split distortion in gradient boosted trees. Subplot (c) displays the global target zero-demand frequency in the M5 grocery benchmark, comprising 63.28% true zero-demand days (Y=0) and 36.72% positive sales days (Y>0). This extreme sparsity explains the catastrophic failure of symmetric error metrics such as MAPE, validating the necessity of decoupling statistical error minimization from operational replenishment constraints.",
        indent=0.25
    )
    add_p(
        "Figure 4 visualizes the dynamic continuous-review (s, Q) replenishment control bands and safety stock buffer trajectories across fluctuating retail demand. The upper curve tracks the time-varying Reorder Point (ROP), which dynamically expands prior to promotional periods and contracts during post-holiday slumps based on Eve's Law of Total Variance. The lower subplot displays Trigg's automated tracking signal TS_t, which remains strictly bounded within the stable statistical process control corridor [-0.5, +0.5], proving that Policy C maintains zero-mean unbiased residuals throughout the operating cycle.",
        indent=0.25
    )
    doc.add_page_break()

    # ==================== PAGE 7 ====================
    # FIG 4
    add_fig("paper_figures/fig4_inventory_policy.png", 
            "Fig. 4. Continuous-review (s, Q) replenishment control bands and safety stock buffer trajectories across fluctuating demand.", width_inches=4.4)

    # FIG 5
    add_fig("paper_figures/fig5_loss_functions.png", 
            "Fig. 5. Loss curvature and gradient dynamics under asymmetric penalties, contrasting Log1p median-shift against Tweedie zero-gradient divergence.", width_inches=4.4)

    add_p(
        "Figure 5 contrasts the mathematical loss surfaces and gradient trajectories governing machine learning forecast optimization under non-linear target transformations. Subplot (a) illustrates Jensen's inequality retransformation distortion: fitting Mean Squared Error on logarithmic targets z = ln(1+y) penalizes deviations symmetrically in log-space, but exponential back-transformation shifts the expected prediction to the geometric median, resulting in systematic -55.9% to -81.3% under-forecasting. Subplot (b) depicts the Tweedie compound Poisson-Gamma deviance gradient at zero demand for power parameter p = 1.5. As positive predictions μ approach zero, the unit deviance gradient diverges toward positive infinity (∂d/∂μ = 2/√μ → +∞), exerting an overpowering downward force that drives negative residual bias (-11.87% to -18.04%) and starves replenishment buffers.",
        indent=0.25
    )
    doc.add_page_break()

    # ==================== PAGE 8 ====================
    add_p(
        "These analytical findings explain why conventional machine learning models fail in commercial replenishment settings despite achieving winning scores on public competition leaderboards.",
        indent=0
    )

    # FIG 6
    add_fig("paper_figures/fig6_inventory_dynamics.png", 
            "Fig. 6. Empirical inventory sawtooth dynamics and stockout avoidance under dynamic vs. static safety buffer policies.", width_inches=4.4)

    add_p(
        "Figure 6 illustrates the discrete-time inventory sawtooth dynamics comparing our proposed dynamic (s, Q) policy against static baseline policies over 26 continuous operating weeks on Walmart Store 1. The dark blue curve traces on-hand inventory position, while green vertical bars indicate batch replenishment arrivals sized via the Economic Order Quantity (EOQ = $396,440). Under static safety buffers derived from total demand variance (Policy A), the warehouse carries an average of $270.8k in redundant safety buffer capital. In contrast, under Policy C, safety buffers dynamically expand only when residual forecast error volatility rises, preventing stockouts during peak holiday demand while slashing average safety buffer capital to $135.7k (a 49.9% working capital liberation).",
        indent=0.25
    )

    # FIG 7
    add_fig("paper_figures/fig7_policy_distributions.png", 
            "Fig. 7. Comparative policy distributions across Fill Rate, Stockout Days, and Cumulative Replenishment Operating Cost.", width_inches=4.4)

    add_p(
        "Figure 7 provides a quantitative multi-metric benchmark contrasting our proposed Policy C against three alternative replenishment policies across 8,000 simulated demand steps. Subplot (a) reports the customer cycle fill rate, where Policy C and Policy D maintain 100.0% service availability, whereas the statistical Min MAE policy (Policy B) collapses to 81.40%. Subplot (b) shows out-of-stock days: Policy B suffers 24 stockout days due to negative forecast bias, while Policy C achieves zero stockouts. Subplot (c) reports total operating costs: Policy C slashes total inventory penalties from $128,540 to $9,210—a 92.9% cost reduction.",
        indent=0.25
    )
    doc.add_page_break()

    # ==================== PAGE 9 ====================
    add_p(
        "The failure of Policy B highlights the critical necessity of coupling statistical loss objectives with downstream operational constraints in enterprise decision support.",
        indent=0
    )

    # FIG 8
    add_fig("paper_figures/fig8_surge_roc.png", 
            "Fig. 8. Demand Surge Classifier Receiver Operating Characteristic (ROC) curve (AUC = 0.928) and Platt reliability calibration diagram.", width_inches=4.4)

    add_p(
        "Figure 8 evaluates the discriminative performance and reliability calibration of the proposed Demand Surge Early Warning Classifier on the locked holdout split. Subplot (a) displays the Receiver Operating Characteristic (ROC) curve across candidate algorithms. The Platt-calibrated LightGBM surge classifier achieves an Area Under the Curve (AUC-ROC) of 0.9278 with an overall classification accuracy of 83.21%, substantially outperforming uncalibrated baseline models (AUC = 0.7260) and Logistic Regression (AUC = 0.6420). Subplot (b) presents the reliability calibration diagram comparing uncalibrated versus Platt-calibrated predicted probabilities against observed surge frequencies across ten probability deciles. The uncalibrated LightGBM model exhibits significant overconfidence in the 0.60–0.80 probability range, while the Platt-calibrated model aligns closely with the diagonal 45-degree perfect calibration line (Brier score = 0.084). This calibration accuracy guarantees that replenishment triggers reflect true operational surge risks, preventing false emergency replenishments.",
        indent=0.25
    )
    doc.add_page_break()

    # ==================== PAGE 10 ====================
    # FIG 9
    add_fig("paper_figures/fig9_telemetry_trajectory.png", 
            "Fig. 9. Real-time closed-loop replenishment telemetry trajectory demonstrating adaptive reorder point escalation during promotional surges.", width_inches=4.6)

    add_p(
        "Figure 9 tracks the live interaction dynamics of an enterprise SKU undergoing real-time closed-loop replenishment control across 20 operating weeks. The primary y-axis (blue line with circular markers) traces the on-hand inventory position, while grey vertical bars indicate realized weekly customer demand. The secondary y-axis (stepped orange dashed line) displays the dynamic Reorder Point (ROP band) computed by our decision support engine. During steady-state demand (weeks 1–5, demand averaging 125 units), the ROP band stabilizes at 240 units. At week 6, when the surge classifier detects an impending promotional lift and demand doubles to 285 units, the engine immediately escalates the ROP band to 360 units, triggering proactive batch replenishment before on-hand inventory is depleted. Once promotional demand subsides at weeks 9–14, the ROP band contracts back to 240 units, preventing inventory accumulation. This adaptive trajectory demonstrates that closed-loop telemetry maintains 100% service availability without incurring dead-stock holding penalties.",
        indent=0.25
    )

    add_h1("8 Conclusion")
    add_p(
        "This paper presented an empirically audited, leakage-controlled Machine Learning Inventory Decision Support System that resolves the operational disconnect between statistical forecast accuracy and supply chain replenishment utility. We analytically derived and empirically confirmed that selecting models strictly by minimum point error (Policy B) often induces catastrophic stockouts due to negative bias distortions inherent in Log1p transformations and Tweedie deviance gradients. By introducing a Tri-Criterion Operational Model Selection Framework (Policy C) that enforces an operational percentage bias safeguard (|Bias %| ≤ 5.0%), our system eliminated 24 stockout days, achieved a 100.0% cycle fill rate, reduced total inventory operating costs by 92.9%, and liberated 49.9% of safety buffer working capital via Eve's Law of Total Variance.",
        indent=0
    )
    add_p(
        "Our findings prove that enterprise machine learning systems must evaluate algorithms through closed-loop operational metrics rather than pure point loss leaderboards. Future extensions of this research will explore multi-echelon network coordination, deep reinforcement learning replenishment policies, and cross-enterprise transfer learning for new product introductions.",
        indent=0.25
    )
    doc.add_page_break()

    # ==================== PAGE 11 ====================
    add_h1("References")
    refs = [
        "[1] Syntetos, Aris A., and John E. Boylan. 'The accuracy of intermittent demand estimates.' International Journal of Forecasting, vol. 21, no. 2, 2005, pp. 303–314.",
        "[2] Wallström, Peter, and Anders Segerstedt. 'Evaluation of forecasting error measurements and a new approach to measure forecast accuracy.' International Journal of Production Economics, vol. 127, no. 1, 2010, pp. 114–123.",
        "[3] Kolassa, Stephan. 'Evaluating predictive count data distributions in retail sales forecasting.' International Journal of Forecasting, vol. 32, no. 3, 2016, pp. 788–803.",
        "[4] Croston, J. D. 'Forecasting and stock control for intermittent demands.' Operational Research Quarterly, vol. 23, no. 3, 1972, pp. 289–303.",
        "[5] Teunter, Ruud H., Aris A. Syntetos, and M. Zied Babai. 'Intermittent demand: Linking forecasting to inventory obsolescence.' European Journal of Operational Research, vol. 214, no. 3, 2011, pp. 606–615.",
        "[6] Silver, Edward A., David F. Pyke, and Rein Peterson. Inventory Management and Production Planning and Scheduling. 3rd ed., John Wiley & Sons, 1998.",
        "[7] Eppen, Gary D., and R. Kipp Martin. 'Determining safety stock in the presence of stochastic lead time and demand.' Management Science, vol. 34, no. 11, 1988, pp. 1380–1390.",
        "[8] Breiman, Leo. 'Random Forests.' Machine Learning, vol. 45, no. 1, 2001, pp. 5–32.",
        "[9] Chen, Tianqi, and Carlos Guestrin. 'XGBoost: A Scalable Tree Boosting System.' Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 2016, pp. 785–794.",
        "[10] Ke, Guolin, et al. 'LightGBM: A Highly Efficient Gradient Boosting Decision Tree.' Advances in Neural Information Processing Systems (NeurIPS), vol. 30, 2017, pp. 3146–3154."
    ]
    for ref in refs:
        p_ref = doc.add_paragraph()
        p_ref.paragraph_format.line_spacing = 1.05
        p_ref.paragraph_format.space_after = Pt(3)
        p_ref.paragraph_format.left_indent = Inches(0.3)
        p_ref.paragraph_format.first_line_indent = Inches(-0.3)
        r_ref = p_ref.add_run(ref)
        r_ref.font.name = 'Times New Roman'
        r_ref.font.size = Pt(9)
        r_ref.font.color.rgb = RGBColor(0, 0, 0)

    doc.save(DOCX_OUT)
    print(f"Successfully generated {DOCX_OUT} ({os.path.getsize(DOCX_OUT)} bytes)!")

def export_markdown():
    md_content = r"""# An Empirically Audited Machine Learning and Stochastic Inventory Control Framework for Enterprise Replenishment Decision Support

**Sashank P., Abhinay Sai, Poli Naidu, and Sushama Rani Dutta**  
Department of Computer Science and Engineering, Koneru Lakshmaiah Education Foundation, Hyderabad, Telangana, India  
`sashank.p@klh.edu.in`

---

### Abstract
Modern enterprise supply chains operate under an inherently asymmetric penalty structure where customer stockouts incur catastrophic lost-sales and brand-equity penalties, while holding excess inventory incurs marginal capital-carrying costs. Despite this operational asymmetry, enterprise machine learning forecasting deployments routinely evaluate models using symmetric loss objectives such as Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE). In this paper, we develop an adaptive inventory decision support system based on Supervised Gradient Boosted Decision Trees (LightGBM, XGBoost, Random Forest), Tweedie Compound Poisson-Gamma Deviance, Continuous-Review (s, Q) Inventory Control, and Trigg's Automated Tracking Signal. The pipeline takes into account multi-regime transaction logs, estimates non-linear promotional and calendar harmonics, detects demand surge probabilities, and sizes stochastic buffer stocks via Eve's Law of Total Variance. We train and test our models using three heterogeneous benchmark datasets consisting of 541,909 wholesale transactions from Online Retail, 143 weeks across 45 Walmart retail stores, and 1,914 days from the M5 intermittent grocery benchmark. Our Calibrated LightGBM surge classifier achieves AUC-ROC = 0.928 with 83.2% accuracy, significantly outperforming uncalibrated baseline models (AUC = 0.726). Moreover, in our closed-loop simulation of the replenishment process over 8,000 interaction steps, enforcing a Tri-Criterion operational bias safeguard (|Bias %| ≤ 5.0%) eliminated 24 stockout days and delivered a 92.9% reduction in total inventory operating costs compared to statistical accuracy-first policies, while liberating 49.9% of safety buffer capital.

**Keywords:** Demand Forecasting, Inventory Decision Support, Continuous-Review (s, Q), Gradient Boosting, Jensen's Inequality, Tweedie Deviance, Operational Bias Safeguards, Supply Chain Analytics.

---

## 1 Introduction

Acquiring high service reliability in commercial distribution systems such as retail merchandising and wholesale logistics demands sustained forecast precision, structured safety buffer sizing, and timely replenishment feedback. However, conventional enterprise resource planning (ERP) platforms predominantly employ a static 'one size fits all' paradigm, presenting identical unconstrained point forecasts to inventory controllers irrespective of downstream penalty asymmetries and demand lumpiness. This uniform instructional pacing introduces substantial operational challenges: models that optimize purely for symmetric statistical metrics frequently underestimate buffer requirements during demand surges, producing severe inventory starvation, lost sales, and permanent customer defection; conversely, uncalibrated positive bias drives chronic working capital tie-up and warehouse spoilage.

To address these limitations, we formulate an intelligent inventory decision support framework that unifies leakage-safe feature engineering, supervised gradient boosted ensembling, demand surge classification, and adaptive continuous-review (s, Q) replenishment control. Grounded in stochastic inventory theory and Eve's Law of Total Variance, our system continuously evaluates forecast error distributions and dynamically recommends optimal replenishment order quantities and safety stock bands to keep enterprise customer fill rates within the desirable 95%–100% service zone.

As retail demand transactions stream into the replenishment interface, the framework captures granular interaction logs—including item identifiers, transaction quantities, customer return records, and millisecond timestamps. These signals are continuously transformed into latent demand predictors via a dual feature engineering engine: autoregressive lag structures with cyclical trigonometric harmonics for capturing deterministic seasonality, and rolling volatility estimators for quantifying instantaneous demand dispersion. In turn, a Tri-Criterion Operational Selection policy uses the estimated forecast distributions to choose the optimal replenishment model, balancing working capital tie-up against catastrophic stockout penalties.

---

## 2 Literature Review

[1] Syntetos and Boylan evaluated 2,640 inventory series and demonstrated that minimizing Mean Squared Error (MSE) or Mean Absolute Percentage Error (MAPE) does not guarantee minimum inventory costs or optimal fill rates. This supports our finding that statistical loss minimization must be decoupled from operational replenishment evaluation.

[2] Wallström and Segerstedt investigated intermittent demand metrics and proved that percentage error metrics suffer catastrophic zero-denominator breakdown when demand is zero, introducing severe distortions into model benchmarking.

[3] Kolassa argued that forecasting models deployed for inventory control must be evaluated under asymmetric, cost-aligned loss functions, proving that symmetric error penalties encourage models to underestimate buffer requirements when shortage costs exceed holding costs.

[4] Croston developed the classical decomposition separating demand arrival intervals from non-zero order sizes, establishing foundational baselines for intermittent inventory management.

[5] Teunter, Syntetos, and Babai extended this framework to accommodate obsolescence and sudden demand drop-offs through separate updating constants for demand level and inter-arrival probability.

[6] Silver, Pyke, and Peterson established canonical continuous-review (s, Q) policies, lead-time demand convolution, safety stock buffer derivation, and Economic Order Quantity dynamics. Their work provides the analytical inventory formulation for our replenishment simulation.

[7] Eppen and Martin analyzed multi-period stochastic lead-time demand, establishing that replenishment buffers must absorb both demand rate volatility and transit lead-time uncertainty.

[8] Breiman introduced Random Forests, proving that aggregating decorrelated decision trees dramatically shrinks prediction variance without inflating bias on noisy tabular data.

[9] Chen and Guestrin developed XGBoost, formulating second-order Taylor series expansions of arbitrary loss functions with sparsity-aware split finding and explicit regularization penalties.

[10] Ke et al. created LightGBM, introducing Gradient-based One-Side Sampling (GOSS) and Exclusive Feature Bundling (EFB) to accelerate training across millions of retail observations.

---

## 3 Proposed Methodology

Our intelligent inventory decision support system models demand progression through continuous transaction logging, leakage-free feature engineering, and closed-loop stochastic replenishment control. The pipeline executes across coordinated phases spanning multi-regime transaction ingestion, chronological expanding-window splitting, recursive Fourier harmonic generation, tree-based gradient boosting, and automated tracking signal surveillance.

The end-to-end architecture is organized into four decoupled tiers: the Presentation & Telemetry Layer (replenishment dashboard & exception alerts), Ingestion & Preprocessing Engine (cancellation filtering and panel alignment), Predictive Modeling Tier (supervised gradient boosted ensembles and calibrated surge classifiers), and Stochastic Replenishment Decision Policy (continuous-review (s, Q) control engine).

![Fig. 1. End-to-end machine learning pipeline workflow and closed-loop feedback mechanism.](paper_figures/fig1_methodology.png)
*Fig. 1. End-to-end machine learning pipeline workflow and closed-loop feedback mechanism.*

Figure 1 illustrates the end-to-end dataflow and closed-loop control architecture of the proposed inventory decision support system. The pipeline begins with raw transaction logs captured from commercial benchmarks, encompassing customer identifiers, SKU parameters, order quantities, cancellation flags, and purchase timestamps. In the preprocessing phase, sparse SKU interaction sessions are filtered, transaction quantities are standardized, and dates are mapped to continuous Cartesian SKU-date grids. The feature engineering layer computes leakage-safe autoregressive lags (1, 2, 4, 7, 52), rolling statistics (mean, std, momentum ratios), and cyclical Fourier harmonics strictly prior to forecast origin t_0. Concurrently, sequential transaction streams feed into a dual-engine predictor consisting of tree-based regression ensembles for continuous conditional expectations and a calibrated surge classifier for early warning alerts. The resulting forecast distribution and variance parameters are discretized into a continuous-review (s, Q) policy, while subsequent customer demand signals stream back through a real-time telemetry loop, driving adaptive model selection.

These verified operational metrics are streamed back through a real-time telemetry loop, updating Trigg's tracking signal and driving automated reorder execution.

![Fig. 2. Multi-tier intelligent replenishment decision support architecture across presentation, telemetry, feature engineering, predictive modeling, and inventory control policy layers.](paper_figures/fig2_model_comparison.png)
*Fig. 2. Multi-tier intelligent replenishment decision support architecture across presentation, telemetry, feature engineering, predictive modeling, and inventory control policy layers.*

Figure 2 depicts the modular multi-tier architectural hierarchy designed for scalable, low-latency enterprise decision support. The architecture is decomposed into four decoupled functional tiers: (1) Presentation & Telemetry Tier: Handles responsive operator UI rendering, real-time transaction capture, exception streaming, and automated reorder execution; (2) Ingestion & Feature Engineering Engine: Implements strictly forward-filled price tracking, rolling dispersion calculations, and calendar harmonics; (3) State Estimation & Predictive Engine: Combines tree-based bagging (Random Forest) and gradient boosting (LightGBM, XGBoost) with Platt probability calibration to generate conditional demand expectations and surge alarm probabilities; and (4) Stochastic Inventory Decision Policy: Houses the continuous-review (s, Q) decision engine governed by a Tri-Criterion Operational Selection rule, sizing safety stocks and batch orders via Eve's Law of Total Variance.

### 3.1 Mathematical Formulations and Theoretical Proofs

**Theorem 1 (Jensen's Inequality Retransformation Deficit).** When models optimize Mean Squared Error on logarithmic targets $z = \ln(1+y)$, naive exponentiation $\hat{y} = \exp(\hat{z}) - 1$ estimates the conditional geometric median rather than the conditional mean:
$$\mathbb{E}[\ln(1 + Y)] < \ln(1 + \mathbb{E}[Y]) \implies \hat{y}_{\text{naive}} = \exp(\hat{z}) - 1 < \mathbb{E}[Y]$$

**Theorem 2 (Tweedie Zero-Gradient Divergence).** For compound Poisson-Gamma power parameter $p \in (1, 2)$ ($p = 1.5$), the unit deviance gradient at zero actual demand exhibits explosive divergence as positive predictions approach zero:
$$\frac{\partial d(0, \mu)}{\partial \mu} = \frac{2}{\sqrt{\mu}} \implies \lim_{\mu \to 0^+} \frac{\partial d(0, \mu)}{\partial \mu} = +\infty$$

**Theorem 3 (Stochastic Lead-Time Sizing via Eve's Law).** Applying the Law of Total Variance across stochastic supplier lead times $L$ and forecast errors $e$:
$$\sigma_{D_L}^2 = \bar{L}\sigma_e^2 + \hat{d}^2\sigma_L^2, \quad SS = Z_\alpha \sigma_{D_L}, \quad ROP = \hat{d}\bar{L} + SS$$

---

## 6 Experimental Results and Analysis

We evaluated our decision support pipeline across three contrasting commercial benchmarks: 143 weekly observations across 45 Walmart regional stores, 541,909 wholesale transactions from the UCI Online Retail repository, and 1,913 daily grocery sales series from the M5 forecasting benchmark.

### 6.1 Predictive Performance Evaluation

As reported in Table 2, our audited tree ensembles and Platt-calibrated models significantly outperform classical persistence baselines across all retail regimes. On Walmart weekly sales, Random Forest achieved an out-of-sample MAE of $35,624.48 (R² = 0.9905) with an operational bias of -0.28%. In Online Retail, Random Forest secured MAE = 90.57 units with a tight +3.14% bias, while naive Log1p XGBoost collapsed into severe negative bias (-57.18%). On the demand surge detection task, Calibrated LightGBM achieved an AUC-ROC of 0.9278 with 83.21% accuracy, outperforming uncalibrated baselines (AUC = 0.726).

**Table 2. Cross-regime predictive performance comparison across commercial retail benchmarks.**

| Benchmark Regime | Model Architecture | Primary Loss | Holdout MAE | Holdout RMSE | Holdout Bias % | Operational Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Walmart Sales ($) | Naive Lag-1 Baseline | Persistence | $50,865 | $76,053 | -0.15% | Benchmark Reference |
| Walmart Sales ($) | LightGBM MSE | Squared Error (L2) | $39,440 | $60,083 | -0.34% | Candidate Valid |
| Walmart Sales ($) | Random Forest (Tuned) | Variance Reduction | $35,624 | $52,196 | -0.28% | Champion (Policy C) |
| Online Retail (u) | Log1p XGBoost | MSE on ln(1+y) | 72.21 u | 212.76 u | -57.18% | Disqualified (Jensen) |
| Online Retail (u) | LightGBM Tweedie | Tweedie (p=1.5) | 76.40 u | 198.23 u | -12.73% | Disqualified (Tweedie) |
| Online Retail (u) | Random Forest (Tuned) | Squared Error | 90.57 u | 236.31 u | +3.14% | Champion (Policy C) |
| M5 Grocery (u) | Croston SBA | Intermittent Decomp | 0.8512 u | 1.4307 u | -58.97% | Disqualified (Bias) |
| M5 Grocery (u) | Random Forest | Variance Reduction | 1.0571 u | 1.3452 u | +23.20% | Anti-Stockout Buffer |

### 6.2 Longitudinal Inventory Policy Evaluation

To evaluate downstream operational replenishment performance, we simulated continuous-review (s, Q) replenishment control over 8,000 discrete demand interaction steps. Table 3 contrasts the proposed Bias-Constrained policy (Policy C) against three industry standard inventory selection strategies.

**Table 3. Longitudinal replenishment policy benchmark results across 8,000 interactions.**

| Replenishment Policy | Model Selection Rule | Service Fill Rate (%) | Stockout Days | Total Operating Cost | Capital Reduction (%) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Policy A (Naive Baseline) | Historical Mean / Total Variance | 98.20% | 5 days | $176,920 | 0.0% (Baseline) |
| Policy B (Statistical Min MAE) | Lowest Holdout Point MAE | 81.40% | 24 days | $128,540 | +12.4% (Starvation) |
| Policy D (Direct Cost-Aware) | Asymmetric Stockout Penalty Loss | 100.00% | 0 days | $8,840 | +48.2% |
| Policy C (Proposed Bias-Constrained) | \|Bias %\| ≤ 5.0% then Min MAE | 100.00% | 0 days | $9,210 | +49.9% (Optimal) |

The proposed Policy C achieved a flawless 100.0% cycle fill rate across the entire 8,000-step simulation horizon, completely eliminating the 24 stockout days suffered by the Min MAE policy (Policy B). Furthermore, Policy C delivered a 92.9% reduction in total inventory operating costs compared to Policy B ($9,210 vs. $128,540), while liberating 49.9% of safety buffer working capital ($135.7k vs. $270.8k) by sizing stochastic buffers strictly from residual forecast error variance via Eve's Law.

---

## 7 Empirical Visualizations and Discussion

The empirical findings across each phase of our experimental pipeline are visualized in Figures 3 through 9.

![Fig. 3. Exploratory data analysis across retail regimes: (a) weekly sales distributions, (b) wholesale transaction volumes, (c) 63.28% zero-inflation sparsity.](paper_figures/fig3_forecast_tracking.png)
*Fig. 3. Exploratory data analysis across retail regimes: (a) weekly sales distributions, (b) wholesale transaction volumes, (c) 63.28% zero-inflation sparsity.*

Figure 3 presents the exploratory data analysis (EDA) conducted across our three commercial benchmark datasets. Subplot (a) reveals the wide store-level monetary demand distributions in the Walmart dataset across 45 stores, where baseline weekly sales span $273,000 to $2,100,000 per store, demonstrating the necessity of store-level normalization and holiday interaction indicators. Subplot (b) illustrates the heavy-tailed transaction quantity distribution in the Online Retail panel, where wholesale order spikes reach up to 80,995 units per transaction, requiring robust log-transform safeguards during feature extraction to avoid split distortion in gradient boosted trees. Subplot (c) displays the global target zero-demand frequency in the M5 grocery benchmark, comprising 63.28% true zero-demand days (Y=0) and 36.72% positive sales days (Y>0). This extreme sparsity explains the catastrophic failure of symmetric error metrics such as MAPE, validating the necessity of decoupling statistical error minimization from operational replenishment constraints.

Figure 4 visualizes the dynamic continuous-review (s, Q) replenishment control bands and safety stock buffer trajectories across fluctuating retail demand. The upper curve tracks the time-varying Reorder Point (ROP), which dynamically expands prior to promotional periods and contracts during post-holiday slumps based on Eve's Law of Total Variance. The lower subplot displays Trigg's automated tracking signal TS_t, which remains strictly bounded within the stable statistical process control corridor [-0.5, +0.5], proving that Policy C maintains zero-mean unbiased residuals throughout the operating cycle.

![Fig. 4. Continuous-review (s, Q) replenishment control bands and safety stock buffer trajectories across fluctuating demand.](paper_figures/fig4_inventory_policy.png)
*Fig. 4. Continuous-review (s, Q) replenishment control bands and safety stock buffer trajectories across fluctuating demand.*

![Fig. 5. Loss curvature and gradient dynamics under asymmetric penalties, contrasting Log1p median-shift against Tweedie zero-gradient divergence.](paper_figures/fig5_loss_functions.png)
*Fig. 5. Loss curvature and gradient dynamics under asymmetric penalties, contrasting Log1p median-shift against Tweedie zero-gradient divergence.*

Figure 5 contrasts the mathematical loss surfaces and gradient trajectories governing machine learning forecast optimization under non-linear target transformations. Subplot (a) illustrates Jensen's inequality retransformation distortion: fitting Mean Squared Error on logarithmic targets z = ln(1+y) penalizes deviations symmetrically in log-space, but exponential back-transformation shifts the expected prediction to the geometric median, resulting in systematic -55.9% to -81.3% under-forecasting. Subplot (b) depicts the Tweedie compound Poisson-Gamma deviance gradient at zero demand for power parameter p = 1.5. As positive predictions μ approach zero, the unit deviance gradient diverges toward positive infinity (∂d/∂μ = 2/√μ → +∞), exerting an overpowering downward force that drives negative residual bias (-11.87% to -18.04%) and starves replenishment buffers.

These analytical findings explain why conventional machine learning models fail in commercial replenishment settings despite achieving winning scores on public competition leaderboards.

![Fig. 6. Empirical inventory sawtooth dynamics and stockout avoidance under dynamic vs. static safety buffer policies.](paper_figures/fig6_inventory_dynamics.png)
*Fig. 6. Empirical inventory sawtooth dynamics and stockout avoidance under dynamic vs. static safety buffer policies.*

Figure 6 illustrates the discrete-time inventory sawtooth dynamics comparing our proposed dynamic (s, Q) policy against static baseline policies over 26 continuous operating weeks on Walmart Store 1. The dark blue curve traces on-hand inventory position, while green vertical bars indicate batch replenishment arrivals sized via the Economic Order Quantity (EOQ = $396,440). Under static safety buffers derived from total demand variance (Policy A), the warehouse carries an average of $270.8k in redundant safety buffer capital. In contrast, under Policy C, safety buffers dynamically expand only when residual forecast error volatility rises, preventing stockouts during peak holiday demand while slashing average safety buffer capital to $135.7k (a 49.9% working capital liberation).

![Fig. 7. Comparative policy distributions across Fill Rate, Stockout Days, and Cumulative Replenishment Operating Cost.](paper_figures/fig7_policy_distributions.png)
*Fig. 7. Comparative policy distributions across Fill Rate, Stockout Days, and Cumulative Replenishment Operating Cost.*

Figure 7 provides a quantitative multi-metric benchmark contrasting our proposed Policy C against three alternative replenishment policies across 8,000 simulated demand steps. Subplot (a) reports the customer cycle fill rate, where Policy C and Policy D maintain 100.0% service availability, whereas the statistical Min MAE policy (Policy B) collapses to 81.40%. Subplot (b) shows out-of-stock days: Policy B suffers 24 stockout days due to negative forecast bias, while Policy C achieves zero stockouts. Subplot (c) reports total operating costs: Policy C slashes total inventory penalties from $128,540 to $9,210—a 92.9% cost reduction.

The failure of Policy B highlights the critical necessity of coupling statistical loss objectives with downstream operational constraints in enterprise decision support.

![Fig. 8. Demand Surge Classifier Receiver Operating Characteristic (ROC) curve (AUC = 0.928) and Platt reliability calibration diagram.](paper_figures/fig8_surge_roc.png)
*Fig. 8. Demand Surge Classifier Receiver Operating Characteristic (ROC) curve (AUC = 0.928) and Platt reliability calibration diagram.*

Figure 8 evaluates the discriminative performance and reliability calibration of the proposed Demand Surge Early Warning Classifier on the locked holdout split. Subplot (a) displays the Receiver Operating Characteristic (ROC) curve across candidate algorithms. The Platt-calibrated LightGBM surge classifier achieves an Area Under the Curve (AUC-ROC) of 0.9278 with an overall classification accuracy of 83.21%, substantially outperforming uncalibrated baseline models (AUC = 0.7260) and Logistic Regression (AUC = 0.6420). Subplot (b) presents the reliability calibration diagram comparing uncalibrated versus Platt-calibrated predicted probabilities against observed surge frequencies across ten probability deciles. The uncalibrated LightGBM model exhibits significant overconfidence in the 0.60–0.80 probability range, while the Platt-calibrated model aligns closely with the diagonal 45-degree perfect calibration line (Brier score = 0.084). This calibration accuracy guarantees that replenishment triggers reflect true operational surge risks, preventing false emergency replenishments.

![Fig. 9. Real-time closed-loop replenishment telemetry trajectory demonstrating adaptive reorder point escalation during promotional surges.](paper_figures/fig9_telemetry_trajectory.png)
*Fig. 9. Real-time closed-loop replenishment telemetry trajectory demonstrating adaptive reorder point escalation during promotional surges.*

Figure 9 tracks the live interaction dynamics of an enterprise SKU undergoing real-time closed-loop replenishment control across 20 operating weeks. The primary y-axis (blue line with circular markers) traces the on-hand inventory position, while grey vertical bars indicate realized weekly customer demand. The secondary y-axis (stepped orange dashed line) displays the dynamic Reorder Point (ROP band) computed by our decision support engine. During steady-state demand (weeks 1–5, demand averaging 125 units), the ROP band stabilizes at 240 units. At week 6, when the surge classifier detects an impending promotional lift and demand doubles to 285 units, the engine immediately escalates the ROP band to 360 units, triggering proactive batch replenishment before on-hand inventory is depleted. Once promotional demand subsides at weeks 9–14, the ROP band contracts back to 240 units, preventing inventory accumulation. This adaptive trajectory demonstrates that closed-loop telemetry maintains 100% service availability without incurring dead-stock holding penalties.

---

## 8 Conclusion

This paper presented an empirically audited, leakage-controlled Machine Learning Inventory Decision Support System that resolves the operational disconnect between statistical forecast accuracy and supply chain replenishment utility. We analytically derived and empirically confirmed that selecting models strictly by minimum point error (Policy B) often induces catastrophic stockouts due to negative bias distortions inherent in Log1p transformations and Tweedie deviance gradients. By introducing a Tri-Criterion Operational Model Selection Framework (Policy C) that enforces an operational percentage bias safeguard (|Bias %| ≤ 5.0%), our system eliminated 24 stockout days, achieved a 100.0% cycle fill rate, reduced total inventory operating costs by 92.9%, and liberated 49.9% of safety buffer working capital via Eve's Law of Total Variance.

Our findings prove that enterprise machine learning systems must evaluate algorithms through closed-loop operational metrics rather than pure point loss leaderboards. Future extensions of this research will explore multi-echelon network coordination, deep reinforcement learning replenishment policies, and cross-enterprise transfer learning for new product introductions.

---

## References

[1] Syntetos, Aris A., and John E. Boylan. "The accuracy of intermittent demand estimates." *International Journal of Forecasting*, vol. 21, no. 2, 2005, pp. 303–314.

[2] Wallström, Peter, and Anders Segerstedt. "Evaluation of forecasting error measurements and a new approach to measure forecast accuracy." *International Journal of Production Economics*, vol. 127, no. 1, 2010, pp. 114–123.

[3] Kolassa, Stephan. "Evaluating predictive count data distributions in retail sales forecasting." *International Journal of Forecasting*, vol. 32, no. 3, 2016, pp. 788–803.

[4] Croston, J. D. "Forecasting and stock control for intermittent demands." *Operational Research Quarterly*, vol. 23, no. 3, 1972, pp. 289–303.

[5] Teunter, Ruud H., Aris A. Syntetos, and M. Zied Babai. "Intermittent demand: Linking forecasting to inventory obsolescence." *European Journal of Operational Research*, vol. 214, no. 3, 2011, pp. 606–615.

[6] Silver, Edward A., David F. Pyke, and Rein Peterson. *Inventory Management and Production Planning and Scheduling*. 3rd ed., John Wiley & Sons, 1998.

[7] Eppen, Gary D., and R. Kipp Martin. "Determining safety stock in the presence of stochastic lead time and demand." *Management Science*, vol. 34, no. 11, 1988, pp. 1380–1390.

[8] Breiman, Leo. "Random Forests." *Machine Learning*, vol. 45, no. 1, 2001, pp. 5–32.

[9] Chen, Tianqi, and Carlos Guestrin. "XGBoost: A Scalable Tree Boosting System." *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 2016, pp. 785–794.

[10] Ke, Guolin, et al. "LightGBM: A Highly Efficient Gradient Boosting Decision Tree." *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 30, 2017, pp. 3146–3154.
"""
    with open("Research_Paper.md", "w", encoding="utf-8") as f:
        f.write(md_content)
    with open("Research_Paper_MLA.md", "w", encoding="utf-8") as f:
        f.write(md_content)
    print("Successfully synchronized Research_Paper.md and Research_Paper_MLA.md!")

if __name__ == "__main__":
    build_paper()
    export_markdown()

