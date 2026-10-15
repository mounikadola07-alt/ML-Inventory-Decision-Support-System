"""
build_executive_summary_handout.py
Generates a polished 1-page Executive Summary Handout for thesis committees,
conference attendees, and journal reviewers.
"""

import os
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

def set_cell_margins(cell, top=80, bottom=80, left=120, right=120):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = parse_xml(f'<w:tcMar {nsdecls("w")}><w:top w:w="{top}" w:type="dxa"/><w:bottom w:w="{bottom}" w:type="dxa"/><w:left w:w="{left}" w:type="dxa"/><w:right w:w="{right}" w:type="dxa"/></w:tcMar>')
    tcPr.append(tcMar)

def set_cell_border(cell, color="000000", sz="4"):
    tcPr = cell._tc.get_or_add_tcPr()
    borders = parse_xml(f'<w:tcBorders {nsdecls("w")}><w:top w:val="single" w:sz="{sz}" w:space="0" w:color="{color}"/><w:left w:val="single" w:sz="{sz}" w:space="0" w:color="{color}"/><w:bottom w:val="single" w:sz="{sz}" w:space="0" w:color="{color}"/><w:right w:val="single" w:sz="{sz}" w:space="0" w:color="{color}"/></w:tcBorders>')
    tcPr.append(borders)

def main():
    doc = docx.Document()
    
    # 0.5 inch compact academic margins
    sections = doc.sections
    for s in sections:
        s.top_margin = Inches(0.5)
        s.bottom_margin = Inches(0.5)
        s.left_margin = Inches(0.6)
        s.right_margin = Inches(0.6)

    # Document Header
    p_top = doc.add_paragraph()
    p_top.paragraph_format.space_after = Pt(2)
    p_top.paragraph_format.space_before = Pt(0)
    run_meta = p_top.add_run("EXECUTIVE RESEARCH BRIEFING & DEFENSE HANDOUT | 2026")
    run_meta.font.name = "Times New Roman"
    run_meta.font.size = Pt(8.5)
    run_meta.font.bold = True
    run_meta.font.color.rgb = RGBColor(100, 100, 100)

    # Title
    p_title = doc.add_paragraph()
    p_title.paragraph_format.space_after = Pt(2)
    p_title.paragraph_format.space_before = Pt(0)
    r_title = p_title.add_run("An Empirically Audited Machine Learning & Stochastic Inventory Control Framework for Enterprise Replenishment Decision Support")
    r_title.font.name = "Times New Roman"
    r_title.font.size = Pt(13)
    r_title.font.bold = True

    # Authors
    p_auth = doc.add_paragraph()
    p_auth.paragraph_format.space_after = Pt(6)
    r_auth = p_auth.add_run("Dola Mounika | Koneru Lakshmaiah Education Foundation")
    r_auth.font.name = "Times New Roman"
    r_auth.font.size = Pt(9.5)
    r_auth.font.italic = True

    # Core Problem & Mathematical Theorems Box
    table_box = doc.add_table(rows=1, cols=2)
    table_box.autofit = False
    
    cell_left = table_box.rows[0].cells[0]
    cell_left.width = Inches(3.6)
    set_cell_margins(cell_left)
    set_cell_border(cell_left, sz="6")
    
    p_box1 = cell_left.paragraphs[0]
    p_box1.paragraph_format.space_after = Pt(3)
    r_h1 = p_box1.add_run("1. THE OPERATIONAL PARADOX")
    r_h1.font.bold = True
    r_h1.font.name = "Times New Roman"
    r_h1.font.size = Pt(10)
    
    p_b1 = cell_left.add_paragraph()
    p_b1.paragraph_format.space_after = Pt(4)
    r_t1 = p_b1.add_run(
        "• Statistical MAE/RMSE models treat under- and over-forecasting identically. In retail operations, "
        "under-forecasting triggers catastrophic stockouts ($p >> h).\n"
        "• Jensen's Inequality Deficit (Theorem 1): Log-transformed regressors log(1+y) incur an operational under-forecasting bias "
        "of up to -57.18% upon exponentiation: E[exp(Z)] > exp(E[Z]).\n"
        "• Solution: Dual-moment smearing correction: Y = exp(u + 0.5*s^2) - 1."
    )
    r_t1.font.name = "Times New Roman"
    r_t1.font.size = Pt(8.5)

    cell_right = table_box.rows[0].cells[1]
    cell_right.width = Inches(3.6)
    set_cell_margins(cell_right)
    set_cell_border(cell_right, sz="6")
    
    p_box2 = cell_right.paragraphs[0]
    p_box2.paragraph_format.space_after = Pt(3)
    r_h2 = p_box2.add_run("2. STOCHASTIC REPLENISHMENT & EVE'S LAW")
    r_h2.font.bold = True
    r_h2.font.name = "Times New Roman"
    r_h2.font.size = Pt(10)
    
    p_b2 = cell_right.add_paragraph()
    p_b2.paragraph_format.space_after = Pt(4)
    r_t2 = p_b2.add_run(
        "• Stochastic Lead-Time Variance: Standard safety stock formulas underestimate stockout risk by >60% under supplier arrival delays.\n"
        "• Eve's Law (Law of Total Variance): s_LTD^2 = L * s_D^2 + u_D^2 * s_L^2.\n"
        "• Platt-Calibrated Surge Trees: GBDT early-warning classifier identifies tail-risk surges (>95th percentile) with ROC AUC = 0.9278, "
        "Accuracy = 83.21%, dynamic buffer expansion."
    )
    r_t2.font.name = "Times New Roman"
    r_t2.font.size = Pt(8.5)

    # Key Empirical Results Table
    p_res = doc.add_paragraph()
    p_res.paragraph_format.space_before = Pt(6)
    p_res.paragraph_format.space_after = Pt(3)
    r_res = p_res.add_run("3. EMPIRICAL BENCHMARK & SIMULATION AUDIT (8,000 STEPS)")
    r_res.font.bold = True
    r_res.font.name = "Times New Roman"
    r_res.font.size = Pt(10)

    t_metrics = doc.add_table(rows=5, cols=5)
    t_metrics.autofit = False
    headers = ["Replenishment Policy", "Service Fill Rate", "Stockout Days", "Total Operating Cost", "Working Capital Impact"]
    rows_data = [
        ["Policy A: Naive Historical", "98.20%", "5 days", "$176.9k", "Heavy buffer bloat ($270.8k)"],
        ["Policy B: Statistical Min MAE", "81.40%", "24 days", "$128.5k", "Severe under-forecasting bias (-18.4%)"],
        ["Policy C: Tri-Criterion Safeguard", "100.00%", "0 days", "$9.2k", "Optimized buffer ($135.7k)"],
        ["Policy D: Dynamic Eve's Law Surge", "100.00%", "0 days", "$8.8k", "95.0% cost reduction, 49.9% capital saved"]
    ]
    
    for c_idx, h_text in enumerate(headers):
        cell = t_metrics.rows[0].cells[c_idx]
        cell.paragraphs[0].text = h_text
        cell.paragraphs[0].runs[0].font.bold = True
        cell.paragraphs[0].runs[0].font.name = "Times New Roman"
        cell.paragraphs[0].runs[0].font.size = Pt(8.5)
        set_cell_margins(cell, top=40, bottom=40, left=60, right=60)
        set_cell_border(cell, sz="6")

    for r_idx, r_data in enumerate(rows_data):
        for c_idx, val in enumerate(r_data):
            cell = t_metrics.rows[r_idx + 1].cells[c_idx]
            cell.paragraphs[0].text = val
            cell.paragraphs[0].runs[0].font.name = "Times New Roman"
            cell.paragraphs[0].runs[0].font.size = Pt(8)
            set_cell_margins(cell, top=30, bottom=30, left=60, right=60)
            set_cell_border(cell, sz="4")

    # Key Takeaways Box
    p_take = doc.add_paragraph()
    p_take.paragraph_format.space_before = Pt(6)
    p_take.paragraph_format.space_after = Pt(2)
    r_take = p_take.add_run("4. STRATEGIC MANAGERIAL & ENTERPRISE ERP TAKEAWAYS")
    r_take.font.bold = True
    r_take.font.name = "Times New Roman"
    r_take.font.size = Pt(10)

    p_ttext = doc.add_paragraph()
    p_ttext.paragraph_format.space_after = Pt(0)
    r_tt = p_ttext.add_run(
        "1. Never evaluate enterprise forecasting models purely on MAE/RMSE; always enforce an operational bias safeguard (|Bias| <= 5%).\n"
        "2. Avoid naive log1p transformations on retail counts without dual-moment retransformation smearing.\n"
        "3. Incorporate lead-time variance via Eve's Law to prevent systemic stockouts during supply chain delivery shocks.\n"
        "4. Fully reproducible: 90/90 passing automated tests, public Google Colab notebook, and Overleaf LaTeX submission bundle."
    )
    r_tt.font.name = "Times New Roman"
    r_tt.font.size = Pt(8.5)

    out_file = "EXECUTIVE_SUMMARY_HANDOUT.docx"
    doc.save(out_file)
    print(f"SUCCESS: Generated 1-page defense handout -> {out_file}")

if __name__ == "__main__":
    main()
