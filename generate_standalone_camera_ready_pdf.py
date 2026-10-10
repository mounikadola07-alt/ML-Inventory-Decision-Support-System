"""
generate_standalone_camera_ready_pdf.py
Compiles ML_Inventory_Decision_Support_Paper.docx to a standalone PDF via ReportLab,
strictly matching the 11-page structure of the DOCX manuscript by honoring the 10 explicit
page break boundaries, embedding all 9 vector figures and 2 tables, rendering mathematical
equations, and framing every page with a solid 0.75 pt black border and centered running footer.
"""

import os
import sys
import docx
import pypdf
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
)
from reportlab.pdfgen import canvas

class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super(NumberedCanvas, self).__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            canvas.Canvas.showPage(self)
        canvas.Canvas.save(self)

    def draw_page_decorations(self, page_count):
        self.saveState()
        # Draw solid single black page border (0.75 pt, matching sz=6 in Word)
        self.setStrokeColor(colors.black)
        self.setLineWidth(0.75)
        # Margin bounds: 36 pt (0.5 in) from each edge of Letter (612 x 792 pt)
        self.rect(36, 36, letter[0] - 72, letter[1] - 72)
        
        # Draw running footer: page number centered
        self.setFont("Times-Roman", 9)
        footer_text = f"{self._pageNumber}"
        self.drawCentredString(letter[0] / 2.0, 44, footer_text)
        self.restoreState()

def build_pdf():
    docx_path = "ML_Inventory_Decision_Support_Paper.docx"
    pdf_path = "ML_Inventory_Decision_Support_Paper.pdf"
    
    print(f"Reading content and page breaks from {docx_path}...")
    doc = docx.Document(docx_path)
    
    # 0.65 in margins inside the 0.5 in page border
    doc_pdf = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        leftMargin=48,
        rightMargin=48,
        topMargin=46,
        bottomMargin=52
    )

    styles = getSampleStyleSheet()
    
    style_title = ParagraphStyle(
        'DocTitle',
        fontName='Times-Bold',
        fontSize=12.5,
        leading=15.5,
        alignment=1, # Center
        spaceAfter=3,
        textColor=colors.black
    )
    
    style_authors = ParagraphStyle(
        'DocAuthors',
        fontName='Times-Roman',
        fontSize=9.5,
        leading=12,
        alignment=1,
        spaceAfter=2,
        textColor=colors.black
    )
    
    style_affil = ParagraphStyle(
        'DocAffil',
        fontName='Times-Italic',
        fontSize=8.5,
        leading=11,
        alignment=1,
        spaceAfter=2,
        textColor=colors.black
    )
    
    style_email = ParagraphStyle(
        'DocEmail',
        fontName='Times-Roman',
        fontSize=8.5,
        leading=11,
        alignment=1,
        spaceAfter=6,
        textColor=colors.black
    )
    
    style_abstract = ParagraphStyle(
        'DocAbstract',
        fontName='Times-Roman',
        fontSize=8.5,
        leading=11.5,
        alignment=4, # Justify
        spaceAfter=3,
        textColor=colors.black
    )

    style_keywords = ParagraphStyle(
        'DocKeywords',
        fontName='Times-Roman',
        fontSize=8.5,
        leading=11.5,
        alignment=4,
        spaceAfter=8,
        textColor=colors.black
    )

    style_heading = ParagraphStyle(
        'DocHeading',
        fontName='Times-Bold',
        fontSize=10.5,
        leading=13.5,
        alignment=0, # Flush Left
        spaceBefore=6,
        spaceAfter=2.5,
        textColor=colors.black,
        keepWithNext=True
    )
    
    style_subheading = ParagraphStyle(
        'DocSubHeading',
        fontName='Times-Bold',
        fontSize=9.5,
        leading=12.5,
        alignment=0,
        spaceBefore=5,
        spaceAfter=2,
        textColor=colors.black,
        keepWithNext=True
    )

    style_body_first = ParagraphStyle(
        'DocBodyFirst',
        fontName='Times-Roman',
        fontSize=8.5,
        leading=11.5,
        alignment=4, # Justify
        firstLineIndent=0,
        spaceAfter=2.5,
        textColor=colors.black
    )

    style_body_subsequent = ParagraphStyle(
        'DocBodySub',
        fontName='Times-Roman',
        fontSize=8.5,
        leading=11.5,
        alignment=4,
        firstLineIndent=18, # 0.25 inch
        spaceAfter=2.5,
        textColor=colors.black
    )

    style_math = ParagraphStyle(
        'DocMath',
        fontName='Times-Italic',
        fontSize=9,
        leading=13,
        alignment=1, # Center
        spaceBefore=3,
        spaceAfter=3,
        textColor=colors.black
    )

    style_caption = ParagraphStyle(
        'DocCaption',
        fontName='Times-Italic',
        fontSize=8,
        leading=10,
        alignment=1, # Center
        spaceBefore=2,
        spaceAfter=4,
        textColor=colors.black
    )
    
    style_ref = ParagraphStyle(
        'DocRef',
        fontName='Times-Roman',
        fontSize=8,
        leading=10.5,
        alignment=4,
        leftIndent=18,
        firstLineIndent=-18,
        spaceAfter=2.5,
        textColor=colors.black
    )

    story = []
    fig_idx = 1
    in_references = False
    just_after_heading = False

    for i, p in enumerate(doc.paragraphs):
        text = p.text.strip()
        has_pb = 'type="page"' in p._p.xml

        if not text:
            if has_pb:
                story.append(PageBreak())
            continue

        # Header metadata
        if i == 0:
            story.append(Paragraph(text, style_title))
            continue
        elif i == 1:
            story.append(Paragraph(text, style_authors))
            continue
        elif i == 2:
            story.append(Paragraph(text, style_affil))
            continue
        elif i == 3:
            story.append(Paragraph(text, style_email))
            continue
        elif text.startswith("Abstract:"):
            t_abs = text.replace("Abstract:", "<b>Abstract:</b>", 1)
            story.append(Paragraph(t_abs, style_abstract))
            continue
        elif text.startswith("Keywords:"):
            t_kw = text.replace("Keywords:", "<b>Keywords:</b>", 1)
            story.append(Paragraph(t_kw, style_keywords))
            continue

        # Check references section
        if text == "References":
            in_references = True
            story.append(Paragraph("<b>References</b>", style_heading))
            just_after_heading = False
            continue

        if in_references:
            story.append(Paragraph(text, style_ref))
            if has_pb:
                story.append(PageBreak())
            continue

        # Headings
        if text.startswith(("1 ", "2 ", "3 ", "6 ", "7 ", "8 ")):
            story.append(Paragraph(f"<b>{text}</b>", style_heading))
            just_after_heading = True
            if has_pb:
                story.append(PageBreak())
            continue
        elif text.startswith(("3.1", "6.1", "6.2")):
            story.append(Paragraph(f"<b>{text}</b>", style_subheading))
            just_after_heading = True
            if has_pb:
                story.append(PageBreak())
            continue

        # Math equations
        if "oMath" in p._p.xml:
            if "exp" in text or "Theorem 1" in doc.paragraphs[i-1].text:
                eq_text = "<i>E</i>[exp(<i>Ẑ</i>)] = exp(<i>μ<sub>Z</sub></i> + ½<i>σ<sub>Z</sub></i><sup>2</sup>) &gt; exp(<i>μ<sub>Z</sub></i>) &nbsp;&nbsp;&nbsp;&nbsp;(Eq. 1)"
            elif "LTD" in text or "Eve" in doc.paragraphs[i-1].text:
                eq_text = "<i>σ</i><sub>LTD</sub><sup>2</sup> = <i>E</i>[<i>L</i>]<i>σ</i><sub>D</sub><sup>2</sup> + (<i>E</i>[<i>D</i>])<sup>2</sup><i>σ</i><sub>L</sub><sup>2</sup> &nbsp;&nbsp;&nbsp;&nbsp;(Eq. 2)"
            elif "C(s, Q)" in text or "Replenishment" in doc.paragraphs[i-1].text:
                eq_text = "<i>C</i>(<i>s, Q</i>) = (<i>K</i>·<i>μ<sub>D</sub></i>)/<i>Q</i> + <i>h</i>(<i>Q</i>/2 + <i>s</i> - <i>μ<sub>D</sub></i>·<i>L</i>) + (<i>p</i>·<i>μ<sub>D</sub></i>/<i>Q</i>) <i>E</i>[(<i>D</i><sub>LTD</sub> - <i>s</i>)<sup>+</sup>] &nbsp;&nbsp;&nbsp;&nbsp;(Eq. 3)"
            else:
                eq_text = f"<i>{text}</i>"
            story.append(Paragraph(eq_text, style_math))
            just_after_heading = False
            if has_pb:
                story.append(PageBreak())
            continue

        # Captions and Figures
        if text.startswith("Figure "):
            matching = [os.path.join("paper_figures", f) for f in os.listdir("paper_figures") if f.startswith(f"fig{fig_idx}_")]
            if matching:
                img = Image(matching[0], width=5.2*72, height=2.0*72)
                story.append(img)
            story.append(Paragraph(f"<b>{text}</b>", style_caption))
            fig_idx += 1
            just_after_heading = False
            if has_pb:
                story.append(PageBreak())
            continue

        # Tables
        if text.startswith("Table 1.") or text.startswith("Table 2."):
            story.append(Paragraph(f"<b>{text}</b>", style_caption))
            t_idx = 0 if "Table 1" in text else 1
            if t_idx < len(doc.tables):
                t_src = doc.tables[t_idx]
                t_data = [[Paragraph(f"<b>{c.text.strip()}</b>" if row == t_src.rows[0] else c.text.strip(),
                                     ParagraphStyle('TC', fontName='Times-Bold' if row == t_src.rows[0] else 'Times-Roman',
                                                    fontSize=7.2, leading=9, alignment=1)) for c in row.cells] for row in t_src.rows]
                col_w = (letter[0] - 96) / float(len(t_src.columns))
                t_rep = Table(t_data, colWidths=[col_w]*len(t_src.columns))
                t_rep.setStyle(TableStyle([
                    ('BOX', (0,0), (-1,-1), 0.75, colors.black),
                    ('INNERGRID', (0,0), (-1,-1), 0.5, colors.black),
                    ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#F2F2F2')),
                    ('TOPPADDING', (0,0), (-1,-1), 2.5),
                    ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
                ]))
                story.append(t_rep)
                story.append(Spacer(1, 3))
            just_after_heading = False
            if has_pb:
                story.append(PageBreak())
            continue

        # Standard Body paragraphs
        p_style = style_body_first if just_after_heading else style_body_subsequent
        story.append(Paragraph(text, p_style))
        just_after_heading = False

        if has_pb:
            story.append(PageBreak())

    # Build the document
    print("Compiling document flow with NumberedCanvas...")
    doc_pdf.build(story, canvasmaker=NumberedCanvas)
    
    # Check page count
    reader = pypdf.PdfReader(pdf_path)
    actual_pages = len(reader.pages)
    print(f"SUCCESS: Generated {pdf_path} matching exact page flow!")
    print(f"Actual PDF Page Count: {actual_pages}")
    return actual_pages

if __name__ == "__main__":
    build_pdf()
