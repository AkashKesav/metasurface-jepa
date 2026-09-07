"""Render the architecture Markdown reference to a compact PDF."""

from pathlib import Path

from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (PageBreak, Paragraph, Preformatted, SimpleDocTemplate,
                                Spacer)


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs" / "UNIFIED_ORIGINAL_ARCHITECTURE.md"
OUTPUT = ROOT / "docs" / "UNIFIED_ORIGINAL_ARCHITECTURE.pdf"


def build_story(text):
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="TitleCenter", parent=styles["Title"], alignment=TA_CENTER))
    styles.add(ParagraphStyle(name="H1x", parent=styles["Heading1"], spaceBefore=14, spaceAfter=7))
    styles.add(ParagraphStyle(name="H2x", parent=styles["Heading2"], spaceBefore=10, spaceAfter=5))
    styles.add(ParagraphStyle(name="Bodyx", parent=styles["BodyText"], leading=14, spaceAfter=6))
    styles.add(ParagraphStyle(name="Codex", parent=styles["Code"], fontName="Courier", fontSize=7.5,
                              leading=9.2, leftIndent=8, rightIndent=8, spaceBefore=4, spaceAfter=7))

    story = []
    in_code = False
    code = []
    for raw in text.splitlines():
        line = raw.rstrip()
        if line.strip() == "```":
            if in_code:
                story.append(Preformatted("\n".join(code), styles["Codex"]))
                code = []
            in_code = not in_code
            continue
        if in_code:
            code.append(line)
            continue
        if not line:
            story.append(Spacer(1, 4))
        elif line.startswith("# "):
            story.append(Paragraph(line[2:], styles["TitleCenter"]))
        elif line.startswith("## "):
            story.append(Paragraph(line[3:], styles["H1x"]))
        elif line.startswith("### "):
            story.append(Paragraph(line[4:], styles["H2x"]))
        elif line.startswith("- "):
            story.append(Paragraph("• " + line[2:], styles["Bodyx"]))
        else:
            safe = (line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
            story.append(Paragraph(safe, styles["Bodyx"]))
    return story


def main():
    doc = SimpleDocTemplate(str(OUTPUT), pagesize=A4, rightMargin=.7*inch,
                            leftMargin=.7*inch, topMargin=.65*inch, bottomMargin=.65*inch,
                            title="Unified JEPA — Original Present Architecture")
    doc.build(build_story(SOURCE.read_text(encoding="utf-8")))
    print(OUTPUT)


if __name__ == "__main__":
    main()
