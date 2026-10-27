"""Generate FICTIONAL engineering documents for the answer evaluation (scripts/eval.py).

Run:  .venv\\Scripts\\python scripts\\make_eval_docs.py
Output: data/eval_docs/
  - KRT_Coupling_Calc_Sheet_scanned.pdf  image-only scan of a calculation sheet, with a BAD hidden
                                         OCR text layer ("I I I" instead of 111, "Ora" instead of σra)
  - KRT_Material_Properties.pdf          table-heavy text PDF
  - KRT_Bolt_Torque_Schedule.xlsx        Excel workbook with two sheets
  - KRT_Engineering_FAQ.docx             Q&A list

All companies, parts and values are made up. These files are safe to commit.
"""

from __future__ import annotations

import io
import random
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "eval_docs"
DPI = 200
A4_PX = (1654, 2339)  # A4 at 200 dpi

CALC_PAGES = [
    [
        "KESTREL RIDGE TECHNOLOGIES LTD. - CALCULATION SHEET CS-042 (FICTIONAL)",
        "Shrink-fit coupling for hub H-42",
        "",
        "8. Interference fit data",
        "   (1) Shaft diameter, 180 mm",
        "   (2) Hub outer diameter, 320 mm",
        "   (3) Contact length, 210 mm",
        "",
        "9. Our proposal of push-up distance, (Pd), mm",
        "   (1) Push-up distance, 10.6 mm",
        "   (2) Grip stress corresponding to Pd, N/mm²   σra = Pd / Pr x σr = 111 N/mm²",
        "   (3) Safety factor against slip,   S = μ x σra x A / F = 2.4",
        "",
        "10. Variables",
        "   A : Interfacial surface area .................. A= 1,118,355",
        "   Pr : Reference push-up distance, mm ........... Pr = 8.2",
        "   σr : Grip stress at Pr, N/mm² ................. σr = 85.8",
        "   μ : Friction coefficient ...................... μ = 0.14",
        "   Θ : Taper angle, deg .......................... Θ = 1.5",
    ],
    [
        "KESTREL RIDGE TECHNOLOGIES LTD. - CALCULATION SHEET CS-042 (FICTIONAL)",
        "",
        "11. Hydraulic oil injection",
        "   (1) Required oil pressure, 142 N/mm²",
        "   (2) Maximum pump pressure, 200 N/mm²",
        "",
        "12. Remarks",
        "   The push-up distance shall be checked with a dial gauge at 20 °C.",
        "   Checked by: J. Doe (fictional)",
    ],
]

# What a poor scanner OCR made of the pages: this goes into the hidden text layer.
BAD_OCR = {"111": "I I I", "σra": "Ora", "σr": "Or", "(1)": "(l)", "10.6": "l0.6", "1,118,355": "l,ll8,355",
           "μ": "u", "Θ": "O", "²": "2", "142": "l42"}


def _font(size: int):
    from PIL import ImageFont

    for name in ("arial.ttf", "Arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    raise SystemExit("No TrueType font with Greek letters found (need Arial or DejaVu Sans).")


def render_scan(lines: list[str], seed: int = 0):
    """A page image that looks like a slightly crooked, noisy scan."""
    from PIL import Image, ImageDraw, ImageFilter

    img = Image.new("L", A4_PX, 255)
    draw = ImageDraw.Draw(img)
    font = _font(30)
    y = 160
    for line in lines:
        draw.text((150, y), line, fill=20, font=font)
        y += 58
    rng = random.Random(seed)
    pixels = img.load()
    for _ in range(4000):  # specks of scanner dust
        pixels[rng.randrange(A4_PX[0]), rng.randrange(A4_PX[1])] = rng.randrange(120, 230)
    img = img.rotate(0.4, fillcolor=255, resample=Image.BICUBIC).filter(ImageFilter.GaussianBlur(0.6))
    return img.convert("RGB")


def corrupt(text: str) -> str:
    for good, bad in BAD_OCR.items():
        text = text.replace(good, bad)
    return text


def make_scanned_pdf(path: Path, pages: list[list[str]], bad_text_layer: bool = True) -> None:
    """Image-only PDF pages, optionally with an invisible (bad) OCR text layer on top."""
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfgen import canvas

    c = canvas.Canvas(str(path), pagesize=A4)
    width, height = A4
    for n, lines in enumerate(pages):
        buf = io.BytesIO()
        render_scan(lines, seed=n).save(buf, format="PNG")
        buf.seek(0)
        c.drawImage(ImageReader(buf), 0, 0, width=width, height=height)
        if bad_text_layer:
            t = c.beginText()
            t.setTextRenderMode(3)  # invisible, like a scanner's OCR layer
            t.setFont("Helvetica", 10.8)
            y_pt = height - 160 * 72 / DPI - 10
            for line in lines:
                t.setTextOrigin(150 * 72 / DPI, y_pt)
                t.textLine(corrupt(line).encode("latin-1", "replace").decode("latin-1"))
                y_pt -= 58 * 72 / DPI
            c.drawText(t)
        c.showPage()
    c.save()


def make_material_pdf(path: Path) -> None:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    styles = getSampleStyleSheet()

    def table(rows):
        t = Table(rows, hAlign="LEFT")
        t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                               ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eef7")),
                               ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"), ("FONTSIZE", (0, 0), (-1, -1), 9)]))
        return t

    story = [
        Paragraph("Kestrel Ridge Technologies Ltd. - Material Properties (FICTIONAL)", styles["Title"]),
        Paragraph("1. Structural and shaft steels", styles["Heading2"]),
        table([["Grade", "Yield strength (MPa)", "Tensile strength (MPa)", "Elongation (%)", "Density (kg/m³)"],
               ["S275", "275", "430", "23", "7850"],
               ["S355", "355", "470", "22", "7850"],
               ["42CrMo4", "650", "900", "12", "7850"],
               ["AISI 316", "205", "515", "40", "8000"]]),
        Spacer(1, 12),
        Paragraph("2. Allowable stresses for coupling design", styles["Heading2"]),
        table([["Component", "Material", "Allowable stress (MPa)", "Safety factor"],
               ["Hub", "42CrMo4", "430", "1.5"],
               ["Shaft", "S355", "236", "1.5"],
               ["Key", "C45", "180", "2.0"]]),
    ]
    SimpleDocTemplate(str(path), pagesize=A4).build(story)


def make_torque_xlsx(path: Path) -> None:
    import openpyxl

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Torque"
    ws.append(["Bolt size", "Property class", "Tightening torque (N·m)", "Preload (kN)"])
    for row in [["M12", "8.8", 79, 37.9], ["M16", "8.8", 195, 70.6], ["M20", "10.9", 540, 167],
                ["M24", "10.9", 935, 241]]:
        ws.append(row)
    ws2 = wb.create_sheet("Inspection")
    ws2.append(["Check", "Interval (months)", "Responsible"])
    for row in [["Coupling alignment check", 6, "Maintenance"], ["Bolt torque re-check", 12, "Maintenance"],
                ["Hub crack inspection", 24, "Quality"]]:
        ws2.append(row)
    wb.save(path)


FAQ = [
    ("What dial gauge resolution is required for push-up measurement?",
     "A dial gauge with 0.01 mm resolution is required."),
    ("Who approves changes to the calculation sheet?",
     "Changes must be approved by the lead mechanical engineer before release."),
    ("How long must calculation records be kept?",
     "Calculation records must be kept for 10 years after decommissioning."),
    ("Can the coupling be mounted without oil injection?",
     "No. Oil injection is mandatory for hub H-42."),
]


def make_faq_docx(path: Path, pairs=FAQ) -> None:
    import docx

    d = docx.Document()
    d.add_heading("Kestrel Ridge Technologies Ltd. - Coupling FAQ (FICTIONAL)", level=1)
    for q, a in pairs:
        d.add_paragraph(f"Q: {q}")
        d.add_paragraph(f"A: {a}")
        d.add_paragraph("")
    d.save(path)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    make_scanned_pdf(OUT_DIR / "KRT_Coupling_Calc_Sheet_scanned.pdf", CALC_PAGES)
    make_material_pdf(OUT_DIR / "KRT_Material_Properties.pdf")
    make_torque_xlsx(OUT_DIR / "KRT_Bolt_Torque_Schedule.xlsx")
    make_faq_docx(OUT_DIR / "KRT_Engineering_FAQ.docx")
    for p in sorted(OUT_DIR.iterdir()):
        print(f"wrote {p.relative_to(OUT_DIR.parent.parent)}")


if __name__ == "__main__":
    main()
