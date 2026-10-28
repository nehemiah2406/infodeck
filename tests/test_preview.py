"""Page previews: the matched line is boxed, a missing line still gives a crop, results are cached."""

import io

import pytest
from PIL import Image

from infodeck import ocr, preview
from infodeck.preview import BOX_COLOR, can_preview, find_line, page_preview

LINE = "(1) Push-up distance, 10.6 mm"


@pytest.fixture
def calc_pdf(tmp_path):
    """A small fictional calculation sheet with a text layer."""
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    path = tmp_path / "calc.pdf"
    c = canvas.Canvas(str(path), pagesize=A4)
    c.setFont("Helvetica", 11)
    lines = ["FICTIONAL CALCULATION SHEET CS-001", "9. Our proposal of push-up distance, (Pd), mm", LINE,
             "(2) Grip stress corresponding to Pd, N/mm2 = 111 N/mm2", "(3) Safety factor against slip = 2.4"]
    for i, text in enumerate(lines * 3):
        c.drawString(60, 780 - i * 22, text if i < 5 else f"Note {i}: filler text for the page")
    c.save()
    return path


def green_pixels(png: bytes) -> int:
    im = Image.open(io.BytesIO(png)).convert("RGB")
    return sum(1 for p in im.getdata() if all(abs(a - b) < 30 for a, b in zip(p, BOX_COLOR)))


def test_text_pdf_line_is_found_and_boxed(calc_pdf, tmp_path):
    pv = page_preview(calc_pdf, 1, LINE, tmp_path / "cache")
    assert pv.boxed and green_pixels(pv.crop) > 100 and green_pixels(pv.full) > 100


def test_line_not_on_the_page_gives_a_crop_without_a_box(calc_pdf, tmp_path):
    pv = page_preview(calc_pdf, 1, "zebra quantum volcano", tmp_path / "cache")
    assert not pv.boxed and green_pixels(pv.crop) == 0 and pv.crop.startswith(b"\x89PNG")


def test_preview_is_cached_next_to_the_parsed_pages(calc_pdf, tmp_path, monkeypatch):
    cache = tmp_path / "parsed" / "doc1"
    assert page_preview(calc_pdf, 1, LINE, cache, cached_only=True) is None  # never renders
    first = page_preview(calc_pdf, 1, LINE, cache)
    assert len(list(cache.glob("preview_p1_*.png"))) == 2  # crop + full page

    def no_render(*a):
        raise AssertionError("should come from the cache")

    monkeypatch.setattr(preview, "_render", no_render)
    again = page_preview(calc_pdf, 1, LINE, cache, cached_only=True)
    assert again.crop == first.crop and again.boxed


def test_scanned_image_uses_ocr_boxes(tmp_path, monkeypatch):
    img = tmp_path / "scan.png"
    Image.new("RGB", (800, 1000), "white").save(img)
    monkeypatch.setattr(preview, "ocr_line_boxes", lambda image: [("Shaft diameter, 180 mm", (50, 80, 400, 110)),
                                                                   (LINE, (50, 300, 420, 330))])
    pv = page_preview(img, 0, LINE, tmp_path / "cache")
    assert pv.boxed and green_pixels(pv.crop) > 100


def test_only_pdfs_and_images_have_previews(tmp_path, calc_pdf):
    doc = tmp_path / "a.docx"
    doc.write_bytes(b"x")
    assert not can_preview(doc, 1) and page_preview(doc, 1, LINE, tmp_path) is None
    assert can_preview(calc_pdf, 1) and not can_preview(calc_pdf, 0)
    assert not can_preview(tmp_path / "missing.pdf", 1)


def test_find_line_prefers_the_row_with_the_value_over_the_header():
    lines = [("Grade Yield strength (MPa) Tensile strength (MPa)", (0, 0, 1, 1)),
             ("S275 275 430", (0, 2, 1, 3)), ("S355 355 470", (0, 4, 1, 5))]
    assert find_line("Grade: S355; Tensile strength (MPa): 470", lines) == (0, 4, 1, 5)
    assert find_line("nothing like this", lines) is None


def test_ocr_line_boxes_groups_words_like_the_upload_text(monkeypatch):
    class Result:
        boxes = [[(10, 10), (60, 10), (60, 30), (10, 30)], [(70, 12), (120, 12), (120, 30), (70, 30)],
                 [(10, 60), (90, 60), (90, 80), (10, 80)]]
        txts = ["Push-up", "distance", "10.6 mm"]

    monkeypatch.setattr(ocr, "_engine", lambda array, **options: Result())
    lines = ocr.ocr_line_boxes(Image.new("RGB", (200, 100)))
    assert lines == [("Push-up distance", (10, 10, 120, 30)), ("10.6 mm", (10, 60, 90, 80))]
    assert ocr.join_ocr_lines(ocr._result_boxes(Result())) == "Push-up distance\n10.6 mm"
