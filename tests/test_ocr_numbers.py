"""Broken OCR numbers ("10).6 mm"): found, read again from the image, and replaced only when
the second read confirms them; never guessed. No network; fictional data only."""

from __future__ import annotations

from PIL import Image

from infodeck import ocr
from infodeck.parsing import NUMBER_CHECK, pages_to_recheck, read_meta, write_meta
from infodeck.store import file_hash
from tests.conftest import FakeLLM, llm_status
from tests.test_fast_upload import PAGES, FakePdfReader, page_text

BROKEN_LINE = "(1) Push-up distance, Pd = 10).6 mm"
CALC_PAGE = "9. Our proposal of push-up distance, (Pd), mm\n{line}\n(2) Grip stress, N/mm²"


# ---- finding broken numbers ------------------------------------------------- #
def test_broken_numbers_are_found():
    assert ocr.broken_numbers(BROKEN_LINE) == ["10).6"]
    assert ocr.broken_numbers("where, f = (dm / 8GH) = (0).534") == ["(0).534"]
    assert ocr.broken_numbers("Pu = 0.8 x (K3 + θ) x10^-3 = 5,940) kN") == ["5,940)"]


def test_normal_brackets_are_not_broken_numbers():
    for line in ["(1) Push-up distance, Pd = 10.6 mm", "1) Remove the cover", "see (Table 3.2)",
                 "K3 : 0.025(Oil injection)", "σe = ( 3 + f4 )^0.5 = 279.51", "Load (kN) 540"]:
        assert ocr.broken_numbers(line) == [], line


# ---- the second read decides ------------------------------------------------- #
def test_number_is_fixed_only_when_every_second_read_agrees():
    scales = []

    def reread(scale):
        scales.append(scale)
        return "(1) Push-up distance, Pd = 10.6 mm"

    assert ocr.recheck_numbers(BROKEN_LINE, reread) == "(1) Push-up distance, Pd = 10.6 mm"
    assert scales == list(ocr.RECHECK_SCALES)  # read again as it is and enlarged


def test_number_is_kept_as_read_when_a_second_read_disagrees():
    # the second read sees other digits: never pick one, keep what was read
    assert ocr.recheck_numbers(BROKEN_LINE, lambda s: "(1) Push-up distance, Pd = 10.8 mm") == BROKEN_LINE
    # one read says 5.940, the other 5,940: a comma or a point changes the value 1000x
    line = "Pu = 0.8 x (K3 + θ) x10^-3 = 5,940) kN"
    reads = {1.0: "Pu = 0.8 x (K3 + θ) x10^-3 = 5.940 kN", 2.0: "Pu = 0.8 x (K3 + θ) x10^-3 = 5,940 kN"}
    assert ocr.recheck_numbers(line, reads.get) == line
    # the second read is still broken, or finds an extra number, or fails
    assert ocr.recheck_numbers(BROKEN_LINE, lambda s: BROKEN_LINE) == BROKEN_LINE
    assert ocr.recheck_numbers(BROKEN_LINE, lambda s: "(1) Push-up distance, Pd = 1 10.6 mm") == BROKEN_LINE

    def crash(scale):
        raise RuntimeError("OCR failed")

    assert ocr.recheck_numbers(BROKEN_LINE, crash) == BROKEN_LINE


def test_only_the_bracket_is_removed_nothing_else_changes():
    line = "(1) Pd = 10).6 mm, f = (0).534"
    reread = lambda s: "(1) Pd = 10.6 mrn, f = 0.534"  # second read has another letter mistake
    assert ocr.recheck_numbers(line, reread) == "(1) Pd = 10.6 mm, f = 0.534"


# ---- through the OCR engine ---------------------------------------------------- #
class FakeOcrEngine:
    """Whole page: one box with a broken number. Crops (recognition only): `crop_text`."""

    def __init__(self, crop_text: str):
        self.crop_text = crop_text
        self.crops: list[tuple[int, int]] = []

    def __call__(self, array, use_det=None, use_cls=None, use_rec=None):
        if use_det is False:  # re-reading one cut-out box
            self.crops.append(array.shape[:2])
            return type("Rec", (), {"txts": (self.crop_text,)})()
        return type("Page", (), {"boxes": [[(10, 20), (400, 20), (400, 50), (10, 50)]], "txts": [BROKEN_LINE]})()


def test_ocr_reads_the_broken_line_again_from_the_page_image(monkeypatch):
    engine = FakeOcrEngine("(1) Push-up distance, Pd = 10.6 mm")
    monkeypatch.setattr(ocr, "_engine", engine)
    assert ocr.ocr_image(Image.new("RGB", (800, 600))) == "(1) Push-up distance, Pd = 10.6 mm"
    # the box was cut out (with a small margin), once as it is and once enlarged 2x
    assert engine.crops == [(49, 409), (98, 818)]
    # page previews use the same text, so the highlight finds the value
    assert ocr.ocr_line_boxes(Image.new("RGB", (800, 600)))[0][0] == "(1) Push-up distance, Pd = 10.6 mm"


def test_ocr_keeps_the_line_when_the_image_does_not_confirm_it(monkeypatch):
    monkeypatch.setattr(ocr, "_engine", FakeOcrEngine("(1) Push-up distance, Pd = 16.6 mm"))
    assert ocr.ocr_image(Image.new("RGB", (800, 600))) == BROKEN_LINE


def test_confirmed_number_gives_an_exact_value_card(engine, tmp_path):
    question = "What is the proposed push-up distance Pd?"
    broken = tmp_path / "broken_calc.txt"
    broken.write_text(CALC_PAGE.format(line=BROKEN_LINE), encoding="utf-8")
    engine.ingest_file(broken)
    assert engine.answer(question, llm_status(FakeLLM())).mode != "exact"  # the bug: no card for "10).6"

    engine.remove_document(file_hash(broken)[:16])
    fixed = tmp_path / "calc.txt"
    fixed.write_text(CALC_PAGE.format(line="(1) Push-up distance, Pd = 10.6 mm"), encoding="utf-8")
    engine.ingest_file(fixed)
    ans = engine.answer(question, llm_status(FakeLLM()))
    assert ans.mode == "exact" and ans.text == "10.6 mm"


# ---- documents read before this fix --------------------------------------------- #
def test_old_cache_lists_only_pages_with_broken_numbers(tmp_path):
    (tmp_path / "page_1.md").write_text("Pd = 10.6 mm", encoding="utf-8")
    (tmp_path / "page_2.md").write_text("title\n" + BROKEN_LINE, encoding="utf-8")
    old = {"parser": "local", "complete": True, "pages": [1, 2]}
    assert pages_to_recheck(tmp_path, old) == [2]
    assert pages_to_recheck(tmp_path, {**old, "number_check": NUMBER_CHECK}) == []  # already re-checked


def test_reupload_of_an_old_cache_reads_only_the_broken_pages_again(engine, monkeypatch, tmp_path):
    data = b"%PDF fictional calc"
    probe = tmp_path / "probe.pdf"
    probe.write_bytes(data)
    folder = engine.settings.parsed_dir / file_hash(probe)[:16]
    folder.mkdir(parents=True)
    for n in range(1, PAGES + 1):  # read completely before this fix; page 3 has a broken number
        text = page_text(n).replace("3.5 mm", "3).5 mm") if n == 3 else page_text(n)
        (folder / f"page_{n}.md").write_text(text, encoding="utf-8")
    write_meta(folder, {"file_name": "calc_scan.pdf", "parser": "local", "complete": True,
                        "page_count": PAGES, "pages": list(range(1, PAGES + 1))})

    reader = FakePdfReader()
    monkeypatch.setattr("infodeck.engine.iter_pdf_pages", reader)
    r = engine.queue_upload("calc_scan.pdf", data)
    assert r.status == "queued" and engine.jobs.wait(20)
    assert engine.jobs.get(r.doc_id).result.status == "added"
    assert sorted(set(range(1, PAGES + 1)) - set(reader.known)) == [3]  # only page 3 read again
    assert "3.5 mm" in (folder / "page_3.md").read_text(encoding="utf-8")
    meta = read_meta(folder)
    assert meta["complete"] and meta["number_check"] == NUMBER_CHECK
