import pytest

from infodeck.chunking import chunk_pages
from infodeck.loaders import Page, load_document
from tests.conftest import SAMPLE_DIR


def make_pages():
    line = "Employees must follow the rules described in this section of the policy. "
    p1 = "1. Purpose and Scope\n" + "\n".join(f"1.{i} PAGEONE {line}" for i in range(1, 20))
    p2 = "\n".join(f"2.{i} PAGETWO {line}" for i in range(1, 20))
    return [Page(1, p1), Page(2, p2)]


def test_chunks_respect_size_limit():
    chunks = chunk_pages(make_pages(), "policy.pdf", size=800, overlap=150)
    assert len(chunks) > 2
    assert all(len(c.text) <= 800 for c in chunks)


def test_consecutive_chunks_overlap():
    chunks = [c for c in chunk_pages(make_pages(), "policy.pdf", size=800, overlap=150) if c.page == 1]
    assert len(chunks) >= 2
    for a, b in zip(chunks, chunks[1:]):
        tail = a.text[-100:]
        assert tail in b.text, "the end of one chunk should be repeated at the start of the next"


def test_every_chunk_keeps_file_name_and_page_and_never_crosses_pages():
    chunks = chunk_pages(make_pages(), "policy.pdf")
    for c in chunks:
        assert c.file_name == "policy.pdf"
        assert c.page in (1, 2)
        assert ("PAGETWO" in c.text) == (c.page == 2)
        assert ("PAGEONE" in c.text) == (c.page == 1)
    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_section_heading_is_tracked():
    chunks = chunk_pages(make_pages(), "policy.pdf")
    assert chunks[0].section == "1. Purpose and Scope"
    assert chunks[-1].section == "1. Purpose and Scope"  # page 2 has no new heading


def test_very_long_line_is_split():
    chunks = chunk_pages([Page(1, "word " * 1000)], "long.txt", size=800, overlap=150)
    assert len(chunks) > 1 and all(len(c.text) <= 800 for c in chunks)


def test_overlap_must_be_smaller_than_size():
    with pytest.raises(ValueError):
        chunk_pages(make_pages(), "x.pdf", size=100, overlap=100)


def test_sample_pdf_pages_tables_and_clean_text():
    doc = load_document(SAMPLE_DIR / "KRT_Leave_Policy.pdf")
    assert doc.page_count == 4
    assert [p.number for p in doc.pages] == [1, 2, 3, 4]
    all_text = "\n".join(p.text for p in doc.pages)
    # tables become readable "Header: value" rows
    assert "Leave type: Sick leave; Entitlement per year: 10 working days" in all_text
    # repeated header/footer lines and undecodable glyphs are removed
    assert "FICTIONAL SAMPLE DOCUMENT" not in all_text
    assert "(cid:" not in all_text
    chunks = chunk_pages(doc.pages, doc.file_name)
    assert all(c.file_name == "KRT_Leave_Policy.pdf" and 1 <= c.page <= 4 for c in chunks)
