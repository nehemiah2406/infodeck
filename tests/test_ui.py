"""The answer UI's HTML pieces: they must show values exactly, escape document text, and use real data."""

from infodeck.engine import Answer
from infodeck.facts import Fact
from infodeck.prompts import NOT_FOUND
from infodeck.store import Passage
from infodeck.ui import (Trace, cited_html, confidence, copy_button_html, copy_label, exact_html, follow_ups,
                         has_numbers, mark_value, not_found_html, snippet, topbar_html, trail_html, trail_steps,
                         unique_sources)

FACT = Fact("value", "Push-up distance", "10.6", unit="mm", symbol="Pd", item="9.(1)",
            line="(1) Push-up distance, 10.6 mm", file_name="calc.pdf", page=1, doc_id="d1")


def passage(text="text", file="a.pdf", page=1, section="", score=0.8):
    return Passage(text, file, page, section, score, "d1")


# ---- confidence ------------------------------------------------------------- #
def test_confidence_levels_come_from_scores_relative_to_the_settings():
    assert confidence(1.0, True, 0.55, 0.80) == "High"
    assert confidence(0.85, True, 0.55, 0.80) == "Medium"
    assert confidence(0.80, False, 0.55, 0.80) == "High"
    assert confidence(0.66, False, 0.55, 0.80) == "Medium"
    assert confidence(0.58, False, 0.55, 0.80) == "Low"


# ---- process trail ------------------------------------------------------------ #
def test_trail_for_exact_value_names_the_item():
    ans = Answer("10.6 mm", "exact", fact=FACT)
    steps = trail_steps(ans, Trace(docs=6))
    assert [s for s, _ in steps] == ["Searched 6 documents", "Matched item 9.(1)", "Value verified in source"]


def test_trail_for_ai_answer_uses_real_counts_and_number_check():
    trace = Trace(docs=1, retrieved=[passage()] * 5)
    ok = trail_steps(Answer("x", "gemini"), trace)
    assert [s for s, _ in ok] == ["Searched 1 document", "Read 5 passages", "Gemini wrote answer", "Numbers checked"]
    bad = trail_steps(Answer("x", "gemini", unverified=["12"]), trace)
    assert bad[-1] == ("Numbers to check", "warn")


def test_trail_for_sources_and_not_found():
    quota = Answer("", "sources", note="AI answer failed: The Gemini quota for this API key is used up for today.")
    assert trail_steps(quota, Trace(docs=2, retrieved=[passage()]))[-1] == ("Quota used up", "warn")
    assert trail_steps(Answer("", "sources", note="No key"), Trace())[-1] == ("AI off", "warn")
    assert trail_steps(Answer(NOT_FOUND, "not_found"), Trace(docs=3))[-1][1] == "bad"


def test_trail_html_shows_elapsed_seconds():
    assert "· 0.4 s" in trail_html([("Searched", "ok")], 0.42)


# ---- exact value --------------------------------------------------------------- #
def test_exact_card_shows_value_and_unit_exactly_and_marks_it_in_the_quote():
    html = exact_html(Answer("10.6 mm", "exact", fact=FACT))
    assert "<span class='num'>10.6</span><span class='unit'>mm</span>" in html
    assert "Item 9.(1)" in html and "Pd" in html and "word for word" in html
    assert "<mark class='val'>10.6</mark>" in html


def test_exact_values_keep_separators_and_document_html_is_escaped():
    assert mark_value("A= 1,118,355 <b>", "1,118,355") == "A= <mark class='val'>1,118,355</mark> &lt;b&gt;"
    fact = Fact("value", "<script>", "1,118,355", line="A= 1,118,355")
    html = exact_html(Answer("1,118,355", "exact", fact=fact))
    assert "<script>" not in html and "1,118,355" in html


def test_copy_button_label_and_safe_attribute():
    assert copy_label(Answer("10.6 mm", "exact", fact=FACT)) == "Copy 10.6 mm"
    assert copy_label(Answer("text", "gemini")) == "Copy"
    html = copy_button_html("Copy", "it's \"quoted\" <b>")
    assert "data-copy='it&#x27;s &quot;quoted&quot; &lt;b&gt;'" in html


# ---- AI answer -------------------------------------------------------------------- #
def test_citations_become_numbered_badges_matching_the_source_cards():
    sources = unique_sources([passage(file="a.pdf", page=2), passage(file="a.pdf", page=2, score=0.9),
                              passage(file="b.pdf", page=3)])
    assert [p.label for p in sources] == ["a.pdf, p.2", "b.pdf, p.3"] and sources[0].score == 0.9
    html = cited_html("Limit is 25 bar [a.pdf, p.2; b.pdf, p.3] and <x> [c.pdf, p.1]", sources)
    assert "title='a.pdf, p.2'>1</span>" in html and "title='b.pdf, p.3'>2</span>" in html
    assert "<span class='chip'>c.pdf, p.1</span>" in html  # not a known source: shown as text
    assert "&lt;x&gt;" in html and "25 bar" in html


def test_has_numbers_ignores_citation_brackets():
    assert has_numbers("It is 25 bar [a.pdf, p.2]")
    assert not has_numbers("Ask the lead engineer [a.pdf, p.2]")


def test_follow_ups_come_from_other_headings_without_repeats():
    ps = [passage(section="3. Annual leave"), passage(section="4. Sick leave"), passage(section="4. Sick leave"),
          passage(section="5. Public holidays, list (2026)"), passage(section="6. Carry forward"),
          passage(section="")]
    qs = follow_ups("How many annual leave days?", ps)
    assert qs == ["What about sick leave?", "What about public holidays?", "What about carry forward?"]


def test_follow_ups_keep_acronyms():
    assert follow_ups("laptop", [passage(section="2. IT equipment")]) == ["What about IT equipment?"]


# ---- sources mode ------------------------------------------------------------------- #
def test_snippet_trims_around_the_match_and_highlights_words():
    text = "\n".join(["Intro line", "Other", "Something", "Hotel limit in Mumbai: INR 7,500 per night",
                      "Meals extra", "Last line", "Footer"])
    html, focus = snippet(text, "What is the hotel limit in Mumbai?")
    assert focus == "Hotel limit in Mumbai: INR 7,500 per night"
    assert "<mark>Hotel</mark>" in html and "<mark>Mumbai</mark>" in html and "7,500" in html
    assert html.startswith("<span class='more'>…</span>") and html.endswith("<span class='more'>…</span>")
    assert "Intro line" not in html and "Footer" not in html


def test_snippet_escapes_and_keeps_short_passages_whole():
    html, _ = snippet("Limit <b>5</b> & more", "limit")
    assert "&lt;b&gt;5&lt;/b&gt; &amp; more" in html and "…" not in html


# ---- not found, top bar ------------------------------------------------------------- #
def test_not_found_card_is_one_line_and_three_tips():
    html = not_found_html()
    assert NOT_FOUND in html and html.count("<li>") == 3


def test_topbar_counts():
    html = topbar_html("InfoDeck", 1, 204, "Gemini ready", "ok")
    assert "1 doc<" in html and "204 pages indexed" in html and "Gemini ready" in html
