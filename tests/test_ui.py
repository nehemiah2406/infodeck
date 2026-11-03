"""The answer card: only the allowed parts (answer line, file + page, exact quote, Copy), values shown
exactly, document text escaped, and a lone passage only when the search match is strong."""

import re

from infodeck.engine import Answer
from infodeck.facts import Fact
from infodeck.prompts import NOT_FOUND
from infodeck.store import Passage
from infodeck.ui import (QUOTE_LABEL, card_html, card_view, clean_ai_text, confidence, copy_button_html,
                         is_strong, mark_value, snippet, snippet_text, topbar_html)

FACT = Fact("value", "Push-up distance", "10.6", unit="mm", symbol="Pd", item="9.(1)",
            line="(1) Push-up distance, 10.6 mm", file_name="calc.pdf", page=21, doc_id="d1",
            formula="Pd = Id x taper", item_title="9. Our proposal of push-up distance")
MIN = 0.55  # MIN_SCORE: a passage alone needs >= 0.70
HOTEL = "4. Accommodation\nHotel limit in Mumbai: INR 7,500 per night\nMeals are extra"
REMOVED = ["Open page", "Passages", "Answered by", "Searched", "Found", "Matched", "AI is busy", "AI answer failed",
           "show more", "All passages", "Exact value", "AI answer", "verified", "High", "Medium", "Low",
           "thumbs", "relevance", "What about", "Item 9", "Pd = Id"]


def passage(text=HOTEL, file="KRT_Travel.pdf", page=3, section="4. Accommodation", score=0.75):
    return Passage(text, file, page, section, score, "d1")


def parts(html: str) -> list[str]:
    """The card's top-level parts, in order."""
    return re.findall(r"<div class='(id-answer|id-where|id-qlabel|id-quote|id-nf)'>", html)


def assert_nothing_removed_is_shown(html: str) -> None:
    for word in REMOVED:
        assert word not in html, word


# ---- exact value --------------------------------------------------------------- #
def test_exact_card_has_only_answer_file_page_and_quote():
    view = card_view(Answer("10.6 mm", "exact", fact=FACT), "What is the push-up distance?", MIN)
    html = card_html(view)
    assert parts(html) == ["id-answer", "id-where", "id-qlabel", "id-quote"]
    assert "<span class='key'>10.6 mm</span>" in html  # the key value in maroon, exactly as in the document
    assert "<span class='file'>calc.pdf</span><span class='sep'>·</span>p. 21" in html
    assert QUOTE_LABEL in html and "<mark class='val'>10.6</mark>" in html
    assert_nothing_removed_is_shown(html)  # no item number, symbol pill, formula or "verified" line
    assert view.copy_text == "10.6 mm"


def test_exact_values_keep_separators_and_document_html_is_escaped():
    assert mark_value("A= 1,118,355 <b>", "1,118,355") == "A= <mark class='val'>1,118,355</mark> &lt;b&gt;"
    fact = Fact("value", "<script>", "1,118,355", line="A= 1,118,355", file_name="x.pdf", page=1)
    html = card_html(card_view(Answer("1,118,355", "exact", fact=fact), "q", MIN))
    assert "<script>" not in html and "<span class='key'>1,118,355</span>" in html


def test_exact_question_and_answer_fact_shows_the_answer_sentence():
    qa = Fact("qa", "Who approves changes?", "The lead engineer approves changes.", file_name="faq.docx",
              line="Q: Who approves changes?\nA: The lead engineer approves changes.")
    html = card_html(card_view(Answer(qa.value, "exact", fact=qa), "Who approves changes?", MIN))
    assert "The lead engineer approves changes." in html and parts(html)[0] == "id-answer"
    assert "p. " not in html  # no page number for a file without pages


# ---- AI answer -------------------------------------------------------------------- #
def test_ai_card_has_the_same_parts_and_drops_citations_but_not_numbers():
    ans = Answer("The hotel limit in Mumbai is **INR 7,500** per night [KRT_Travel.pdf, p.3].", "groq",
                 [passage()], answered_by="Groq")
    view = card_view(ans, "What is the hotel limit in Mumbai?", MIN)
    html = card_html(view)
    assert parts(html) == ["id-answer", "id-where", "id-qlabel", "id-quote"]
    assert view.answer_text == "The hotel limit in Mumbai is INR 7,500 per night."
    assert "<span class='key'>7,500</span>" in html and "[KRT_Travel.pdf" not in html
    assert "<mark>Mumbai</mark>" in html  # the quote: exact passage lines, question words marked
    assert "4. Accommodation" in view.quote_text  # passage text is quoted as is ...
    assert "<span class='file'>KRT_Travel.pdf</span><span class='sep'>·</span>p. 3</div>" in html  # ... no section
    assert_nothing_removed_is_shown(html.replace("4. Accommodation", ""))


def test_clean_ai_text_keeps_every_digit():
    text = "Torque is 1,118.5 N·m [a.pdf, p.2; b.pdf, p.3], see 3.2 [c.docx]."
    assert clean_ai_text(text) == "Torque is 1,118.5 N·m, see 3.2."
    # some models (seen with Groq) cite with full-width brackets
    assert clean_ai_text("It is 22 days 【KRT_Leave_Policy.pdf, p.2】.") == "It is 22 days."
    assert clean_ai_text("Keep [this note] and 【this】.") == "Keep [this note] and 【this】."  # not citations


# ---- AI failed / off: a lone passage only when strong ------------------------------------ #
def test_strong_passage_is_shown_alone_without_any_message():
    ans = Answer("", "sources", [passage(score=0.72)], note="AI is busy, showing the matching passages.",
                 ai_failed=True)
    html = card_html(card_view(ans, "What is the hotel limit in Mumbai?", MIN))
    assert parts(html) == ["id-where", "id-qlabel", "id-quote"]  # no answer line, no message
    assert "7,500" in html
    assert_nothing_removed_is_shown(html.replace("4. Accommodation", ""))


def test_weak_match_shows_no_quote_only_the_not_found_line():
    for score in (0.69, 0.60, 0.56):
        ans = Answer("", "sources", [passage(score=score)], note="AI is busy", ai_failed=True)
        html = card_html(card_view(ans, "What is the hotel limit in Mumbai?", MIN))
        assert html == f"<div class='id-nf'>{NOT_FOUND}</div>", score
        assert "7,500" not in html and QUOTE_LABEL not in html


def test_strong_means_the_high_confidence_level():
    assert is_strong(passage(score=0.70), 0.55) and not is_strong(passage(score=0.6999), 0.55)
    assert is_strong(passage(score=0.40), 0.25)  # follows MIN_SCORE
    assert confidence(0.80, False, 0.55, 0.80) == "High" and confidence(0.66, False, 0.55, 0.80) == "Medium"
    assert confidence(1.0, True, 0.55, 0.80) == "High" and confidence(0.85, True, 0.55, 0.80) == "Medium"


def test_not_found_and_error_are_one_line():
    assert card_html(card_view(Answer(NOT_FOUND, "not_found"), "q", MIN)) == f"<div class='id-nf'>{NOT_FOUND}</div>"
    view = card_view(Answer("Sorry, <b>", "error"), "q", MIN)
    assert card_html(view) == "<div class='id-nf'>Sorry, &lt;b&gt;</div>" and view.copy_text == ""


# ---- pieces ------------------------------------------------------------------------- #
def test_copy_button_safe_attribute():
    html = copy_button_html("Copy", "it's \"quoted\" <b>")
    assert "data-copy='it&#x27;s &quot;quoted&quot; &lt;b&gt;'" in html


def test_snippet_trims_around_the_match_and_highlights_words():
    text = "\n".join(["Intro line", "Other", "Something", "Hotel limit in Mumbai: INR 7,500 per night",
                      "Meals extra", "Last line", "Footer"])
    html, focus = snippet(text, "What is the hotel limit in Mumbai?")
    assert focus == "Hotel limit in Mumbai: INR 7,500 per night"
    assert "<mark>Hotel</mark>" in html and "<mark>Mumbai</mark>" in html and "7,500" in html
    assert html.startswith("<span class='more'>…</span>") and html.endswith("<span class='more'>…</span>")
    assert "Intro line" not in html and "Footer" not in html
    plain = snippet_text(text, "What is the hotel limit in Mumbai?")  # the same lines, as plain text
    assert plain == "Other\nSomething\nHotel limit in Mumbai: INR 7,500 per night"
    assert html == "<span class='more'>…</span><br>Other<br>Something<br>" \
                   "<mark>Hotel</mark> <mark>limit</mark> in <mark>Mumbai</mark>: INR 7,500 per night" \
                   "<br><span class='more'>…</span>"


def test_snippet_escapes_and_keeps_short_passages_whole():
    html, _ = snippet("Limit <b>5</b> & more", "limit")
    assert "&lt;b&gt;5&lt;/b&gt; &amp; more" in html and "…" not in html


def test_topbar_is_the_logo_only():
    assert topbar_html("InfoDeck") == "<div class='id-topbar'><div class='id-logo'>InfoDeck</div></div>"
