"""Optional check with the real embedding model (slow: loads bge-small).

Run with:  .venv\\Scripts\\python -m pytest -m slow
"""

from dataclasses import replace

import pytest

from infodeck.config import get_settings
from infodeck.engine import InfoDeckEngine
from infodeck.store import VectorStore
from tests.conftest import SAMPLE_DIR

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def real_engine(tmp_path_factory):
    from infodeck.embeddings import CrossEncoderReranker, SentenceTransformerEmbedder

    s = replace(get_settings(), chroma_dir=tmp_path_factory.mktemp("chroma"))
    store = VectorStore(s.chroma_dir, SentenceTransformerEmbedder(s.embedding_model))
    engine = InfoDeckEngine(store, s, CrossEncoderReranker(s.rerank_model))
    engine.ingest_folder(SAMPLE_DIR)
    return engine


@pytest.mark.parametrize("question, file_name, page", [
    ("Do I need a medical certificate for sick leave?", "KRT_Leave_Policy.pdf", 2),
    ("What is the hotel limit per night in Mumbai?", "KRT_Travel_and_Expense_Policy.pdf", 2),
    ("How long must my password be?", "KRT_IT_Security_Policy.pdf", 1),
])
def test_right_page_is_retrieved(real_engine, question, file_name, page):
    hits = real_engine.retrieve(question)
    assert any(h.file_name == file_name and h.page == page for h in hits)


@pytest.mark.parametrize("question, needle", [
    ("How many sick days do I get per year?", "Sick leave; Entitlement per year: 10 working days"),
    ("What should I do if I lose my company laptop?", "Lost or stolen Company devices"),
])
def test_reranker_puts_the_answer_first(real_engine, question, needle):
    assert needle in real_engine.retrieve(question)[0].text.replace("\n", " ")


def test_off_topic_question_is_not_found(real_engine):
    assert real_engine.retrieve("Who won the cricket world cup in 2011?") == []
