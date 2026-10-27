"""ChromaDB vector store: add, search, list and remove documents.

Every document is identified by the SHA-256 hash of its bytes, so uploading the
same file twice (even under a different name) never creates duplicates.
"""

from __future__ import annotations

import hashlib
import threading
from dataclasses import dataclass
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings

from .chunking import Chunk
from .embeddings import Embedder

COLLECTION = "infodeck"


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


@dataclass
class Passage:
    text: str 
    file_name: str
    page: int
    section: str
    score: float  # cosine similarity, 1.0 = identical meaning
    doc_id: str

    @property
    def label(self) -> str:
        """Citation label, e.g. 'KRT_Leave_Policy.pdf, p.2'."""
        return f"{self.file_name}, p.{self.page}" if self.page else self.file_name


@dataclass
class DocumentInfo:
    doc_id: str
    file_name: str
    pages: int
    chunks: int
    source_path: str


class VectorStore:
    def __init__(self, path: Path, embedder: Embedder):
        Path(path).mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(path), settings=ChromaSettings(anonymized_telemetry=False)
        )
        self._col = self._client.get_or_create_collection(
            COLLECTION, configuration={"hnsw": {"space": "cosine"}}
        )
        self.embedder = embedder
        self._lock = threading.Lock()  # Streamlit serves several users from one process

    # ---- documents ------------------------------------------------------- #
    def has_document(self, doc_id: str) -> bool:
        return bool(self._col.get(where={"doc_id": doc_id}, limit=1)["ids"])

    def find_by_name(self, file_name: str) -> list[str]:
        metas = self._col.get(where={"file_name": file_name}, include=["metadatas"])["metadatas"]
        return sorted({m["doc_id"] for m in metas})

    def add_document(self, doc_id: str, chunks: list[Chunk], page_count: int, source_path: str) -> int:
        if not chunks:
            return 0
        title = Path(chunks[0].file_name).stem.replace("_", " ")
        # The embedded text gets a short "title | section" header so each chunk carries its
        # context; the stored/displayed text stays exactly as in the document.
        to_embed = [f"{title} | {c.section}\n{c.text}" if c.section else f"{title}\n{c.text}" for c in chunks]
        vectors = self.embedder.embed_documents(to_embed)
        with self._lock:
            self._col.add(
                ids=[f"{doc_id}-{c.index}" for c in chunks],
                embeddings=vectors,
                documents=[c.text for c in chunks],
                metadatas=[{
                    "doc_id": doc_id, "file_name": c.file_name, "page": c.page, "section": c.section,
                    "chunk_index": c.index, "page_count": page_count, "source_path": source_path,
                } for c in chunks],
            )
        return len(chunks)

    def remove_document(self, doc_id: str) -> None:
        with self._lock:
            self._col.delete(where={"doc_id": doc_id})

    def list_documents(self) -> list[DocumentInfo]:
        metas = self._col.get(include=["metadatas"])["metadatas"]
        docs: dict[str, DocumentInfo] = {}
        for m in metas:
            d = docs.get(m["doc_id"])
            if d is None:
                docs[m["doc_id"]] = DocumentInfo(m["doc_id"], m["file_name"], m["page_count"], 1, m["source_path"])
            else:
                d.chunks += 1
        return sorted(docs.values(), key=lambda d: d.file_name.lower())

    def count(self) -> int:
        return self._col.count()

    # ---- search ---------------------------------------------------------- #
    def search(self, question: str, k: int = 5) -> list[Passage]:
        if self.count() == 0:
            return []
        res = self._col.query(
            query_embeddings=[self.embedder.embed_query(question)],
            n_results=min(k, self.count()),
            include=["documents", "metadatas", "distances"],
        )
        return [
            Passage(text=doc, file_name=m["file_name"], page=m["page"], section=m["section"],
                    score=round(1.0 - dist, 4), doc_id=m["doc_id"])
            for doc, m, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])
        ]
