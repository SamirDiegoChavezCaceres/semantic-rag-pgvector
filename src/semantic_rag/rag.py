"""The RAG orchestrator: chunk, embed, store, and retrieve with a threshold.

The one opinionated part is :meth:`SemanticRAG.search`. A plain nearest-
neighbour search *always* returns its ``k`` closest chunks, however far away
they are, so an off-topic question still walks away with text it can present
as an answer. Here, if the nearest chunk is farther than ``max_distance`` the
result is marked ``found = False`` and carries no context, so the caller (and
the model) can honestly say "nothing in the knowledge base covers that".
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from .embeddings import Embedder, get_embedder
from .store import Document, NumpyStore, SearchHit, VectorStore

# Default L2 threshold. With L2-normalized embeddings, distance runs 0 (same)
# to ~1.41 (unrelated). Calibrate against your own corpus; see the README.
DEFAULT_MAX_DISTANCE = 1.0


@dataclass
class RagResult:
    """Outcome of a retrieval.

    ``found`` is False when the search ran fine but nothing cleared the
    distance threshold - a legitimate empty answer, not an error.
    """

    found: bool
    hits: List[SearchHit] = field(default_factory=list)
    message: str = ""

    @property
    def context(self) -> str:
        return "\n\n---\n\n".join(hit.text for hit in self.hits)


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> List[str]:
    """Split on paragraphs, then pack into ~``chunk_size`` windows with overlap."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: List[str] = []
    buffer = ""
    for paragraph in paragraphs:
        if buffer and len(buffer) + len(paragraph) + 2 > chunk_size:
            chunks.append(buffer)
            buffer = buffer[-overlap:] + "\n\n" + paragraph if overlap else paragraph
        else:
            buffer = f"{buffer}\n\n{paragraph}" if buffer else paragraph
    if buffer:
        chunks.append(buffer)
    return chunks


class SemanticRAG:
    def __init__(
        self,
        embedder: Optional[Embedder] = None,
        store: Optional[VectorStore] = None,
        max_distance: float = DEFAULT_MAX_DISTANCE,
    ) -> None:
        self.embedder = embedder or get_embedder()
        self.store = store or NumpyStore()
        self.max_distance = max_distance

    def ingest_texts(self, texts: List[str], source: str = "doc") -> int:
        docs: List[Document] = []
        for text in texts:
            for i, chunk in enumerate(chunk_text(text)):
                docs.append(
                    Document(doc_id=f"{source}:{len(docs)}:{i}", text=chunk,
                             metadata={"source": source})
                )
        return self.store.add_documents(docs, self.embedder)

    def ingest_directory(self, path: str, patterns=(".md", ".txt")) -> int:
        root = Path(path)
        total = 0
        for file in sorted(root.rglob("*")):
            if file.suffix.lower() in patterns and file.is_file():
                text = file.read_text(encoding="utf-8")
                chunks = chunk_text(text)
                docs = [
                    Document(doc_id=f"{file.name}:{i}", text=chunk,
                             metadata={"source": file.name})
                    for i, chunk in enumerate(chunks)
                ]
                total += self.store.add_documents(docs, self.embedder)
        return total

    def search(self, query: str, k: int = 3) -> RagResult:
        query_vector = self.embedder.embed_query(query)
        candidates = self.store.search(query_vector, k=k)
        if not candidates:
            return RagResult(found=False, message="The knowledge base is empty.")
        kept = [hit for hit in candidates if hit.distance <= self.max_distance]
        if not kept:
            return RagResult(
                found=False,
                message=(
                    "No relevant document found "
                    f"(nearest distance {candidates[0].distance:.3f} "
                    f"> threshold {self.max_distance:.3f})."
                ),
            )
        return RagResult(found=True, hits=kept)
