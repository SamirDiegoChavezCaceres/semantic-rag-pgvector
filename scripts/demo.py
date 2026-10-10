"""A guided walkthrough of the RAG: retrieval with a threshold, and dedup.

    python scripts/demo.py

Uses OpenAI embeddings when OPENAI_API_KEY is set (see .env.example), otherwise
an offline hashing embedder. Force a backend with
RAG_EMBEDDER=openai|sentence-transformers|hashing.
"""

from __future__ import annotations

import os
from pathlib import Path

from semantic_rag import Document, HashingEmbedder, SemanticRAG, get_embedder
from semantic_rag.store import NumpyStore

CORPUS = Path(__file__).resolve().parents[1] / "corpus"

QUERIES = [
    ("How do I calculate ROI?", "in-domain"),
    ("What is the middle of the funnel?", "in-domain"),
    ("What is the capital of France?", "off-domain"),
]


def pick_embedder():
    """The real path is semantic embeddings; use OpenAI when a key is set, else
    fall back to the offline hashing embedder so the demo still runs."""
    try:
        from dotenv import find_dotenv, load_dotenv

        load_dotenv(find_dotenv(usecwd=True))
    except Exception:
        pass
    choice = os.getenv("RAG_EMBEDDER")
    if not choice:
        choice = "openai" if os.getenv("OPENAI_API_KEY") else "auto"
    try:
        return get_embedder(choice)
    except Exception:
        return get_embedder("hashing")


def rule(title: str) -> None:
    print(f"\n=== {title} ===")


def main() -> None:
    embedder = pick_embedder()
    offline = isinstance(embedder, HashingEmbedder)
    # Calibrated per embedder: the in-domain/out-of-domain gap differs by model.
    max_distance = 1.2 if offline else 1.15

    rule("1. Ingest the corpus")
    rag = SemanticRAG(embedder=embedder, max_distance=max_distance)
    n = rag.ingest_directory(str(CORPUS))
    print(f"embedder : {type(embedder).__name__}" + ("  (offline, mechanics only)" if offline else ""))
    print(f"ingested : {n} chunks from {CORPUS.name}/")
    print(f"threshold: L2 distance <= {max_distance}")

    rule("2. Retrieval with a threshold")
    for query, kind in QUERIES:
        result = rag.search(query, k=3)
        if result.found:
            hit = result.hits[0]
            print(f"[{kind:10}] {query}")
            print(f"             -> match in {hit.metadata.get('source')} (distance {hit.distance:.3f})")
        else:
            print(f"[{kind:10}] {query}")
            print(f"             -> no answer ({result.message})")

    rule("3. Dedup: the same text across keys is embedded once")
    store = NumpyStore()
    blurb = "people interested in fitness and wellness"
    docs = [Document(doc_id=f"segment:{c}", text=blurb, metadata={"country": c})
            for c in ("MX", "CL", "CO", "PE")]
    docs.append(Document(doc_id="segment:cooking", text="people interested in cooking"))
    store.add_documents(docs, embedder)
    stats = store.stats()
    print(f"documents stored : {stats['documents']}")
    print(f"unique embeddings: {stats['unique_embeddings']}  (4 countries share 1 vector)")
    print(f"dedup ratio      : {stats['dedup_ratio']}x  -> {stats['storage_saved_pct']}% storage saved")


if __name__ == "__main__":
    main()
