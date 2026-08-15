"""End-to-end demo: ingest the sample corpus and run a few queries.

    python scripts/demo.py

By default it uses the dependency-free hashing embedder and the in-memory
store, so it runs with nothing but numpy installed. Install the ``semantic``
extra and set ``RAG_EMBEDDER=sentence-transformers`` for real semantics.
"""

from __future__ import annotations

import os
from pathlib import Path

from semantic_rag import HashingEmbedder, SemanticRAG, get_embedder

CORPUS = Path(__file__).resolve().parents[1] / "corpus"

QUERIES = [
    "How do I calculate ROI?",            # in-domain
    "What is the middle of the funnel?",  # in-domain
    "What is the capital of France?",     # off-domain -> should find nothing
]


def main() -> None:
    embedder = get_embedder(os.getenv("RAG_EMBEDDER", "auto"))
    # The threshold depends on the embedder. The hashing fallback has a narrow
    # relevant/irrelevant band, so it needs a looser cut; real embeddings
    # (sentence-transformers) separate much more cleanly. Calibrate on yours.
    if isinstance(embedder, HashingEmbedder):
        max_distance = 1.2
        print(
            "NOTE: using the offline hashing embedder (bag-of-words). It shows "
            "the mechanics only.\n      Install the 'semantic' extra and set "
            "RAG_EMBEDDER=sentence-transformers for real semantics.\n"
        )
    else:
        max_distance = 1.0

    rag = SemanticRAG(embedder=embedder, max_distance=max_distance)
    ingested = rag.ingest_directory(str(CORPUS))
    print(f"Ingested {ingested} chunks (threshold {max_distance}). "
          f"Store stats: {rag.store.stats()}\n")

    for query in QUERIES:
        result = rag.search(query, k=3)
        print(f"Q: {query}")
        if result.found:
            best = result.hits[0]
            print(f"   found (distance {best.distance:.3f}) from {best.metadata.get('source')}")
            print(f"   > {best.text.splitlines()[0][:80]}")
        else:
            print(f"   no answer: {result.message}")
        print()


if __name__ == "__main__":
    main()
