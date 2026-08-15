"""A small RAG with a distance threshold and content-hash dedup.

Two ideas this package exists to show:

1. A retrieval **distance threshold**: if the nearest chunk is still too far,
   we return "nothing relevant" instead of handing the model its closest
   neighbour and letting it answer anyway. See ``SemanticRAG`` and
   ``RagResult``.
2. **Content-hash dedup**: identical chunk text is embedded once and shared,
   so a corpus with repeated passages stores far fewer vectors. See
   ``NumpyStore`` / ``PgVectorStore`` and their ``stats()``.
"""

from .embeddings import Embedder, HashingEmbedder, OpenAIEmbedder, get_embedder
from .rag import RagResult, SemanticRAG
from .store import Document, NumpyStore, SearchHit, VectorStore, content_hash

__all__ = [
    "Embedder",
    "HashingEmbedder",
    "get_embedder",
    "SemanticRAG",
    "RagResult",
    "Document",
    "SearchHit",
    "VectorStore",
    "NumpyStore",
    "content_hash",
]
