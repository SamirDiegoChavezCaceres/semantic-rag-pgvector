"""Vector stores with content-hash deduplication.

Two backends sit behind one small interface:

* :class:`NumpyStore` - brute-force L2, no external service. Default for tests
  and the demo.
* :class:`PgVectorStore` - PostgreSQL + pgvector with an HNSW index. The
  production shape.

Both deduplicate embeddings. Identical chunk text hashes to the same key, is
embedded once, and is shared by every document that maps to it. A corpus with
repeated passages (the same blurb stored under many keys) ends up with far
fewer vectors than documents. ``stats()`` reports the ratio.

Schema (mirrors the two tables in both backends):

* ``content_embeddings``: ``content_hash`` -> ``embedding``
* ``doc_map``: ``doc_id`` -> (``content_hash``, ``text``, ``metadata``)
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Protocol

import numpy as np

from .embeddings import Embedder


def content_hash(text: str) -> str:
    """Stable short hash of a chunk's text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


@dataclass
class Document:
    doc_id: str
    text: str
    metadata: dict = field(default_factory=dict)


@dataclass
class SearchHit:
    doc_id: str
    text: str
    distance: float
    metadata: dict


class VectorStore(Protocol):
    def add_documents(self, docs: List[Document], embedder: Embedder) -> int: ...

    def search(self, query_vector: np.ndarray, k: int) -> List[SearchHit]: ...

    def stats(self) -> dict: ...


def _dedup(docs: List[Document]) -> Dict[str, List[Document]]:
    """Group documents by the hash of their text."""
    groups: Dict[str, List[Document]] = {}
    for doc in docs:
        groups.setdefault(content_hash(doc.text), []).append(doc)
    return groups


class NumpyStore:
    """In-memory brute-force store. Zero external dependencies."""

    def __init__(self) -> None:
        self._embeddings: Dict[str, np.ndarray] = {}
        self._text: Dict[str, str] = {}
        self._doc_hash: Dict[str, str] = {}
        self._doc_meta: Dict[str, dict] = {}

    def add_documents(self, docs: List[Document], embedder: Embedder) -> int:
        groups = _dedup(docs)
        new_hashes = [h for h in groups if h not in self._embeddings]
        if new_hashes:
            texts = [groups[h][0].text for h in new_hashes]
            vectors = embedder.embed_documents(texts)
            for h, vector in zip(new_hashes, vectors):
                self._embeddings[h] = np.asarray(vector, dtype=np.float32)
                self._text[h] = groups[h][0].text
        for h, members in groups.items():
            for doc in members:
                self._doc_hash[doc.doc_id] = h
                self._doc_meta[doc.doc_id] = doc.metadata
        return len(docs)

    def search(self, query_vector: np.ndarray, k: int) -> List[SearchHit]:
        query = np.asarray(query_vector, dtype=np.float32)
        hits: List[SearchHit] = []
        for doc_id, h in self._doc_hash.items():
            distance = float(np.linalg.norm(query - self._embeddings[h]))
            hits.append(
                SearchHit(
                    doc_id=doc_id,
                    text=self._text[h],
                    distance=distance,
                    metadata=self._doc_meta[doc_id],
                )
            )
        hits.sort(key=lambda hit: hit.distance)
        return hits[:k]

    def stats(self) -> dict:
        documents = len(self._doc_hash)
        unique = len(self._embeddings)
        ratio = documents / unique if unique else 0.0
        return {
            "documents": documents,
            "unique_embeddings": unique,
            "dedup_ratio": round(ratio, 2),
            "storage_saved_pct": round((1 - 1 / ratio) * 100, 1) if ratio > 1 else 0.0,
            "backend": "numpy",
        }


class PgVectorStore:
    """PostgreSQL + pgvector store with an HNSW index (L2).

    Needs the ``pg`` extra (``pip install .[pg]``) and a running Postgres with
    the ``vector`` extension. See ``docker-compose.yml``.
    """

    def __init__(self, dsn: str, dim: int, table_prefix: str = "rag") -> None:
        import psycopg  # lazy import

        self._psycopg = psycopg
        self.dsn = dsn
        self.dim = dim
        self.embeddings_table = f"{table_prefix}_content_embeddings"
        self.doc_table = f"{table_prefix}_doc_map"
        self._init_schema()

    def _connect(self):
        return self._psycopg.connect(self.dsn)

    def _init_schema(self) -> None:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute(
                f"CREATE TABLE IF NOT EXISTS {self.embeddings_table} ("
                f"  content_hash TEXT PRIMARY KEY,"
                f"  embedding vector({self.dim}) NOT NULL,"
                f"  content_text TEXT NOT NULL)"
            )
            cur.execute(
                f"CREATE INDEX IF NOT EXISTS {self.embeddings_table}_hnsw "
                f"ON {self.embeddings_table} USING hnsw (embedding vector_l2_ops)"
            )
            cur.execute(
                f"CREATE TABLE IF NOT EXISTS {self.doc_table} ("
                f"  doc_id TEXT PRIMARY KEY,"
                f"  content_hash TEXT NOT NULL REFERENCES {self.embeddings_table},"
                f"  metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb)"
            )
            cur.execute(
                f"CREATE INDEX IF NOT EXISTS {self.doc_table}_hash "
                f"ON {self.doc_table} (content_hash)"
            )
            conn.commit()

    @staticmethod
    def _vec_literal(vector: np.ndarray) -> str:
        return "[" + ",".join(str(float(v)) for v in vector) + "]"

    def add_documents(self, docs: List[Document], embedder: Embedder) -> int:
        import json

        groups = _dedup(docs)
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(f"SELECT content_hash FROM {self.embeddings_table}")
            known = {row[0] for row in cur.fetchall()}
            new_hashes = [h for h in groups if h not in known]
            if new_hashes:
                vectors = embedder.embed_documents(
                    [groups[h][0].text for h in new_hashes]
                )
                for h, vector in zip(new_hashes, vectors):
                    cur.execute(
                        f"INSERT INTO {self.embeddings_table} "
                        f"(content_hash, embedding, content_text) "
                        f"VALUES (%s, %s::vector, %s) ON CONFLICT DO NOTHING",
                        (h, self._vec_literal(vector), groups[h][0].text),
                    )
            for h, members in groups.items():
                for doc in members:
                    cur.execute(
                        f"INSERT INTO {self.doc_table} (doc_id, content_hash, metadata) "
                        f"VALUES (%s, %s, %s) ON CONFLICT (doc_id) DO UPDATE SET "
                        f"content_hash = EXCLUDED.content_hash, metadata = EXCLUDED.metadata",
                        (doc.doc_id, h, json.dumps(doc.metadata)),
                    )
            conn.commit()
        return len(docs)

    def search(self, query_vector: np.ndarray, k: int) -> List[SearchHit]:
        literal = self._vec_literal(np.asarray(query_vector, dtype=np.float32))
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                f"SELECT d.doc_id, e.content_text, d.metadata, "
                f"       e.embedding <-> %s::vector AS distance "
                f"FROM {self.doc_table} d "
                f"JOIN {self.embeddings_table} e ON d.content_hash = e.content_hash "
                f"ORDER BY distance ASC LIMIT %s",
                (literal, k),
            )
            return [
                SearchHit(doc_id=r[0], text=r[1], distance=float(r[3]), metadata=r[2])
                for r in cur.fetchall()
            ]

    def stats(self) -> dict:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {self.doc_table}")
            documents = cur.fetchone()[0]
            cur.execute(f"SELECT COUNT(*) FROM {self.embeddings_table}")
            unique = cur.fetchone()[0]
        ratio = documents / unique if unique else 0.0
        return {
            "documents": documents,
            "unique_embeddings": unique,
            "dedup_ratio": round(ratio, 2),
            "storage_saved_pct": round((1 - 1 / ratio) * 100, 1) if ratio > 1 else 0.0,
            "backend": "pgvector",
        }
