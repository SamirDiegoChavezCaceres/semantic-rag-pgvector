"""Swappable embedding providers.

The pipeline never hardcodes an embedding backend. All providers satisfy the
same :class:`Embedder` protocol, so the same code runs with a dependency-free
hashing embedder (tests / offline), a real sentence-transformers model, or a
hosted API.
"""

from __future__ import annotations

import hashlib
import os
import re
from typing import List, Protocol, runtime_checkable

import numpy as np

_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")

# A tiny English stop-word list. Dropping these keeps the hashing embedder from
# scoring two sentences as similar just because they share "the", "of", "is".
_STOPWORDS = frozenset(
    """
    a an and are as at be by do does for from how i in is it its of on or
    per that the this to was what when where which who why with you your
    """.split()
)


@runtime_checkable
class Embedder(Protocol):
    """Anything that turns text into L2-normalized vectors."""

    dim: int

    def embed_documents(self, texts: List[str]) -> List[np.ndarray]: ...

    def embed_query(self, text: str) -> np.ndarray: ...


def _tokenize(text: str) -> List[str]:
    return [tok for tok in _TOKEN_RE.findall(text.lower()) if tok not in _STOPWORDS]


class HashingEmbedder:
    """Deterministic, dependency-free embedder (the hashing trick).

    It is not semantically smart, but it is stable and needs no model download
    or API key, which makes it ideal for tests and for demonstrating the
    *mechanics* (distance threshold, dedup). Vectors are L2-normalized, so two
    texts that share no tokens sit about ``sqrt(2)`` apart.
    """

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def _embed_one(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        for token in _tokenize(text):
            bucket = int(hashlib.md5(token.encode()).hexdigest(), 16) % self.dim
            vec[bucket] += 1.0
        norm = float(np.linalg.norm(vec))
        if norm > 0.0:
            vec /= norm
        return vec

    def embed_documents(self, texts: List[str]) -> List[np.ndarray]:
        return [self._embed_one(text) for text in texts]

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed_one(text)


class SentenceTransformerEmbedder:
    """Real semantic embeddings via sentence-transformers (optional extra).

    Install with ``pip install .[semantic]``. The model is downloaded on first
    use and cached locally.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer  # lazy import

        self._model = SentenceTransformer(model_name)
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def embed_documents(self, texts: List[str]) -> List[np.ndarray]:
        vectors = self._model.encode(texts, normalize_embeddings=True)
        return [np.asarray(v, dtype=np.float32) for v in vectors]

    def embed_query(self, text: str) -> np.ndarray:
        vector = self._model.encode([text], normalize_embeddings=True)[0]
        return np.asarray(vector, dtype=np.float32)


class OpenAIEmbedder:
    """Embeddings from the OpenAI API (the ``openai`` extra).

    Reads ``OPENAI_API_KEY`` from the environment or a local ``.env`` file.
    Vectors are L2-normalized so the distance threshold keeps its meaning.
    """

    def __init__(self, model: str = None, dimensions: int = None) -> None:
        try:
            from dotenv import load_dotenv

            load_dotenv()
        except Exception:
            pass
        from openai import OpenAI  # lazy import

        self._client = OpenAI()
        self.model = model or os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
        self.dim = dimensions or int(os.getenv("OPENAI_EMBED_DIM", "1536"))

    def _normalize(self, vector) -> np.ndarray:
        vec = np.asarray(vector, dtype=np.float32)
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm > 0 else vec

    def embed_documents(self, texts: List[str]) -> List[np.ndarray]:
        resp = self._client.embeddings.create(
            model=self.model, input=list(texts), dimensions=self.dim
        )
        return [self._normalize(item.embedding) for item in resp.data]

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([text])[0]


# Strategy + Factory: each provider is one entry in this registry, and
# get_embedder picks one by name. Adding Anthropic/Gemini later is a one-liner.
_PROVIDERS = {
    "openai": OpenAIEmbedder,
    "sentence-transformers": SentenceTransformerEmbedder,
    "hashing": HashingEmbedder,
}


def get_embedder(prefer: str = "auto") -> Embedder:
    """Return an embedder by provider name.

    - ``"openai"``: the OpenAI API (needs ``OPENAI_API_KEY``; see ``.env.example``).
    - ``"sentence-transformers"``: the local model (requires the extra).
    - ``"hashing"``: the dependency-free offline embedder.
    - ``"auto"`` (default): sentence-transformers if installed, else hashing. It
      never reaches for OpenAI on its own, so it cannot surprise you with a bill.
    """
    if prefer == "auto":
        try:
            return SentenceTransformerEmbedder()
        except Exception:
            return HashingEmbedder()
    try:
        return _PROVIDERS[prefer]()
    except KeyError:
        raise ValueError(
            f"unknown embedder '{prefer}'; options: {', '.join(_PROVIDERS)} or 'auto'"
        )
