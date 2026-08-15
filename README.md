# semantic-rag-pgvector

[![CI](https://github.com/SamirDiegoChavezCaceres/semantic-rag-pgvector/actions/workflows/ci.yml/badge.svg)](https://github.com/SamirDiegoChavezCaceres/semantic-rag-pgvector/actions/workflows/ci.yml) ![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg) ![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

A small Retrieval-Augmented Generation core that does two things most tutorials
skip:

1. **It knows when to say "I don't know."** A distance threshold means an
   off-topic question gets *no* context instead of the nearest-but-irrelevant
   chunk dressed up as an answer.
2. **It deduplicates embeddings by content hash.** Identical chunk text is
   embedded once and shared, so a corpus with repeated passages stores far
   fewer vectors.

The examples run over a small marketing corpus included in the repo.

## Demo

![demo](assets/demo.gif)

Generate it with [VHS](https://github.com/charmbracelet/vhs): `vhs demo.tape`.

## Why a threshold

A plain vector search always returns its `k` nearest neighbours, no matter how
far away they are. Ask an unrelated question and it still hands back *something*
and the model answers from it. That is the quiet failure mode behind a lot of
RAG hallucinations.

Here, retrieval returns a typed result:

```python
result = rag.search("What is the capital of France?")
if result.found:
    answer = llm(prompt + result.context)
else:
    answer = "Nothing in the knowledge base covers that."   # ran, found nothing
```

If the nearest chunk is farther than `max_distance`, `found` is `False` and the
result carries no context. "Ran and found nothing" is reported differently from
"failed," so the model can be told to stop instead of improvising.

### Calibrating the threshold

With L2-normalized embeddings, distance runs from 0 (identical) to ~1.41
(unrelated). Pick the cut by measuring your own corpus: run a batch of
in-domain and out-of-domain questions and look at the gap.

The gap depends heavily on the embedder. Real sentence embeddings separate
cleanly; the offline bag-of-words fallback in this repo leaves only a narrow
band, which is itself a useful lesson: **a weak embedder makes a good threshold
impossible.** In the original system, measured in-domain distances sat around
0.18-0.31 and out-of-domain around 0.42-0.51, so a cut near 0.38 was safe.

## Why dedup

The same text often appears under many keys (the same audience blurb across
several countries, the same boilerplate across documents). Embedding each copy
wastes storage and an API call. The store keeps two tables:

- `content_embeddings`: `content_hash -> embedding`
- `doc_map`: `doc_id -> content_hash`

`stats()` reports the dedup ratio and storage saved. In the system this is based
on, ~3,000 documents collapsed to ~760 unique embeddings (a 2x+ saving).

## Architecture

```
          text ─► chunk ─► hash ─► embed (unique only) ─► store
                                                           │
query ─► embed ─► search (L2) ─► threshold filter ─► RagResult(found, context)
```

Everything is swappable behind small interfaces:

| Piece     | Default (no deps)      | Production option                  |
|-----------|------------------------|------------------------------------|
| Embedder  | `HashingEmbedder`      | `SentenceTransformerEmbedder` / `OpenAIEmbedder` |
| Store     | `NumpyStore` (in-mem)  | `PgVectorStore` (Postgres + HNSW)  |

## Quickstart

```bash
pip install -e .            # core (numpy only)
python scripts/demo.py      # ingest the sample corpus, run a few queries
```

Real semantic embeddings (local model, no key):

```bash
pip install -e ".[semantic]"
RAG_EMBEDDER=sentence-transformers python scripts/demo.py
```

OpenAI embeddings (needs a key):

```bash
pip install -e ".[openai]"
cp .env.example .env          # then put your OPENAI_API_KEY in .env
RAG_EMBEDDER=openai python scripts/demo.py
```

Production store (Postgres + pgvector):

```bash
docker compose up -d
pip install -e ".[pg]"
```

```python
from semantic_rag import SemanticRAG, get_embedder
from semantic_rag.store import PgVectorStore

embedder = get_embedder("sentence-transformers")
store = PgVectorStore("postgresql://rag:rag@localhost:5432/rag", dim=embedder.dim)
rag = SemanticRAG(embedder=embedder, store=store, max_distance=1.0)
rag.ingest_directory("corpus")
print(rag.search("How is ROI calculated?").context)
```

## Tests

```bash
pip install -e ".[dev]"
pytest
```

The suite covers the two behaviours that matter: the threshold drops an
off-domain query, and dedup collapses identical text into one embedding.

## Limitations and next steps

- The distance threshold has to be recalibrated per embedder and per corpus; a
  value tuned for one does not carry over to another.
- The hashing embedder only shows the mechanics; real retrieval quality needs
  sentence-transformers or OpenAI.
- Next: rerank the top-k candidates, and support metadata filters before the
  vector search.

## License

MIT.
