from semantic_rag import Document, HashingEmbedder, NumpyStore


def test_identical_text_is_embedded_once():
    store = NumpyStore()
    docs = [
        Document(doc_id="cl", text="people interested in fitness"),
        Document(doc_id="mx", text="people interested in fitness"),  # same text
        Document(doc_id="co", text="people interested in cooking"),
    ]
    store.add_documents(docs, HashingEmbedder())

    stats = store.stats()
    assert stats["documents"] == 3
    assert stats["unique_embeddings"] == 2       # the two CL/MX docs share one
    assert stats["dedup_ratio"] == 1.5
    assert stats["storage_saved_pct"] > 0


def test_repeated_ingest_does_not_duplicate_embeddings():
    store = NumpyStore()
    embedder = HashingEmbedder()
    doc = [Document(doc_id="a", text="return on investment")]
    store.add_documents(doc, embedder)
    store.add_documents([Document(doc_id="b", text="return on investment")], embedder)

    assert store.stats()["unique_embeddings"] == 1
    assert store.stats()["documents"] == 2
