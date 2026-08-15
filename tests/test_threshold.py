from semantic_rag import HashingEmbedder, NumpyStore, SemanticRAG


def _rag(max_distance: float) -> SemanticRAG:
    rag = SemanticRAG(
        embedder=HashingEmbedder(),
        store=NumpyStore(),
        max_distance=max_distance,
    )
    rag.ingest_texts(
        [
            "Return on investment measures profit earned per unit of cost.",
            "The marketing funnel has three stages: awareness, consideration, decision.",
        ],
        source="kb",
    )
    return rag


def test_in_domain_query_is_found():
    rag = _rag(max_distance=1.2)
    result = rag.search("investment profit cost", k=3)
    assert result.found
    assert result.context


def test_off_domain_query_returns_nothing():
    # A query sharing no tokens with the corpus sits ~sqrt(2) away, past 1.2.
    rag = _rag(max_distance=1.2)
    result = rag.search("photosynthesis chlorophyll sunlight", k=3)
    assert not result.found
    assert "threshold" in result.message


def test_threshold_is_tunable():
    # Loosen the threshold and the same off-domain query now slips through.
    rag = _rag(max_distance=2.0)
    result = rag.search("photosynthesis chlorophyll sunlight", k=3)
    assert result.found
