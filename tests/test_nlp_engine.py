"""Focused unit tests for nlp_engine helpers and aggregation logic."""

from __future__ import annotations

from contextlib import nullcontext
from unittest.mock import MagicMock, patch

from nlp_engine import NlpEngine


class _DummyTracer:
    class _DummySpan:
        def __enter__(self) -> _DummyTracer._DummySpan:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def set_attribute(self, _key: str, _value: object) -> None:
            return None

    def start_as_current_span(self, _name: str) -> _DummySpan:
        return self._DummySpan()


class _DummyEmbedding:
    def __init__(self, values: list[float]) -> None:
        self._values = values

    def tolist(self) -> list[float]:
        return self._values


class _DummyPoint:
    def __init__(self, score: float, payload: dict | None) -> None:
        self.score = score
        self.payload = payload


class _DummyResults:
    def __init__(self, points: list[_DummyPoint]) -> None:
        self.points = points


class _DummyQdrant:
    def __init__(self, payload: dict | None, score: float = 0.99, has_points: bool = True) -> None:
        self._payload = payload
        self._score = score
        self.has_points = has_points
        self.upserts: list[dict[str, object]] = []

    def query_points(self, **_kwargs: object) -> _DummyResults:
        if not self.has_points:
            return _DummyResults([])
        return _DummyResults([_DummyPoint(self._score, self._payload)])

    def upsert(self, **kwargs: object) -> None:
        self.upserts.append(kwargs)


def _build_engine_for_unit_tests() -> NlpEngine:
    engine = NlpEngine.__new__(NlpEngine)
    engine._tracer = _DummyTracer()
    return engine


def test_engine_init_builds_all_models_and_cache() -> None:
    """Engine initialization should wire all models and create cache collection."""
    builder = MagicMock()
    builder.with_preloaded_language_models.return_value = builder
    builder.build.return_value = MagicMock()

    qdrant_client = MagicMock()
    qdrant_client.collection_exists.return_value = False

    with (
        patch("nlp_engine.trace.get_tracer", return_value=_DummyTracer()),
        patch("nlp_engine.LanguageDetectorBuilder.from_languages", return_value=builder),
        patch("nlp_engine.TextEmbedding", return_value=MagicMock()),
        patch(
            "nlp_engine.pipeline",
            side_effect=[MagicMock(), MagicMock(), MagicMock()],
        ) as pipeline_mock,
        patch("nlp_engine.QdrantClient", return_value=qdrant_client),
    ):
        NlpEngine()

    assert pipeline_mock.call_count == 3
    qdrant_client.create_collection.assert_called_once()


def test_analyze_toxicity_non_toxic() -> None:
    """Non-toxic label should stay below threshold and report False."""
    engine = _build_engine_for_unit_tests()
    engine._toxicity = lambda *_args, **_kwargs: [
        [
            {"label": "non-toxic", "score": 0.96},
            {"label": "toxic", "score": 0.04},
        ]
    ]

    score, is_toxic = engine.analyze_toxicity("I appreciate your help.")

    assert score == 0.04
    assert is_toxic is False


def test_analyze_toxicity_toxic() -> None:
    """Toxic label above threshold should report True."""
    engine = _build_engine_for_unit_tests()
    engine._toxicity = lambda *_args, **_kwargs: [
        [
            {"label": "toxic", "score": 0.91},
            {"label": "non-toxic", "score": 0.09},
        ]
    ]

    score, is_toxic = engine.analyze_toxicity("I hate you.")

    assert score == 0.91
    assert is_toxic is True


def test_cache_lookup_rejects_mismatched_labels() -> None:
    """Cache hit must be invalidated when categories/tonalities do not match."""
    engine = _build_engine_for_unit_tests()
    engine._qdrant = _DummyQdrant(
        payload={
            "requested_categories": ["finance"],
            "requested_tonalities": ["formal"],
        },
        score=0.98,
    )

    result = engine._cache_lookup(
        _DummyEmbedding([0.1, 0.2, 0.3]),
        categories=["sports"],
        tonalities=["informal"],
    )

    assert result is None


def test_cache_lookup_rejects_invalid_payload() -> None:
    """Cache payloads must be dict-like to be accepted."""
    engine = _build_engine_for_unit_tests()
    engine._qdrant = _DummyQdrant(payload="invalid", score=0.98)

    result = engine._cache_lookup(
        _DummyEmbedding([0.1, 0.2, 0.3]),
        categories=["sports"],
        tonalities=["informal"],
    )

    assert result is None


def test_infer_sentence_uses_cache_hit() -> None:
    """A matching cache hit should skip model inference and return cached payload."""
    engine = _build_engine_for_unit_tests()
    cached_payload = {
        "sentiment_label": "positive",
        "sentiment_score": 0.8,
        "toxicity_score": 0.1,
        "is_toxic": False,
        "category_scores": [{"category": "tech", "score": 0.7}],
        "tonality_scores": [{"tonality": "formal", "score": 0.6}],
        "requested_categories": ["tech"],
        "requested_tonalities": ["formal"],
    }
    engine._qdrant = _DummyQdrant(payload=cached_payload, score=0.99)

    result = engine._infer_sentence(
        sentence="AI helps medicine.",
        categories=["tech"],
        tonalities=["formal"],
        embedding=_DummyEmbedding([0.1, 0.2]),
    )

    assert result == cached_payload


def test_analyze_end_to_end_with_cache_store() -> None:
    """Analyze should aggregate toxicity/categories/tonality and store cache entries."""
    engine = _build_engine_for_unit_tests()

    iso = MagicMock()
    iso.name = "EN"
    detected = MagicMock()
    detected.iso_code_639_1 = iso

    engine._lingua = MagicMock()
    engine._lingua.detect_language_of.return_value = detected
    engine._embedder = MagicMock()
    engine._embedder.embed.return_value = [
        _DummyEmbedding([0.1, 0.2]),
        _DummyEmbedding([0.3, 0.4]),
    ]
    engine._qdrant = _DummyQdrant(payload=None, score=0.0, has_points=False)
    engine._sentiment = lambda _text, truncation=True: [{"label": "4 stars", "score": 0.88}]
    engine._toxicity = lambda _text, top_k=None, truncation=True: [
        [
            {"label": "toxic", "score": 0.81},
            {"label": "non-toxic", "score": 0.19},
        ]
    ]

    def classifier(
        _text: str,
        candidate_labels: list[str],
        truncation: bool = True,
    ) -> dict[str, list[float] | list[str]]:
        assert truncation is True
        return {
            "labels": candidate_labels,
            "scores": [0.72 for _ in candidate_labels],
        }

    engine._classifier = classifier

    with patch("nlp_engine.sent_tokenize", return_value=["One.", "Two."]):
        result = engine.analyze(
            text="One. Two.",
            categories=["technology"],
            tonalities=["formal"],
        )

    assert result["language"] == "en"
    assert result["sentiment"] == "positive"
    assert result["is_toxic"] is True
    assert result["highest_toxicity_score"] == 0.81
    assert len(result["matched_categories"]) == 1
    assert len(result["matched_tonalities"]) == 1
    assert len(engine._qdrant.upserts) == 2


def test_analyze_batch_uses_shared_labels() -> None:
    """Batch analyze should call single-text analyze for each text."""
    engine = _build_engine_for_unit_tests()
    engine.analyze = MagicMock(
        side_effect=[
            {"language": "en", "processing_time_ms": 10.0},
            {"language": "de", "processing_time_ms": 11.0},
        ]
    )

    results = engine.analyze_batch(
        texts=["hello", "hallo"],
        categories=["tech"],
        tonalities=["formal"],
    )

    assert len(results) == 2
    assert results[0]["language"] == "en"
    assert results[1]["language"] == "de"


def test_aggregate_toxicity_any_sentence_toxic() -> None:
    """Aggregate toxicity should mark the document toxic if any sentence is toxic."""
    aggregated = NlpEngine._aggregate_toxicity(
        [
            {"toxicity_score": 0.24, "is_toxic": False},
            {"toxicity_score": 0.85, "is_toxic": True},
        ]
    )

    assert aggregated["is_toxic"] is True
    assert aggregated["highest_toxicity_score"] == 0.85


def test_point_id_changes_with_requested_labels() -> None:
    """Same text with different labels must produce different point IDs."""
    engine = _build_engine_for_unit_tests()

    id_1 = engine._point_id("same sentence", ["finance"], ["formal"])
    id_2 = engine._point_id("same sentence", ["sports"], ["formal"])
    id_3 = engine._point_id("same sentence", ["finance"], ["aggressive"])

    assert id_1 != id_2
    assert id_1 != id_3


def test_cache_hit_with_reordered_labels() -> None:
    """Cache lookup must hit when the same labels are supplied in a different order."""
    engine = _build_engine_for_unit_tests()
    # Payload stored with labels in sorted order (as the fixed code now writes them).
    cached_payload = {
        "sentiment_label": "positive",
        "sentiment_score": 0.8,
        "toxicity_score": 0.05,
        "is_toxic": False,
        "category_scores": [{"category": "finance", "score": 0.7}],
        "tonality_scores": [],
        "requested_categories": ["finance", "tech"],  # sorted
        "requested_tonalities": [],
    }
    engine._qdrant = _DummyQdrant(payload=cached_payload, score=0.99)

    result = engine._cache_lookup(
        _DummyEmbedding([0.1, 0.2, 0.3]),
        categories=["tech", "finance"],  # reversed order
        tonalities=[],
    )

    assert result is not None, "Expected a cache hit for reordered labels"
    assert result["requested_categories"] == ["finance", "tech"]


def test_analyze_toxicity_passes_truncation_flag() -> None:
    """analyze_toxicity must pass truncation=True to the pipeline, not slice the string."""
    engine = _build_engine_for_unit_tests()
    received_kwargs: dict = {}

    def fake_toxicity(text: str, **kwargs: object) -> list:
        received_kwargs["text"] = text
        received_kwargs.update(kwargs)
        return [[{"label": "non-toxic", "score": 0.95}, {"label": "toxic", "score": 0.05}]]

    engine._toxicity = fake_toxicity  # type: ignore[method-assign]
    long_text = "word " * 600  # >512 chars
    engine.analyze_toxicity(long_text)

    assert received_kwargs["truncation"] is True, "truncation=True must be forwarded to the pipeline"
    assert received_kwargs["text"] == long_text, "Full text must reach the pipeline; no pre-slicing"
