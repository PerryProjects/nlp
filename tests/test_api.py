"""Comprehensive tests for the NLP microservice REST API."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from unittest.mock import MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture()
def mock_engine() -> MagicMock:
    """Return a mocked NlpEngine with a canned .analyze() response."""
    engine = MagicMock()
    engine.analyze.return_value = {
        "language": "en",
        "sentiment": "positive",
        "sentiment_score": 0.55,
        "matched_categories": [
            {"category": "technology", "score": 0.82},
            {"category": "science", "score": 0.65},
        ],
        "matched_tonalities": [],
        "is_toxic": False,
        "highest_toxicity_score": 0.04,
        "processing_time_ms": 123.45,
    }
    return engine


@pytest.fixture()
async def client(mock_engine: MagicMock) -> AsyncGenerator[AsyncClient]:
    """Create an async test client with the engine mocked out."""
    with (
        patch("main_rest.NlpEngine", return_value=mock_engine),
        patch("main_rest.setup_telemetry"),
    ):
        from main_rest import app

        async with app.router.lifespan_context(app):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as ac:
                yield ac


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_health(client: AsyncClient) -> None:
    """GET /health returns 200 with status ok."""
    resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# Analyze endpoint – happy path
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_analyze_success(client: AsyncClient, mock_engine: MagicMock) -> None:
    """POST /analyze returns a structured analysis result."""
    payload = {
        "text": "Artificial intelligence is transforming technology and science.",
        "categories": ["technology", "science", "sports"],
    }
    resp = await client.post("/analyze", json=payload)
    assert resp.status_code == 200

    body = resp.json()
    assert body["language"] == "en"
    assert body["sentiment"] == "positive"
    assert isinstance(body["sentiment_score"], float)
    assert len(body["matched_categories"]) == 2
    assert body["matched_categories"][0]["category"] == "technology"
    assert body["is_toxic"] is False
    assert body["highest_toxicity_score"] == 0.04
    assert body["processing_time_ms"] > 0

    mock_engine.analyze.assert_called_once_with(payload["text"], payload["categories"], [])


# ---------------------------------------------------------------------------
# Analyze endpoint – empty categories
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_analyze_no_categories(client: AsyncClient, mock_engine: MagicMock) -> None:
    """POST /analyze works when no categories are provided."""
    mock_engine.analyze.return_value = {
        "language": "fr",
        "sentiment": "neutral",
        "sentiment_score": 0.5,
        "matched_categories": [],
        "matched_tonalities": [],
        "is_toxic": False,
        "highest_toxicity_score": 0.01,
        "processing_time_ms": 50.0,
    }
    resp = await client.post("/analyze", json={"text": "Bonjour le monde"})
    assert resp.status_code == 200
    assert resp.json()["matched_categories"] == []


# ---------------------------------------------------------------------------
# Analyze endpoint - German input
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_analyze_german_text(client: AsyncClient, mock_engine: MagicMock) -> None:
    """POST /analyze supports German text payloads."""
    mock_engine.analyze.return_value = {
        "language": "de",
        "sentiment": "positive",
        "sentiment_score": 0.74,
        "matched_categories": [
            {"category": "technologie", "score": 0.88},
            {"category": "wirtschaft", "score": 0.69},
        ],
        "matched_tonalities": [],
        "is_toxic": False,
        "highest_toxicity_score": 0.06,
        "processing_time_ms": 95.2,
    }
    payload = {
        "text": "Die kuenstliche Intelligenz veraendert die Technologiebranche in Deutschland.",
        "categories": ["technologie", "wirtschaft", "sport"],
    }

    resp = await client.post("/analyze", json=payload)
    assert resp.status_code == 200

    body = resp.json()
    assert body["language"] == "de"
    assert body["matched_categories"][0]["category"] == "technologie"
    assert body["processing_time_ms"] > 0
    mock_engine.analyze.assert_called_with(payload["text"], payload["categories"], [])


# ---------------------------------------------------------------------------
# Analyze endpoint – toxicity fields
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_analyze_toxicity_flags(client: AsyncClient, mock_engine: MagicMock) -> None:
    """POST /analyze surfaces toxicity score and toxic boolean."""
    mock_engine.analyze.return_value = {
        "language": "en",
        "sentiment": "negative",
        "sentiment_score": 0.91,
        "matched_categories": [],
        "matched_tonalities": [],
        "is_toxic": True,
        "highest_toxicity_score": 0.93,
        "processing_time_ms": 66.0,
    }

    resp = await client.post("/analyze", json={"text": "You are stupid."})
    assert resp.status_code == 200
    body = resp.json()
    assert body["is_toxic"] is True
    assert body["highest_toxicity_score"] == 0.93


# ---------------------------------------------------------------------------
# Analyze endpoint – batch happy path
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_analyze_batch_success(client: AsyncClient, mock_engine: MagicMock) -> None:
    """POST /analyze/batch returns one result per text."""
    mock_engine.analyze_batch.return_value = [
        {
            "language": "en",
            "sentiment": "positive",
            "sentiment_score": 0.8,
            "matched_categories": [{"category": "tech", "score": 0.9}],
            "matched_tonalities": [{"tonality": "formal", "score": 0.7}],
            "is_toxic": False,
            "highest_toxicity_score": 0.05,
            "processing_time_ms": 40.0,
        },
        {
            "language": "fr",
            "sentiment": "neutral",
            "sentiment_score": 0.51,
            "matched_categories": [{"category": "tech", "score": 0.5}],
            "matched_tonalities": [{"tonality": "informal", "score": 0.6}],
            "is_toxic": True,
            "highest_toxicity_score": 0.84,
            "processing_time_ms": 44.0,
        },
    ]

    payload = {
        "texts": ["AI helps medicine.", "Je te deteste."],
        "categories": ["tech"],
        "tonalities": ["formal", "informal"],
    }
    resp = await client.post("/analyze/batch", json=payload)
    assert resp.status_code == 200

    body = resp.json()
    assert len(body["results"]) == 2
    assert body["results"][0]["language"] == "en"
    assert body["results"][1]["is_toxic"] is True
    mock_engine.analyze_batch.assert_called_once_with(
        payload["texts"],
        payload["categories"],
        payload["tonalities"],
    )


# ---------------------------------------------------------------------------
# Analyze endpoint – batch validation
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_analyze_batch_empty_texts_rejected(client: AsyncClient) -> None:
    """POST /analyze/batch rejects empty texts list with 422."""
    resp = await client.post("/analyze/batch", json={"texts": []})
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_analyze_batch_empty_string_item_rejected(client: AsyncClient) -> None:
    """POST /analyze/batch rejects a batch containing an empty string with 422."""
    resp = await client.post("/analyze/batch", json={"texts": ["valid text", ""]})
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_analyze_batch_too_large_rejected(client: AsyncClient) -> None:
    """POST /analyze/batch rejects batches with more than 100 texts with 422."""
    resp = await client.post("/analyze/batch", json={"texts": ["x"] * 101})
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Analyze endpoint – validation error (empty text)
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_analyze_empty_text(client: AsyncClient) -> None:
    """POST /analyze rejects empty text with 422."""
    resp = await client.post("/analyze", json={"text": "", "categories": []})
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Analyze endpoint – missing text field
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_analyze_missing_text(client: AsyncClient) -> None:
    """POST /analyze rejects missing text with 422."""
    resp = await client.post("/analyze", json={"categories": ["foo"]})
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Scalar docs endpoint
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_scalar_docs(client: AsyncClient) -> None:
    """GET /docs returns the Scalar API reference HTML."""
    resp = await client.get("/docs")
    assert resp.status_code == 200
    assert "api-reference" in resp.text
    assert "scalar" in resp.text.lower()


# ---------------------------------------------------------------------------
# OpenAPI schema is served
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_openapi_json(client: AsyncClient) -> None:
    """GET /openapi.json returns a valid OpenAPI document."""
    resp = await client.get("/openapi.json")
    assert resp.status_code == 200
    schema = resp.json()
    assert schema["info"]["title"] == "NLP Microservice"
    assert "/analyze" in schema["paths"]


# ---------------------------------------------------------------------------
# Request ID propagation
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_request_id_header(client: AsyncClient) -> None:
    """Custom x-request-id header is echoed back."""
    resp = await client.get("/health", headers={"x-request-id": "test-123"})
    assert resp.headers["x-request-id"] == "test-123"


@pytest.mark.asyncio
async def test_request_id_generated(client: AsyncClient) -> None:
    """A request without x-request-id gets one generated."""
    resp = await client.get("/health")
    assert "x-request-id" in resp.headers
    assert len(resp.headers["x-request-id"]) > 0


# ---------------------------------------------------------------------------
# Config module
# ---------------------------------------------------------------------------
def test_settings_defaults() -> None:
    """Settings loads with default values when no .env present."""
    from config import Settings

    s = Settings()
    assert s.rest_port == 8000
    assert s.grpc_port == 50051
    assert s.embedding_model == "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    assert s.classifier_model == "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
    assert s.toxicity_model == "unitary/multilingual-toxic-xlm-roberta"
    assert s.toxicity_threshold == 0.8
    assert 0 < s.qdrant_cache_threshold < 1


# ---------------------------------------------------------------------------
# Engine-not-initialized guard
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_analyze_engine_not_initialized(client: AsyncClient) -> None:
    """POST /analyze returns 503 when the engine is None."""
    import main_rest

    original = main_rest.engine
    main_rest.engine = None
    try:
        resp = await client.post("/analyze", json={"text": "hello"})
        assert resp.status_code == 503
    finally:
        main_rest.engine = original


@pytest.mark.asyncio
async def test_analyze_batch_engine_not_initialized(client: AsyncClient) -> None:
    """POST /analyze/batch returns 503 when the engine is None."""
    import main_rest

    original = main_rest.engine
    main_rest.engine = None
    try:
        resp = await client.post("/analyze/batch", json={"texts": ["hello"]})
        assert resp.status_code == 503
    finally:
        main_rest.engine = original
