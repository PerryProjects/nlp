"""FastAPI REST server with Scalar API reference."""

from __future__ import annotations

import asyncio
import uuid
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Annotated

import nltk
import structlog
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from opentelemetry import trace
from pydantic import BaseModel, Field

from config import settings
from nlp_engine import NlpEngine
from observability import instrument_fastapi, setup_logging, setup_telemetry

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Awaitable, Callable

    from starlette.requests import Request
    from starlette.responses import Response

logger = structlog.get_logger()
tracer = trace.get_tracer(__name__)

# ---------------------------------------------------------------------------
# Global engine reference (initialized in lifespan)
# ---------------------------------------------------------------------------
engine: NlpEngine | None = None


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[None]:
    """Startup / shutdown lifecycle hook."""
    global engine
    setup_logging()
    setup_telemetry()
    nltk.download("punkt_tab", quiet=True)
    engine = NlpEngine()
    logger.info("rest_server_ready", port=settings.rest_port)
    yield
    logger.info("rest_server_shutdown")


# ---------------------------------------------------------------------------
# FastAPI app – Swagger UI disabled in favour of Scalar
# ---------------------------------------------------------------------------
app = FastAPI(
    title="NLP Microservice",
    version="1.0.0",
    docs_url=None,
    redoc_url=None,
    lifespan=lifespan,
)
instrument_fastapi(app)


# ---------------------------------------------------------------------------
# Request ID middleware
# ---------------------------------------------------------------------------
@app.middleware("http")
async def request_id_middleware(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    """Inject a unique request-id into structlog context and response headers."""
    req_id = request.headers.get("x-request-id", str(uuid.uuid4()))
    structlog.contextvars.bind_contextvars(request_id=req_id)
    response = await call_next(request)
    response.headers["x-request-id"] = req_id
    structlog.contextvars.unbind_contextvars("request_id")
    return response


# ---------------------------------------------------------------------------
# Scalar API Reference (replaces Swagger UI)
# ---------------------------------------------------------------------------
_SCALAR_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>NLP Microservice – API Reference</title>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
</head>
<body>
    <script id="api-reference" data-url="/openapi.json"></script>
    <script src="https://cdn.jsdelivr.net/npm/@scalar/api-reference"></script>
</body>
</html>
"""


@app.get("/docs", include_in_schema=False, response_class=HTMLResponse)
async def scalar_docs() -> HTMLResponse:
    """Serve the Scalar API reference UI."""
    return HTMLResponse(content=_SCALAR_HTML)


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------
class AnalyzeRequest(BaseModel):
    """Input payload for the /analyze endpoint."""

    text: str = Field(..., min_length=1, description="The text to analyse")
    categories: list[str] = Field(default_factory=list, description="Categories for zero-shot")
    tonalities: list[str] = Field(default_factory=list, description="Tonality labels for zero-shot")


class CategoryScore(BaseModel):
    """A single category with its confidence score."""

    category: str
    score: float


class TonalityScore(BaseModel):
    """A single tonality with its confidence score."""

    tonality: str
    score: float


class AnalyzeResponse(BaseModel):
    """Structured analysis result."""

    language: str
    sentiment: str
    sentiment_score: float
    matched_categories: list[CategoryScore]
    matched_tonalities: list[TonalityScore]
    is_toxic: bool
    highest_toxicity_score: float
    processing_time_ms: float


class BatchAnalyzeRequest(BaseModel):
    """Batch input payload for /analyze/batch endpoint."""

    texts: list[Annotated[str, Field(min_length=1)]] = Field(
        ..., min_length=1, max_length=100, description="Texts to analyse in one request"
    )
    categories: list[str] = Field(default_factory=list, description="Categories for zero-shot")
    tonalities: list[str] = Field(default_factory=list, description="Tonality labels for zero-shot")


class BatchAnalyzeResponse(BaseModel):
    """Batch response containing one analysis result per input text."""

    results: list[AnalyzeResponse]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
async def health() -> dict[str, str]:
    """Liveness probe."""
    return {"status": "ok"}


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(payload: AnalyzeRequest) -> AnalyzeResponse:
    """Run full NLP analysis on the provided text."""
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    log = logger.bind(endpoint="/analyze", text_length=len(payload.text))
    log.info("rest_request_received")
    with tracer.start_as_current_span("rest_analyze") as span:
        span.set_attribute("text.length", len(payload.text))
        span.set_attribute("categories.count", len(payload.categories))
        span.set_attribute("tonalities.count", len(payload.tonalities))
        result = await asyncio.to_thread(
            engine.analyze,
            payload.text,
            payload.categories,
            payload.tonalities,
        )
    log.info("rest_request_complete", processing_time_ms=result["processing_time_ms"])
    return AnalyzeResponse(**result)


@app.post("/analyze/batch", response_model=BatchAnalyzeResponse)
async def analyze_batch(payload: BatchAnalyzeRequest) -> BatchAnalyzeResponse:
    """Run NLP analysis on a batch of texts with shared labels."""
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    log = logger.bind(endpoint="/analyze/batch", batch_size=len(payload.texts))
    log.info("rest_batch_request_received")

    with tracer.start_as_current_span("rest_analyze_batch") as span:
        span.set_attribute("batch.size", len(payload.texts))
        span.set_attribute("categories.count", len(payload.categories))
        span.set_attribute("tonalities.count", len(payload.tonalities))
        results = await asyncio.to_thread(
            engine.analyze_batch,
            payload.texts,
            payload.categories,
            payload.tonalities,
        )

    log.info("rest_batch_request_complete", batch_size=len(results))
    return BatchAnalyzeResponse(results=[AnalyzeResponse(**result) for result in results])


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run(
        "main_rest:app",
        host=settings.rest_host,
        port=settings.rest_port,
        log_level=settings.log_level.lower(),
    )
