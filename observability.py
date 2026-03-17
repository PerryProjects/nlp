"""OpenTelemetry and structlog configuration."""

from __future__ import annotations

import logging

import structlog
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

from config import settings


def setup_telemetry() -> None:
    """Initialize OpenTelemetry tracing with OTLP exporter."""
    resource = Resource.create({"service.name": settings.otel_service_name})
    provider = TracerProvider(resource=resource)

    # Always add console exporter for local development visibility
    provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))

    # OTLP exporter (Jaeger / collector) — best-effort; silently degrades if unavailable
    try:
        otlp_exporter = OTLPSpanExporter(endpoint=settings.otel_exporter_endpoint, insecure=True)
        provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
    except Exception:
        structlog.get_logger().warning("otlp_exporter_unavailable")

    trace.set_tracer_provider(provider)


def instrument_fastapi(app: object) -> None:
    """Attach OpenTelemetry instrumentation to a FastAPI app."""
    FastAPIInstrumentor.instrument_app(app)  # type: ignore[arg-type]


def setup_logging() -> None:
    """Configure structlog for JSON-formatted, structured logging."""
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelName(settings.log_level),
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
