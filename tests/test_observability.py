"""Unit tests for observability setup helpers."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import observability


def test_setup_telemetry_configures_provider_and_exporters() -> None:
    """setup_telemetry wires tracer provider and both span processors."""
    provider = MagicMock()

    with (
        patch.object(observability.Resource, "create", return_value=MagicMock()),
        patch.object(observability, "TracerProvider", return_value=provider),
        patch.object(observability, "BatchSpanProcessor", side_effect=lambda exporter: exporter),
        patch.object(observability, "ConsoleSpanExporter", return_value=MagicMock()),
        patch.object(observability, "OTLPSpanExporter", return_value=MagicMock()),
        patch.object(observability.trace, "set_tracer_provider") as set_provider,
    ):
        observability.setup_telemetry()

    assert provider.add_span_processor.call_count == 2
    set_provider.assert_called_once_with(provider)


def test_setup_telemetry_handles_otlp_exporter_failure() -> None:
    """OTLP exporter errors should not break telemetry setup."""
    provider = MagicMock()
    logger = MagicMock()

    with (
        patch.object(observability.Resource, "create", return_value=MagicMock()),
        patch.object(observability, "TracerProvider", return_value=provider),
        patch.object(observability, "BatchSpanProcessor", side_effect=lambda exporter: exporter),
        patch.object(observability, "ConsoleSpanExporter", return_value=MagicMock()),
        patch.object(observability, "OTLPSpanExporter", side_effect=RuntimeError("boom")),
        patch.object(observability.structlog, "get_logger", return_value=logger),
        patch.object(observability.trace, "set_tracer_provider"),
    ):
        observability.setup_telemetry()

    logger.warning.assert_called_once_with("otlp_exporter_unavailable")


def test_instrument_fastapi_calls_instrumentor() -> None:
    """instrument_fastapi should delegate to FastAPIInstrumentor."""
    app = MagicMock()

    with patch.object(observability.FastAPIInstrumentor, "instrument_app") as instrument:
        observability.instrument_fastapi(app)

    instrument.assert_called_once_with(app)


def test_setup_logging_configures_structlog() -> None:
    """setup_logging should call structlog.configure once."""
    with patch.object(observability.structlog, "configure") as configure:
        observability.setup_logging()

    configure.assert_called_once()
