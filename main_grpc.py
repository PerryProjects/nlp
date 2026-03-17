"""gRPC server for the NLP microservice."""

from __future__ import annotations

import time
from concurrent import futures

import grpc

# Generated stubs — run `make proto` or the grpcio-tools command to regenerate.
import nlp_service_pb2
import nlp_service_pb2_grpc
import nltk
import structlog
from opentelemetry import trace
from opentelemetry.instrumentation.grpc import GrpcInstrumentorServer

from config import settings
from nlp_engine import NlpEngine
from observability import setup_logging, setup_telemetry

logger = structlog.get_logger()
tracer = trace.get_tracer(__name__)


class NlpServiceServicer(nlp_service_pb2_grpc.NlpServiceServicer):
    """gRPC servicer that delegates to NlpEngine."""

    def __init__(self, engine: NlpEngine) -> None:
        self._engine = engine

    @staticmethod
    def _to_analyze_response(result: dict, processing_time_ms: float) -> nlp_service_pb2.AnalyzeResponse:
        """Map internal result dict to protobuf AnalyzeResponse."""
        category_scores = [
            nlp_service_pb2.CategoryScore(category=c["category"], score=c["score"])
            for c in result["matched_categories"]
        ]
        tonality_scores = [
            nlp_service_pb2.TonalityScore(tonality=t["tonality"], score=t["score"])
            for t in result["matched_tonalities"]
        ]
        return nlp_service_pb2.AnalyzeResponse(
            language=result["language"],
            sentiment=result["sentiment"],
            sentiment_score=result["sentiment_score"],
            matched_categories=category_scores,
            matched_tonalities=tonality_scores,
            is_toxic=result["is_toxic"],
            highest_toxicity_score=result["highest_toxicity_score"],
            processing_time_ms=processing_time_ms,
        )

    def analyze(
        self,
        request: nlp_service_pb2.AnalyzeRequest,
        _context: grpc.ServicerContext,
    ) -> nlp_service_pb2.AnalyzeResponse:
        """Handle an Analyze RPC."""
        log = logger.bind(method="Analyze", text_length=len(request.text))
        log.info("grpc_request_received")

        with tracer.start_as_current_span("grpc_analyze") as span:
            span.set_attribute("text.length", len(request.text))
            span.set_attribute("categories.count", len(request.categories))
            span.set_attribute("tonalities.count", len(request.tonalities))

            t0 = time.perf_counter()
            result = self._engine.analyze(request.text, list(request.categories), list(request.tonalities))
            elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)

        log.info("grpc_request_complete", processing_time_ms=elapsed_ms)
        return self._to_analyze_response(result, elapsed_ms)

    def analyze_batch(
        self,
        request: nlp_service_pb2.AnalyzeBatchRequest,
        _context: grpc.ServicerContext,
    ) -> nlp_service_pb2.AnalyzeBatchResponse:
        """Handle an AnalyzeBatch RPC."""
        log = logger.bind(method="AnalyzeBatch", batch_size=len(request.texts))
        log.info("grpc_batch_request_received")

        with tracer.start_as_current_span("grpc_analyze_batch") as span:
            span.set_attribute("batch.size", len(request.texts))
            span.set_attribute("categories.count", len(request.categories))
            span.set_attribute("tonalities.count", len(request.tonalities))

            t0 = time.perf_counter()
            results = self._engine.analyze_batch(
                list(request.texts),
                list(request.categories),
                list(request.tonalities),
            )
            elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)

        log.info("grpc_batch_request_complete", processing_time_ms=elapsed_ms)
        responses = [
            self._to_analyze_response(result, result["processing_time_ms"])
            for result in results
        ]
        return nlp_service_pb2.AnalyzeBatchResponse(results=responses)

    # gRPC expects RPC handlers with the exact method name from the proto.
    Analyze = analyze
    AnalyzeBatch = analyze_batch


def serve() -> None:
    """Start the gRPC server."""
    setup_logging()
    setup_telemetry()
    nltk.download("punkt_tab", quiet=True)

    engine = NlpEngine()

    # Instrument gRPC server with OpenTelemetry
    grpc_instrumentor = GrpcInstrumentorServer()
    grpc_instrumentor.instrument()

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    nlp_service_pb2_grpc.add_NlpServiceServicer_to_server(NlpServiceServicer(engine), server)
    bind_address = f"{settings.grpc_host}:{settings.grpc_port}"
    server.add_insecure_port(bind_address)
    server.start()
    logger.info("grpc_server_started", address=bind_address)
    server.wait_for_termination()


if __name__ == "__main__":  # pragma: no cover
    serve()
