"""Benchmark tests for critical NLP engine hot paths."""

from __future__ import annotations

import os
from time import perf_counter

import pytest
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


def _build_engine() -> NlpEngine:
    engine = NlpEngine.__new__(NlpEngine)
    engine._tracer = _DummyTracer()
    return engine


def test_benchmark_point_id_generation() -> None:
    """Benchmark cache key generation for sentence-level requests."""
    engine = _build_engine()
    start = perf_counter()
    result = ""
    for _ in range(20_000):
        result = engine._point_id(
            "The same sentence can be evaluated under many label sets.",
            ["technology", "science", "finance"],
            ["formal", "objective"],
        )
    elapsed = perf_counter() - start

    assert isinstance(result, str)
    assert len(result) == 32
    assert elapsed < 1.0


def test_benchmark_aggregate_categories() -> None:
    """Benchmark aggregation of sentence-level category scores."""
    results = [
        {
            "category_scores": [
                {"category": "technology", "score": 0.81},
                {"category": "science", "score": 0.64},
                {"category": "finance", "score": 0.22},
            ]
        }
        for _ in range(500)
    ]
    categories = ["technology", "science", "finance"]

    start = perf_counter()
    aggregated = []
    for _ in range(200):
        aggregated = NlpEngine._aggregate_categories(results, categories)
    elapsed = perf_counter() - start

    assert len(aggregated) == 3
    assert aggregated[0]["category"] == "technology"
    assert elapsed < 1.0


def test_benchmark_analyze_batch_dispatch() -> None:
    """Benchmark batch dispatch overhead when per-item analyze is cheap."""
    engine = _build_engine()

    def fake_analyze(text: str, categories: list[str], tonalities: list[str]) -> dict:
        return {
            "language": "en",
            "sentiment": "positive",
            "sentiment_score": 0.9,
            "matched_categories": [{"category": c, "score": 0.5} for c in categories],
            "matched_tonalities": [{"tonality": t, "score": 0.4} for t in tonalities],
            "is_toxic": False,
            "highest_toxicity_score": 0.03,
            "processing_time_ms": float(len(text)),
        }

    engine.analyze = fake_analyze  # type: ignore[method-assign]

    texts = [f"Sample text number {idx}." for idx in range(100)]
    categories = ["technology", "science"]
    tonalities = ["formal", "objective"]

    start = perf_counter()
    output = []
    for _ in range(200):
        output = engine.analyze_batch(texts, categories, tonalities)
    elapsed = perf_counter() - start

    assert len(output) == len(texts)
    assert elapsed < 2.0


def test_benchmark_onnx_vs_pytorch_text_classification() -> None:
    """Compare ONNX Runtime vs default Transformers backend for one model.

    Run this test explicitly with:
    RUN_ONNX_BENCHMARKS=1 pytest tests/test_benchmark.py -k onnx_vs_pytorch -s
    """
    if os.getenv("RUN_ONNX_BENCHMARKS") != "1":
        pytest.skip("Set RUN_ONNX_BENCHMARKS=1 to run ONNX comparison benchmark")

    try:
        from optimum.onnxruntime import ORTModelForSequenceClassification
        from transformers import AutoTokenizer, pipeline
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"Optional ONNX benchmark dependencies missing: {exc}")

    model_id = os.getenv("BENCHMARK_MODEL_ID", "distilbert-base-uncased-finetuned-sst-2-english")
    sample_text = (
        "Artificial intelligence systems can process language quickly and accurately "
        "for many enterprise use cases."
    )
    loops = int(os.getenv("BENCHMARK_LOOPS", "30"))

    # Baseline: default Transformers backend
    torch_classifier = pipeline("text-classification", model=model_id, device=-1)
    torch_classifier(sample_text, truncation=True)

    start = perf_counter()
    for _ in range(loops):
        torch_classifier(sample_text, truncation=True)
    torch_elapsed = perf_counter() - start

    # ONNX Runtime path for the same model id
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    onnx_model = ORTModelForSequenceClassification.from_pretrained(model_id, export=True)
    onnx_classifier = pipeline(
        "text-classification",
        model=onnx_model,
        tokenizer=tokenizer,
        device=-1,
    )
    onnx_classifier(sample_text, truncation=True)

    start = perf_counter()
    for _ in range(loops):
        onnx_classifier(sample_text, truncation=True)
    onnx_elapsed = perf_counter() - start

    torch_per_req_ms = (torch_elapsed / loops) * 1000
    onnx_per_req_ms = (onnx_elapsed / loops) * 1000
    speedup = torch_per_req_ms / onnx_per_req_ms if onnx_per_req_ms else float("inf")

    print(
        "ONNX benchmark:",
        {
            "model_id": model_id,
            "loops": loops,
            "torch_per_req_ms": round(torch_per_req_ms, 3),
            "onnx_per_req_ms": round(onnx_per_req_ms, 3),
            "onnx_speedup_x": round(speedup, 3),
        },
    )

    assert torch_per_req_ms > 0
    assert onnx_per_req_ms > 0
