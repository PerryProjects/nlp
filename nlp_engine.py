"""Core NLP engine: language, sentiment, toxicity, and classification with semantic caching."""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import TYPE_CHECKING

import structlog
from fastembed import TextEmbedding
from lingua import Language, LanguageDetectorBuilder
from nltk.tokenize import sent_tokenize
from opentelemetry import trace
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from transformers import AutoTokenizer, pipeline

from config import settings

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

logger = structlog.get_logger()

# ---------------------------------------------------------------------------
# Embedding dimension for paraphrase-multilingual-MiniLM-L12-v2
# ---------------------------------------------------------------------------
_EMBEDDING_DIM = 384
_MODELS_ROOT = Path("/models")
_CLASSIFIER_ONNX_DIR = _MODELS_ROOT / "classifier"
_SENTIMENT_ONNX_DIR = _MODELS_ROOT / "sentiment"
_TOXICITY_ONNX_DIR = _MODELS_ROOT / "toxicity"


class NlpEngine:
    """Orchestrates language, sentiment, toxicity, and zero-shot classification."""

    def __init__(self) -> None:
        log = logger.bind(component="NlpEngine")
        log.info("initializing_models")

        self._tracer = trace.get_tracer(__name__)

        self._lingua = (
            LanguageDetectorBuilder.from_languages(
                Language.ENGLISH,
                Language.GERMAN,
                Language.SPANISH,
                Language.FRENCH,
            )
            .with_preloaded_language_models()
            .build()
        )
        log.info("lingua_ready")

        self._embedder = TextEmbedding(model_name=settings.embedding_model)
        log.info("embedder_ready", model=settings.embedding_model)

        self._classifier = self._build_sequence_pipeline(
            task="zero-shot-classification",
            local_dir=_CLASSIFIER_ONNX_DIR,
            fallback_model=settings.classifier_model,
        )
        log.info("classifier_ready", model=settings.classifier_model)

        self._sentiment = self._build_sequence_pipeline(
            task="sentiment-analysis",
            local_dir=_SENTIMENT_ONNX_DIR,
            fallback_model=settings.sentiment_model,
        )
        log.info("sentiment_ready", model=settings.sentiment_model)

        self._toxicity = self._build_sequence_pipeline(
            task="text-classification",
            local_dir=_TOXICITY_ONNX_DIR,
            fallback_model=settings.toxicity_model,
        )
        log.info(
            "toxicity_ready",
            model=settings.toxicity_model,
            threshold=settings.toxicity_threshold,
        )

        self._qdrant = QdrantClient(":memory:")
        self._init_qdrant_collection()
        log.info("qdrant_cache_ready", collection=settings.qdrant_collection)

    def _build_sequence_pipeline(self, task: str, local_dir: Path, fallback_model: str) -> object:
        """Build a text pipeline from local ONNX artifacts, fallback to model id if missing."""
        if local_dir.exists():
            try:
                from optimum.onnxruntime import ORTModelForSequenceClassification
            except ImportError as exc:  # pragma: no cover - docker runtime dependency
                msg = "optimum[onnxruntime] is required when local ONNX model paths are present"
                raise RuntimeError(msg) from exc

            tokenizer = AutoTokenizer.from_pretrained(local_dir.as_posix())
            model = ORTModelForSequenceClassification.from_pretrained(local_dir.as_posix())
            logger.info("onnx_model_loaded", task=task, path=local_dir.as_posix())
            return pipeline(task, model=model, tokenizer=tokenizer, device=-1)

        logger.warning(
            "onnx_model_missing_fallback",
            task=task,
            path=local_dir.as_posix(),
            fallback_model=fallback_model,
        )
        return pipeline(task, model=fallback_model, device=-1)

    # ------------------------------------------------------------------
    # Qdrant cache helpers
    # ------------------------------------------------------------------

    def _init_qdrant_collection(self) -> None:
        if not self._qdrant.collection_exists(settings.qdrant_collection):
            self._qdrant.create_collection(
                collection_name=settings.qdrant_collection,
                vectors_config=VectorParams(size=_EMBEDDING_DIM, distance=Distance.COSINE),
            )

    @staticmethod
    def _point_id(text: str, categories: list[str], tonalities: list[str]) -> str:
        """Deterministic point ID from text and requested labels."""
        unique_string = f"{text}_{sorted(categories)}_{sorted(tonalities)}"
        return hashlib.sha256(unique_string.encode()).hexdigest()[:32]

    def _embed(self, texts: list[str]) -> list[NDArray[np.float32]]:
        with self._tracer.start_as_current_span("embed_texts"):
            return list(self._embedder.embed(texts))

    def _cache_lookup(
        self,
        embedding: NDArray[np.float32],
        categories: list[str],
        tonalities: list[str],
    ) -> dict | None:
        """Search Qdrant for a cached result above the similarity threshold."""
        with self._tracer.start_as_current_span("cache_lookup"):
            results = self._qdrant.query_points(
                collection_name=settings.qdrant_collection,
                query=embedding.tolist(),
                limit=1,
            )
            if results.points and results.points[0].score >= settings.qdrant_cache_threshold:
                payload = results.points[0].payload
                if not isinstance(payload, dict):
                    logger.info("cache_miss", reason="invalid_payload")
                    return None

                cached_categories = payload.get("requested_categories")
                cached_tonalities = payload.get("requested_tonalities")
                if cached_categories == sorted(categories) and cached_tonalities == sorted(tonalities):
                    logger.info("cache_hit", score=results.points[0].score)
                    return payload

                logger.info(
                    "cache_miss",
                    reason="request_labels_mismatch",
                    score=results.points[0].score,
                )
                return None
            logger.info("cache_miss")
            return None

    def _cache_store(
        self,
        text: str,
        categories: list[str],
        tonalities: list[str],
        embedding: NDArray[np.float32],
        payload: dict,
    ) -> None:
        """Upsert inference result into Qdrant."""
        with self._tracer.start_as_current_span("cache_store"):
            self._qdrant.upsert(
                collection_name=settings.qdrant_collection,
                points=[
                    PointStruct(
                        id=self._point_id(text, categories, tonalities),
                        vector=embedding.tolist(),
                        payload=payload,
                    )
                ],
            )

    # ------------------------------------------------------------------
    # Individual NLP tasks
    # ------------------------------------------------------------------

    def detect_language(self, text: str) -> str:
        """Detect the dominant language of the input text."""
        with self._tracer.start_as_current_span("detect_language"):
            detected = self._lingua.detect_language_of(text)
            lang = detected.iso_code_639_1.name.lower() if detected else "unknown"
            logger.info("language_detected", language=lang)
            return lang

    _SENTIMENT_MAP: dict[str, str] = {
        "1 star": "negative",
        "2 stars": "negative",
        "3 stars": "neutral",
        "4 stars": "positive",
        "5 stars": "positive",
    }

    def analyze_sentiment(self, text: str) -> tuple[str, float]:
        """Return (label, score) for overall sentiment."""
        with self._tracer.start_as_current_span("analyze_sentiment"):
            result = self._sentiment(text, truncation=True)[0]
            raw_label: str = result["label"]
            score: float = result["score"]
            label = self._SENTIMENT_MAP.get(raw_label, raw_label)
            logger.info("sentiment_analyzed", label=label, score=round(score, 4))
            return label, score

    def classify_categories(self, text: str, categories: list[str]) -> list[dict[str, float | str]]:
        """Zero-shot classify text against provided categories."""
        with self._tracer.start_as_current_span("classify_categories"):
            result = self._classifier(
                text,
                candidate_labels=categories,
                truncation=True,
            )
            scored = [
                {"category": lbl, "score": round(sc, 4)}
                for lbl, sc in zip(result["labels"], result["scores"], strict=False)
            ]
            logger.info("categories_classified", count=len(scored))
            return scored

    def analyze_toxicity(self, text: str) -> tuple[float, bool]:
        """Return (toxicity_score, is_toxic) based on configured threshold."""
        with self._tracer.start_as_current_span("analyze_toxicity"):
            raw_result = self._toxicity(text, top_k=None, truncation=True)
            candidates = raw_result[0] if raw_result and isinstance(raw_result[0], list) else raw_result

            toxicity_score = 0.0
            for item in candidates:
                label = str(item.get("label", "")).lower()
                score = float(item.get("score", 0.0))
                if "toxic" in label and "non" not in label:
                    toxicity_score = max(toxicity_score, score)

            is_toxic = toxicity_score >= settings.toxicity_threshold
            logger.info(
                "toxicity_analyzed",
                toxicity_score=round(toxicity_score, 4),
                is_toxic=is_toxic,
            )
            return toxicity_score, is_toxic

    # ------------------------------------------------------------------
    # Sentence-level cached inference
    # ------------------------------------------------------------------

    def classify_tonality(self, text: str, tonalities: list[str]) -> list[dict[str, float | str]]:
        """Zero-shot classify text against provided tonality labels."""
        with self._tracer.start_as_current_span("classify_tonality"):
            result = self._classifier(
                text,
                candidate_labels=tonalities,
                truncation=True,
            )
            scored = [
                {"tonality": lbl, "score": round(sc, 4)}
                for lbl, sc in zip(result["labels"], result["scores"], strict=False)
            ]
            logger.info("tonality_classified", count=len(scored))
            return scored

    def _infer_sentence(
        self,
        sentence: str,
        categories: list[str],
        tonalities: list[str],
        embedding: NDArray[np.float32],
    ) -> dict:
        """Run sentiment + classification + tonality on a single sentence, with cache."""
        cached = self._cache_lookup(embedding, categories, tonalities)
        if cached is not None:
            return cached

        sentiment_label, sentiment_score = self.analyze_sentiment(sentence)
        toxicity_score, is_toxic = self.analyze_toxicity(sentence)
        cat_scores = self.classify_categories(sentence, categories) if categories else []
        ton_scores = self.classify_tonality(sentence, tonalities) if tonalities else []

        payload = {
            "sentiment_label": sentiment_label,
            "sentiment_score": sentiment_score,
            "toxicity_score": toxicity_score,
            "is_toxic": is_toxic,
            "category_scores": cat_scores,
            "tonality_scores": ton_scores,
            "requested_categories": sorted(categories),
            "requested_tonalities": sorted(tonalities),
        }
        self._cache_store(sentence, categories, tonalities, embedding, payload)
        return payload

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def analyze(self, text: str, categories: list[str], tonalities: list[str] | None = None) -> dict:
        """Full analysis pipeline: language, sentiment, categories, tonality with caching."""
        if tonalities is None:
            tonalities = []
        with self._tracer.start_as_current_span("nlp_analyze") as span:
            t0 = time.perf_counter()
            span.set_attribute("text.length", len(text))
            span.set_attribute("categories.count", len(categories))
            span.set_attribute("tonalities.count", len(tonalities))

            # 1. Language detection (on full text)
            language = self.detect_language(text)

            # 2. Sentence tokenization via NLTK
            with self._tracer.start_as_current_span("sentence_tokenize"):
                sentences = sent_tokenize(text)
                logger.info("sentences_split", count=len(sentences))

            # 3. Batch-embed all sentences
            embeddings = self._embed(sentences)

            # 4. Per-sentence inference (with cache)
            sentence_results = [
                self._infer_sentence(sent, categories, tonalities, emb)
                for sent, emb in zip(sentences, embeddings, strict=False)
            ]

            # 5. Aggregate sentiment
            aggregated_sentiment = self._aggregate_sentiment(sentence_results)

            # 6. Aggregate categories
            aggregated_categories = self._aggregate_categories(sentence_results, categories)

            # 7. Aggregate tonalities
            aggregated_tonalities = self._aggregate_tonalities(sentence_results, tonalities)

            # 8. Aggregate toxicity
            aggregated_toxicity = self._aggregate_toxicity(sentence_results)

            elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
            span.set_attribute("processing_time_ms", elapsed_ms)
            span.set_attribute("toxicity.is_toxic", aggregated_toxicity["is_toxic"])
            span.set_attribute(
                "toxicity.highest_score",
                aggregated_toxicity["highest_toxicity_score"],
            )

            result = {
                "language": language,
                "sentiment": aggregated_sentiment["label"],
                "sentiment_score": aggregated_sentiment["score"],
                "matched_categories": aggregated_categories,
                "matched_tonalities": aggregated_tonalities,
                "is_toxic": aggregated_toxicity["is_toxic"],
                "highest_toxicity_score": aggregated_toxicity["highest_toxicity_score"],
                "processing_time_ms": elapsed_ms,
            }
            logger.info("analysis_complete", processing_time_ms=elapsed_ms)
            return result

    def analyze_batch(
        self,
        texts: list[str],
        categories: list[str],
        tonalities: list[str] | None = None,
    ) -> list[dict]:
        """Analyze a batch of texts with shared categories and tonalities."""
        if tonalities is None:
            tonalities = []

        with self._tracer.start_as_current_span("nlp_analyze_batch") as span:
            span.set_attribute("batch.size", len(texts))
            span.set_attribute("categories.count", len(categories))
            span.set_attribute("tonalities.count", len(tonalities))

            results = [self.analyze(text, categories, tonalities) for text in texts]
            logger.info("batch_analysis_complete", batch_size=len(results))
            return results

    # ------------------------------------------------------------------
    # Aggregation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _aggregate_sentiment(results: list[dict]) -> dict[str, float | str]:
        """Pick the most frequent sentiment label and average its score."""
        if not results:
            return {"label": "unknown", "score": 0.0}

        label_scores: dict[str, list[float]] = {}
        for r in results:
            lbl = r["sentiment_label"]
            label_scores.setdefault(lbl, []).append(r["sentiment_score"])

        best_label = max(label_scores, key=lambda k: len(label_scores[k]))
        avg_score = round(sum(label_scores[best_label]) / len(label_scores[best_label]), 4)
        return {"label": best_label, "score": avg_score}

    @staticmethod
    def _aggregate_categories(
        results: list[dict], categories: list[str]
    ) -> list[dict[str, float | str]]:
        """Average category scores across sentences."""
        if not results or not categories:
            return []

        cat_accum: dict[str, list[float]] = {c: [] for c in categories}
        for r in results:
            for cs in r.get("category_scores", []):
                cat = cs["category"]
                if cat in cat_accum:
                    cat_accum[cat].append(cs["score"])

        aggregated = [
            {"category": cat, "score": round(sum(scores) / len(scores), 4)}
            for cat, scores in cat_accum.items()
            if scores
        ]
        aggregated.sort(key=lambda x: x["score"], reverse=True)
        return aggregated

    @staticmethod
    def _aggregate_tonalities(
        results: list[dict], tonalities: list[str]
    ) -> list[dict[str, float | str]]:
        """Average tonality scores across sentences."""
        if not results or not tonalities:
            return []

        ton_accum: dict[str, list[float]] = {t: [] for t in tonalities}
        for r in results:
            for ts in r.get("tonality_scores", []):
                ton = ts["tonality"]
                if ton in ton_accum:
                    ton_accum[ton].append(ts["score"])

        aggregated = [
            {"tonality": ton, "score": round(sum(scores) / len(scores), 4)}
            for ton, scores in ton_accum.items()
            if scores
        ]
        aggregated.sort(key=lambda x: x["score"], reverse=True)
        return aggregated

    @staticmethod
    def _aggregate_toxicity(results: list[dict]) -> dict[str, float | bool]:
        """Aggregate toxicity signal across sentences."""
        if not results:
            return {"is_toxic": False, "highest_toxicity_score": 0.0}

        highest = max(float(r.get("toxicity_score", 0.0)) for r in results)
        is_toxic = any(bool(r.get("is_toxic", False)) for r in results)
        return {"is_toxic": is_toxic, "highest_toxicity_score": round(highest, 4)}
