"""Application configuration loaded from environment variables."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration sourced from .env file."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Server
    rest_host: str = "0.0.0.0"
    rest_port: int = 8000
    grpc_host: str = "0.0.0.0"
    grpc_port: int = 50051

    # Models
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    classifier_model: str = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
    sentiment_model: str = "nlptown/bert-base-multilingual-uncased-sentiment"
    toxicity_model: str = "unitary/multilingual-toxic-xlm-roberta"
    toxicity_threshold: float = 0.8

    # Qdrant
    qdrant_collection: str = "nlp_cache"
    qdrant_cache_threshold: float = 0.92

    # Observability
    log_level: str = "INFO"
    otel_service_name: str = "nlp-microservice"
    otel_exporter_endpoint: str = "http://jaeger:4317"


settings = Settings()
