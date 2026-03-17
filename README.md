# NLP Microservice

A production-ready, hybrid **REST + gRPC** NLP microservice that performs language detection, sentiment analysis, toxicity detection, and zero-shot text classification with **semantic caching** powered by Qdrant.

---

## Architecture

```
┌──────────────┐      ┌──────────────┐      ┌───────────────────────┐
│ REST Client  │────▶│ FastAPI      │─────▶│                       │
│ (HTTP/JSON)  │      │ main_rest.py │      │      NLP Engine       │
└──────────────┘      └──────────────┘      │     nlp_engine.py     │
                                            │                       │
┌──────────────┐      ┌──────────────┐      │   ┌─────────────────┐ │
│ gRPC Client  │────▶│ gRPC Server  │─────▶│   │ Lingua          │ │
│ (Protobuf)   │      │ main_grpc.py │      │   │ NLTK            │ │
└──────────────┘      └──────────────┘      │   │ FastEmbed       │ │
                                            │   │ HF Models       │ │
                                            │   │ Qdrant Cache    │ │
                                            │   └─────────────────┘ │
                                            └───────────┬───────────┘
                                                        │
                                            ┌───────────▼────────────┐
                                            │ OpenTelemetry          │
                                            │ Jaeger / OTLP Exporter │
                                            └────────────────────────┘
```

### Pipeline Flow

1. **Language Detection** — Lingua detects the dominant language of the full text.
2. **Sentence Tokenization** — NLTK splits the text into sentences.
3. **Batch Embedding** — FastEmbed (`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`) encodes all sentences.
4. **Semantic Cache Lookup** — Each sentence embedding is searched in Qdrant (in-memory). On a cache hit (cosine similarity ≥ threshold), the cached inference is reused.
5. **AI Inference** — Cache misses go through:
   - **Sentiment** via `nlptown/bert-base-multilingual-uncased-sentiment`
  - **Toxicity detection** via `unitary/multilingual-toxic-xlm-roberta`
  - **Zero-shot classification** via `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli`
6. **Cache Store** — New inferences are stored in Qdrant for future reuse.
7. **Aggregation** — Sentence-level results are aggregated into a single response.

---

## Tech Stack

| Layer             | Technology                                         |
|-------------------|----------------------------------------------------|
| REST API          | FastAPI + Uvicorn                                   |
| gRPC API          | grpcio + grpcio-tools                               |
| API Docs          | Scalar (replaces Swagger UI)                        |
| Language Detection| Lingua                                              |
| Sentence Splitting| NLTK                                                |
| Embeddings        | FastEmbed (sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2) |
| Vector Cache      | Qdrant (in-memory)                                  |
| NLP Inference     | Hugging Face Transformers (mDeBERTa XNLI, mBERT sentiment, multilingual toxicity) |
| Observability     | OpenTelemetry → Jaeger, structlog (JSON)            |
| Linting           | Ruff (strict config)                                |
| CI/CD             | GitLab CI                                           |
| Containerization  | Docker multi-stage + Docker Compose                 |

---

## Quick Start

### Prerequisites

- Python 3.11+
- Docker & Docker Compose (for containerized deployment)

### Local Development

```bash
# 1. Clone and enter the project
cd nlp-microservice

# 2. Create a virtual environment
python -m venv .venv && source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Generate gRPC stubs
make proto

# 5. Copy environment config
cp .env.example .env

# 6. Start the REST server
make run-rest

# 7. (In another terminal) Start the gRPC server
make run-grpc
```

### Docker Compose

```bash
# Build and start everything (REST, gRPC, Jaeger)
docker compose up --build -d

# View logs
docker compose logs -f nlp-rest

# Access services
# REST API:  http://localhost:8000
# API Docs:  http://localhost:8000/docs
# Jaeger UI: http://localhost:16686
# gRPC:      localhost:50051
```

---

## API Reference

### REST: `POST /analyze`

**Request:**
```json
{
  "text": "Artificial intelligence is making remarkable progress in healthcare and drug discovery.",
  "categories": ["technology", "healthcare", "finance", "sports"],
  "tonalities": ["formal", "objective", "enthusiastic"]
}
```

**Response:**
```json
{
  "language": "en",
  "sentiment": "positive",
  "sentiment_score": 0.62,
  "is_toxic": false,
  "highest_toxicity_score": 0.07,
  "matched_categories": [
    {"category": "healthcare", "score": 0.78},
    {"category": "technology", "score": 0.71},
    {"category": "finance", "score": 0.15},
    {"category": "sports", "score": 0.04}
  ],
  "processing_time_ms": 1842.37
}
```

### REST: `POST /analyze/batch`

**Request:**
```json
{
  "texts": [
    "AI systems are helping doctors find diseases faster.",
    "You are useless and should disappear."
  ],
  "categories": ["technology", "healthcare"],
  "tonalities": ["formal", "aggressive"]
}
```

**Response:**
```json
{
  "results": [
    {
      "language": "en",
      "sentiment": "positive",
      "sentiment_score": 0.73,
      "matched_categories": [{"category": "healthcare", "score": 0.81}],
      "matched_tonalities": [{"tonality": "formal", "score": 0.76}],
      "is_toxic": false,
      "highest_toxicity_score": 0.06,
      "processing_time_ms": 412.15
    },
    {
      "language": "en",
      "sentiment": "negative",
      "sentiment_score": 0.89,
      "matched_categories": [{"category": "technology", "score": 0.39}],
      "matched_tonalities": [{"tonality": "aggressive", "score": 0.88}],
      "is_toxic": true,
      "highest_toxicity_score": 0.93,
      "processing_time_ms": 405.72
    }
  ]
}
```

### REST: `GET /health`
Returns `{"status": "ok"}`.

### REST: `GET /docs`
Serves the **Scalar** interactive API reference.

### gRPC: `NlpService.Analyze`
Defined in `nlp_service.proto` — accepts `AnalyzeRequest`, returns `AnalyzeResponse`.

### gRPC: `NlpService.AnalyzeBatch`
Defined in `nlp_service.proto` — accepts `AnalyzeBatchRequest`, returns `AnalyzeBatchResponse`.

---

## Configuration

All settings are loaded from `.env` via **pydantic-settings**. See `.env.example` for the full list:

| Variable                 | Default                                           | Description                        |
|--------------------------|---------------------------------------------------|------------------------------------|
| `REST_HOST`              | `0.0.0.0`                                        | REST server bind host              |
| `REST_PORT`              | `8000`                                            | REST server port                   |
| `GRPC_HOST`              | `0.0.0.0`                                        | gRPC server bind host              |
| `GRPC_PORT`              | `50051`                                           | gRPC server port                   |
| `EMBEDDING_MODEL`        | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | FastEmbed model name               |
| `CLASSIFIER_MODEL`       | `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli`       | Zero-shot classifier model         |
| `SENTIMENT_MODEL`        | `nlptown/bert-base-multilingual-uncased-sentiment`| Sentiment analysis model           |
| `TOXICITY_MODEL`         | `unitary/multilingual-toxic-xlm-roberta`         | Toxicity classification model      |
| `TOXICITY_THRESHOLD`     | `0.8`                                            | Score threshold for `is_toxic`     |
| `QDRANT_COLLECTION`      | `nlp_cache`                                       | Qdrant collection name             |
| `QDRANT_CACHE_THRESHOLD` | `0.92`                                            | Min cosine similarity for cache hit|
| `LOG_LEVEL`              | `INFO`                                            | Logging verbosity                  |
| `OTEL_SERVICE_NAME`      | `nlp-microservice`                                | OpenTelemetry service name         |
| `OTEL_EXPORTER_ENDPOINT` | `http://jaeger:4317`                              | OTLP collector endpoint            |

---

## Observability

### Structured Logging

All logs are JSON-formatted via **structlog** with meaningful fields:

```json
{"event": "cache_hit", "score": 0.96, "request_id": "abc-123", "timestamp": "2026-03-17T10:00:00Z", "level": "info"}
```

### Distributed Tracing

**OpenTelemetry** instruments both FastAPI and gRPC. Spans include:
- `nlp_analyze` (root span per request)
- `detect_language`, `sentence_tokenize`, `embed_texts`
- `cache_lookup`, `cache_store`
- `analyze_sentiment`, `analyze_toxicity`, `classify_categories`, `classify_tonality`
- `rest_analyze`, `rest_analyze_batch`, `grpc_analyze`, `grpc_analyze_batch`

View traces in **Jaeger** at `http://localhost:16686`.

---

## CI/CD (GitLab)

The `.gitlab-ci.yml` defines three stages:

| Stage   | What it does                             |
|---------|------------------------------------------|
| `lint`  | Runs `ruff check` and `ruff format --check` |
| `test`  | Installs deps, generates proto, runs `pytest` with coverage gate |
| `build` | Builds and pushes Docker image to registry |

### Coverage Gate

Coverage is enforced with pytest-cov and fails if total coverage drops below 90%.

```bash
pytest --cov=. --cov-report=term-missing --cov-fail-under=90
```

### ONNX vs Default Backend Benchmark

Use this optional benchmark to compare ONNX Runtime vs the default Transformers backend:

```bash
pip install optimum[onnxruntime]
RUN_ONNX_BENCHMARKS=1 pytest tests/test_benchmark.py -k onnx_vs_pytorch -s
# or
make bench-onnx
```

Optional environment knobs:
- `BENCHMARK_MODEL_ID` (default: `distilbert-base-uncased-finetuned-sst-2-english`)
- `BENCHMARK_LOOPS` (default: `30`)

---

## Project Structure

```
├── .env.example          # Environment variable template
├── .gitlab-ci.yml        # GitLab CI/CD pipeline
├── config.py             # Pydantic Settings (loads .env)
├── docker-compose.yml    # Multi-service orchestration
├── Dockerfile            # Multi-stage container build
├── main_grpc.py          # gRPC server entrypoint
├── main_rest.py          # FastAPI REST server entrypoint
├── Makefile              # Developer convenience commands
├── nlp_engine.py         # Core NLP pipeline with semantic cache
├── nlp_service.proto     # gRPC service definition
├── observability.py      # OTel + structlog setup
├── pyproject.toml        # Ruff & pytest configuration
├── README.md             # This file
├── requirements.txt      # Pinned dependencies
└── tests/
    ├── test_api.py       # REST integration and schema tests
    ├── test_grpc.py      # gRPC service mapping tests
    └── test_nlp_engine.py # NLP engine unit tests
```

---

## License

Internal use only. All rights reserved.
