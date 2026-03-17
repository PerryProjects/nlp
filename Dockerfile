# =============================================================================
# Production Dockerfile for NLP Microservice with pre-baked ONNX models
# =============================================================================

# --- Stage 1: Build proto stubs ---
FROM python:3.11-slim AS proto-builder

WORKDIR /build
COPY nlp_service.proto .
RUN python -m pip install --no-cache-dir grpcio-tools==1.72.0 protobuf==6.30.0
RUN python -m grpc_tools.protoc \
    --proto_path=. \
    --python_out=. \
    --grpc_python_out=. \
    nlp_service.proto

# --- Stage 2: Runtime ---
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    NLTK_DATA=/usr/local/share/nltk_data

WORKDIR /app

# Install minimal system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies with CPU-only torch first
COPY requirements.txt .
RUN python -m pip install --no-cache-dir --upgrade pip && \
    python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch && \
    python -m pip install --no-cache-dir -r requirements.txt && \
    python -m pip install --no-cache-dir "optimum[onnxruntime]"

# Export ONNX models at build time
COPY scripts/export_onnx.py /tmp/export_onnx.py
RUN python /tmp/export_onnx.py --output-dir /models

# Pre-download NLTK data into shared location
RUN python -c "import nltk; nltk.download('punkt_tab', download_dir='/usr/local/share/nltk_data')"

# Copy generated proto stubs
COPY --from=proto-builder /build/nlp_service_pb2.py /build/nlp_service_pb2_grpc.py ./

# Copy application source
COPY config.py observability.py nlp_engine.py main_rest.py main_grpc.py ./

EXPOSE 8000 50051

CMD ["uvicorn", "main_rest:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
