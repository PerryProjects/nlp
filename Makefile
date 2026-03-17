# Makefile for NLP Microservice

.PHONY: proto lint format test run-rest run-grpc bench-onnx docker-build docker-up

proto: ## Generate gRPC Python stubs from .proto
	python -m grpc_tools.protoc \
		--proto_path=. \
		--python_out=. \
		--grpc_python_out=. \
		nlp_service.proto

lint: ## Run ruff linter
	ruff check .

format: ## Format code with ruff
	ruff format .

test: ## Run tests
	pytest tests/ -v

run-rest: ## Start the REST server
	uvicorn main_rest:app --host 0.0.0.0 --port 8000 --reload

run-grpc: ## Start the gRPC server
	python main_grpc.py

bench-onnx: ## Run ONNX vs default backend benchmark (optional)
	RUN_ONNX_BENCHMARKS=1 pytest tests/test_benchmark.py -k onnx_vs_pytorch -s

docker-build: ## Build Docker image
	docker build -t nlp-microservice .

docker-up: ## Start all services with Docker Compose
	docker compose up --build -d
