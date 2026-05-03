.PHONY: help setup generate-data train build-index run-api run-dashboard run-mlflow demo test lint format clean docker-up docker-down

# ── Variables ───────────────────────────────────────────────────────────────────
PYTHON   := python
PIP      := pip
UVICORN  := uvicorn
STREAMLIT := streamlit

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ""

# ── Setup ───────────────────────────────────────────────────────────────────────
setup:  ## Install all dependencies
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

# ── Data Pipeline ───────────────────────────────────────────────────────────────
generate-data:  ## Generate synthetic sensor data (writes CSV + DB)
	$(PYTHON) data/generate_data.py

generate-data-csv:  ## Generate data CSV only (no DB write)
	$(PYTHON) data/generate_data.py --csv-only

# ── ML ──────────────────────────────────────────────────────────────────────────
train:  ## Train all ML models and log to MLflow
	$(PYTHON) ml/train.py

build-index:  ## Build FAISS RAG index from maintenance docs
	$(PYTHON) rag/build_index.py

# ── Services ─────────────────────────────────────────────────────────────────────
run-api:  ## Start FastAPI server on :8000 (with auto-reload)
	$(UVICORN) api.main:app --reload --host 0.0.0.0 --port 8000

run-api-prod:  ## Start FastAPI server (production mode)
	$(UVICORN) api.main:app --host 0.0.0.0 --port 8000 --workers 2

run-dashboard:  ## Start Streamlit dashboard on :8501
	$(STREAMLIT) run dashboard/app.py --server.port 8501 --server.address 0.0.0.0

run-mlflow:  ## Start MLflow tracking UI on :5000
	mlflow ui --port 5000

# ── Demo ────────────────────────────────────────────────────────────────────────
demo:  ## Run full end-to-end demo (requires trained models + FAISS index)
	$(PYTHON) demo_run.py

pipeline:  ## Full pipeline from scratch: data → train → index → demo
	$(MAKE) generate-data
	$(MAKE) train
	$(MAKE) build-index
	$(MAKE) demo

# ── Quality ──────────────────────────────────────────────────────────────────────
lint:  ## Lint with ruff
	ruff check . --output-format=concise

format:  ## Auto-fix lint issues with ruff
	ruff check . --fix

test:  ## Run tests with coverage
	pytest tests/ -v --cov=. --cov-report=term-missing --tb=short

test-fast:  ## Run tests (no coverage, faster)
	pytest tests/ -v --tb=short

# ── Docker ───────────────────────────────────────────────────────────────────────
docker-up:  ## Build and start all Docker services
	docker-compose up --build -d

docker-down:  ## Stop and remove Docker services
	docker-compose down

docker-logs:  ## Follow logs from all services
	docker-compose logs -f

# ── Cleanup ───────────────────────────────────────────────────────────────────────
clean:  ## Remove generated artefacts (models, FAISS index, pycache)
	rm -rf ml/models/*.pkl rag/faiss_index/ data/vigilai*.db data/vigilai_test.db
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
