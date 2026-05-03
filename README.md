# VigilAI — Equipment Intelligence Platform

> **End-to-end AI-powered industrial equipment monitoring** — from raw sensor telemetry to plain-English diagnostic reports via a LangGraph multi-agent pipeline.

[![CI](https://github.com/Puneetdivedi/VigilAI/actions/workflows/ci.yml/badge.svg)](https://github.com/Puneetdivedi/VigilAI/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110-green?logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-1.33-red?logo=streamlit)
![MLflow](https://img.shields.io/badge/MLflow-2.12-blue?logo=mlflow)
![Docker](https://img.shields.io/badge/Docker-ready-blue?logo=docker)

---

## 🏗️ Architecture

```
Sensor Data (synthetic) ──▶ ML Models (RF / XGBoost / IsoForest)
                                │
                                ▼
                        LangGraph Pipeline
                        ┌───────────────────────────────────┐
                        │ 1. anomaly_detector_node (ML)      │
                        │ 2. rag_retriever_node  (FAISS)     │
                        │ 3. report_writer_node  (Gemini LLM)│
                        └───────────────────────────────────┘
                                │
                                ▼
                  FastAPI REST API  ◀──▶  Streamlit Dashboard
                        │
                        ▼
              SQLite (via SQLAlchemy ORM)
                        │
                        ▼
               MLflow Experiment Tracking
```

---

## 🚀 Quickstart

### 1. Clone & Install

```bash
git clone https://github.com/Puneetdivedi/VigilAI.git
cd VigilAI
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env — set GOOGLE_API_KEY (or GROQ_API_KEY) for LLM support
```

LLM priority: **Google Gemini → Groq → Ollama → rule-based fallback**

### 3. Run the Full Pipeline

```bash
# Generate synthetic sensor data
python data/generate_data.py

# Train ML models (logs to MLflow)
python ml/train.py

# Build the FAISS RAG index
python rag/build_index.py

# (Optional) Run a single end-to-end demo
python demo_run.py
```

### 4. Start Services

```bash
# API server (port 8000)
uvicorn api.main:app --reload --port 8000

# Dashboard (port 8501)
streamlit run dashboard/app.py

# MLflow tracking UI (port 5000)
mlflow ui
```

Or use **Docker Compose**:

```bash
docker-compose up --build
```

---

## 🧩 Project Structure

```
VigilAI/
├── agents/           # LangGraph nodes + graph + prompts
├── api/              # FastAPI app, routers, Pydantic models
│   └── routers/      # sensors / ml / agents endpoints
├── dashboard/        # Premium Streamlit UI
├── data/             # Synthetic data generator + CSV output
├── db/               # SQLAlchemy ORM models
├── docker/           # Dockerfiles (API + Dashboard)
├── knowledge_base/   # Maintenance manual chunks for RAG
├── ml/               # Feature engineering, training, prediction
├── mlflow_tracking/  # MLflow setup utilities
├── rag/              # FAISS index builder + retriever
├── tests/            # pytest suites (ML, API, agents)
├── .github/          # GitHub Actions CI/CD
├── docker-compose.yml
├── Makefile
└── requirements.txt
```

---

## 🤖 LLM Integration

VigilAI tries LLM backends in priority order:

| Priority | Provider | Env Var |
|----------|----------|---------|
| 1 | Google Gemini (`gemini-1.5-flash`) | `GOOGLE_API_KEY` |
| 2 | Groq (`llama3-8b-8192`) | `GROQ_API_KEY` |
| 3 | Ollama (local `llama3`) | `OLLAMA_BASE_URL` |
| 4 | Rule-based fallback | — (no config needed) |

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

---

## 📊 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | API health check |
| GET | `/sensors/latest` | Fetch recent sensor readings |
| POST | `/sensors/ingest` | Ingest new sensor reading |
| POST | `/ml/predict` | Run ML fault prediction |
| POST | `/agents/diagnose` | Full agent diagnostic pipeline |
| GET | `/agents/reports` | Fetch all stored agent reports |

Interactive docs: **`http://localhost:8000/docs`**

---

## 📦 Makefile Shortcuts

```bash
make generate-data   # Synthetic sensor data
make train           # Train ML models
make build-index     # Build FAISS RAG index
make run-api         # Start FastAPI server
make run-dashboard   # Start Streamlit dashboard
make test            # Run pytest
make lint            # Run ruff linter
```

---

## 📄 License

MIT — see [LICENSE](LICENSE)
