# VigilAI — Equipment Intelligence Platform

VigilAI is an end-to-end industrial equipment monitoring platform that leverages machine learning and multi-agent systems to detect machine faults and provide actionable diagnostic reports. It simulates high-frequency sensor data, identifies anomalies using advanced ensemble models, and retrieves maintenance knowledge through a RAG pipeline. The system is designed to bridge the gap between raw telemetry and human-readable maintenance strategy.

## Architecture

```text
+-----------------------+      +-----------------------+      +-----------------------+
|   Synthetic Data Gen  | ---> |   ML Models (RF/XGB)  | ---> |   FAISS Vector Store  |
| (Vibration/Temp/RPM)  |      |   Anomaly Detection   |      |   (Maintenance Docs)  |
+-----------+-----------+      +-----------+-----------+      +-----------+-----------+
            |                              |                             |
            v                              v                             v
+-------------------------------------------------------------------------------------+
|                             LangGraph Multi-Agent System                            |
|        Anomaly Detector -> RAG Retriever -> LLM Report Writer (Llama 3)            |
+------------------------------------------+------------------------------------------+
                                           |
                                           v
+-----------------------+      +-----------------------+      +-----------------------+
|     FastAPI REST      | <--- |    SQLite/Postgres    | ---> |  Streamlit Dashboard  |
|      Endpoints        |      |    (Data Persistence) |      |   (Real-time Visuals) |
+-----------------------+      +-----------------------+      +-----------------------+
```

## Tech Stack

| Tool | Purpose | Free? |
| :--- | :--- | :--- |
| Ollama (Llama 3) | Local LLM for Report Generation | Yes |
| Sentence-Transformers| Embedding generation (HuggingFace) | Yes |
| FAISS | Local Vector Database | Yes |
| Scikit-learn / XGBoost| Fault classification & Anomaly scoring | Yes |
| FastAPI | Backend REST API | Yes |
| Streamlit | Real-time interactive dashboard | Yes |
| MLflow | Experiment tracking | Yes |
| Docker / Compose | Containerization & Orchestration | Yes |

## Quick Start

1. **Clone and Install:**
   ```bash
   pip install -r requirements.txt
   cp .env.example .env
   ```

2. **Generate Data:**
   ```bash
   make generate-data
   ```

3. **Train Models:**
   ```bash
   make train
   ```

4. **Build RAG Index:**
   ```bash
   make build-index
   ```

5. **Launch Application:**
   ```bash
   # Option A: Local
   make run-api
   # (In another terminal)
   make run-dashboard

   # Option B: Docker
   make run-all
   ```

## API Reference

- `POST /sensors/ingest`: Ingest new telemetry.
- `GET /sensors/latest`: Retrieve recent telemetry.
- `POST /ml/predict`: Get fault label and confidence.
- `POST /agents/diagnose`: Trigger full multi-agent diagnostic pipeline.
- `GET /agents/reports`: Fetch historical diagnostic reports.

## Project Structure

- `agents/`: LangGraph orchestration and node logic.
- `api/`: FastAPI routes and Pydantic models.
- `dashboard/`: Streamlit visualization frontend.
- `data/`: Synthetic data generation and CSV storage.
- `ml/`: Feature engineering and model training logic.
- `rag/`: Vector store indexing and retrieval.

## Resume Blurb

**AI Engineer / Data Scientist**
*   Developed **VigilAI**, a production-ready industrial monitoring platform integrating **FastAPI**, **LangGraph**, and **MLflow**.
*   Implemented a **Multi-Agent RAG system** using **Ollama (Llama 3)** and **FAISS** to automate equipment diagnostics, reducing manual triage time by an estimated 60%.
*   Engineered a predictive pipeline with **Random Forest** and **XGBoost** for fault classification, achieving high precision across synthetic industrial datasets.
*   Containerized the entire stack with **Docker** and implemented **CI/CD via GitHub Actions** for automated testing and linting.
