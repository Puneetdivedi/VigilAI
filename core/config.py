"""
Centralised application configuration using Pydantic Settings.
All values are read from environment variables (or .env file).
"""
from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Single source of truth for every configurable value in VigilAI."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Application ────────────────────────────────────────────────────────
    app_name: str = "VigilAI"
    app_version: str = "3.0.0"
    environment: Literal["development", "staging", "production"] = "development"
    debug: bool = False

    # ── Database ───────────────────────────────────────────────────────────
    database_url: str = Field(
        default="sqlite:///./data/vigilai.db",
        description="SQLAlchemy connection string",
    )
    db_pool_size: int = 5
    db_max_overflow: int = 10
    db_pool_timeout: int = 30

    # ── LLM providers (priority: Gemini → Groq → Ollama → fallback) ───────
    google_api_key: str = ""
    groq_api_key: str = ""
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3"
    gemini_model: str = "gemini-1.5-flash"
    groq_model: str = "llama3-8b-8192"
    llm_temperature: float = 0.2
    llm_timeout_seconds: int = 60

    # ── MLflow ─────────────────────────────────────────────────────────────
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_experiment_name: str = "VigilAI-FaultDetection"

    # ── ML ─────────────────────────────────────────────────────────────────
    models_dir: str = "ml/models"
    anomaly_threshold: float = 0.5      # score above this triggers "High" alert
    fault_ratio: float = 0.15           # fraction of synthetic faults
    num_synthetic_rows: int = 10_000

    # ── API ────────────────────────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 1
    api_url: str = "http://localhost:8000"
    cors_origins: list[str] = ["*"]
    sensor_history_limit: int = 200     # /sensors/latest max rows
    prediction_history_limit: int = 100
    report_history_limit: int = 50

    # ── RAG ────────────────────────────────────────────────────────────────
    faiss_index_dir: str = "rag/faiss_index"
    embedding_model: str = "all-MiniLM-L6-v2"
    rag_top_k: int = 3

    # ── Logging ────────────────────────────────────────────────────────────
    log_level: str = "INFO"
    log_format: str = "json"            # "json" | "text"

    @field_validator("log_level")
    @classmethod
    def _validate_log_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in allowed:
            raise ValueError(f"log_level must be one of {allowed}")
        return upper


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Returns a cached singleton Settings instance."""
    return Settings()
