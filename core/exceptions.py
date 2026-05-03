"""
Custom exception hierarchy for VigilAI.
Raise typed exceptions; let the FastAPI exception handler convert them to HTTP responses.
"""
from __future__ import annotations


class VigilAIBaseError(Exception):
    """Root exception for all domain errors."""
    http_status: int = 500

    def __init__(self, message: str, details: dict | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


# ── Data / Ingestion ──────────────────────────────────────────────────────────

class DataIngestionError(VigilAIBaseError):
    """Raised when sensor data cannot be persisted."""
    http_status = 422


class InvalidSensorReadingError(DataIngestionError):
    """Raised when sensor values are physically impossible."""
    http_status = 400


# ── ML ────────────────────────────────────────────────────────────────────────

class ModelNotFoundError(VigilAIBaseError):
    """Raised when a trained model artefact is missing from disk."""
    http_status = 503


class PredictionError(VigilAIBaseError):
    """Raised when the prediction pipeline encounters a runtime error."""
    http_status = 500


class FeatureExtractionError(VigilAIBaseError):
    """Raised when feature engineering fails for a reading."""
    http_status = 422


# ── RAG ───────────────────────────────────────────────────────────────────────

class FAISSIndexNotFoundError(VigilAIBaseError):
    """Raised when the FAISS vector store is not built yet."""
    http_status = 503


class RetrievalError(VigilAIBaseError):
    """Raised when RAG retrieval fails at runtime."""
    http_status = 500


# ── Agent / LLM ───────────────────────────────────────────────────────────────

class LLMUnavailableError(VigilAIBaseError):
    """Raised when no LLM backend is reachable."""
    http_status = 503


class DiagnosticPipelineError(VigilAIBaseError):
    """Raised when the LangGraph diagnostic pipeline fails."""
    http_status = 500


# ── Configuration ─────────────────────────────────────────────────────────────

class ConfigurationError(VigilAIBaseError):
    """Raised on invalid or missing configuration."""
    http_status = 500
