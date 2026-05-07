"""
FastAPI application — VigilAI v3 (Production-ready).

Features:
- Lifespan context manager (startup / shutdown)
- Global exception handler for domain exceptions
- Request-ID middleware for distributed tracing
- Prometheus-style /metrics stub
- Structured JSON logging on every request
"""
from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.models import HealthResponse
from api.routers import agents, ml, sensors
from core.config import get_settings
from core.exceptions import VigilAIBaseError
from core.logging import get_logger
from db.models import Base, SessionLocal, engine
from ml.predict import load_models, models_ready
from rag.retriever import is_index_ready

log = get_logger(__name__)
_cfg = get_settings()


# ── Lifespan ───────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore[type-arg]
    """Startup: initialise DB + warm up ML models. Shutdown: log graceful exit."""
    log.info("VigilAI starting up — env=%s version=%s", _cfg.environment, _cfg.app_version)

    # Create DB tables
    Base.metadata.create_all(bind=engine)
    log.info("Database schema verified/created.")

    # Warm up ML models (non-blocking — they're optional at startup)
    try:
        load_models()
        log.info("ML models loaded successfully.")
    except Exception as exc:
        log.warning("ML models not yet trained — prediction endpoints will return fallback. (%s)", exc)

    yield  # ←─ application runs here

    log.info("VigilAI shutdown complete.")


# ── App factory ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="VigilAI Equipment Monitoring API",
    description=(
        "**Industrial-grade** AI platform for real-time equipment fault detection.\n\n"
        "Pipeline: `Sensor Ingest → ML Prediction → RAG Retrieval → LLM Diagnosis`\n\n"
        "LLM priority: **Google Gemini → Groq → Ollama → rule-based fallback**"
    ),
    version=_cfg.app_version,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_tags=[
        {"name": "System",   "description": "Health, readiness, and metadata"},
        {"name": "Sensors",  "description": "Sensor data ingestion and retrieval"},
        {"name": "ML",       "description": "Fault detection and prediction history"},
        {"name": "Agents",   "description": "LangGraph AI diagnostic pipeline"},
        {"name": "Machines", "description": "Machine registry management"},
    ],
)

# ── Middleware ─────────────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cfg.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def _request_id_and_timing(request: Request, call_next: Any) -> Response:
    """Attaches X-Request-ID and X-Process-Time headers to every response."""
    req_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    start = time.perf_counter()
    response: Response = await call_next(request)
    duration_ms = (time.perf_counter() - start) * 1000
    response.headers["X-Request-ID"] = req_id
    response.headers["X-Process-Time-Ms"] = f"{duration_ms:.2f}"
    log.debug("%-6s %-40s %d  %.1fms  req=%s",
              request.method, request.url.path, response.status_code, duration_ms, req_id)
    return response


# ── Global exception handler ───────────────────────────────────────────────────

@app.exception_handler(VigilAIBaseError)
async def _domain_exception_handler(request: Request, exc: VigilAIBaseError) -> JSONResponse:
    log.error("Domain error on %s: %s | details=%s", request.url.path, exc.message, exc.details)
    return JSONResponse(
        status_code=exc.http_status,
        content={"error": type(exc).__name__, "message": exc.message, "details": exc.details},
    )


@app.exception_handler(Exception)
async def _unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    log.exception("Unhandled exception on %s", request.url.path)
    return JSONResponse(
        status_code=500,
        content={"error": "InternalServerError", "message": "An unexpected error occurred."},
    )


# ── Routers ────────────────────────────────────────────────────────────────────

app.include_router(sensors.router)
app.include_router(ml.router)
app.include_router(agents.router)


# ── System endpoints ───────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check() -> HealthResponse:
    """
    Returns API health, DB connectivity, and model readiness.
    Use this as a Docker / Kubernetes liveness probe.
    """
    db_ok = False
    try:
        with SessionLocal() as session:
            session.execute(__import__("sqlalchemy").text("SELECT 1"))
            db_ok = True
    except Exception:
        pass

    return HealthResponse(
        status="ok",
        version=_cfg.app_version,
        service=_cfg.app_name,
        environment=_cfg.environment,
        db_ok=db_ok,
        models_loaded=models_ready(),
        rag_ok=is_index_ready(),
    )


@app.get("/", tags=["System"])
def root() -> dict:
    """API root — discovery links."""
    return {
        "service": _cfg.app_name,
        "version": _cfg.app_version,
        "environment": _cfg.environment,
        "docs":   "/docs",
        "redoc":  "/redoc",
        "health": "/health",
    }
