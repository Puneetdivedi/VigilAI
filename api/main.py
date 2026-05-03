"""
FastAPI application entry point for VigilAI.
Includes startup initialization, CORS, and health/metadata endpoints.
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from api.routers import sensors, ml, agents
from db.models import Base, engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize DB tables on startup."""
    Base.metadata.create_all(bind=engine)
    print("[VigilAI] Database initialized.")
    yield
    print("[VigilAI] Shutting down.")


app = FastAPI(
    title="VigilAI Equipment Monitoring API",
    description=(
        "End-to-end industrial equipment monitoring platform. "
        "Ingest sensor data → ML fault detection → RAG-powered diagnostics → Agent reports."
    ),
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(sensors.router, prefix="", tags=["Sensors"])
app.include_router(ml.router, prefix="", tags=["ML"])
app.include_router(agents.router, prefix="", tags=["Agents"])


@app.get("/health", tags=["System"])
def health_check():
    """Returns API health status."""
    return {"status": "ok", "version": "2.0.0", "service": "VigilAI"}


@app.get("/", tags=["System"])
def root():
    """API root — redirect to docs."""
    return {
        "message": "VigilAI Equipment Intelligence API",
        "docs": "/docs",
        "redoc": "/redoc",
        "health": "/health",
    }
