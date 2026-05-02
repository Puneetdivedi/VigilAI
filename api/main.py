"""
FastAPI application entry point.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routers import sensors, ml, agents

app = FastAPI(
    title="VigilAI Equipment Monitoring API",
    description="API for ingesting sensor data, running ML models, and generating agent diagnostics.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(sensors.router)
app.include_router(ml.router)
app.include_router(agents.router)

@app.get("/health")
def health_check():
    return {"status": "ok"}
