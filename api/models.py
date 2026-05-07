"""
Industrial-grade Pydantic API models for VigilAI.
Includes strict validation, field constraints, and OpenAPI examples.
"""
from __future__ import annotations

import json
from datetime import datetime
from typing import Annotated, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

# ── Shared constraints ─────────────────────────────────────────────────────────
_MACHINE_PATTERN = r"^MACH_\d{3,}$"

PositiveFloat = Annotated[float, Field(ge=0.0)]


# ── Request Models ─────────────────────────────────────────────────────────────

class SensorReadingCreate(BaseModel):
    """Validated payload for ingesting a single sensor snapshot."""

    model_config = {"json_schema_extra": {
        "example": {
            "machine_id": "MACH_001",
            "vibration": 12.5,
            "temperature": 65.2,
            "rpm": 3000.0,
            "pressure": 1.2,
        }
    }}

    machine_id: str = Field(
        ...,
        min_length=3,
        max_length=64,
        description="Unique machine identifier (e.g. MACH_001)",
    )
    vibration: PositiveFloat = Field(..., description="Vibration RMS in Hz", le=500.0)
    temperature: PositiveFloat = Field(..., description="Surface temperature in °C", le=500.0)
    rpm: PositiveFloat = Field(..., description="Rotational speed in RPM", le=100_000.0)
    pressure: PositiveFloat = Field(..., description="System pressure in bar", le=50.0)
    timestamp: Optional[datetime] = Field(
        default=None, description="ISO-8601 timestamp; defaults to server time"
    )

    @field_validator("vibration")
    @classmethod
    def _vib_not_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError("vibration must be non-negative")
        return round(v, 4)

    @field_validator("temperature")
    @classmethod
    def _temp_plausible(cls, v: float) -> float:
        if v > 400:
            raise ValueError("temperature > 400 °C is outside sensor range")
        return round(v, 2)


class PaginationParams(BaseModel):
    """Standard pagination query parameters."""
    limit: int = Field(default=100, ge=1, le=1000)
    offset: int = Field(default=0, ge=0)


# ── Response Models ────────────────────────────────────────────────────────────

class SensorReadingResponse(BaseModel):
    """Returned sensor reading including server-assigned id and timestamp."""
    id: int
    machine_id: str
    vibration: float
    temperature: float
    rpm: float
    pressure: float
    timestamp: datetime

    model_config = {"from_attributes": True}


class MLPredictionResponse(BaseModel):
    """ML fault prediction result."""
    fault_label: str = Field(..., description="Predicted fault type")
    anomaly_score: float = Field(..., ge=0.0, le=1.0, description="Isolation Forest anomaly score (1 = critical)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Classifier confidence")
    model_version: str = Field(default="unknown", description="Model artefact version tag")

    model_config = {"from_attributes": True}


class AgentDiagnosisResponse(BaseModel):
    """Full AI diagnostic report returned by the agent pipeline."""
    fault_summary: str = Field(..., description="Plain-English fault summary")
    severity: str = Field(..., description="Low | Medium | High | Critical")
    recommended_actions: List[str] = Field(..., description="Ordered list of maintenance actions")
    estimated_downtime_risk: str = Field(..., description="Expected downtime exposure")
    llm_provider: str = Field(default="fallback", description="LLM backend that generated this report")
    pipeline_duration_ms: Optional[float] = Field(default=None, description="End-to-end pipeline time in ms")

    @field_validator("severity")
    @classmethod
    def _validate_severity(cls, v: str) -> str:
        allowed = {"Low", "Medium", "High", "Critical"}
        if v not in allowed:
            return "Medium"   # safe default rather than raising
        return v


class AgentReportResponse(BaseModel):
    """Stored agent report returned from the database."""
    id: int
    machine_id: str
    fault_label: str
    anomaly_score: float
    confidence: float
    severity: str
    summary: str
    recommended_actions: str        # JSON string stored in DB
    estimated_downtime_risk: str
    llm_provider: str
    pipeline_duration_ms: Optional[float]
    timestamp: datetime

    model_config = {"from_attributes": True}

    @property
    def actions_list(self) -> List[str]:
        """Convenience: decode recommended_actions JSON to list."""
        try:
            return json.loads(self.recommended_actions)
        except Exception:
            return [self.recommended_actions]


class FaultPredictionResponse(BaseModel):
    """Historical fault prediction record."""
    id: int
    reading_id: Optional[int]
    machine_id: str
    fault_label: str
    anomaly_score: float
    confidence: float
    model_version: str
    timestamp: datetime

    model_config = {"from_attributes": True}


class HealthResponse(BaseModel):
    """API health check payload."""
    status: str
    version: str
    service: str
    environment: str
    db_ok: bool
    models_loaded: bool
    rag_ok: bool


class MachineMetaResponse(BaseModel):
    """Machine metadata record."""
    id: int
    machine_id: str
    machine_type: str
    location: str
    manufacturer: str
    install_date: Optional[datetime]
    last_maintenance: Optional[datetime]
    notes: Optional[str]
    created_at: datetime

    model_config = {"from_attributes": True}
