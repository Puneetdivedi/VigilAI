"""
Pydantic models for the FastAPI application.
"""
from pydantic import BaseModel, Field
from datetime import datetime
from typing import List, Optional

class SensorReadingCreate(BaseModel):
    """Model for incoming sensor data."""
    machine_id: str = Field(..., example="MACH_001")
    vibration: float = Field(..., example=12.5)
    temperature: float = Field(..., example=65.2)
    rpm: float = Field(..., example=3000.0)
    pressure: float = Field(..., example=1.2)
    timestamp: Optional[datetime] = None

class SensorReadingResponse(SensorReadingCreate):
    """Model for returning sensor data."""
    id: int
    timestamp: datetime

    class Config:
        from_attributes = True

class MLPredictionResponse(BaseModel):
    """Model for ML prediction results."""
    fault_label: str
    anomaly_score: float
    confidence: float

class AgentDiagnosisResponse(BaseModel):
    """Model for returning the complete Agent diagnosis."""
    fault_summary: str
    severity: str
    recommended_actions: List[str]
    estimated_downtime_risk: str

class AgentReportResponse(BaseModel):
    """Model for returning Agent reports from the database."""
    id: int
    machine_id: str
    fault_label: str
    anomaly_score: float
    severity: str
    summary: str
    recommended_actions: str
    estimated_downtime_risk: str
    timestamp: datetime

    class Config:
        from_attributes = True
