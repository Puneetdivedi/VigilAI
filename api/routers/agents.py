"""
Router for Agent endpoints.
"""
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from typing import List
import pandas as pd
import json

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from api.models import SensorReadingCreate, AgentDiagnosisResponse, AgentReportResponse
from agents.graph import run_diagnostic_pipeline
from db.models import SessionLocal, SensorReading, AgentReport

router = APIRouter(prefix="/agents", tags=["agents"])

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.post("/diagnose", response_model=AgentDiagnosisResponse)
def diagnose(reading: SensorReadingCreate, db: Session = Depends(get_db)):
    recent = db.query(SensorReading).filter(SensorReading.machine_id == reading.machine_id).order_by(SensorReading.timestamp.desc()).limit(10).all()
    history_df = pd.DataFrame([r.__dict__ for r in recent]) if recent else pd.DataFrame()
    
    final_state = run_diagnostic_pipeline(reading.model_dump(), history_df)
    report = final_state.get("report_dict", {})
    
    return AgentDiagnosisResponse(
        fault_summary=report.get("fault_summary", ""),
        severity=report.get("severity", "Unknown"),
        recommended_actions=report.get("recommended_actions", []),
        estimated_downtime_risk=report.get("estimated_downtime_risk", "")
    )

@router.get("/reports", response_model=List[AgentReportResponse])
def get_reports(db: Session = Depends(get_db)):
    reports = db.query(AgentReport).order_by(AgentReport.timestamp.desc()).limit(20).all()
    # Need to process recommended_actions if we want to return it as a list, but models expects string or handles it
    return reports
