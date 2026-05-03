"""
Agent diagnostic pipeline endpoints.
"""
from __future__ import annotations

import json
from typing import List, Optional

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from api.models import AgentDiagnosisResponse, AgentReportResponse, SensorReadingCreate
from core.exceptions import DiagnosticPipelineError
from core.logging import get_logger
from agents.graph import run_diagnostic_pipeline
from db.models import AgentReport, SensorReading, get_db

log = get_logger(__name__)
router = APIRouter(prefix="/agents", tags=["Agents"])


@router.post(
    "/diagnose",
    response_model=AgentDiagnosisResponse,
    summary="Run full AI diagnostic pipeline",
)
def diagnose(
    reading: SensorReadingCreate,
    db: Session = Depends(get_db),
) -> AgentDiagnosisResponse:
    """
    Runs the complete LangGraph multi-agent diagnostic pipeline:

    1. **Anomaly Detector** — ML fault label + anomaly score
    2. **RAG Retriever** — relevant maintenance manual context (FAISS)
    3. **Report Writer** — LLM-generated structured diagnostic report

    The report is persisted to the database and returned in the response.
    LLM priority: **Gemini → Groq → Ollama → rule-based fallback**.
    """
    # Load recent sensor history for rolling features
    recent_rows = (
        db.query(SensorReading)
        .filter(SensorReading.machine_id == reading.machine_id)
        .order_by(SensorReading.timestamp.desc())
        .limit(20)
        .all()
    )
    history_df = pd.DataFrame([
        {c: getattr(r, c) for c in ("machine_id", "vibration", "temperature", "rpm", "pressure", "timestamp")}
        for r in recent_rows
    ]) if recent_rows else pd.DataFrame()

    try:
        final_state = run_diagnostic_pipeline(reading.model_dump(), history_df)
    except Exception as exc:
        log.exception("Diagnostic pipeline failed for machine=%s", reading.machine_id)
        raise DiagnosticPipelineError(f"Pipeline error: {exc}") from exc

    report = final_state.get("report_dict", {})
    if not report:
        raise DiagnosticPipelineError("Pipeline returned an empty report.")

    return AgentDiagnosisResponse(
        fault_summary=report.get("fault_summary", "No summary."),
        severity=report.get("severity", "Medium"),
        recommended_actions=report.get("recommended_actions", []),
        estimated_downtime_risk=report.get("estimated_downtime_risk", "Unknown."),
        llm_provider=report.get("llm_provider", "fallback"),
        pipeline_duration_ms=report.get("pipeline_duration_ms"),
    )


@router.get(
    "/reports",
    response_model=List[AgentReportResponse],
    summary="Fetch stored diagnostic reports",
)
def get_reports(
    machine_id: Optional[str] = Query(default=None, description="Filter by machine ID"),
    severity: Optional[str] = Query(default=None, description="Filter by severity (Low/Medium/High/Critical)"),
    limit: int = Query(default=50, ge=1, le=500),
    db: Session = Depends(get_db),
) -> list[AgentReport]:
    """Returns historical agent diagnostic reports, newest first."""
    q = db.query(AgentReport).order_by(AgentReport.timestamp.desc())
    if machine_id:
        q = q.filter(AgentReport.machine_id == machine_id)
    if severity:
        q = q.filter(AgentReport.severity == severity)
    return q.limit(limit).all()


@router.get(
    "/reports/{report_id}",
    response_model=AgentReportResponse,
    summary="Fetch a single diagnostic report",
)
def get_report(report_id: int, db: Session = Depends(get_db)) -> AgentReport:
    """Returns a single agent report by its database ID."""
    report = db.get(AgentReport, report_id)
    if report is None:
        raise HTTPException(status_code=404, detail=f"Report {report_id} not found.")
    return report


@router.get(
    "/stats",
    summary="Diagnostic report statistics",
)
def report_stats(db: Session = Depends(get_db)) -> dict:
    """Aggregate statistics over all stored diagnostic reports."""
    from sqlalchemy import func
    total = db.query(func.count(AgentReport.id)).scalar() or 0
    by_severity = dict(
        db.query(AgentReport.severity, func.count(AgentReport.id))
        .group_by(AgentReport.severity)
        .all()
    )
    by_fault = dict(
        db.query(AgentReport.fault_label, func.count(AgentReport.id))
        .group_by(AgentReport.fault_label)
        .all()
    )
    avg_score = db.query(func.avg(AgentReport.anomaly_score)).scalar()
    return {
        "total_reports": total,
        "by_severity": by_severity,
        "by_fault_label": by_fault,
        "avg_anomaly_score": round(float(avg_score), 4) if avg_score else 0.0,
    }
