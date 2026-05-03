"""
Sensor ingestion and retrieval endpoints.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from api.models import SensorReadingCreate, SensorReadingResponse
from core.config import get_settings
from core.exceptions import DataIngestionError, InvalidSensorReadingError
from core.logging import get_logger
from db.models import SensorReading, get_db

log = get_logger(__name__)
_cfg = get_settings()
router = APIRouter(prefix="/sensors", tags=["Sensors"])


@router.post(
    "/ingest",
    response_model=SensorReadingResponse,
    status_code=201,
    summary="Ingest a sensor reading",
)
def ingest_sensor_data(
    reading: SensorReadingCreate,
    db: Session = Depends(get_db),
) -> SensorReading:
    """
    Persist one sensor telemetry snapshot to the database.

    - Rejects physically impossible values (handled by Pydantic validators).
    - Defaults `timestamp` to current UTC time if not provided.
    """
    try:
        payload = reading.model_dump()
        if payload.get("timestamp") is None:
            payload["timestamp"] = datetime.now(tz=timezone.utc)

        db_reading = SensorReading(**payload)
        db.add(db_reading)
        db.commit()
        db.refresh(db_reading)
        log.info("Ingested reading id=%d machine=%s", db_reading.id, db_reading.machine_id)
        return db_reading
    except Exception as exc:
        db.rollback()
        log.exception("Failed to ingest reading for machine=%s", reading.machine_id)
        raise DataIngestionError(f"Could not persist reading: {exc}") from exc


@router.get(
    "/latest",
    response_model=List[SensorReadingResponse],
    summary="Fetch recent sensor readings",
)
def get_latest_sensors(
    machine_id: Optional[str] = Query(default=None, description="Filter by machine ID"),
    limit: int = Query(default=200, ge=1, le=1000, description="Max rows to return"),
    db: Session = Depends(get_db),
) -> list[SensorReading]:
    """
    Returns the most recent sensor readings, optionally filtered by machine.
    Results are ordered newest-first.
    """
    q = db.query(SensorReading).order_by(SensorReading.timestamp.desc())
    if machine_id:
        q = q.filter(SensorReading.machine_id == machine_id)
    return q.limit(limit).all()


@router.get(
    "/machines",
    response_model=List[str],
    summary="List all known machine IDs",
)
def list_machines(db: Session = Depends(get_db)) -> list[str]:
    """Returns a deduplicated list of machine IDs seen in sensor data."""
    rows = db.query(SensorReading.machine_id).distinct().all()
    return [r[0] for r in rows]


@router.delete(
    "/{reading_id}",
    status_code=204,
    summary="Delete a sensor reading",
)
def delete_reading(reading_id: int, db: Session = Depends(get_db)) -> None:
    """Hard-delete a sensor reading by ID (admin use only)."""
    reading = db.get(SensorReading, reading_id)
    if reading is None:
        raise HTTPException(status_code=404, detail=f"Reading {reading_id} not found.")
    db.delete(reading)
    db.commit()
    log.info("Deleted reading id=%d", reading_id)
