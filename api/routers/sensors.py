"""
Router for sensor endpoints.
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from api.models import SensorReadingCreate, SensorReadingResponse
from db.models import SensorReading, SessionLocal

router = APIRouter(prefix="/sensors", tags=["sensors"])

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.post("/ingest", response_model=SensorReadingResponse)
def ingest_sensor_data(reading: SensorReadingCreate, db: Session = Depends(get_db)):
    db_reading = SensorReading(**reading.model_dump())
    db.add(db_reading)
    db.commit()
    db.refresh(db_reading)
    return db_reading

@router.get("/latest", response_model=List[SensorReadingResponse])
def get_latest_sensors(db: Session = Depends(get_db)):
    return db.query(SensorReading).order_by(SensorReading.timestamp.desc()).limit(100).all()
