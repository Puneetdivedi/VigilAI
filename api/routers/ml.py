"""
Router for ML endpoints.
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
import pandas as pd

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from api.models import SensorReadingCreate, MLPredictionResponse
from ml.predict import predict
from ml.feature_engineering import extract_features_single
from db.models import SessionLocal, SensorReading, FaultPrediction

router = APIRouter(prefix="/ml", tags=["ml"])

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.post("/predict", response_model=MLPredictionResponse)
def predict_fault(reading: SensorReadingCreate, db: Session = Depends(get_db)):
    # Get recent history for rolling features
    recent = db.query(SensorReading).filter(SensorReading.machine_id == reading.machine_id).order_by(SensorReading.timestamp.desc()).limit(10).all()
    history_df = pd.DataFrame([r.__dict__ for r in recent]) if recent else pd.DataFrame()
    
    features = extract_features_single(reading.model_dump(), history_df)
    prediction = predict(features)
    
    # Save prediction to DB
    db_pred = FaultPrediction(
        machine_id=reading.machine_id,
        fault_label=prediction["fault_label"],
        anomaly_score=prediction["anomaly_score"]
    )
    db.add(db_pred)
    db.commit()
    
    return prediction

@router.get("/history")
def get_prediction_history(db: Session = Depends(get_db)):
    preds = db.query(FaultPrediction).order_by(FaultPrediction.timestamp.desc()).limit(50).all()
    return preds
