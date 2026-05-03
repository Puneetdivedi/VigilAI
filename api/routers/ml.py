"""
ML fault prediction endpoints.
"""
from __future__ import annotations

from typing import List, Optional

import pandas as pd
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from api.models import FaultPredictionResponse, MLPredictionResponse, SensorReadingCreate
from core.exceptions import FeatureExtractionError, ModelNotFoundError, PredictionError
from core.logging import get_logger
from db.models import FaultPrediction, SensorReading, get_db
from ml.feature_engineering import extract_features_single
from ml.predict import predict, models_ready

log = get_logger(__name__)
router = APIRouter(prefix="/ml", tags=["ML"])

_MODEL_VERSION = "v3.0"


@router.post(
    "/predict",
    response_model=MLPredictionResponse,
    summary="Run fault prediction on a sensor reading",
)
def predict_fault(
    reading: SensorReadingCreate,
    db: Session = Depends(get_db),
) -> MLPredictionResponse:
    """
    Runs the fault detection pipeline on a single sensor snapshot:

    1. Fetches recent history for the same machine (rolling window features).
    2. Extracts signal features (RMS, kurtosis, FFT peak, etc.).
    3. Runs Random Forest / XGBoost classifier + Isolation Forest anomaly scorer.
    4. Persists the prediction and returns structured results.
    """
    if not models_ready():
        raise ModelNotFoundError(
            "ML models are not loaded. Run `python ml/train.py` first.",
            details={"hint": "make train"},
        )

    # Fetch recent history for rolling feature calculations
    recent_rows = (
        db.query(SensorReading)
        .filter(SensorReading.machine_id == reading.machine_id)
        .order_by(SensorReading.timestamp.desc())
        .limit(20)
        .all()
    )
    if recent_rows:
        history_df = pd.DataFrame([
            {c: getattr(r, c) for c in ("machine_id", "vibration", "temperature", "rpm", "pressure", "timestamp")}
            for r in recent_rows
        ])
    else:
        history_df = pd.DataFrame()

    try:
        features = extract_features_single(reading.model_dump(), history_df)
    except Exception as exc:
        log.exception("Feature extraction failed for machine=%s", reading.machine_id)
        raise FeatureExtractionError(f"Feature extraction error: {exc}") from exc

    try:
        prediction = predict(features)
    except Exception as exc:
        log.exception("Prediction failed for machine=%s", reading.machine_id)
        raise PredictionError(f"Prediction runtime error: {exc}") from exc

    # Persist to DB
    try:
        db_pred = FaultPrediction(
            machine_id=reading.machine_id,
            fault_label=prediction["fault_label"],
            anomaly_score=prediction["anomaly_score"],
            confidence=prediction.get("confidence", 0.0),
            model_version=_MODEL_VERSION,
        )
        db.add(db_pred)
        db.commit()
        db.refresh(db_pred)
        log.info(
            "Prediction id=%d machine=%s label=%s score=%.3f",
            db_pred.id, db_pred.machine_id, db_pred.fault_label, db_pred.anomaly_score,
        )
    except Exception as exc:
        db.rollback()
        log.warning("Could not persist prediction: %s", exc)

    return MLPredictionResponse(
        fault_label=prediction["fault_label"],
        anomaly_score=prediction["anomaly_score"],
        confidence=prediction.get("confidence", 0.0),
        model_version=_MODEL_VERSION,
    )


@router.get(
    "/history",
    response_model=List[FaultPredictionResponse],
    summary="Fetch prediction history",
)
def get_prediction_history(
    machine_id: Optional[str] = Query(default=None, description="Filter by machine ID"),
    limit: int = Query(default=100, ge=1, le=1000),
    db: Session = Depends(get_db),
) -> list[FaultPrediction]:
    """Returns historical fault predictions, newest first."""
    q = db.query(FaultPrediction).order_by(FaultPrediction.timestamp.desc())
    if machine_id:
        q = q.filter(FaultPrediction.machine_id == machine_id)
    return q.limit(limit).all()


@router.get(
    "/status",
    summary="ML model status",
)
def model_status() -> dict:
    """Returns whether ML models are loaded and ready for inference."""
    return {"models_loaded": models_ready(), "model_version": _MODEL_VERSION}
