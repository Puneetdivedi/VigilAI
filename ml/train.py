"""
Industrial-grade ML training pipeline for VigilAI.

Trains Random Forest, XGBoost, and Isolation Forest.
Logs all experiments to MLflow with full metadata.
Saves the best classification model, Isolation Forest, and LabelEncoder as pickles.
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.config import get_settings
from core.logging import Timer, get_logger
from ml.feature_engineering import extract_features
from mlflow_tracking.setup import setup_mlflow

log = get_logger(__name__)
_cfg = get_settings()

FEATURE_COLS = [
    "vibration", "temperature", "rpm", "pressure",
    "temp_rolling_mean", "pressure_dev",
    "vib_kurtosis", "vib_rms", "vib_fft_peak",
]
MODELS_DIR = Path(_cfg.models_dir)


def _log_classification_report(y_true, y_pred, le: LabelEncoder) -> None:
    """Logs per-class metrics to MLflow as individual scalars."""
    labels = le.classes_
    report = classification_report(y_true, y_pred, target_names=labels, output_dict=True)
    for label, metrics in report.items():
        if isinstance(metrics, dict):
            for metric, value in metrics.items():
                mlflow.log_metric(f"{label}_{metric}".replace(" ", "_"), round(value, 4))


def _train_random_forest(X_train, X_test, y_train, y_test, le) -> tuple:
    log.info("Training Random Forest...")
    with mlflow.start_run(run_name="RandomForest", nested=True):
        params = {"n_estimators": 200, "max_depth": None, "min_samples_split": 5, "random_state": 42, "n_jobs": -1}
        mlflow.log_params(params)
        mlflow.log_param("model_type", "RandomForest")

        with Timer("RandomForest training", log):
            rf = RandomForestClassifier(**params)
            rf.fit(X_train, y_train)

        preds = rf.predict(X_test)
        acc = accuracy_score(y_test, preds)
        f1  = f1_score(y_test, preds, average="weighted", zero_division=0)
        prec = precision_score(y_test, preds, average="weighted", zero_division=0)
        rec  = recall_score(y_test, preds, average="weighted", zero_division=0)

        mlflow.log_metrics({"accuracy": acc, "f1_weighted": f1, "precision_weighted": prec, "recall_weighted": rec})
        _log_classification_report(y_test, preds, le)
        mlflow.sklearn.log_model(rf, "random_forest")

        log.info("RandomForest — acc=%.4f  f1=%.4f", acc, f1)
        return rf, f1


def _train_xgboost(X_train, X_test, y_train, y_test, le) -> tuple:
    log.info("Training XGBoost...")
    with mlflow.start_run(run_name="XGBoost", nested=True):
        params = {
            "n_estimators": 200, "max_depth": 6, "learning_rate": 0.05,
            "subsample": 0.8, "colsample_bytree": 0.8,
            "random_state": 42, "eval_metric": "mlogloss", "n_jobs": -1,
        }
        mlflow.log_params(params)
        mlflow.log_param("model_type", "XGBoost")

        with Timer("XGBoost training", log):
            xgb = XGBClassifier(**params)
            xgb.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

        preds = xgb.predict(X_test)
        acc  = accuracy_score(y_test, preds)
        f1   = f1_score(y_test, preds, average="weighted", zero_division=0)
        prec = precision_score(y_test, preds, average="weighted", zero_division=0)
        rec  = recall_score(y_test, preds, average="weighted", zero_division=0)

        mlflow.log_metrics({"accuracy": acc, "f1_weighted": f1, "precision_weighted": prec, "recall_weighted": rec})
        _log_classification_report(y_test, preds, le)
        mlflow.xgboost.log_model(xgb, "xgboost")

        log.info("XGBoost — acc=%.4f  f1=%.4f", acc, f1)
        return xgb, f1


def _train_isolation_forest(X_train, contamination: float) -> IsolationForest:
    log.info("Training Isolation Forest (contamination=%.2f)...", contamination)
    with mlflow.start_run(run_name="IsolationForest", nested=True):
        params = {"contamination": contamination, "n_estimators": 200, "random_state": 42, "n_jobs": -1}
        mlflow.log_params(params)

        with Timer("IsolationForest training", log):
            iso = IsolationForest(**params)
            iso.fit(X_train)

        mlflow.sklearn.log_model(iso, "isolation_forest")
        log.info("IsolationForest training complete.")
        return iso


def train_models(
    data_path: str | None = None,
    fault_ratio: float | None = None,
    test_size: float = 0.2,
) -> None:
    """
    Full training pipeline: load data → feature engineering → train → evaluate → save.

    Args:
        data_path:   Path to sensor_readings.csv (defaults to data/sensor_readings.csv).
        fault_ratio: Contamination fraction for Isolation Forest.
        test_size:   Train/test split ratio.
    """
    setup_mlflow()

    # ── Load data ──────────────────────────────────────────────────────────────
    csv_path = data_path or Path(__file__).parent.parent / "data" / "sensor_readings.csv"
    if not Path(csv_path).exists():
        log.error("Data not found at %s. Run `python data/generate_data.py` first.", csv_path)
        return

    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    log.info("Loaded %d rows from %s", len(df), csv_path)

    # ── Feature engineering ────────────────────────────────────────────────────
    log.info("Extracting features...")
    with Timer("Feature engineering", log):
        df = extract_features(df)

    X = df[FEATURE_COLS].fillna(0.0)
    y = df["label"]

    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    log.info("Classes: %s", le.classes_)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=test_size, random_state=42, stratify=y_enc
    )
    log.info("Train: %d  Test: %d", len(X_train), len(X_test))

    contamination = fault_ratio or _cfg.fault_ratio

    # ── Training runs under a parent MLflow experiment ─────────────────────────
    with mlflow.start_run(run_name="VigilAI_Training_Session"):
        mlflow.log_param("total_rows", len(df))
        mlflow.log_param("train_rows", len(X_train))
        mlflow.log_param("test_rows", len(X_test))
        mlflow.log_param("features", FEATURE_COLS)
        mlflow.log_param("classes", le.classes_.tolist())
        mlflow.log_param("contamination", contamination)

        rf, rf_f1   = _train_random_forest(X_train, X_test, y_train, y_test, le)
        xgb, xgb_f1 = _train_xgboost(X_train, X_test, y_train, y_test, le)
        iso          = _train_isolation_forest(X_train, contamination)

        # ── Select best model ──────────────────────────────────────────────────
        best_model, best_name, best_f1 = (
            (rf, "RandomForest", rf_f1) if rf_f1 >= xgb_f1 else (xgb, "XGBoost", xgb_f1)
        )
        mlflow.log_param("best_model", best_name)
        mlflow.log_metric("best_f1", best_f1)
        log.info("Best model: %s  (F1=%.4f)", best_name, best_f1)

    # ── Save artefacts ─────────────────────────────────────────────────────────
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    artefacts = {
        "best_model.pkl":       best_model,
        "isolation_forest.pkl": iso,
        "label_encoder.pkl":    le,
    }
    for filename, obj in artefacts.items():
        path = MODELS_DIR / filename
        with open(path, "wb") as fh:
            pickle.dump(obj, fh, protocol=pickle.HIGHEST_PROTOCOL)
        log.info("Saved artefact → %s", path)

    log.info("Training pipeline complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VigilAI ML Training")
    parser.add_argument("--data",         type=str,   default=None,  help="Path to sensor_readings.csv")
    parser.add_argument("--fault-ratio",  type=float, default=None,  help="Isolation Forest contamination")
    parser.add_argument("--test-size",    type=float, default=0.2,   help="Test split fraction")
    args = parser.parse_args()
    train_models(args.data, args.fault_ratio, args.test_size)
