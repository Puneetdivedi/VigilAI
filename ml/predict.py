"""
Industrial-grade ML prediction module for VigilAI.

Thread-safe model loading with a module-level lock.
Exposes `predict()`, `load_models()`, and `models_ready()`.
"""
from __future__ import annotations

import os
import pickle
import threading
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from core.config import get_settings
from core.exceptions import ModelNotFoundError, PredictionError
from core.logging import get_logger

log = get_logger(__name__)
_cfg = get_settings()

# ── Thread safety ──────────────────────────────────────────────────────────────
_lock = threading.Lock()

# ── Module-level model handles ─────────────────────────────────────────────────
_best_model = None
_iso_forest = None
_label_encoder = None

FEATURE_COLS = [
    "vibration", "temperature", "rpm", "pressure",
    "temp_rolling_mean", "pressure_dev",
    "vib_kurtosis", "vib_rms", "vib_fft_peak",
]

_MODELS_DIR = Path(_cfg.models_dir)


# ── Public API ─────────────────────────────────────────────────────────────────

def models_ready() -> bool:
    """Returns True only when all three model artefacts are loaded in memory."""
    return _best_model is not None and _iso_forest is not None and _label_encoder is not None


def load_models(force: bool = False) -> None:
    """
    Loads trained model artefacts from disk into module-level singletons.

    Thread-safe: uses a lock so concurrent startup calls don't double-load.

    Args:
        force: If True, reloads even if models are already in memory.

    Raises:
        ModelNotFoundError: If any artefact file is missing.
    """
    global _best_model, _iso_forest, _label_encoder

    if models_ready() and not force:
        return  # fast path — already loaded

    with _lock:
        if models_ready() and not force:
            return  # double-checked locking

        required = {
            "best_model.pkl": "_best_model",
            "isolation_forest.pkl": "_iso_forest",
            "label_encoder.pkl": "_label_encoder",
        }
        for filename, attr in required.items():
            path = _MODELS_DIR / filename
            if not path.exists():
                raise ModelNotFoundError(
                    f"Model artefact not found: {path}",
                    details={"hint": "Run `python ml/train.py` to train the models.", "path": str(path)},
                )
            with open(path, "rb") as fh:
                obj = pickle.load(fh)  # noqa: S301 — trusted internal artefacts
            if attr == "_best_model":
                _best_model = obj
            elif attr == "_iso_forest":
                _iso_forest = obj
            else:
                _label_encoder = obj
            log.info("Loaded %s from %s", attr, path)


def predict(features_dict: dict) -> dict:
    """
    Runs fault classification and anomaly scoring for a feature vector.

    Args:
        features_dict: Dict containing at minimum all keys in FEATURE_COLS.

    Returns:
        Dict with keys: fault_label, anomaly_score, confidence.

    Raises:
        ModelNotFoundError: If models are not loaded.
        PredictionError: If inference fails at runtime.
    """
    if not models_ready():
        load_models()

    if not models_ready():
        raise ModelNotFoundError("Models failed to load — cannot run prediction.")

    # Build feature DataFrame (enforce column order, fill missing with 0)
    try:
        df = pd.DataFrame(
            [{col: features_dict.get(col, 0.0) for col in FEATURE_COLS}],
            columns=FEATURE_COLS,
        ).fillna(0.0)
    except Exception as exc:
        raise PredictionError(f"Failed to construct feature DataFrame: {exc}") from exc

    # ── Fault classification ───────────────────────────────────────────────────
    try:
        if hasattr(_best_model, "predict_proba"):
            proba: np.ndarray = _best_model.predict_proba(df)[0]
            pred_idx = int(np.argmax(proba))
            confidence = float(proba[pred_idx])
        else:
            raw = _best_model.predict(df)[0]
            pred_idx = int(raw)
            confidence = 1.0

        fault_label: str = _label_encoder.inverse_transform([pred_idx])[0]
    except Exception as exc:
        raise PredictionError(f"Classifier inference error: {exc}") from exc

    # ── Anomaly scoring (Isolation Forest) ────────────────────────────────────
    # decision_function: higher = more normal; we invert and clip to [0, 1]
    try:
        iso_score: float = float(_iso_forest.decision_function(df)[0])
        # Map: score=0.0 → anomaly=0.5 (boundary), very negative → close to 1
        anomaly_score = float(np.clip(0.5 - iso_score, 0.0, 1.0))
    except Exception as exc:
        log.warning("Isolation Forest scoring failed: %s — using 0.0", exc)
        anomaly_score = 0.0

    log.debug(
        "Prediction: label=%s score=%.3f confidence=%.3f",
        fault_label, anomaly_score, confidence,
    )

    return {
        "fault_label": fault_label,
        "anomaly_score": round(anomaly_score, 4),
        "confidence": round(confidence, 4),
    }
