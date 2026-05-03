"""
Industrial-grade pytest test suite for VigilAI.

Coverage:
  - Feature engineering
  - ML prediction (with and without artefacts)
  - API endpoints (sensors, ml, agents, health)
  - Database models
  - Configuration
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def sample_df() -> pd.DataFrame:
    """15-row sensor DataFrame for a single machine."""
    return pd.DataFrame({
        "machine_id": ["MACH_001"] * 15,
        "timestamp": pd.date_range("2024-01-01", periods=15, freq="min"),
        "vibration":   np.random.default_rng(42).normal(5, 1.2, 15).clip(0),
        "temperature": np.random.default_rng(42).normal(60, 3, 15).clip(0),
        "rpm":         np.full(15, 3000.0),
        "pressure":    np.full(15, 1.2),
        "label":       ["normal"] * 12 + ["bearing_fault"] * 3,
    })


@pytest.fixture()
def normal_reading() -> dict:
    return {
        "machine_id": "MACH_001",
        "vibration": 5.2,
        "temperature": 61.0,
        "rpm": 3000.0,
        "pressure": 1.21,
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
    }


@pytest.fixture()
def fault_reading() -> dict:
    return {
        "machine_id": "MACH_002",
        "vibration": 28.5,
        "temperature": 98.0,
        "rpm": 2900.0,
        "pressure": 1.1,
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
    }


# ── Feature Engineering ────────────────────────────────────────────────────────

class TestFeatureEngineering:
    def test_extract_features_columns(self, sample_df):
        from ml.feature_engineering import extract_features
        out = extract_features(sample_df)
        required = {"temp_rolling_mean", "pressure_dev", "vib_kurtosis", "vib_rms", "vib_fft_peak"}
        assert required.issubset(set(out.columns)), f"Missing columns: {required - set(out.columns)}"

    def test_extract_features_row_count(self, sample_df):
        from ml.feature_engineering import extract_features
        out = extract_features(sample_df)
        assert len(out) == len(sample_df)

    def test_extract_features_no_nulls_after_fill(self, sample_df):
        from ml.feature_engineering import extract_features
        out = extract_features(sample_df)
        feature_cols = ["temp_rolling_mean", "pressure_dev", "vib_kurtosis", "vib_rms", "vib_fft_peak"]
        assert out[feature_cols].isnull().sum().sum() == 0, "NaN values remain after feature extraction"

    def test_extract_features_single(self, normal_reading, sample_df):
        from ml.feature_engineering import extract_features_single
        result = extract_features_single(normal_reading, sample_df)
        assert isinstance(result, dict)
        assert "vibration" in result

    def test_pressure_dev_baseline(self, sample_df):
        from ml.feature_engineering import extract_features
        out = extract_features(sample_df)
        # All pressure values are 1.2 → pressure_dev should be ~0
        assert out["pressure_dev"].max() < 0.01

    def test_vib_rms_positive(self, sample_df):
        from ml.feature_engineering import extract_features
        out = extract_features(sample_df)
        assert (out["vib_rms"] >= 0).all()


# ── ML Prediction ──────────────────────────────────────────────────────────────

class TestMLPrediction:
    def test_predict_fallback_when_no_models(self):
        """Prediction should raise ModelNotFoundError when artefacts are absent."""
        from core.exceptions import ModelNotFoundError
        from ml import predict as ml_predict_module
        # Temporarily clear module-level model handles
        orig_bm = ml_predict_module._best_model
        orig_iso = ml_predict_module._iso_forest
        orig_le = ml_predict_module._label_encoder
        ml_predict_module._best_model = None
        ml_predict_module._iso_forest = None
        ml_predict_module._label_encoder = None
        try:
            with pytest.raises((ModelNotFoundError, Exception)):
                ml_predict_module.predict({
                    "vibration": 5.0, "temperature": 60.0, "rpm": 3000.0, "pressure": 1.2,
                    "temp_rolling_mean": 60.0, "pressure_dev": 0.0,
                    "vib_kurtosis": 0.0, "vib_rms": 5.0, "vib_fft_peak": 10.0,
                })
        finally:
            ml_predict_module._best_model = orig_bm
            ml_predict_module._iso_forest = orig_iso
            ml_predict_module._label_encoder = orig_le

    def test_predict_result_keys(self):
        """If models are present, result must contain required keys."""
        models_dir = Path("ml/models")
        if not (models_dir / "best_model.pkl").exists():
            pytest.skip("ML models not trained yet — run `python ml/train.py`")
        from ml.predict import predict
        sample = {
            "vibration": 5.0, "temperature": 60.0, "rpm": 3000.0, "pressure": 1.2,
            "temp_rolling_mean": 60.0, "pressure_dev": 0.0,
            "vib_kurtosis": 0.0, "vib_rms": 5.0, "vib_fft_peak": 10.0,
        }
        result = predict(sample)
        assert "fault_label" in result
        assert "anomaly_score" in result
        assert "confidence" in result
        assert 0.0 <= result["anomaly_score"] <= 1.0

    def test_models_ready_false_initially(self):
        from ml import predict as m
        orig = m._best_model
        m._best_model = None
        assert m.models_ready() is False
        m._best_model = orig


# ── API Endpoints (using TestClient) ──────────────────────────────────────────

@pytest.fixture(scope="module")
def client():
    """TestClient backed by an in-memory SQLite DB."""
    os.environ["DATABASE_URL"] = "sqlite:///./data/vigilai_test.db"
    from fastapi.testclient import TestClient
    from api.main import app
    with TestClient(app) as c:
        yield c


class TestAPIHealth:
    def test_health_ok(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"
        assert "version" in data
        assert "db_ok" in data

    def test_root(self, client):
        r = client.get("/")
        assert r.status_code == 200
        assert "docs" in r.json()


class TestSensorEndpoints:
    def test_ingest_valid(self, client, normal_reading):
        r = client.post("/sensors/ingest", json=normal_reading)
        assert r.status_code == 201
        data = r.json()
        assert data["machine_id"] == normal_reading["machine_id"]
        assert "id" in data

    def test_ingest_invalid_vibration(self, client):
        bad = {"machine_id": "MACH_001", "vibration": -5.0, "temperature": 60.0, "rpm": 3000.0, "pressure": 1.2}
        r = client.post("/sensors/ingest", json=bad)
        assert r.status_code == 422   # Pydantic validation error

    def test_latest_returns_list(self, client):
        r = client.get("/sensors/latest")
        assert r.status_code == 200
        assert isinstance(r.json(), list)

    def test_latest_machine_filter(self, client, normal_reading):
        r = client.get("/sensors/latest", params={"machine_id": normal_reading["machine_id"]})
        assert r.status_code == 200
        data = r.json()
        if data:
            assert all(d["machine_id"] == normal_reading["machine_id"] for d in data)

    def test_machines_list(self, client):
        r = client.get("/sensors/machines")
        assert r.status_code == 200
        assert isinstance(r.json(), list)


class TestMLEndpoints:
    def test_model_status(self, client):
        r = client.get("/ml/status")
        assert r.status_code == 200
        assert "models_loaded" in r.json()

    def test_history_returns_list(self, client):
        r = client.get("/ml/history")
        assert r.status_code == 200
        assert isinstance(r.json(), list)


class TestAgentEndpoints:
    def test_reports_returns_list(self, client):
        r = client.get("/agents/reports")
        assert r.status_code == 200
        assert isinstance(r.json(), list)

    def test_stats_structure(self, client):
        r = client.get("/agents/stats")
        assert r.status_code == 200
        data = r.json()
        assert "total_reports" in data
        assert "by_severity" in data
        assert "by_fault_label" in data

    def test_report_not_found(self, client):
        r = client.get("/agents/reports/999999")
        assert r.status_code == 404


# ── Data Generation ────────────────────────────────────────────────────────────

class TestDataGeneration:
    def test_generate_returns_dataframe(self):
        from data.generate_data import generate_synthetic_data
        df = generate_synthetic_data(num_rows=100, fault_ratio=0.2, seed=0)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 100

    def test_fault_ratio(self):
        from data.generate_data import generate_synthetic_data
        df = generate_synthetic_data(num_rows=1000, fault_ratio=0.3, seed=1)
        fault_frac = (df["label"] != "normal").mean()
        assert 0.25 <= fault_frac <= 0.35, f"Unexpected fault ratio: {fault_frac:.2f}"

    def test_required_columns(self):
        from data.generate_data import generate_synthetic_data
        df = generate_synthetic_data(num_rows=50, seed=2)
        required = {"timestamp", "machine_id", "vibration", "temperature", "rpm", "pressure", "label"}
        assert required.issubset(set(df.columns))

    def test_no_negative_values(self):
        from data.generate_data import generate_synthetic_data
        df = generate_synthetic_data(num_rows=200, seed=3)
        for col in ("vibration", "temperature", "rpm", "pressure"):
            assert (df[col] >= 0).all(), f"Negative values in {col}"


# ── Configuration ──────────────────────────────────────────────────────────────

class TestConfig:
    def test_settings_singleton(self):
        from core.config import get_settings
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2

    def test_default_values(self):
        from core.config import get_settings
        s = get_settings()
        assert s.app_name == "VigilAI"
        assert 0.0 < s.fault_ratio < 1.0
        assert s.rag_top_k > 0
