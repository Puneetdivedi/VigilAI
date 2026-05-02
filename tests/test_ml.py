"""
Unit tests for ML models.
"""
import pytest
import pandas as pd
import numpy as np
import os
import sys
import pickle

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ml.feature_engineering import extract_features
from ml.predict import predict

def test_feature_engineering():
    """Test that features are extracted correctly."""
    data = {
        'machine_id': ['MACH_001'] * 15,
        'timestamp': pd.date_range(start='2024-01-01', periods=15, freq='min'),
        'vibration': np.random.normal(5, 1, 15),
        'temperature': np.random.normal(60, 5, 15),
        'rpm': [3000] * 15,
        'pressure': [1.2] * 15
    }
    df = pd.DataFrame(data)
    features_df = extract_features(df)
    
    assert 'temp_rolling_mean' in features_df.columns
    assert 'vib_rms' in features_df.columns
    assert len(features_df) == 15

def test_predict_fallback():
    """Test prediction fallback when models are not loaded."""
    # This might fail if models actually exist, but we can mock or check keys
    sample_features = {
        'vibration': 5.0, 'temperature': 60.0, 'rpm': 3000.0, 'pressure': 1.2,
        'temp_rolling_mean': 60.0, 'pressure_dev': 0.0, 'vib_kurtosis': 0.0,
        'vib_rms': 5.0, 'vib_fft_peak': 10.0
    }
    res = predict(sample_features)
    assert "fault_label" in res
    assert "anomaly_score" in res
