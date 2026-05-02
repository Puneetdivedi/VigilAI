"""
Prediction script for ML models.
"""
import os
import pickle
import pandas as pd
import numpy as np

# Load models at module level to avoid reloading
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')
best_model = None
iso_forest = None
label_encoder = None

def load_models():
    """Loads the trained models from disk."""
    global best_model, iso_forest, label_encoder
    
    try:
        if best_model is None:
            with open(os.path.join(MODELS_DIR, 'best_model.pkl'), 'rb') as f:
                best_model = pickle.load(f)
        if iso_forest is None:
            with open(os.path.join(MODELS_DIR, 'isolation_forest.pkl'), 'rb') as f:
                iso_forest = pickle.load(f)
        if label_encoder is None:
            with open(os.path.join(MODELS_DIR, 'label_encoder.pkl'), 'rb') as f:
                label_encoder = pickle.load(f)
    except Exception as e:
        print(f"Error loading models: {e}")

def predict(features_dict: dict) -> dict:
    """
    Predicts fault label and anomaly score for a given set of features.
    """
    load_models()
    
    if best_model is None or iso_forest is None or label_encoder is None:
        return {"fault_label": "unknown", "anomaly_score": 0.0, "confidence": 0.0}
        
    feature_cols = ['vibration', 'temperature', 'rpm', 'pressure', 
                    'temp_rolling_mean', 'pressure_dev', 'vib_kurtosis', 'vib_rms', 'vib_fft_peak']
    
    # Ensure features are in the right order
    df = pd.DataFrame([features_dict], columns=feature_cols).fillna(0)
    
    # Predict fault type
    try:
        proba = best_model.predict_proba(df)[0]
        pred_idx = np.argmax(proba)
        fault_label = label_encoder.inverse_transform([pred_idx])[0]
        confidence = float(proba[pred_idx])
    except AttributeError:
        # Fallback if model doesn't support predict_proba
        pred_idx = best_model.predict(df)[0]
        fault_label = label_encoder.inverse_transform([pred_idx])[0]
        confidence = 1.0
        
    # Anomaly score from Isolation Forest (lower is more anomalous, standardizing to 0-1 where 1 is highly anomalous)
    iso_score = iso_forest.decision_function(df)[0]
    # Map score roughly to 0-1
    anomaly_score = float(np.clip(0.5 - iso_score, 0, 1))
    
    return {
        "fault_label": fault_label,
        "anomaly_score": anomaly_score,
        "confidence": confidence
    }
