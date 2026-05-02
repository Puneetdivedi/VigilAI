"""
Trains Random Forest, XGBoost, and Isolation Forest models.
Logs metrics to MLflow and saves the best model.
"""
import pandas as pd
import numpy as np
import os
import sys
import pickle
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder
import mlflow
import mlflow.sklearn
import mlflow.xgboost

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ml.feature_engineering import extract_features
from mlflow_tracking.setup import setup_mlflow

def train_models():
    """Trains and logs models to MLflow."""
    setup_mlflow()
    
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'sensor_readings.csv')
    if not os.path.exists(data_path):
        print("Data not found. Run generate_data.py first.")
        return
        
    df = pd.read_csv(data_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    print("Extracting features...")
    df = extract_features(df)
    
    feature_cols = ['vibration', 'temperature', 'rpm', 'pressure', 
                    'temp_rolling_mean', 'pressure_dev', 'vib_kurtosis', 'vib_rms', 'vib_fft_peak']
    
    X = df[feature_cols].fillna(0)
    y = df['label']
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    
    best_f1 = 0
    best_model = None
    best_model_name = ""
    
    # 1. Random Forest
    print("Training Random Forest...")
    with mlflow.start_run(run_name="RandomForest"):
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        preds = rf.predict(X_test)
        
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average='weighted')
        prec = precision_score(y_test, preds, average='weighted')
        rec = recall_score(y_test, preds, average='weighted')
        
        mlflow.log_params({"n_estimators": 100})
        mlflow.log_metrics({"accuracy": acc, "f1": f1, "precision": prec, "recall": rec})
        mlflow.sklearn.log_model(rf, "random_forest_model")
        
        if f1 > best_f1:
            best_f1 = f1
            best_model = rf
            best_model_name = "RandomForest"
            
    # 2. XGBoost
    print("Training XGBoost...")
    with mlflow.start_run(run_name="XGBoost"):
        xgb = XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='mlogloss')
        xgb.fit(X_train, y_train)
        preds = xgb.predict(X_test)
        
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average='weighted')
        prec = precision_score(y_test, preds, average='weighted', zero_division=0)
        rec = recall_score(y_test, preds, average='weighted')
        
        mlflow.log_params({"n_estimators": 100})
        mlflow.log_metrics({"accuracy": acc, "f1": f1, "precision": prec, "recall": rec})
        mlflow.xgboost.log_model(xgb, "xgboost_model")
        
        if f1 > best_f1:
            best_f1 = f1
            best_model = xgb
            best_model_name = "XGBoost"
            
    # 3. Isolation Forest
    print("Training Isolation Forest (Anomaly Detection)...")
    with mlflow.start_run(run_name="IsolationForest"):
        iso = IsolationForest(contamination=0.15, random_state=42)
        iso.fit(X_train)
        # -1 is anomaly, 1 is normal
        preds = iso.predict(X_test)
        
        mlflow.log_params({"contamination": 0.15})
        mlflow.sklearn.log_model(iso, "isolation_forest_model")
        
        # Save Isolation Forest separately
        models_dir = os.path.join(os.path.dirname(__file__), 'models')
        os.makedirs(models_dir, exist_ok=True)
        with open(os.path.join(models_dir, 'isolation_forest.pkl'), 'wb') as f:
            pickle.dump(iso, f)

    # Save Best Model and Label Encoder
    print(f"Saving best model ({best_model_name}) with F1: {best_f1:.4f}")
    with open(os.path.join(models_dir, 'best_model.pkl'), 'wb') as f:
        pickle.dump(best_model, f)
        
    with open(os.path.join(models_dir, 'label_encoder.pkl'), 'wb') as f:
        pickle.dump(le, f)

if __name__ == "__main__":
    train_models()
