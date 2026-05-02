"""
MLflow configuration setup.
"""
import mlflow
import os

def setup_mlflow():
    """Configures MLflow tracking URI."""
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("VigilAI_Equipment_Monitoring")
