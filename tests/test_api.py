"""
Unit tests for API endpoints.
"""
import pytest
from fastapi.testclient import TestClient
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from api.main import app

client = TestClient(app)

def test_health_check():
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_sensors_latest():
    """Test getting latest sensor readings."""
    response = client.get("/sensors/latest")
    assert response.status_code == 200
    assert isinstance(response.json(), list)

def test_ingest_sensors():
    """Test ingesting sensor data."""
    payload = {
        "machine_id": "TEST_MACH",
        "vibration": 10.5,
        "temperature": 75.0,
        "rpm": 3200,
        "pressure": 1.5
    }
    response = client.post("/sensors/ingest", json=payload)
    assert response.status_code == 200
    assert response.json()["machine_id"] == "TEST_MACH"
