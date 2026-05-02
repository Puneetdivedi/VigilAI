"""
SQLAlchemy ORM models for the VigilAI database.
"""
from sqlalchemy import Column, Integer, String, Float, DateTime, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
import os
from datetime import datetime

# Read from env or use sqlite by default
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./data/vigilai.db")

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

class SensorReading(Base):
    """Stores raw sensor data from machines."""
    __tablename__ = "sensor_readings"

    id = Column(Integer, primary_key=True, index=True)
    machine_id = Column(String, index=True)
    vibration = Column(Float)
    temperature = Column(Float)
    rpm = Column(Float)
    pressure = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow)

class FaultPrediction(Base):
    """Stores ML predictions based on sensor readings."""
    __tablename__ = "fault_predictions"

    id = Column(Integer, primary_key=True, index=True)
    reading_id = Column(Integer)  # References SensorReading.id
    machine_id = Column(String, index=True)
    fault_label = Column(String)
    anomaly_score = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow)

class AgentReport(Base):
    """Stores LLM-generated diagnostic reports."""
    __tablename__ = "agent_reports"

    id = Column(Integer, primary_key=True, index=True)
    machine_id = Column(String, index=True)
    fault_label = Column(String)
    anomaly_score = Column(Float)
    severity = Column(String) # Low, Medium, High
    summary = Column(String)
    recommended_actions = Column(String) # JSON or newline separated
    estimated_downtime_risk = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)

# Create all tables on import
Base.metadata.create_all(bind=engine)
