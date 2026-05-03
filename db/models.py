"""
SQLAlchemy ORM models for VigilAI — industrial-grade schema.
Includes proper indices, FK constraints, audit timestamps, and connection pooling.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    create_engine,
    event,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker

from core.config import get_settings

_cfg = get_settings()

# ── Engine ─────────────────────────────────────────────────────────────────────
_connect_args: dict = {}
if "sqlite" in _cfg.database_url:
    _connect_args = {"check_same_thread": False}

engine = create_engine(
    _cfg.database_url,
    connect_args=_connect_args,
    pool_pre_ping=True,          # validate connections before use
    pool_size=_cfg.db_pool_size if "sqlite" not in _cfg.database_url else 1,
    max_overflow=_cfg.db_max_overflow if "sqlite" not in _cfg.database_url else 0,
    pool_timeout=_cfg.db_pool_timeout,
    echo=_cfg.debug,
)

# Enable WAL mode for SQLite (better concurrent read performance)
if "sqlite" in _cfg.database_url:
    @event.listens_for(engine, "connect")
    def _set_sqlite_pragma(dbapi_conn, _connection_record):  # type: ignore[no-untyped-def]
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


# ── Base ───────────────────────────────────────────────────────────────────────
class Base(DeclarativeBase):
    pass


def _utcnow() -> datetime:
    return datetime.now(tz=timezone.utc)


# ── Models ─────────────────────────────────────────────────────────────────────

class SensorReading(Base):
    """
    Raw telemetry snapshot from an industrial machine.

    Indexed on (machine_id, timestamp DESC) for time-series queries.
    """
    __tablename__ = "sensor_readings"
    __table_args__ = (
        Index("ix_sensor_machine_ts", "machine_id", "timestamp"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    machine_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    vibration: Mapped[float] = mapped_column(Float, nullable=False)
    temperature: Mapped[float] = mapped_column(Float, nullable=False)
    rpm: Mapped[float] = mapped_column(Float, nullable=False)
    pressure: Mapped[float] = mapped_column(Float, nullable=False)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow, index=True
    )

    # Relationships (back-populated for convenience)
    fault_predictions: Mapped[list["FaultPrediction"]] = relationship(
        "FaultPrediction", back_populates="sensor_reading", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<SensorReading id={self.id} machine={self.machine_id} ts={self.timestamp}>"


class FaultPrediction(Base):
    """
    ML fault prediction result linked to a specific sensor reading.
    """
    __tablename__ = "fault_predictions"
    __table_args__ = (
        Index("ix_fault_machine_ts", "machine_id", "timestamp"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    reading_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("sensor_readings.id", ondelete="SET NULL"), nullable=True, index=True
    )
    machine_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    fault_label: Mapped[str] = mapped_column(String(64), nullable=False)
    anomaly_score: Mapped[float] = mapped_column(Float, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    model_version: Mapped[str] = mapped_column(String(32), nullable=False, default="unknown")
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow, index=True
    )

    sensor_reading: Mapped[Optional[SensorReading]] = relationship(
        "SensorReading", back_populates="fault_predictions"
    )

    def __repr__(self) -> str:
        return f"<FaultPrediction id={self.id} label={self.fault_label} score={self.anomaly_score:.3f}>"


class AgentReport(Base):
    """
    LLM-generated structured diagnostic report for a machine fault event.
    Uses Text column for summary/actions to support large outputs.
    """
    __tablename__ = "agent_reports"
    __table_args__ = (
        Index("ix_report_machine_ts", "machine_id", "timestamp"),
        Index("ix_report_severity", "severity"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    machine_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    fault_label: Mapped[str] = mapped_column(String(64), nullable=False)
    anomaly_score: Mapped[float] = mapped_column(Float, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    severity: Mapped[str] = mapped_column(String(16), nullable=False)   # Low | Medium | High | Critical
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    recommended_actions: Mapped[str] = mapped_column(Text, nullable=False)  # JSON array
    estimated_downtime_risk: Mapped[str] = mapped_column(Text, nullable=False)
    llm_provider: Mapped[str] = mapped_column(String(32), nullable=False, default="fallback")
    pipeline_duration_ms: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow, index=True
    )

    def __repr__(self) -> str:
        return f"<AgentReport id={self.id} machine={self.machine_id} severity={self.severity}>"


class MachineMeta(Base):
    """
    Static metadata registry for monitored machines.
    """
    __tablename__ = "machine_meta"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    machine_id: Mapped[str] = mapped_column(String(64), nullable=False, unique=True, index=True)
    machine_type: Mapped[str] = mapped_column(String(64), nullable=False, default="Industrial Motor")
    location: Mapped[str] = mapped_column(String(128), nullable=False, default="Plant Floor A")
    manufacturer: Mapped[str] = mapped_column(String(128), nullable=False, default="Unknown")
    install_date: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    last_maintenance: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_utcnow)

    def __repr__(self) -> str:
        return f"<MachineMeta id={self.id} machine={self.machine_id}>"


# ── Dependency helper (used by FastAPI routers) ───────────────────────────────
def get_db():
    """FastAPI dependency — yields a DB session and guarantees cleanup."""
    db = SessionLocal()
    try:
        yield db
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
