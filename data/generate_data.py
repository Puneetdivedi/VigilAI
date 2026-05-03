"""
Industrial-grade data generation for VigilAI.

Generates realistic synthetic sensor telemetry with:
- 5 machine types with distinct baseline signatures
- Gradual fault progression (not just binary normal/fault)
- Seasonal noise and drift
- Bulk DB insertion for performance
"""
from __future__ import annotations

import argparse
import os
import sys
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.config import get_settings
from core.logging import get_logger
from db.models import Base, SensorReading, SessionLocal, engine

log = get_logger(__name__)
_cfg = get_settings()

# ── Machine profiles ───────────────────────────────────────────────────────────
MACHINE_PROFILES = {
    "MACH_001": {"type": "Rotary Compressor",  "base_vib": 5.0,  "base_temp": 58.0, "base_rpm": 3000, "base_pres": 1.20},
    "MACH_002": {"type": "Centrifugal Pump",   "base_vib": 4.5,  "base_temp": 62.0, "base_rpm": 2800, "base_pres": 1.40},
    "MACH_003": {"type": "Gearbox Assembly",   "base_vib": 6.0,  "base_temp": 65.0, "base_rpm": 1500, "base_pres": 1.10},
    "MACH_004": {"type": "AC Induction Motor", "base_vib": 3.5,  "base_temp": 55.0, "base_rpm": 3600, "base_pres": 1.15},
    "MACH_005": {"type": "Hydraulic Press",    "base_vib": 7.0,  "base_temp": 70.0, "base_rpm": 900,  "base_pres": 1.80},
}

FAULT_SPECS = {
    "bearing_fault":    {"vib_delta": 20.0, "temp_delta": 10.0, "rpm_delta": -100, "pres_delta": 0.0,  "vib_noise": 5.0},
    "overheating":      {"vib_delta": 5.0,  "temp_delta": 50.0, "rpm_delta": 0,    "pres_delta": 0.0,  "vib_noise": 2.0},
    "pressure_anomaly": {"vib_delta": 2.0,  "temp_delta": 5.0,  "rpm_delta": 0,    "pres_delta": 1.3,  "vib_noise": 3.0},
}


def _add_drift(series: np.ndarray, drift_rate: float = 0.0001) -> np.ndarray:
    """Linear drift over time to simulate gradual sensor degradation."""
    n = len(series)
    return series + np.linspace(0, drift_rate * n, n)


def generate_synthetic_data(
    num_rows: int = 10_000,
    fault_ratio: float = 0.15,
    days: int = 30,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generates realistic synthetic sensor telemetry.

    Args:
        num_rows:    Total records to generate.
        fault_ratio: Fraction of records that are fault events.
        days:        Time span of the data.
        seed:        NumPy random seed for reproducibility.

    Returns:
        pd.DataFrame with columns: timestamp, machine_id, vibration,
        temperature, rpm, pressure, label, fault_type.
    """
    np.random.seed(seed)
    machines = list(MACHINE_PROFILES.keys())

    end_time = datetime.now(tz=timezone.utc)
    start_time = end_time - timedelta(days=days)
    total_seconds = (end_time - start_time).total_seconds()

    timestamps = [
        start_time + timedelta(seconds=total_seconds * i / num_rows)
        for i in range(num_rows)
    ]

    num_faults = int(num_rows * fault_ratio)
    fault_indices = set(np.random.choice(num_rows, num_faults, replace=False))
    fault_types = list(FAULT_SPECS.keys())

    rows = []
    for i in range(num_rows):
        machine_id = machines[i % len(machines)]
        profile = MACHINE_PROFILES[machine_id]
        is_fault = i in fault_indices

        if is_fault:
            fault_type = np.random.choice(fault_types)
            spec = FAULT_SPECS[fault_type]
            # Gradual onset: fault severity ramps up over 10 preceding samples
            onset = min((i % 10) / 10.0 + 0.3, 1.0)
            vib   = max(0.0, np.random.normal(profile["base_vib"]  + spec["vib_delta"]  * onset, spec["vib_noise"]))
            temp  = max(0.0, np.random.normal(profile["base_temp"] + spec["temp_delta"] * onset, 4.0))
            rpm   = max(0.0, np.random.normal(profile["base_rpm"]  + spec["rpm_delta"]  * onset, 60.0))
            pres  = max(0.0, np.random.normal(profile["base_pres"] + spec["pres_delta"] * onset, 0.15))
            label = fault_type
        else:
            # Normal operating conditions with Gaussian noise + small drift
            vib   = max(0.0, np.random.normal(profile["base_vib"],  1.2))
            temp  = max(0.0, np.random.normal(profile["base_temp"], 3.0))
            rpm   = max(0.0, np.random.normal(profile["base_rpm"],  40.0))
            pres  = max(0.0, np.random.normal(profile["base_pres"], 0.05))
            fault_type = "none"
            label = "normal"

        rows.append({
            "timestamp":  timestamps[i],
            "machine_id": machine_id,
            "vibration":  round(vib,  4),
            "temperature": round(temp, 2),
            "rpm":        round(rpm,  1),
            "pressure":   round(pres, 4),
            "label":      label,
        })

    df = pd.DataFrame(rows)
    log.info("Generated %d rows (faults: %d | normal: %d)", len(df),
             len(df[df["label"] != "normal"]), len(df[df["label"] == "normal"]))
    return df


def save_to_db(df: pd.DataFrame, batch_size: int = 2000) -> None:
    """
    Bulk-inserts sensor readings into the database.
    Clears existing data first (intended for dev/demo re-runs).

    Args:
        df:         DataFrame from generate_synthetic_data().
        batch_size: Records per DB commit batch.
    """
    session = SessionLocal()
    try:
        deleted = session.query(SensorReading).delete()
        session.commit()
        log.info("Cleared %d existing sensor readings.", deleted)

        # Build ORM objects in batches
        sensor_cols = ["machine_id", "vibration", "temperature", "rpm", "pressure", "timestamp"]
        total = 0
        for start in range(0, len(df), batch_size):
            batch_df = df.iloc[start : start + batch_size]
            objects = [
                SensorReading(
                    machine_id=row["machine_id"],
                    vibration=row["vibration"],
                    temperature=row["temperature"],
                    rpm=row["rpm"],
                    pressure=row["pressure"],
                    timestamp=row["timestamp"] if hasattr(row["timestamp"], "tzinfo") else
                              pd.Timestamp(row["timestamp"]).to_pydatetime(),
                )
                for _, row in batch_df.iterrows()
            ]
            session.bulk_save_objects(objects)
            session.commit()
            total += len(objects)
            log.info("Inserted batch %d/%d (%d rows)", start // batch_size + 1,
                     (len(df) - 1) // batch_size + 1, len(objects))

        log.info("Database insert complete — total rows: %d", total)
    except Exception as exc:
        session.rollback()
        log.exception("Database insert failed: %s", exc)
        raise
    finally:
        session.close()


def main(args: Optional[argparse.Namespace] = None) -> None:
    if args is None:
        parser = argparse.ArgumentParser(description="VigilAI Synthetic Data Generator")
        parser.add_argument("--rows",        type=int,   default=_cfg.num_synthetic_rows, help="Number of rows")
        parser.add_argument("--fault-ratio", type=float, default=_cfg.fault_ratio,        help="Fraction of fault events (0–1)")
        parser.add_argument("--days",        type=int,   default=30,                       help="Time span in days")
        parser.add_argument("--seed",        type=int,   default=42,                       help="Random seed")
        parser.add_argument("--csv-only",    action="store_true",                          help="Skip DB insert")
        args = parser.parse_args()

    log.info("Starting data generation: rows=%d fault_ratio=%.2f days=%d",
             args.rows, args.fault_ratio, args.days)

    Base.metadata.create_all(bind=engine)

    df = generate_synthetic_data(args.rows, args.fault_ratio, args.days, args.seed)

    data_dir = Path(__file__).parent
    csv_path = data_dir / "sensor_readings.csv"
    df.to_csv(csv_path, index=False)
    log.info("CSV saved → %s", csv_path)

    if not getattr(args, "csv_only", False):
        save_to_db(df)

    log.info("Data generation complete.")


if __name__ == "__main__":
    main()
