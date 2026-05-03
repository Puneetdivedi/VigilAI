"""
VigilAI — Production demo runner.

Executes the full pipeline end-to-end:
  1. Generate data (if not present)
  2. Train models (if not present)
  3. Build FAISS index (if not present)
  4. Run agent diagnostic pipeline on a synthetic fault reading
  5. Print a formatted diagnostic report
"""
from __future__ import annotations

import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))

from core.config import get_settings
from core.logging import get_logger
from db.models import Base, engine

log = get_logger(__name__)
_cfg = get_settings()

BANNER = """
╔══════════════════════════════════════════════════════════════╗
║         VigilAI — Equipment Intelligence Platform v3         ║
║     Industrial AI: Fault Detection + RAG + LLM Diagnostics   ║
╚══════════════════════════════════════════════════════════════╝"""

FAULT_SCENARIOS = [
    {
        "name": "Bearing Fault",
        "reading": {
            "machine_id": "MACH_001",
            "vibration": 31.5,
            "temperature": 78.2,
            "rpm": 2880.0,
            "pressure": 1.15,
        },
    },
    {
        "name": "Overheating",
        "reading": {
            "machine_id": "MACH_003",
            "vibration": 9.5,
            "temperature": 118.4,
            "rpm": 1490.0,
            "pressure": 1.12,
        },
    },
    {
        "name": "Pressure Anomaly",
        "reading": {
            "machine_id": "MACH_005",
            "vibration": 13.2,
            "temperature": 69.0,
            "rpm": 902.0,
            "pressure": 3.85,
        },
    },
]


def _ensure_prerequisites() -> bool:
    """Returns True if all required artefacts are present."""
    csv = Path("data/sensor_readings.csv")
    best = Path(_cfg.models_dir) / "best_model.pkl"
    iso  = Path(_cfg.models_dir) / "isolation_forest.pkl"
    le   = Path(_cfg.models_dir) / "label_encoder.pkl"
    faiss = Path(_cfg.faiss_index_dir)

    missing = []
    if not csv.exists():       missing.append("data/sensor_readings.csv (run: python data/generate_data.py)")
    if not best.exists():      missing.append("ml/models/best_model.pkl (run: python ml/train.py)")
    if not iso.exists():       missing.append("ml/models/isolation_forest.pkl")
    if not le.exists():        missing.append("ml/models/label_encoder.pkl")
    if not faiss.exists():     missing.append("rag/faiss_index/ (run: python rag/build_index.py)")

    if missing:
        log.error("Missing prerequisites:")
        for m in missing:
            log.error("  • %s", m)
        return False
    return True


def _print_report(scenario_name: str, state: dict, duration: float) -> None:
    report = state.get("report_dict", {})
    fault  = state.get("fault_label", "unknown")
    score  = state.get("anomaly_score", 0.0)
    conf   = state.get("confidence", 0.0)

    sev_icons = {"High": "🔴", "Critical": "🚨", "Medium": "🟠", "Low": "🟢"}
    sev  = report.get("severity", "Unknown")
    icon = sev_icons.get(sev, "⚪")

    print(f"\n{'─'*62}")
    print(f"  SCENARIO : {scenario_name}")
    print(f"  FAULT    : {fault.upper()}")
    print(f"  SCORE    : {score:.4f}  |  CONF: {conf:.2%}  |  TIME: {duration:.2f}s")
    print(f"  SEVERITY : {icon} {sev}")
    print(f"{'─'*62}")
    print(f"  SUMMARY  : {report.get('fault_summary', 'N/A')[:200]}")
    print()
    print("  RECOMMENDED ACTIONS:")
    for i, action in enumerate(report.get("recommended_actions", []), 1):
        print(f"    {i}. {action}")
    print()
    print(f"  DOWNTIME RISK : {report.get('estimated_downtime_risk', 'N/A')}")
    print(f"  LLM PROVIDER  : {report.get('llm_provider', 'fallback')}")
    print(f"{'─'*62}")


def run_demo() -> None:
    print(BANNER)
    print(f"\n  Timestamp : {datetime.now(tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"  Version   : {_cfg.app_version}")
    print(f"  Env       : {_cfg.environment}")

    if not _ensure_prerequisites():
        print("\n  ⚠️  Please fix the above prerequisites and re-run.")
        sys.exit(1)

    # Initialise DB
    Base.metadata.create_all(bind=engine)

    from agents.graph import run_diagnostic_pipeline

    for scenario in FAULT_SCENARIOS:
        name    = scenario["name"]
        reading = scenario["reading"]
        reading["timestamp"] = datetime.now(tz=timezone.utc).isoformat()

        print(f"\n[→] Running scenario: {name}  (machine={reading['machine_id']})")
        print(f"    Vibration={reading['vibration']} Hz  |  Temp={reading['temperature']} °C  |  "
              f"RPM={reading['rpm']}  |  Pressure={reading['pressure']} bar")

        try:
            t0 = time.perf_counter()
            final_state = run_diagnostic_pipeline(reading)
            elapsed = time.perf_counter() - t0
            _print_report(name, final_state, elapsed)
        except Exception as exc:
            log.exception("Demo scenario '%s' failed: %s", name, exc)

    print("\n✅ Demo complete. Reports saved to database.\n")


if __name__ == "__main__":
    run_demo()
