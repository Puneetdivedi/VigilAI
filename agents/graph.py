"""
LangGraph multi-agent diagnostic pipeline — industrial grade.

Nodes:
  1. anomaly_detector_node  — ML fault classification + anomaly scoring
  2. rag_retriever_node     — FAISS knowledge-base retrieval
  3. report_writer_node     — LLM-generated structured diagnostic report

LLM priority: Gemini → Groq → Ollama → rule-based fallback
"""
from __future__ import annotations

import json
import os
import re
import time
import threading
from typing import Any, Dict, Optional

import pandas as pd

# Project imports
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.config import get_settings
from core.logging import get_logger, Timer
from core.exceptions import LLMUnavailableError
from ml.predict import predict, load_models
from ml.feature_engineering import extract_features_single
from rag.retriever import retrieve_context
from .prompts import DIAGNOSIS_PROMPT

log = get_logger(__name__)
_cfg = get_settings()

from tenacity import retry, stop_after_attempt, wait_exponential

# ── LLM Factory ────────────────────────────────────────────────────────────────

class LLMFactory:
    """Manages LLM instances with prioritized fallback logic."""
    
    _instance: Optional[Any] = None
    _provider: str = "fallback"
    _lock = threading.Lock()

    @classmethod
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True
    )
    def _create_gemini(cls):
        from langchain_google_genai import ChatGoogleGenerativeAI
        if not _cfg.google_api_key:
            return None
        llm = ChatGoogleGenerativeAI(
            model=_cfg.gemini_model,
            google_api_key=_cfg.google_api_key,
            temperature=_cfg.llm_temperature,
            timeout=_cfg.llm_timeout_seconds,
        )
        llm.invoke("ping")
        return llm

    @classmethod
    def get_llm(cls) -> tuple[Any, str]:
        """Returns (llm_instance, provider_name). Thread-safe singleton."""
        if cls._instance is not None:
            return cls._instance, cls._provider
            
        with cls._lock:
            if cls._instance is not None:
                return cls._instance, cls._provider
            
            # Priority 1: Gemini
            try:
                llm = cls._create_gemini()
                if llm:
                    cls._instance, cls._provider = llm, "gemini"
                    log.info("LLM: Using Google Gemini (%s)", _cfg.gemini_model)
                    return cls._instance, cls._provider
            except Exception as exc:
                log.warning("LLM: Gemini unavailable — %s", exc)

            # Priority 2: Groq
            try:
                from langchain_groq import ChatGroq
                if _cfg.groq_api_key:
                    llm = ChatGroq(model=_cfg.groq_model, groq_api_key=_cfg.groq_api_key)
                    llm.invoke("ping")
                    cls._instance, cls._provider = llm, "groq"
                    log.info("LLM: Using Groq (%s)", _cfg.groq_model)
                    return cls._instance, cls._provider
            except Exception as exc:
                log.warning("LLM: Groq unavailable — %s", exc)

            # Priority 3: Ollama
            try:
                from langchain_community.llms import Ollama
                llm = Ollama(model=_cfg.ollama_model, base_url=_cfg.ollama_base_url)
                llm.invoke("ping")
                cls._instance, cls._provider = llm, "ollama"
                log.info("LLM: Using Ollama (%s)", _cfg.ollama_model)
                return cls._instance, cls._provider
            except Exception as exc:
                log.warning("LLM: Ollama unavailable — %s", exc)

        log.warning("LLM: No provider available — using rule-based fallback.")
        return None, "fallback"

def get_llm() -> tuple[Any, str]:
    return LLMFactory.get_llm()


# ── Rule-based fallback ────────────────────────────────────────────────────────

_SEVERITY_MAP = {
    "bearing_fault": "High",
    "overheating": "High",
    "pressure_anomaly": "Medium",
    "normal": "Low",
}

_ACTION_MAP = {
    "bearing_fault": [
        "Immediately shut down the machine and lock-out/tag-out (LOTO).",
        "Inspect bearings for wear and apply correct industrial grease.",
        "Replace bearings if vibration RMS exceeds 15 Hz sustained.",
    ],
    "overheating": [
        "Stop the machine and allow to cool for ≥2 hours using industrial fans.",
        "Check coolant levels, inspect for leaks, and clean cooling fins.",
        "Verify thermal sensor calibration; check ambient temperature < 40 °C.",
    ],
    "pressure_anomaly": [
        "Isolate the compressor line and check all O-rings for cracks.",
        "Manually actuate the pressure relief valve to clear any debris.",
        "Recalibrate pressure transducer against a certified reference standard.",
    ],
    "normal": [
        "Continue standard preventive maintenance schedule.",
        "Log current readings in the maintenance ledger.",
        "Verify next scheduled inspection date is on calendar.",
    ],
}


def _rule_based_report(fault_label: str, anomaly_score: float, context: str) -> dict:
    severity = _SEVERITY_MAP.get(fault_label, "High" if anomaly_score > 0.7 else "Medium" if anomaly_score > 0.3 else "Low")
    actions = _ACTION_MAP.get(fault_label, ["Manually inspect the machine.", "Review sensor calibration."])
    risk_map = {"High": "High — immediate shutdown recommended.", "Medium": "Moderate — schedule within 48 hours.", "Low": "Low — continue monitoring."}
    first_ctx = (context or "No context.").split("\n")[0][:200]
    return {
        "fault_summary": (
            f"ML classifier detected '{fault_label}' with anomaly score {anomaly_score:.2f}. "
            f"Maintenance context: {first_ctx}"
        ),
        "severity": severity,
        "recommended_actions": actions,
        "estimated_downtime_risk": risk_map.get(severity, "Unknown."),
        "llm_provider": "fallback",
    }


def _parse_llm_response(text: str) -> dict:
    """Parse structured fields out of an LLM response string."""
    defaults = {
        "fault_summary": text[:300].strip(),
        "severity": "Medium",
        "recommended_actions": ["Inspect machine.", "Review sensor data.", "Consult maintenance manual."],
        "estimated_downtime_risk": "Moderate risk.",
    }
    try:
        patterns = {
            "fault_summary": r"Fault Summary:?\s*(.*?)(?=Severity:|$)",
            "severity_raw": r"Severity:?\s*(.*?)(?=Recommended Actions:|$)",
            "actions_raw": r"Recommended Actions:?\s*(.*?)(?=Estimated Downtime Risk:|$)",
            "estimated_downtime_risk": r"Estimated Downtime Risk:?\s*(.*)",
        }
        m = {k: re.search(v, text, re.IGNORECASE | re.DOTALL) for k, v in patterns.items()}

        if m["fault_summary"]:
            defaults["fault_summary"] = m["fault_summary"].group(1).strip()[:500]
        if m["severity_raw"]:
            sev = m["severity_raw"].group(1).strip().lower()
            defaults["severity"] = (
                "Critical" if "critical" in sev else
                "High" if "high" in sev else
                "Low" if "low" in sev else "Medium"
            )
        if m["actions_raw"]:
            raw_actions = m["actions_raw"].group(1).strip()
            actions = [a.strip("- •*1234567890.\n\r") for a in raw_actions.split("\n") if a.strip()]
            if actions:
                defaults["recommended_actions"] = actions[:5]
        if m["estimated_downtime_risk"]:
            defaults["estimated_downtime_risk"] = m["estimated_downtime_risk"].group(1).strip()[:200]
    except Exception as exc:
        log.warning("LLM response parse failed: %s", exc)
    return defaults


# ── Graph State ────────────────────────────────────────────────────────────────

try:
    from typing import TypedDict
except ImportError:
    from typing_extensions import TypedDict


class DiagnosticState(TypedDict, total=False):
    raw_reading: Dict[str, Any]
    recent_history: Optional[Any]
    fault_label: Optional[str]
    anomaly_score: Optional[float]
    confidence: Optional[float]
    context: Optional[str]
    report_dict: Optional[Dict[str, Any]]
    pipeline_start_ms: float


# ── Nodes ──────────────────────────────────────────────────────────────────────

def anomaly_detector_node(state: DiagnosticState) -> DiagnosticState:
    """Node 1: Feature extraction + ML prediction."""
    log.info("[Node] anomaly_detector_node — machine=%s", state["raw_reading"].get("machine_id"))
    raw = state["raw_reading"]
    history = state.get("recent_history") or pd.DataFrame([raw])

    features = extract_features_single(raw, history)
    result = predict(features)

    state["fault_label"] = result["fault_label"]
    state["anomaly_score"] = result["anomaly_score"]
    state["confidence"] = result.get("confidence", 0.0)
    return state


def rag_retriever_node(state: DiagnosticState) -> DiagnosticState:
    """Node 2: FAISS RAG — fetch maintenance manual context."""
    fault_label = state.get("fault_label", "normal")
    log.info("[Node] rag_retriever_node — fault=%s", fault_label)

    if fault_label == "normal":
        state["context"] = "Machine operating normally. Apply standard preventive maintenance schedule."
    else:
        query = f"{fault_label} fault: diagnosis, root cause, and repair procedure"
        context = retrieve_context(query)
        state["context"] = context or "No relevant maintenance context found."
    return state


def report_writer_node(state: DiagnosticState) -> DiagnosticState:
    """Node 3: LLM-powered structured diagnostic report."""
    machine_id = state["raw_reading"].get("machine_id", "UNKNOWN")
    fault_label = state.get("fault_label", "unknown")
    anomaly_score = state.get("anomaly_score", 0.0)
    context = state.get("context", "No context.")
    log.info("[Node] report_writer_node — machine=%s fault=%s score=%.3f",
             machine_id, fault_label, anomaly_score)

    llm, provider = get_llm()

    if llm is None:
        report = _rule_based_report(fault_label, anomaly_score, context)
    else:
        prompt_str = DIAGNOSIS_PROMPT.format(
            machine_id=machine_id,
            fault_label=fault_label,
            anomaly_score=anomaly_score,
            context=context,
        )
        try:
            with Timer("LLM inference", log):
                response = llm.invoke(prompt_str)
            text = response.content if hasattr(response, "content") else str(response)
            report = _parse_llm_response(text)
            report["llm_provider"] = provider
        except Exception as exc:
            log.error("LLM inference failed: %s — falling back to rule-based.", exc)
            report = _rule_based_report(fault_label, anomaly_score, context)

    state["report_dict"] = report
    return state


# ── Graph builder ──────────────────────────────────────────────────────────────

def _build_graph():
    from langgraph.graph import StateGraph, END
    builder = StateGraph(DiagnosticState)
    builder.add_node("anomaly_detector", anomaly_detector_node)
    builder.add_node("rag_retriever", rag_retriever_node)
    builder.add_node("report_writer", report_writer_node)
    builder.set_entry_point("anomaly_detector")
    builder.add_edge("anomaly_detector", "rag_retriever")
    builder.add_edge("rag_retriever", "report_writer")
    builder.add_edge("report_writer", END)
    return builder.compile()


# ── Public pipeline function ───────────────────────────────────────────────────

def run_diagnostic_pipeline(reading: dict, recent_history=None) -> dict:
    """
    Executes the full LangGraph diagnostic pipeline and persists the report.

    Args:
        reading: Raw sensor dict (machine_id, vibration, temperature, rpm, pressure).
        recent_history: Optional pd.DataFrame of recent readings for the same machine.

    Returns:
        Final LangGraph state dict (contains report_dict, fault_label, etc.).
    """
    from db.models import SessionLocal, AgentReport

    t0 = time.perf_counter()
    graph = _build_graph()

    initial_state: DiagnosticState = {
        "raw_reading": reading,
        "recent_history": recent_history,
        "fault_label": None,
        "anomaly_score": None,
        "confidence": None,
        "context": None,
        "report_dict": None,
    }

    final_state = graph.invoke(initial_state)
    duration_ms = (time.perf_counter() - t0) * 1000

    report = final_state.get("report_dict", {})
    if "pipeline_duration_ms" not in report:
        report["pipeline_duration_ms"] = round(duration_ms, 1)
    if report:
        final_state["report_dict"] = report

    log.info(
        "Pipeline complete — machine=%s fault=%s severity=%s duration=%.0fms",
        reading.get("machine_id"), final_state.get("fault_label"),
        report.get("severity"), duration_ms,
    )

    # Persist report to DB
    session = SessionLocal()
    try:
        db_report = AgentReport(
            machine_id=reading.get("machine_id", "UNKNOWN"),
            fault_label=final_state.get("fault_label", "unknown"),
            anomaly_score=final_state.get("anomaly_score", 0.0),
            confidence=final_state.get("confidence", 0.0),
            severity=report.get("severity", "Unknown"),
            summary=report.get("fault_summary", ""),
            recommended_actions=json.dumps(report.get("recommended_actions", [])),
            estimated_downtime_risk=report.get("estimated_downtime_risk", ""),
            llm_provider=report.get("llm_provider", "fallback"),
            pipeline_duration_ms=report.get("pipeline_duration_ms"),
        )
        session.add(db_report)
        session.commit()
        session.refresh(db_report)
        log.info("Report saved — id=%d", db_report.id)
    except Exception as exc:
        log.error("Failed to persist report: %s", exc)
        session.rollback()
    finally:
        session.close()

    return final_state
