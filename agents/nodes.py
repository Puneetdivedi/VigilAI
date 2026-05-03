"""
Agent nodes for LangGraph.
Uses Google Gemini as the primary LLM with Ollama as fallback.
"""
import os
import re
import json
from typing import Dict, Any

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ml.predict import predict
from ml.feature_engineering import extract_features_single
from rag.retriever import retrieve_context
from agents.prompts import DIAGNOSIS_PROMPT


def _init_gemini():
    """Initialize Google Gemini LLM via langchain-google-genai."""
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        api_key = os.getenv("GOOGLE_API_KEY", "")
        if not api_key:
            return None
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=api_key,
            temperature=0.3,
        )
        return llm
    except Exception as e:
        print(f"[LLM] Gemini init failed: {e}")
        return None


def _init_groq():
    """Initialize Groq LLM as secondary fallback."""
    try:
        from langchain_groq import ChatGroq
        api_key = os.getenv("GROQ_API_KEY", "")
        if not api_key:
            return None
        llm = ChatGroq(model="llama3-8b-8192", groq_api_key=api_key, temperature=0.3)
        return llm
    except Exception as e:
        print(f"[LLM] Groq init failed: {e}")
        return None


def _init_ollama():
    """Initialize Ollama LLM as last-resort fallback."""
    try:
        from langchain_community.llms import Ollama
        ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        llm = Ollama(model="llama3", base_url=ollama_base_url)
        # Quick health check
        llm.invoke("ping")
        return llm
    except Exception as e:
        print(f"[LLM] Ollama init failed: {e}")
        return None


def _load_llm():
    """Load the best available LLM: Gemini → Groq → Ollama → None."""
    llm = _init_gemini()
    if llm:
        print("[LLM] Using Google Gemini")
        return llm
    llm = _init_groq()
    if llm:
        print("[LLM] Using Groq")
        return llm
    llm = _init_ollama()
    if llm:
        print("[LLM] Using Ollama")
        return llm
    print("[LLM] No LLM available — using rule-based fallback.")
    return None


llm = _load_llm()


def _rule_based_report(fault_label: str, anomaly_score: float, context: str) -> Dict[str, Any]:
    """Generates a deterministic report when no LLM is available."""
    severity_map = {"bearing_fault": "High", "overheating": "High", "pressure_anomaly": "Medium", "normal": "Low"}
    severity = severity_map.get(fault_label, "High" if anomaly_score > 0.7 else "Medium" if anomaly_score > 0.3 else "Low")

    action_map = {
        "bearing_fault": [
            "Immediately shut down machine and inspect bearings.",
            "Check lubrication levels and apply correct industrial grease.",
            "Replace bearings if vibration RMS exceeds threshold.",
        ],
        "overheating": [
            "Stop the machine and allow it to cool for at least 2 hours.",
            "Check coolant fluid levels and inspect for leaks.",
            "Clean all cooling fins and verify ambient temperature.",
        ],
        "pressure_anomaly": [
            "Inspect all O-rings and pneumatic seals for cracks.",
            "Manually actuate the pressure relief valve to clear debris.",
            "Recalibrate the pressure transducer and verify readings.",
        ],
        "normal": [
            "Continue standard preventive maintenance schedule.",
            "Log current readings in the maintenance ledger.",
            "Perform next scheduled full inspection on time.",
        ],
    }

    actions = action_map.get(fault_label, ["Manually inspect the machine.", "Review sensor calibration.", "Consult maintenance manual."])
    risk = "High — immediate action required." if severity == "High" else "Moderate — schedule maintenance soon." if severity == "Medium" else "Low — continue monitoring."

    first_context_line = context.split("\n")[0][:150] if context else "No manual context available."
    summary = (
        f"ML model detected a '{fault_label}' fault with anomaly score {anomaly_score:.2f}. "
        f"Maintenance context: {first_context_line}"
    )

    return {
        "fault_summary": summary,
        "severity": severity,
        "recommended_actions": actions,
        "estimated_downtime_risk": risk,
    }


def _parse_llm_response(response_text: str) -> Dict[str, Any]:
    """Parses structured fields from LLM output."""
    report = {
        "fault_summary": "Issue detected.",
        "severity": "Medium",
        "recommended_actions": ["Inspect machine.", "Review sensor data.", "Consult maintenance manual."],
        "estimated_downtime_risk": "Moderate risk.",
    }
    try:
        summary_match = re.search(r"Fault Summary:?\s*(.*?)(?=Severity:|$)", response_text, re.IGNORECASE | re.DOTALL)
        severity_match = re.search(r"Severity:?\s*(.*?)(?=Recommended Actions:|$)", response_text, re.IGNORECASE | re.DOTALL)
        actions_match = re.search(r"Recommended Actions:?\s*(.*?)(?=Estimated Downtime Risk:|$)", response_text, re.IGNORECASE | re.DOTALL)
        risk_match = re.search(r"Estimated Downtime Risk:?\s*(.*)", response_text, re.IGNORECASE | re.DOTALL)

        if summary_match:
            report["fault_summary"] = summary_match.group(1).strip()
        if severity_match:
            sev = severity_match.group(1).strip().lower()
            report["severity"] = "High" if "high" in sev else "Low" if "low" in sev else "Medium"
        if actions_match:
            raw = actions_match.group(1).strip()
            actions = [a.strip("- *•1234567890.\n") for a in raw.split("\n") if a.strip()]
            report["recommended_actions"] = actions[:3] if actions else report["recommended_actions"]
        if risk_match:
            report["estimated_downtime_risk"] = risk_match.group(1).strip()[:200]
    except Exception as e:
        print(f"[LLM] Response parse error: {e}")
    return report


# ─── Node Functions ────────────────────────────────────────────────────────────

def anomaly_detector_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Node 1: Runs ML prediction to detect fault type and anomaly score."""
    print("[Node] anomaly_detector_node")
    raw_reading = state["raw_reading"]

    import pandas as pd
    recent_history = state.get("recent_history") or pd.DataFrame([raw_reading])
    features = extract_features_single(raw_reading, recent_history)
    prediction = predict(features)

    state["fault_label"] = prediction["fault_label"]
    state["anomaly_score"] = prediction["anomaly_score"]
    return state


def rag_retriever_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Node 2: Retrieves relevant maintenance manual context via FAISS RAG."""
    print("[Node] rag_retriever_node")
    fault_label = state.get("fault_label", "normal")

    if fault_label == "normal":
        state["context"] = "Machine operating normally. Standard preventive maintenance applies."
    else:
        query = f"{fault_label} fault diagnosis and repair procedure"
        state["context"] = retrieve_context(query)

    return state


def report_writer_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Node 3: Generates a structured diagnostic report using the LLM."""
    print("[Node] report_writer_node")
    machine_id = state["raw_reading"]["machine_id"]
    fault_label = state.get("fault_label", "unknown")
    anomaly_score = state.get("anomaly_score", 0.0)
    context = state.get("context", "No context retrieved.")

    if llm is None:
        state["report_dict"] = _rule_based_report(fault_label, anomaly_score, context)
        return state

    prompt_str = DIAGNOSIS_PROMPT.format(
        machine_id=machine_id,
        fault_label=fault_label,
        anomaly_score=anomaly_score,
        context=context,
    )

    try:
        response = llm.invoke(prompt_str)
        response_text = response.content if hasattr(response, "content") else str(response)
        state["report_dict"] = _parse_llm_response(response_text)
    except Exception as e:
        print(f"[LLM] Invocation error: {e}")
        state["report_dict"] = _rule_based_report(fault_label, anomaly_score, context)

    return state
