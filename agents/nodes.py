"""
Agent nodes for LangGraph.
"""
import os
import re
from typing import Dict, Any
from langchain_community.llms import Ollama

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ml.predict import predict
from ml.feature_engineering import extract_features_single
from rag.retriever import retrieve_context
from agents.prompts import DIAGNOSIS_PROMPT

def init_llm():
    """Initializes the LLM (Ollama)."""
    ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    try:
        llm = Ollama(model="llama3", base_url=ollama_base_url)
        # Test connection
        llm.invoke("test")
        return llm
    except Exception as e:
        print(f"Warning: LLM unavailable ({e}). Falling back.")
        return None

llm = init_llm()

def anomaly_detector_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Runs ML prediction on sensor data."""
    print("Executing anomaly_detector_node")
    raw_reading = state["raw_reading"]
    recent_history = state.get("recent_history", None)
    
    # We'd typically use recent history to calculate rolling features.
    # For simplicity if history isn't provided, we just pass the raw reading as the feature set 
    # (assuming extract_features_single handles it).
    import pandas as pd
    if recent_history is None:
        recent_history = pd.DataFrame([raw_reading])
        
    features = extract_features_single(raw_reading, recent_history)
    prediction = predict(features)
    
    state["fault_label"] = prediction["fault_label"]
    state["anomaly_score"] = prediction["anomaly_score"]
    return state

def rag_retriever_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Retrieves relevant maintenance context."""
    print("Executing rag_retriever_node")
    fault_label = state.get("fault_label", "normal")
    
    if fault_label == "normal":
        state["context"] = "Machine operating normally. Standard preventive maintenance applies."
    else:
        query = f"{fault_label} diagnosis and troubleshooting"
        state["context"] = retrieve_context(query)
        
    return state

def report_writer_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Generates the diagnostic report using LLM."""
    print("Executing report_writer_node")
    machine_id = state["raw_reading"]["machine_id"]
    fault_label = state["fault_label"]
    anomaly_score = state["anomaly_score"]
    context = state["context"]
    
    if llm is None:
        # Fallback if LLM is down
        severity = "High" if anomaly_score > 0.7 else "Medium" if anomaly_score > 0.3 else "Low"
        if fault_label == "normal": severity = "Low"
        
        state["report_dict"] = {
            "fault_summary": f"LLM unavailable — showing ML-only diagnosis. Detected: {fault_label}.",
            "severity": severity,
            "recommended_actions": ["Inspect machine manually.", "Check sensor calibration.", "Restart diagnostics when LLM is back online."],
            "estimated_downtime_risk": "Unknown due to LLM unavailability."
        }
        return state

    prompt_value = DIAGNOSIS_PROMPT.format(
        machine_id=machine_id,
        fault_label=fault_label,
        anomaly_score=anomaly_score,
        context=context
    )
    
    response = llm.invoke(prompt_value)
    
    # Parse the text to extract fields
    report_dict = {
        "fault_summary": "Issue detected.",
        "severity": "Medium",
        "recommended_actions": ["Inspect machine."],
        "estimated_downtime_risk": "Moderate risk."
    }
    
    # Simple regex parsing
    try:
        summary_match = re.search(r'Fault Summary:?\s*(.*?)(?=Severity:|\n2\.|\Z)', response, re.IGNORECASE | re.DOTALL)
        severity_match = re.search(r'Severity:?\s*(.*?)(?=Recommended Actions:|\n3\.|\Z)', response, re.IGNORECASE | re.DOTALL)
        actions_match = re.search(r'Recommended Actions:?\s*(.*?)(?=Estimated Downtime Risk:|\n4\.|\Z)', response, re.IGNORECASE | re.DOTALL)
        risk_match = re.search(r'Estimated Downtime Risk:?\s*(.*)', response, re.IGNORECASE | re.DOTALL)
        
        if summary_match: report_dict["fault_summary"] = summary_match.group(1).strip()
        if severity_match: 
            sev_text = severity_match.group(1).strip().lower()
            if "high" in sev_text: report_dict["severity"] = "High"
            elif "medium" in sev_text: report_dict["severity"] = "Medium"
            elif "low" in sev_text: report_dict["severity"] = "Low"
            else: report_dict["severity"] = severity_match.group(1).strip()
        
        if actions_match:
            actions_text = actions_match.group(1).strip()
            actions = [a.strip("- *123456789.") for a in actions_text.split('\n') if a.strip()]
            report_dict["recommended_actions"] = actions[:3] if len(actions) >= 3 else actions
            
        if risk_match: report_dict["estimated_downtime_risk"] = risk_match.group(1).strip()
        
    except Exception as e:
        print(f"Error parsing LLM response: {e}")
        report_dict["fault_summary"] = response[:200]
        
    state["report_dict"] = report_dict
    return state
