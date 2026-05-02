"""
LangGraph StateGraph for diagnostic pipeline.
"""
from typing import Dict, Any, TypedDict, Optional
import json
from langgraph.graph import StateGraph, END

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from agents.nodes import anomaly_detector_node, rag_retriever_node, report_writer_node
from db.models import SessionLocal, AgentReport

class DiagnosticState(TypedDict):
    raw_reading: Dict[str, Any]
    recent_history: Optional[Any]
    fault_label: Optional[str]
    anomaly_score: Optional[float]
    context: Optional[str]
    report_dict: Optional[Dict[str, Any]]

def build_graph():
    """Builds the LangGraph state graph."""
    builder = StateGraph(DiagnosticState)
    
    builder.add_node("anomaly_detector", anomaly_detector_node)
    builder.add_node("rag_retriever", rag_retriever_node)
    builder.add_node("report_writer", report_writer_node)
    
    builder.set_entry_point("anomaly_detector")
    builder.add_edge("anomaly_detector", "rag_retriever")
    builder.add_edge("rag_retriever", "report_writer")
    builder.add_edge("report_writer", END)
    
    return builder.compile()

def run_diagnostic_pipeline(reading: Dict[str, Any], recent_history=None):
    """Executes the pipeline and saves the report to DB."""
    graph = build_graph()
    
    initial_state = {
        "raw_reading": reading,
        "recent_history": recent_history,
        "fault_label": None,
        "anomaly_score": None,
        "context": None,
        "report_dict": None
    }
    
    final_state = graph.invoke(initial_state)
    report = final_state.get("report_dict", {})
    
    # Save to SQLite
    session = SessionLocal()
    try:
        db_report = AgentReport(
            machine_id=reading["machine_id"],
            fault_label=final_state.get("fault_label", "unknown"),
            anomaly_score=final_state.get("anomaly_score", 0.0),
            severity=report.get("severity", "Unknown"),
            summary=report.get("fault_summary", "No summary"),
            recommended_actions=json.dumps(report.get("recommended_actions", [])),
            estimated_downtime_risk=report.get("estimated_downtime_risk", "Unknown")
        )
        session.add(db_report)
        session.commit()
        session.refresh(db_report)
    except Exception as e:
        print(f"Failed to save report to DB: {e}")
        session.rollback()
    finally:
        session.close()
        
    return final_state
