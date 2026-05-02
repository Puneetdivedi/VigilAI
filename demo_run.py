"""
Demo script to run the full VigilAI pipeline once and display results.
"""
import os
import sys
import json
import time

# Add root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))

from agents.graph import run_diagnostic_pipeline
from db.models import Base, engine

def run_demo():
    print("="*60)
    print(" VIGILAI - EQUIPMENT INTELLIGENCE DEMO ")
    print("="*60)
    
    # Ensure tables exist
    Base.metadata.create_all(bind=engine)
    
    # Sample anomalous reading (High vibration, high temp)
    sample_reading = {
        "machine_id": "MACH_001",
        "vibration": 28.5,
        "temperature": 95.2,
        "rpm": 2950.0,
        "pressure": 1.1,
        "timestamp": "2024-05-01T10:00:00"
    }
    
    print(f"\n[1/3] Processing Sample Reading for {sample_reading['machine_id']}...")
    print(f"      Vibration: {sample_reading['vibration']} Hz, Temp: {sample_reading['temperature']} C")
    
    print("\n[2/3] Running Multi-Agent Diagnostic Pipeline...")
    print("      (Anomaly Detector -> RAG Retriever -> Report Writer)")
    
    try:
        start_time = time.time()
        final_state = run_diagnostic_pipeline(sample_reading)
        duration = time.time() - start_time
        
        print(f"\n[3/3] Pipeline Completed in {duration:.2f} seconds.")
        
        report = final_state.get("report_dict", {})
        
        print("\n" + "-"*40)
        print(" FINAL DIAGNOSTIC REPORT ")
        print("-"*40)
        print(f"FAULT TYPE: {final_state.get('fault_label', 'Unknown')}")
        print(f"ANOMALY SCORE: {final_state.get('anomaly_score', 0.0):.4f}")
        print(f"SEVERITY: {report.get('severity', 'N/A')}")
        print("\nSUMMARY:")
        print(report.get("fault_summary", "No summary generated."))
        
        print("\nRECOMMENDED ACTIONS:")
        actions = report.get("recommended_actions", [])
        if isinstance(actions, list):
            for i, action in enumerate(actions, 1):
                print(f" {i}. {action}")
        else:
            print(f" - {actions}")
            
        print(f"\nDOWNTIME RISK: {report.get('estimated_downtime_risk', 'N/A')}")
        print("-"*40)
        
    except Exception as e:
        print(f"\n[!] Error during demo run: {e}")
        print("    Ensure models are trained and FAISS index is built.")

if __name__ == "__main__":
    run_demo()
