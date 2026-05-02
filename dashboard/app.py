"""
Streamlit Dashboard for VigilAI.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import os
import json

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="VigilAI — Equipment Intelligence", layout="wide", page_icon="🏭")

def fetch_data():
    try:
        sensors = requests.get(f"{API_URL}/sensors/latest").json()
        reports = requests.get(f"{API_URL}/agents/reports").json()
        return sensors, reports
    except Exception as e:
        st.error(f"Failed to connect to API: {e}")
        return [], []

def run_diagnosis(reading):
    try:
        res = requests.post(f"{API_URL}/agents/diagnose", json=reading)
        return res.json()
    except Exception as e:
        return None

def main():
    st.title("🏭 VigilAI — Equipment Intelligence Dashboard")
    
    # Auto-refresh logic (simplistic stream using rerun every 30s)
    # Using experimental rerun has been replaced by st.rerun() in newer versions
    import time
    
    sensors, reports = fetch_data()
    
    if not sensors:
        st.warning("No sensor data available. Run 'make generate-data' and 'make run-api'.")
        return

    df_sensors = pd.DataFrame(sensors)
    df_reports = pd.DataFrame(reports)
    
    # Sidebar
    st.sidebar.header("Filters & Actions")
    machines = df_sensors['machine_id'].unique().tolist()
    selected_machine = st.sidebar.selectbox("Select Machine", ["All"] + machines)
    
    st.sidebar.markdown("---")
    if st.sidebar.button("Run Diagnosis on Latest Reading"):
        if selected_machine != "All":
            latest = df_sensors[df_sensors['machine_id'] == selected_machine].iloc[0]
            reading = {
                "machine_id": latest['machine_id'],
                "vibration": latest['vibration'],
                "temperature": latest['temperature'],
                "rpm": latest['rpm'],
                "pressure": latest['pressure'],
                "timestamp": latest['timestamp']
            }
            with st.spinner("Running Agent Diagnosis..."):
                diag = run_diagnosis(reading)
            if diag:
                st.sidebar.success("Diagnosis Complete!")
                st.sidebar.json(diag)
        else:
            st.sidebar.error("Please select a specific machine to diagnose.")

    # KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    total_readings = len(df_sensors)
    if not df_reports.empty:
        faults_detected = len(df_reports[df_reports['fault_label'] != 'normal'])
        anomaly_rate = (faults_detected / len(df_reports)) * 100 if len(df_reports) > 0 else 0
    else:
        faults_detected = 0
        anomaly_rate = 0.0
    active_machines = len(machines)
    
    col1.metric("Total Recent Readings", total_readings)
    col2.metric("Faults Diagnosed", faults_detected)
    col3.metric("Anomaly Rate", f"{anomaly_rate:.1f}%")
    col4.metric("Active Machines", active_machines)
    
    st.markdown("---")
    
    # Charts
    c1, c2 = st.columns(2)
    
    with c1:
        st.subheader("Sensor Values Over Time")
        plot_df = df_sensors if selected_machine == "All" else df_sensors[df_sensors['machine_id'] == selected_machine]
        if not plot_df.empty:
            fig = px.line(plot_df, x='timestamp', y=['vibration', 'temperature'], color='machine_id',
                          title="Vibration & Temperature Trends")
            st.plotly_chart(fig, use_container_width=True)
            
    with c2:
        st.subheader("Fault Type Distribution")
        if not df_reports.empty:
            fault_counts = df_reports['fault_label'].value_counts().reset_index()
            fault_counts.columns = ['Fault Type', 'Count']
            fig2 = px.bar(fault_counts, x='Fault Type', y='Count', color='Fault Type', title="Fault Types Diagnosed")
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No reports available to display fault distribution.")
            
    st.markdown("---")
    
    # Agent Reports Panel
    st.subheader("Recent Agent Reports")
    if not df_reports.empty:
        for idx, row in df_reports.head(5).iterrows():
            severity_color = "🟢" if row['severity'].lower() == 'low' else "🟠" if row['severity'].lower() == 'medium' else "🔴"
            with st.expander(f"{severity_color} {row['timestamp'][:16]} | {row['machine_id']} | {row['fault_label']}"):
                st.markdown(f"**Summary:** {row['summary']}")
                st.markdown(f"**Anomaly Score:** {row['anomaly_score']:.2f}")
                st.markdown(f"**Downtime Risk:** {row['estimated_downtime_risk']}")
                st.markdown("**Recommended Actions:**")
                try:
                    actions = json.loads(row['recommended_actions'])
                    for act in actions:
                        st.markdown(f"- {act}")
                except:
                    st.markdown(row['recommended_actions'])
    else:
        st.info("No agent reports generated yet.")

    time.sleep(30)
    st.rerun()

if __name__ == "__main__":
    main()
