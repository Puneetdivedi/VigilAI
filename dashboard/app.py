"""
VigilAI v3 Industrial Command Center.

A professional, high-performance monitoring dashboard featuring:
- Glassmorphism UI with HSL-based design system
- Tabbed navigation for multi-dimensional analytics
- Real-time machine health grid
- AI-driven diagnostic report explorer
- Dynamic latency and system metrics
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from plotly.subplots import make_subplots

# ── Configuration ─────────────────────────────────────────────────────────────
API_URL = os.getenv("API_URL", "http://localhost:8000")
REFRESH_SECS = int(os.getenv("DASHBOARD_REFRESH", "30"))

st.set_page_config(
    page_title="VigilAI | Industrial Command Center",
    layout="wide",
    page_icon="🦾",
    initial_sidebar_state="expanded",
)

# ── Premium Design System ───────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --primary: hsl(199, 89%, 48%);
    --primary-glow: hsla(199, 89%, 48%, 0.3);
    --secondary: hsl(230, 58%, 53%);
    --accent: hsl(280, 80%, 65%);
    --bg-dark: hsl(222, 47%, 7%);
    --bg-card: hsla(222, 47%, 11%, 0.85);
    --border: hsla(217, 33%, 17%, 0.8);
    --text-main: hsl(210, 40%, 98%);
    --text-muted: hsl(215, 20%, 65%);
}

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}

.stApp {
    background: radial-gradient(circle at 0% 0%, hsl(222, 47%, 10%) 0%, hsl(222, 47%, 5%) 100%);
    color: var(--text-main);
}

/* Hide Streamlit elements */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 2.5rem; max-width: 1800px; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background-color: hsl(222, 47%, 4%) !important;
    border-right: 1px solid var(--border) !important;
}

/* ── Glass Cards ── */
.glass-card {
    background: var(--bg-card);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.5rem;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

.glass-card:hover {
    border-color: hsla(199, 89%, 48%, 0.4);
    box-shadow: 0 12px 48px 0 hsla(199, 89%, 48%, 0.15);
    transform: translateY(-2px);
}

/* ── Header ── */
.v-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 2rem;
    padding-bottom: 1rem;
    border-bottom: 1px solid var(--border);
}

.v-logo {
    font-size: 1.8rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    background: linear-gradient(135deg, var(--primary) 0%, var(--accent) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.v-status-pill {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.4rem 1rem;
    border-radius: 100px;
    background: hsla(142, 70%, 45%, 0.1);
    border: 1px solid hsla(142, 70%, 45%, 0.2);
    color: hsl(142, 70%, 45%);
    font-size: 0.75rem;
    font-weight: 600;
}

.v-status-pill.offline {
    background: hsla(0, 70%, 45%, 0.1);
    border: 1px solid hsla(0, 70%, 45%, 0.2);
    color: hsl(0, 70%, 45%);
}

/* ── KPIs ── */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
    gap: 1.25rem;
    margin-bottom: 2rem;
}

.kpi-stat {
    display: flex;
    flex-direction: column;
}

.kpi-label {
    font-size: 0.75rem;
    font-weight: 600;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.5rem;
}

.kpi-value {
    font-size: 2.2rem;
    font-weight: 700;
    line-height: 1;
    margin-bottom: 0.25rem;
}

.kpi-trend {
    font-size: 0.8rem;
    font-weight: 500;
}

.trend-up { color: hsl(142, 70%, 45%); }
.trend-down { color: hsl(0, 70%, 45%); }

/* ── Machine Grid ── */
.machine-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
    gap: 1rem;
    margin-top: 1rem;
}

.machine-item {
    background: hsla(222, 47%, 15%, 0.5);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
}

.machine-id {
    font-size: 0.9rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
    display: block;
}

/* ── Report Cards ── */
.report-card {
    margin-bottom: 1rem;
}

.report-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.75rem;
}

.badge {
    padding: 0.25rem 0.75rem;
    border-radius: 6px;
    font-size: 0.65rem;
    font-weight: 700;
    text-transform: uppercase;
}

.badge-critical { background: hsla(0, 84%, 60%, 0.2); color: hsl(0, 84%, 60%); border: 1px solid hsla(0, 84%, 60%, 0.3); }
.badge-high     { background: hsla(20, 90%, 60%, 0.2); color: hsl(20, 90%, 60%); border: 1px solid hsla(20, 90%, 60%, 0.3); }
.badge-medium   { background: hsla(45, 90%, 60%, 0.2); color: hsl(45, 90%, 60%); border: 1px solid hsla(45, 90%, 60%, 0.3); }
.badge-low      { background: hsla(142, 70%, 45%, 0.2); color: hsl(142, 70%, 45%); border: 1px solid hsla(142, 70%, 45%, 0.3); }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 2rem;
    background-color: transparent;
}

.stTabs [data-baseweb="tab"] {
    height: 3rem;
    white-space: pre-wrap;
    background-color: transparent;
    border-radius: 4px 4px 0 0;
    gap: 1rem;
    padding-top: 10px;
    padding-bottom: 10px;
}

.stTabs [aria-selected="true"] {
    background-color: hsla(199, 89%, 48%, 0.05) !important;
    border-bottom: 2px solid var(--primary) !important;
}

/* ── Animations ── */
@keyframes pulse-dot {
    0% { transform: scale(0.9); opacity: 1; }
    50% { transform: scale(1.1); opacity: 0.7; }
    100% { transform: scale(0.9); opacity: 1; }
}

.pulse {
    animation: pulse-dot 2s infinite;
}

</style>
""", unsafe_allow_html=True)

# ── API Helpers ───────────────────────────────────────────────────────────────
def _get(path: str, params: dict | None = None):
    try:
        r = requests.get(f"{API_URL}{path}", params=params, timeout=6)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None

def _post(path: str, payload: dict):
    try:
        r = requests.post(f"{API_URL}{path}", json=payload, timeout=90)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}

@st.cache_data(ttl=5)
def fetch_sensors(machine_id=None, limit=200):
    params = {"limit": limit}
    if machine_id: params["machine_id"] = machine_id
    return _get("/sensors/latest", params) or []

@st.cache_data(ttl=5)
def fetch_reports(machine_id=None, severity=None, limit=50):
    params = {"limit": limit}
    if machine_id: params["machine_id"] = machine_id
    if severity: params["severity"] = severity
    return _get("/agents/reports", params) or []

@st.cache_data(ttl=10)
def fetch_stats():
    return _get("/agents/stats") or {}

@st.cache_data(ttl=30)
def fetch_machines():
    return _get("/sensors/machines") or []

@st.cache_data(ttl=10)
def fetch_health():
    return _get("/health") or {}

# ── Design Components ───────────────────────────────────────────────────────────
def render_kpi(label, value, trend=None, color_class="blue"):
    trend_html = ""
    if trend:
        direction = "up" if trend > 0 else "down"
        trend_html = f'<span class="kpi-trend trend-{direction}">{"↑" if trend > 0 else "↓"} {abs(trend)}%</span>'
    
    st.markdown(f"""
    <div class="glass-card">
        <div class="kpi-stat">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            {trend_html}
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_report_card(report):
    sev = report.get("severity", "Medium")
    badge_cls = f"badge-{sev.lower()}"
    ts = str(report.get("timestamp", ""))[:16].replace("T", " ")
    
    try:
        actions = json.loads(report["recommended_actions"]) if isinstance(report["recommended_actions"], str) else report["recommended_actions"]
    except:
        actions = [str(report.get("recommended_actions", ""))]
    
    actions_html = "".join(f'<div style="font-size:0.8rem; color:var(--text-muted); margin-bottom:4px;">▹ {a}</div>' for a in actions[:3])
    
    st.markdown(f"""
    <div class="glass-card report-card">
        <div class="report-header">
            <span style="font-weight:700; color:var(--primary);">⚙️ {report.get("machine_id")}</span>
            <span class="badge {badge_cls}">{sev}</span>
        </div>
        <div style="font-size:0.75rem; color:var(--text-muted); margin-bottom:0.75rem;">
            {ts} | {report.get("fault_label")} | Score: {report.get("anomaly_score",0):.3f}
        </div>
        <div style="font-size:0.85rem; line-height:1.5; margin-bottom:1rem;">
            {report.get("summary")}
        </div>
        <div style="margin-bottom:0.75rem;">
            {actions_html}
        </div>
        <div style="font-size:0.75rem; border-top:1px solid var(--border); padding-top:0.75rem; color:var(--text-muted);">
            <span style="color:var(--accent);">Risk:</span> {report.get("estimated_downtime_risk")}
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div style="font-size:1.5rem; font-weight:800; color:var(--primary); margin-bottom:1.5rem;">VigilAI</div>', unsafe_allow_html=True)
    
    if health.get("status") == "ok":
        st.markdown('<div class="v-status-pill"><span class="pulse">●</span> API ONLINE</div>', unsafe_allow_html=True)
        # Detailed status
        c_status1, c_status2 = st.columns(2)
        with c_status1:
            st.caption("ML Models")
            st.markdown("✅ Ready" if health.get("models_loaded") else "❌ Missing")
        with c_status2:
            st.caption("RAG Index")
            st.markdown("✅ Ready" if health.get("rag_ok") else "❌ Missing")
    else:
        st.markdown('<div class="v-status-pill offline"><span class="pulse">●</span> API OFFLINE</div>', unsafe_allow_html=True)
    
    st.markdown("### Control Plane")
    refresh = st.toggle("Auto-Refresh", value=True)
    refresh_rate = st.slider("Rate (s)", 5, 60, REFRESH_SECS)
    
    st.divider()
    
    st.markdown("### Quick Diagnostic")
    machines = fetch_machines()
    diag_machine = st.selectbox("Select Target", machines if machines else ["No Data"])
    if st.button("🚀 Trigger AI Analysis", use_container_width=True, type="primary"):
        if diag_machine and diag_machine != "No Data":
            with st.spinner("Executing LangGraph Pipeline..."):
                latest = fetch_sensors(machine_id=diag_machine, limit=1)
                if latest:
                    payload = {k: latest[0][k] for k in ("machine_id", "vibration", "temperature", "rpm", "pressure")}
                    res = _post("/agents/diagnose", payload)
                    if "error" not in res:
                        st.success("Analysis Complete")
                        st.balloons()
                    else:
                        st.error("Pipeline Failure")
                else:
                    st.error("No telemetry for target")
        else:
            st.warning("Select valid machine")

    st.divider()
    st.caption(f"Last updated: {datetime.now().strftime('%H:%M:%S')}")
    st.caption(f"API: {API_URL}")

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="v-header">
    <div class="v-logo">Industrial Command Center</div>
    <div style="font-size:0.8rem; color:var(--text-muted);">Powered by LangGraph & Gemini 1.5</div>
</div>
""", unsafe_allow_html=True)

# ── Main Layout ───────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Executive Summary", 
    "📈 Performance Analytics", 
    "🧠 AI Diagnostics", 
    "🛠 System Registry"
])

# Load Data
stats = fetch_stats()
sensors = fetch_sensors()
reports = fetch_reports()
df_s = pd.DataFrame(sensors)
df_r = pd.DataFrame(reports)

# ── Tab 1: Executive Summary ──────────────────────────────────────────────────
with tab1:
    # KPI Row
    c1, c2, c3, c4 = st.columns(4)
    with c1: render_kpi("Active Telemetry", f"{len(df_s):,}", trend=12)
    with c2: 
        total_reports = stats.get("total_reports", 0)
        render_kpi("Incident Reports", f"{total_reports}", color_class="red")
    with c3:
        avg_score = stats.get("avg_anomaly_score", 0.0)
        render_kpi("Avg Anomaly Index", f"{avg_score:.3f}")
    with c4:
        avg_temp = df_s["temperature"].mean() if not df_s.empty else 0.0
        render_kpi("Thermal Baseline", f"{avg_temp:.1f}°C")

    st.markdown("<br>", unsafe_allow_html=True)
    
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        st.markdown("### Global Sensor Dynamics")
        if not df_s.empty:
            fig = px.line(df_s, x="timestamp", y="vibration", color="machine_id",
                         template="plotly_dark", color_discrete_sequence=px.colors.qualitative.Prism)
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=0, r=0, t=30, b=0),
                height=400,
                xaxis=dict(showgrid=False),
                yaxis=dict(gridcolor="rgba(255,255,255,0.05)")
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Awaiting sensor data...")

    with col_right:
        st.markdown("### Severity Distribution")
        by_sev = stats.get("by_severity", {})
        if by_sev:
            fig = px.pie(
                values=list(by_sev.values()), 
                names=list(by_sev.keys()),
                hole=0.7,
                color_discrete_sequence=["#ef4444", "#f87171", "#fb923c", "#4ade80"]
            )
            fig.update_layout(
                showlegend=False,
                paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=0, r=0, t=0, b=0),
                height=300
            )
            fig.add_annotation(text=f"Total<br>{sum(by_sev.values())}", x=0.5, y=0.5, showarrow=False, font_size=20)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No report metadata")

    st.markdown("### Fleet Overview")
    machine_ids = df_s["machine_id"].unique() if not df_s.empty else []
    cols = st.columns(6)
    for i, mid in enumerate(machine_ids[:12]):
        with cols[i % 6]:
            m_data = df_s[df_s["machine_id"] == mid].iloc[0] if not df_s.empty else {}
            status = "NORMAL" if m_data.get("vibration", 0) < 5 else "WARNING"
            st.markdown(f"""
            <div class="machine-item">
                <span class="machine-id">{mid}</span>
                <div style="font-size:0.7rem; color:{'#4ade80' if status=='NORMAL' else '#f87171'}">{status}</div>
                <div style="font-size:1.1rem; font-weight:600;">{m_data.get('rpm',0):.0f} <span style="font-size:0.6rem; color:var(--text-muted);">RPM</span></div>
            </div>
            """, unsafe_allow_html=True)

# ── Tab 2: Performance Analytics ──────────────────────────────────────────────
with tab2:
    st.markdown("### Deep Signal Analysis")
    if not df_s.empty:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Pressure vs RPM Correlation")
            fig = px.scatter(df_s, x="rpm", y="pressure", color="vibration", 
                           size="temperature", hover_data=["machine_id"])
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=500)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.markdown("#### Thermal Profile Histogram")
            fig = px.histogram(df_s, x="temperature", color="machine_id", marginal="box")
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=500)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Telemetry offline")

# ── Tab 3: AI Diagnostics ─────────────────────────────────────────────────────
with tab3:
    st.markdown("### Intelligent Diagnostic Ledger")
    
    # Filter Row
    fc1, fc2, fc3 = st.columns([2, 2, 1])
    with fc1:
        f_machine = st.selectbox("Filter Machine", ["All"] + machines, key="f_m")
    with fc2:
        f_sev = st.selectbox("Filter Severity", ["All", "Critical", "High", "Medium", "Low"], key="f_s")
    
    filtered_reports = fetch_reports(
        machine_id=None if f_machine == "All" else f_machine,
        severity=None if f_sev == "All" else f_sev
    )
    
    if filtered_reports:
        # Render in 2 columns
        rc1, rc2 = st.columns(2)
        for i, rep in enumerate(filtered_reports):
            with (rc1 if i % 2 == 0 else rc2):
                render_report_card(rep)
    else:
        st.info("No matching diagnostic reports found.")

# ── Tab 4: System Registry ────────────────────────────────────────────────────
with tab4:
    st.markdown("### Equipment Inventory & Network Status")
    
    col_sys1, col_sys2 = st.columns([1, 1])
    
    with col_sys1:
        st.markdown("#### Registered Assets")
        if machines:
            for m in machines:
                with st.expander(f"⚙️ {m}", expanded=False):
                    st.write(f"ID: {m}")
                    st.write("Type: Industrial Actuator")
                    st.write("Location: Sector 7G")
                    st.button(f"Manage {m}", key=f"btn_{m}")
        else:
            st.write("No machines registered in database.")

    with col_sys2:
        st.markdown("#### API & Infrastructure Metrics")
        if health:
            st.json(health)
        else:
            st.error("API unreachable")

# ── Auto Refresh ──────────────────────────────────────────────────────────────
if refresh:
    time.sleep(refresh_rate)
    st.rerun()
