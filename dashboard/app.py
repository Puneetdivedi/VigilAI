"""
VigilAI — Premium Streamlit Dashboard
Real-time industrial equipment monitoring with AI-powered diagnostics.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import os
import json
import time
from datetime import datetime

# ─── Config ───────────────────────────────────────────────────────────────────
API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="VigilAI — Equipment Intelligence",
    layout="wide",
    page_icon="🏭",
    initial_sidebar_state="expanded",
)

# ─── Premium CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}
.stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #0d1526 50%, #0a1020 100%);
    color: #e2e8f0;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 2rem 2rem 2rem; max-width: 1600px; }

/* ── Custom header ── */
.vigil-header {
    background: linear-gradient(90deg, rgba(56,189,248,0.12) 0%, rgba(99,102,241,0.12) 100%);
    border: 1px solid rgba(56,189,248,0.25);
    border-radius: 16px;
    padding: 1.5rem 2rem;
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    gap: 1rem;
    backdrop-filter: blur(10px);
}
.vigil-header h1 {
    margin: 0;
    font-size: 2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #38bdf8, #818cf8, #c084fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.vigil-header p {
    margin: 0.2rem 0 0 0;
    color: #94a3b8;
    font-size: 0.9rem;
}
.status-dot {
    width: 10px; height: 10px;
    border-radius: 50%;
    background: #22c55e;
    box-shadow: 0 0 8px #22c55e;
    animation: pulse 2s infinite;
    display: inline-block;
    margin-right: 6px;
}
@keyframes pulse { 0%,100%{opacity:1;} 50%{opacity:0.4;} }

/* ── KPI Cards ── */
.kpi-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin-bottom: 1.5rem; }
.kpi-card {
    background: rgba(15, 23, 42, 0.8);
    border: 1px solid rgba(56,189,248,0.15);
    border-radius: 14px;
    padding: 1.25rem 1.5rem;
    position: relative;
    overflow: hidden;
    transition: transform 0.2s, border-color 0.2s;
}
.kpi-card:hover { transform: translateY(-3px); border-color: rgba(56,189,248,0.4); }
.kpi-card::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, #38bdf8, #818cf8);
}
.kpi-label { font-size: 0.75rem; font-weight: 500; color: #64748b; text-transform: uppercase; letter-spacing: 0.08em; }
.kpi-value { font-size: 2.2rem; font-weight: 800; color: #f0f9ff; margin: 0.25rem 0; line-height: 1; }
.kpi-sub { font-size: 0.78rem; color: #64748b; }
.kpi-icon { position: absolute; top: 1rem; right: 1.25rem; font-size: 1.8rem; opacity: 0.25; }

/* ── Section titles ── */
.section-title {
    font-size: 1rem; font-weight: 600; color: #94a3b8;
    text-transform: uppercase; letter-spacing: 0.06em;
    margin: 0 0 1rem 0; display: flex; align-items: center; gap: 0.5rem;
}
.section-title::before { content: ''; width: 3px; height: 1em; background: #38bdf8; border-radius: 2px; display: inline-block; }

/* ── Report cards ── */
.report-card {
    background: rgba(15, 23, 42, 0.85);
    border: 1px solid rgba(99,102,241,0.2);
    border-radius: 12px;
    padding: 1rem 1.25rem;
    margin-bottom: 0.75rem;
    transition: border-color 0.2s;
}
.report-card:hover { border-color: rgba(99,102,241,0.5); }
.report-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem; }
.severity-badge {
    font-size: 0.7rem; font-weight: 700; padding: 0.2em 0.8em;
    border-radius: 999px; text-transform: uppercase; letter-spacing: 0.08em;
}
.badge-high { background: rgba(239,68,68,0.2); color: #f87171; border: 1px solid rgba(239,68,68,0.4); }
.badge-medium { background: rgba(251,146,60,0.2); color: #fb923c; border: 1px solid rgba(251,146,60,0.4); }
.badge-low { background: rgba(34,197,94,0.2); color: #4ade80; border: 1px solid rgba(34,197,94,0.4); }
.report-machine { font-size: 0.85rem; font-weight: 600; color: #38bdf8; }
.report-summary { font-size: 0.82rem; color: #94a3b8; line-height: 1.5; }
.report-actions { list-style: none; padding: 0; margin: 0.5rem 0 0 0; }
.report-actions li { font-size: 0.78rem; color: #cbd5e1; padding: 0.15rem 0; padding-left: 1rem; position: relative; }
.report-actions li::before { content: '→'; position: absolute; left: 0; color: #38bdf8; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a0e1a 0%, #0d1526 100%) !important;
    border-right: 1px solid rgba(56,189,248,0.12) !important;
}
section[data-testid="stSidebar"] .block-container { padding: 1.5rem 1rem; }
.sidebar-logo { font-size: 1.5rem; font-weight: 800; text-align: center; padding: 0.5rem 0 1.5rem 0;
    background: linear-gradient(135deg, #38bdf8, #818cf8); -webkit-background-clip: text;
    -webkit-text-fill-color: transparent; background-clip: text; }

/* ── Plotly chart container ── */
.js-plotly-plot { border-radius: 12px; overflow: hidden; }

/* ── Alerts ── */
.alert { padding: 0.75rem 1rem; border-radius: 10px; margin-bottom: 1rem; font-size: 0.85rem; }
.alert-warn { background: rgba(251,146,60,0.1); border: 1px solid rgba(251,146,60,0.3); color: #fed7aa; }
.alert-info { background: rgba(56,189,248,0.1); border: 1px solid rgba(56,189,248,0.3); color: #bae6fd; }
</style>
""", unsafe_allow_html=True)

# ─── Plotly theme ─────────────────────────────────────────────────────────────
CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(15,23,42,0.6)",
    font=dict(family="Inter", color="#94a3b8", size=12),
    margin=dict(l=10, r=10, t=40, b=10),
    xaxis=dict(gridcolor="rgba(148,163,184,0.08)", linecolor="rgba(148,163,184,0.1)"),
    yaxis=dict(gridcolor="rgba(148,163,184,0.08)", linecolor="rgba(148,163,184,0.1)"),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#94a3b8")),
    title_font=dict(color="#e2e8f0", size=14, family="Inter"),
)

COLOR_SEQ = ["#38bdf8", "#818cf8", "#c084fc", "#f472b6", "#34d399", "#fb923c"]

# ─── Data Layer ───────────────────────────────────────────────────────────────
@st.cache_data(ttl=15)
def fetch_sensors():
    try:
        r = requests.get(f"{API_URL}/sensors/latest", timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception:
        return []


@st.cache_data(ttl=15)
def fetch_reports():
    try:
        r = requests.get(f"{API_URL}/agents/reports", timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception:
        return []


def run_diagnosis(reading: dict):
    try:
        r = requests.post(f"{API_URL}/agents/diagnose", json=reading, timeout=60)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-logo">⚡ VigilAI</div>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("**Machine Filter**")
    sensors_raw = fetch_sensors()
    df_s = pd.DataFrame(sensors_raw) if sensors_raw else pd.DataFrame()
    machines = df_s["machine_id"].unique().tolist() if not df_s.empty else []
    selected_machine = st.selectbox("Select Machine", ["All"] + machines, label_visibility="collapsed")

    st.markdown("---")
    st.markdown("**Run Agent Diagnosis**")
    diagnose_btn = st.button("🤖 Diagnose Latest Reading", use_container_width=True)
    diag_result = None
    if diagnose_btn:
        if selected_machine != "All" and not df_s.empty:
            latest = df_s[df_s["machine_id"] == selected_machine].iloc[0]
            reading = {
                "machine_id": latest["machine_id"],
                "vibration": float(latest["vibration"]),
                "temperature": float(latest["temperature"]),
                "rpm": float(latest["rpm"]),
                "pressure": float(latest["pressure"]),
            }
            with st.spinner("Running AI diagnostic pipeline..."):
                diag_result = run_diagnosis(reading)
            if diag_result and "error" not in diag_result:
                st.success("✅ Diagnosis complete!")
            else:
                st.error("❌ Diagnosis failed — check API logs.")
        else:
            st.warning("Select a specific machine first.")

    if diag_result and "error" not in diag_result:
        st.markdown("---")
        st.markdown("**Latest Diagnosis**")
        sev = diag_result.get("severity", "Unknown")
        badge_cls = "badge-high" if sev == "High" else "badge-medium" if sev == "Medium" else "badge-low"
        st.markdown(f'<span class="severity-badge {badge_cls}">{sev}</span>', unsafe_allow_html=True)
        st.caption(diag_result.get("fault_summary", "")[:200])

    st.markdown("---")
    auto_refresh = st.toggle("Auto-Refresh (30s)", value=False)
    st.caption(f"Last updated: {datetime.now().strftime('%H:%M:%S')}")


# ─── Main Content ─────────────────────────────────────────────────────────────

# Header
st.markdown("""
<div class="vigil-header">
    <div>
        <h1>🏭 VigilAI</h1>
        <p><span class="status-dot"></span>Equipment Intelligence Platform — Real-time Monitoring & AI Diagnostics</p>
    </div>
</div>
""", unsafe_allow_html=True)

# Check API connectivity
if not sensors_raw:
    st.markdown("""
    <div class="alert alert-warn">
        ⚠️ <strong>API not reachable.</strong> Run <code>make run-api</code> and <code>make generate-data</code> first.
    </div>
    """, unsafe_allow_html=True)
    st.stop()

reports_raw = fetch_reports()
df_r = pd.DataFrame(reports_raw) if reports_raw else pd.DataFrame()

# Filter by machine
df_filtered = df_s if selected_machine == "All" else df_s[df_s["machine_id"] == selected_machine]

# ─── KPI Section ─────────────────────────────────────────────────────────────
total_readings = len(df_filtered)
active_machines = len(machines)
faults_detected = len(df_r[df_r["fault_label"] != "normal"]) if not df_r.empty else 0
anomaly_rate = (faults_detected / len(df_r) * 100) if not df_r.empty and len(df_r) > 0 else 0.0
avg_temp = df_filtered["temperature"].mean() if not df_filtered.empty else 0.0
avg_vib = df_filtered["vibration"].mean() if not df_filtered.empty else 0.0

kpi_html = f"""
<div class="kpi-grid">
    <div class="kpi-card">
        <div class="kpi-icon">📡</div>
        <div class="kpi-label">Sensor Readings</div>
        <div class="kpi-value">{total_readings:,}</div>
        <div class="kpi-sub">filtered view</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-icon">🔴</div>
        <div class="kpi-label">Faults Detected</div>
        <div class="kpi-value">{faults_detected}</div>
        <div class="kpi-sub">{anomaly_rate:.1f}% anomaly rate</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-icon">🌡️</div>
        <div class="kpi-label">Avg Temperature</div>
        <div class="kpi-value">{avg_temp:.1f}°C</div>
        <div class="kpi-sub">across selected machines</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-icon">⚙️</div>
        <div class="kpi-label">Active Machines</div>
        <div class="kpi-value">{active_machines}</div>
        <div class="kpi-sub">online & reporting</div>
    </div>
</div>
"""
st.markdown(kpi_html, unsafe_allow_html=True)

# ─── Charts Row 1 ─────────────────────────────────────────────────────────────
c1, c2 = st.columns([3, 2], gap="medium")

with c1:
    st.markdown('<p class="section-title">Sensor Trends</p>', unsafe_allow_html=True)
    if not df_filtered.empty and "timestamp" in df_filtered.columns:
        df_plot = df_filtered.copy()
        df_plot["timestamp"] = pd.to_datetime(df_plot["timestamp"])
        df_plot = df_plot.sort_values("timestamp")

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                            subplot_titles=("Vibration (Hz)", "Temperature (°C)"))
        for i, mid in enumerate(df_plot["machine_id"].unique()):
            sub = df_plot[df_plot["machine_id"] == mid]
            color = COLOR_SEQ[i % len(COLOR_SEQ)]
            fig.add_trace(go.Scatter(x=sub["timestamp"], y=sub["vibration"], name=mid,
                                     line=dict(color=color, width=1.5), mode="lines"), row=1, col=1)
            fig.add_trace(go.Scatter(x=sub["timestamp"], y=sub["temperature"], name=mid,
                                     line=dict(color=color, width=1.5, dash="dot"), mode="lines",
                                     showlegend=False), row=2, col=1)
        fig.update_layout(**CHART_LAYOUT, height=360)
        fig.update_layout(legend=dict(orientation="h", y=1.12))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No time-series data available for this selection.")

with c2:
    st.markdown('<p class="section-title">Fault Distribution</p>', unsafe_allow_html=True)
    if not df_r.empty:
        fault_counts = df_r["fault_label"].value_counts().reset_index()
        fault_counts.columns = ["Fault", "Count"]
        fig2 = px.pie(fault_counts, names="Fault", values="Count",
                      color_discrete_sequence=COLOR_SEQ, hole=0.55)
        fig2.update_traces(textinfo="percent+label", textfont_size=11)
        fig2.update_layout(**CHART_LAYOUT, height=360,
                           annotations=[dict(text=f"<b>{len(df_r)}</b><br>reports", x=0.5, y=0.5,
                                             font_size=14, showarrow=False, font_color="#e2e8f0")])
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No agent reports yet.")

# ─── Charts Row 2 ─────────────────────────────────────────────────────────────
c3, c4 = st.columns(2, gap="medium")

with c3:
    st.markdown('<p class="section-title">RPM vs Pressure Scatter</p>', unsafe_allow_html=True)
    if not df_filtered.empty:
        color_col = "machine_id" if selected_machine == "All" else None
        fig3 = px.scatter(df_filtered, x="rpm", y="pressure",
                          color="machine_id" if color_col else None,
                          color_discrete_sequence=COLOR_SEQ,
                          opacity=0.7, size_max=6,
                          labels={"rpm": "RPM", "pressure": "Pressure (bar)"})
        fig3.update_traces(marker=dict(size=5))
        fig3.update_layout(**CHART_LAYOUT, height=280)
        st.plotly_chart(fig3, use_container_width=True)

with c4:
    st.markdown('<p class="section-title">Anomaly Score Distribution</p>', unsafe_allow_html=True)
    if not df_r.empty and "anomaly_score" in df_r.columns:
        fig4 = px.histogram(df_r, x="anomaly_score", nbins=20,
                            color_discrete_sequence=["#818cf8"])
        fig4.update_traces(marker_line_color="rgba(99,102,241,0.8)", marker_line_width=0.5)
        fig4.add_vline(x=0.5, line_dash="dash", line_color="#f472b6",
                       annotation_text="Threshold", annotation_position="top right")
        fig4.update_layout(**CHART_LAYOUT, height=280,
                           xaxis_title="Anomaly Score", yaxis_title="Count")
        st.plotly_chart(fig4, use_container_width=True)

# ─── Agent Reports Panel ───────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<p class="section-title">Recent Agent Diagnostic Reports</p>', unsafe_allow_html=True)

if not df_r.empty:
    display_reports = df_r.head(8)
    for _, row in display_reports.iterrows():
        sev = str(row.get("severity", "Medium"))
        badge_cls = "badge-high" if sev.lower() == "high" else "badge-medium" if sev.lower() == "medium" else "badge-low"
        try:
            actions = json.loads(row["recommended_actions"]) if isinstance(row["recommended_actions"], str) else row["recommended_actions"]
        except Exception:
            actions = [str(row["recommended_actions"])]

        actions_html = "".join(f"<li>{a}</li>" for a in (actions[:3] if actions else []))
        ts = str(row.get("timestamp", ""))[:16]
        anomaly_sc = float(row.get("anomaly_score", 0.0))
        downtime = str(row.get("estimated_downtime_risk", "—"))[:100]

        st.markdown(f"""
        <div class="report-card">
            <div class="report-header">
                <span class="report-machine">⚙️ {row.get('machine_id', '—')}</span>
                <span class="severity-badge {badge_cls}">{sev}</span>
            </div>
            <div style="font-size:0.75rem; color:#475569; margin-bottom:0.4rem;">
                🕐 {ts} &nbsp;|&nbsp; Fault: <strong style="color:#e2e8f0">{row.get('fault_label','—')}</strong>
                &nbsp;|&nbsp; Anomaly Score: <strong style="color:#38bdf8">{anomaly_sc:.2f}</strong>
            </div>
            <div class="report-summary">{str(row.get('summary', '—'))[:200]}</div>
            <ul class="report-actions">{actions_html}</ul>
            <div style="font-size:0.75rem; color:#64748b; margin-top:0.4rem;">
                ⏱ Downtime Risk: {downtime}
            </div>
        </div>
        """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="alert alert-info">
        ℹ️ No agent reports yet. Click <strong>Diagnose Latest Reading</strong> in the sidebar to generate your first AI diagnostic report.
    </div>
    """, unsafe_allow_html=True)

# ─── Raw Data Table ───────────────────────────────────────────────────────────
with st.expander("📊 Raw Sensor Data Table", expanded=False):
    if not df_filtered.empty:
        st.dataframe(
            df_filtered.style.format({"vibration": "{:.2f}", "temperature": "{:.1f}", "rpm": "{:.0f}", "pressure": "{:.3f}"}),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No data for current filter.")

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding:2rem 0 1rem 0; color:#334155; font-size:0.75rem;">
    VigilAI Equipment Intelligence Platform &nbsp;|&nbsp; Powered by LangGraph + FAISS + Gemini &nbsp;|&nbsp;
    <a href="https://github.com/Puneetdivedi/VigilAI" style="color:#38bdf8; text-decoration:none;">GitHub</a>
</div>
""", unsafe_allow_html=True)

# ─── Auto-refresh ─────────────────────────────────────────────────────────────
if auto_refresh:
    time.sleep(30)
    st.cache_data.clear()
    st.rerun()
