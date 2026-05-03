"""
Production-grade Streamlit Dashboard for VigilAI v3.

Features:
- Premium dark glassmorphism UI
- Real-time KPI cards with trend indicators
- Multi-panel interactive Plotly charts
- AI diagnostic report cards with severity badges
- Machine selector with live stats
- Auto-refresh toggle
- /agents/stats integration
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

# ── Config ─────────────────────────────────────────────────────────────────────
API_URL = os.getenv("API_URL", "http://localhost:8000")
PAGE_LIMIT = int(os.getenv("DASHBOARD_LIMIT", "300"))
REFRESH_SECS = int(os.getenv("DASHBOARD_REFRESH", "30"))

st.set_page_config(
    page_title="VigilAI — Equipment Intelligence",
    layout="wide",
    page_icon="🏭",
    initial_sidebar_state="expanded",
)

# ── Premium CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:ital,wght@0,300;0,400;0,500;0,600;0,700;0,800;1,400&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }
.stApp { background: radial-gradient(ellipse at top left, #0d1117 0%, #0a0f1e 40%, #060b18 100%); color: #e2e8f0; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.25rem 2rem 3rem; max-width: 1700px; }
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#080d1a 0%,#0b1120 100%) !important;
    border-right: 1px solid rgba(56,189,248,.1) !important;
}

/* ── Header ── */
.vigil-hero {
    background: linear-gradient(135deg,rgba(56,189,248,.08) 0%,rgba(99,102,241,.08) 50%,rgba(192,132,252,.06) 100%);
    border: 1px solid rgba(56,189,248,.2);
    border-radius: 20px; padding: 1.75rem 2.25rem; margin-bottom: 1.5rem;
    backdrop-filter: blur(12px);
}
.vigil-hero h1 {
    margin:0; font-size:2.2rem; font-weight:800; line-height:1.1;
    background:linear-gradient(135deg,#38bdf8 0%,#818cf8 50%,#c084fc 100%);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;
}
.vigil-hero .sub { color:#64748b; font-size:.88rem; margin:.35rem 0 0; }
.live-dot { width:9px; height:9px; border-radius:50%; background:#22c55e;
    box-shadow:0 0 10px #22c55e; display:inline-block; margin-right:7px;
    animation: blink 1.8s ease-in-out infinite; }
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:.3} }

/* ── KPI Cards ── */
.kpi-row { display:grid; grid-template-columns:repeat(4,1fr); gap:1rem; margin-bottom:1.4rem; }
.kpi { background:rgba(10,18,35,.85); border:1px solid rgba(56,189,248,.12);
    border-radius:14px; padding:1.2rem 1.5rem; position:relative; overflow:hidden;
    transition:transform .18s,border-color .18s,box-shadow .18s; cursor:default; }
.kpi:hover { transform:translateY(-4px); border-color:rgba(56,189,248,.35);
    box-shadow:0 8px 32px rgba(56,189,248,.08); }
.kpi::after { content:''; position:absolute; top:0; left:0; right:0; height:3px;
    border-radius:2px 2px 0 0; }
.kpi-blue::after  { background:linear-gradient(90deg,#38bdf8,#0ea5e9); }
.kpi-red::after   { background:linear-gradient(90deg,#f87171,#ef4444); }
.kpi-amber::after { background:linear-gradient(90deg,#fb923c,#f59e0b); }
.kpi-green::after { background:linear-gradient(90deg,#34d399,#10b981); }
.kpi-icon { position:absolute; top:1.1rem; right:1.2rem; font-size:1.9rem; opacity:.18; }
.kpi-label { font-size:.7rem; font-weight:600; color:#475569; text-transform:uppercase; letter-spacing:.1em; }
.kpi-value { font-size:2.4rem; font-weight:800; color:#f1f5f9; margin:.2rem 0; line-height:1; }
.kpi-delta { font-size:.75rem; color:#64748b; }
.kpi-delta .up   { color:#4ade80; }
.kpi-delta .down { color:#f87171; }

/* ── Section headers ── */
.sec-head { font-size:.78rem; font-weight:700; color:#64748b; text-transform:uppercase;
    letter-spacing:.1em; display:flex; align-items:center; gap:.5rem; margin:0 0 .75rem; }
.sec-head::before { content:''; width:3px; height:14px; background:#38bdf8;
    border-radius:2px; display:inline-block; }

/* ── Report cards ── */
.rcard { background:rgba(10,18,35,.9); border:1px solid rgba(99,102,241,.18);
    border-radius:14px; padding:1rem 1.25rem; margin-bottom:.7rem;
    transition:border-color .18s, box-shadow .18s; }
.rcard:hover { border-color:rgba(99,102,241,.4); box-shadow:0 4px 20px rgba(0,0,0,.3); }
.rcard-top { display:flex; justify-content:space-between; align-items:center; margin-bottom:.4rem; }
.rmachine { font-size:.88rem; font-weight:700; color:#38bdf8; }
.rmeta { font-size:.73rem; color:#475569; margin-bottom:.4rem; }
.rsummary { font-size:.82rem; color:#94a3b8; line-height:1.55; }
.ractions { list-style:none; padding:0; margin:.5rem 0 0; }
.ractions li { font-size:.77rem; color:#cbd5e1; padding:.12rem 0 .12rem 1.1rem; position:relative; }
.ractions li::before { content:"▸"; position:absolute; left:0; color:#38bdf8; }
.rrisk { font-size:.73rem; color:#475569; margin-top:.4rem; }
.badge { font-size:.65rem; font-weight:700; padding:.2em .75em; border-radius:999px; text-transform:uppercase; letter-spacing:.07em; }
.badge-c { background:rgba(239,68,68,.2); color:#f87171; border:1px solid rgba(239,68,68,.4); }
.badge-h { background:rgba(239,68,68,.12); color:#fca5a5; border:1px solid rgba(239,68,68,.25); }
.badge-m { background:rgba(251,146,60,.15); color:#fb923c; border:1px solid rgba(251,146,60,.3); }
.badge-l { background:rgba(34,197,94,.12); color:#4ade80; border:1px solid rgba(34,197,94,.25); }

/* ── Alert boxes ── */
.alert { padding:.85rem 1.1rem; border-radius:10px; margin-bottom:1rem; font-size:.84rem; }
.alert-warn { background:rgba(251,146,60,.08); border:1px solid rgba(251,146,60,.3); color:#fed7aa; }
.alert-info { background:rgba(56,189,248,.08); border:1px solid rgba(56,189,248,.25); color:#bae6fd; }
.alert-ok   { background:rgba(34,197,94,.08);  border:1px solid rgba(34,197,94,.25);  color:#bbf7d0; }

/* ── Sidebar ── */
.sb-logo { font-size:1.6rem; font-weight:800; text-align:center; padding:.5rem 0 1.2rem;
    background:linear-gradient(135deg,#38bdf8,#818cf8); -webkit-background-clip:text;
    -webkit-text-fill-color:transparent; background-clip:text; }
.sb-section { font-size:.7rem; font-weight:700; color:#334155; text-transform:uppercase;
    letter-spacing:.1em; margin:.8rem 0 .4rem; }
</style>
""", unsafe_allow_html=True)

# ── Plotly Theme ───────────────────────────────────────────────────────────────
CHART = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(10,18,35,.6)",
    font=dict(family="Inter", color="#64748b", size=11),
    margin=dict(l=8, r=8, t=36, b=8),
    xaxis=dict(gridcolor="rgba(148,163,184,.06)", linecolor="rgba(148,163,184,.08)", tickfont=dict(color="#475569")),
    yaxis=dict(gridcolor="rgba(148,163,184,.06)", linecolor="rgba(148,163,184,.08)", tickfont=dict(color="#475569")),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#94a3b8", size=10)),
    title_font=dict(color="#94a3b8", size=12, family="Inter"),
)
COLORS = ["#38bdf8", "#818cf8", "#c084fc", "#f472b6", "#34d399", "#fb923c", "#fbbf24"]


# ── API helpers ────────────────────────────────────────────────────────────────
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


@st.cache_data(ttl=10)
def fetch_sensors(machine_id=None, limit=PAGE_LIMIT):
    params = {"limit": limit}
    if machine_id:
        params["machine_id"] = machine_id
    return _get("/sensors/latest", params) or []


@st.cache_data(ttl=10)
def fetch_reports(machine_id=None, severity=None, limit=80):
    params = {"limit": limit}
    if machine_id:
        params["machine_id"] = machine_id
    if severity:
        params["severity"] = severity
    return _get("/agents/reports", params) or []


@st.cache_data(ttl=30)
def fetch_stats():
    return _get("/agents/stats") or {}


@st.cache_data(ttl=60)
def fetch_machines():
    return _get("/sensors/machines") or []


@st.cache_data(ttl=60)
def fetch_health():
    return _get("/health") or {}


# ── Severity badge helper ──────────────────────────────────────────────────────
def severity_badge(sev: str) -> str:
    cls = {"Critical": "badge-c", "High": "badge-h", "Medium": "badge-m", "Low": "badge-l"}.get(sev, "badge-m")
    return f'<span class="badge {cls}">{sev}</span>'


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sb-logo">⚡ VigilAI</div>', unsafe_allow_html=True)

    # Health indicator
    health = fetch_health()
    if health.get("status") == "ok":
        st.markdown('<div class="alert alert-ok" style="padding:.5rem .8rem;font-size:.76rem;">🟢 API Online</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="alert alert-warn" style="padding:.5rem .8rem;font-size:.76rem;">🔴 API Offline</div>', unsafe_allow_html=True)

    st.markdown('<div class="sb-section">Machine</div>', unsafe_allow_html=True)
    machines = fetch_machines()
    selected_machine = st.selectbox("Select Machine", ["All"] + machines, label_visibility="collapsed")

    st.markdown('<div class="sb-section">Severity Filter</div>', unsafe_allow_html=True)
    sev_filter = st.selectbox("Severity", ["All", "Critical", "High", "Medium", "Low"], label_visibility="collapsed")

    st.markdown('<div class="sb-section">AI Diagnosis</div>', unsafe_allow_html=True)
    run_diag = st.button("🤖 Run Diagnosis", use_container_width=True, type="primary")

    diag_result = None
    if run_diag:
        if selected_machine != "All":
            rows = fetch_sensors(machine_id=selected_machine, limit=1)
            if rows:
                r = rows[0]
                payload = {k: r[k] for k in ("machine_id", "vibration", "temperature", "rpm", "pressure")}
                with st.spinner("Running AI pipeline…"):
                    diag_result = _post("/agents/diagnose", payload)
                st.cache_data.clear()
                if "error" not in (diag_result or {}):
                    st.success("✅ Diagnosis complete")
                else:
                    st.error("❌ Pipeline error — check API logs")
        else:
            st.warning("Select a specific machine first.")

    if diag_result and "error" not in diag_result:
        sev = diag_result.get("severity", "")
        st.markdown(f'{severity_badge(sev)} &nbsp;{diag_result.get("fault_summary","")[:120]}',
                    unsafe_allow_html=True)

    st.markdown("---")
    auto_refresh = st.toggle(f"Auto-Refresh ({REFRESH_SECS}s)", value=False)
    st.caption(f"Updated {datetime.now().strftime('%H:%M:%S')}")

    if health:
        st.markdown(f'<div style="font-size:.7rem;color:#334155;margin-top:.5rem;">v{health.get("version","?")} · {health.get("environment","?")}</div>',
                    unsafe_allow_html=True)


# ── Main Content ───────────────────────────────────────────────────────────────

# Hero header
st.markdown("""
<div class="vigil-hero">
  <h1>🏭 VigilAI</h1>
  <div class="sub"><span class="live-dot"></span>Industrial Equipment Intelligence — Real-time Monitoring & AI-Powered Diagnostics</div>
</div>
""", unsafe_allow_html=True)

# Fetch data
all_sensors = fetch_sensors(machine_id=None if selected_machine == "All" else selected_machine)
all_reports = fetch_reports(
    machine_id=None if selected_machine == "All" else selected_machine,
    severity=None if sev_filter == "All" else sev_filter,
)
stats = fetch_stats()

if not all_sensors:
    st.markdown("""
    <div class="alert alert-warn">
        ⚠️ <strong>No sensor data found.</strong>
        Run <code>python data/generate_data.py</code> then <code>uvicorn api.main:app --reload</code>.
    </div>""", unsafe_allow_html=True)
    st.stop()

df_s = pd.DataFrame(all_sensors)
df_r = pd.DataFrame(all_reports) if all_reports else pd.DataFrame()

# ── KPI Row ────────────────────────────────────────────────────────────────────
total_readings = len(df_s)
active_machines = len(df_s["machine_id"].unique()) if not df_s.empty else 0
total_reports   = stats.get("total_reports", len(df_r))
by_sev          = stats.get("by_severity", {})
high_critical   = by_sev.get("High", 0) + by_sev.get("Critical", 0)
avg_score       = stats.get("avg_anomaly_score", 0.0)
avg_temp        = df_s["temperature"].mean() if not df_s.empty else 0.0

kpi_html = f"""
<div class="kpi-row">
  <div class="kpi kpi-blue">
    <div class="kpi-icon">📡</div>
    <div class="kpi-label">Sensor Readings</div>
    <div class="kpi-value">{total_readings:,}</div>
    <div class="kpi-delta">Latest snapshot</div>
  </div>
  <div class="kpi kpi-red">
    <div class="kpi-icon">⚠️</div>
    <div class="kpi-label">High / Critical Faults</div>
    <div class="kpi-value">{high_critical}</div>
    <div class="kpi-delta">of {total_reports} total reports</div>
  </div>
  <div class="kpi kpi-amber">
    <div class="kpi-icon">🌡️</div>
    <div class="kpi-label">Avg Temperature</div>
    <div class="kpi-value">{avg_temp:.1f}°C</div>
    <div class="kpi-delta">across selected machines</div>
  </div>
  <div class="kpi kpi-green">
    <div class="kpi-icon">⚙️</div>
    <div class="kpi-label">Active Machines</div>
    <div class="kpi-value">{active_machines}</div>
    <div class="kpi-delta">Avg anomaly score: {avg_score:.3f}</div>
  </div>
</div>
"""
st.markdown(kpi_html, unsafe_allow_html=True)

# ── Charts Row 1: Time Series + Fault Donut ────────────────────────────────────
c1, c2 = st.columns([3, 2], gap="medium")

with c1:
    st.markdown('<p class="sec-head">Sensor Trends (Vibration & Temperature)</p>', unsafe_allow_html=True)
    if "timestamp" in df_s.columns and not df_s.empty:
        df_plot = df_s.copy()
        df_plot["timestamp"] = pd.to_datetime(df_plot["timestamp"])
        df_plot = df_plot.sort_values("timestamp")

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                            subplot_titles=["Vibration (Hz)", "Temperature (°C)"])
        for i, mid in enumerate(df_plot["machine_id"].unique()):
            sub = df_plot[df_plot["machine_id"] == mid]
            col = COLORS[i % len(COLORS)]
            fig.add_trace(go.Scatter(x=sub["timestamp"], y=sub["vibration"], name=mid,
                                     line=dict(color=col, width=1.5), mode="lines"), row=1, col=1)
            fig.add_trace(go.Scatter(x=sub["timestamp"], y=sub["temperature"], name=mid,
                                     line=dict(color=col, width=1.5, dash="dot"), mode="lines",
                                     showlegend=False), row=2, col=1)
        fig.update_layout(**CHART, height=380, legend=dict(orientation="h", y=1.1, x=0))
        for ann in fig.layout.annotations:
            ann.font.color = "#64748b"
            ann.font.size  = 11
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No time-series data available.")

with c2:
    st.markdown('<p class="sec-head">Fault Distribution</p>', unsafe_allow_html=True)
    by_fault = stats.get("by_fault_label", {})
    if by_fault:
        labels = list(by_fault.keys())
        values = list(by_fault.values())
        fig2 = go.Figure(go.Pie(labels=labels, values=values, hole=0.58,
                                marker=dict(colors=COLORS, line=dict(color="rgba(0,0,0,.4)", width=1.5)),
                                textinfo="percent", textfont_size=11,
                                hovertemplate="%{label}: %{value}<extra></extra>"))
        fig2.update_layout(**CHART, height=380,
                           annotations=[dict(text=f"<b>{sum(values)}</b><br>reports",
                                             x=.5, y=.5, showarrow=False,
                                             font=dict(size=13, color="#e2e8f0"))])
        st.plotly_chart(fig2, use_container_width=True)
    elif not df_r.empty and "fault_label" in df_r.columns:
        fc = df_r["fault_label"].value_counts().reset_index()
        fc.columns = ["Fault", "Count"]
        fig2 = px.pie(fc, names="Fault", values="Count", color_discrete_sequence=COLORS, hole=0.55)
        fig2.update_layout(**CHART, height=380)
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No diagnostic reports yet.")

# ── Charts Row 2: Scatter + Anomaly Histogram ─────────────────────────────────
c3, c4 = st.columns(2, gap="medium")

with c3:
    st.markdown('<p class="sec-head">RPM vs Pressure by Machine</p>', unsafe_allow_html=True)
    if not df_s.empty:
        fig3 = px.scatter(df_s, x="rpm", y="pressure", color="machine_id",
                          color_discrete_sequence=COLORS, opacity=.65,
                          labels={"rpm": "RPM", "pressure": "Pressure (bar)"})
        fig3.update_traces(marker=dict(size=4))
        fig3.update_layout(**CHART, height=260)
        st.plotly_chart(fig3, use_container_width=True)

with c4:
    st.markdown('<p class="sec-head">Anomaly Score Distribution</p>', unsafe_allow_html=True)
    if not df_r.empty and "anomaly_score" in df_r.columns:
        fig4 = px.histogram(df_r, x="anomaly_score", nbins=25, color_discrete_sequence=["#818cf8"])
        fig4.update_traces(marker_line_color="rgba(129,140,248,.8)", marker_line_width=.5)
        fig4.add_vline(x=0.5, line_dash="dash", line_color="#f472b6",
                       annotation_text="Alert threshold", annotation_position="top right",
                       annotation_font_color="#f472b6")
        fig4.update_layout(**CHART, height=260, xaxis_title="Anomaly Score", yaxis_title="Count")
        st.plotly_chart(fig4, use_container_width=True)
    else:
        st.info("Anomaly score data unavailable.")

# ── Severity Summary Bar ───────────────────────────────────────────────────────
if by_sev:
    st.markdown('<p class="sec-head">Reports by Severity</p>', unsafe_allow_html=True)
    sev_order = ["Critical", "High", "Medium", "Low"]
    sev_colors_map = {"Critical": "#ef4444", "High": "#f87171", "Medium": "#fb923c", "Low": "#4ade80"}
    sev_df = pd.DataFrame([{"Severity": s, "Count": by_sev.get(s, 0)} for s in sev_order if s in by_sev])
    if not sev_df.empty:
        sev_df["Color"] = sev_df["Severity"].map(sev_colors_map)
        fig5 = px.bar(sev_df, x="Severity", y="Count", color="Severity",
                      color_discrete_map=sev_colors_map, text="Count")
        fig5.update_traces(textposition="outside", textfont_color="#94a3b8")
        fig5.update_layout(**CHART, height=200, showlegend=False, margin=dict(l=8, r=8, t=20, b=8))
        st.plotly_chart(fig5, use_container_width=True)

# ── Agent Reports Panel ────────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<p class="sec-head">AI Diagnostic Reports</p>', unsafe_allow_html=True)

if not df_r.empty:
    for _, row in df_r.head(10).iterrows():
        sev = str(row.get("severity", "Medium"))
        try:
            actions = json.loads(row["recommended_actions"]) if isinstance(row["recommended_actions"], str) else row["recommended_actions"]
        except Exception:
            actions = [str(row.get("recommended_actions", ""))]

        actions_html = "".join(f"<li>{a}</li>" for a in (actions[:4] if actions else []))
        ts = str(row.get("timestamp", ""))[:16]
        score = float(row.get("anomaly_score", 0.0))
        conf  = float(row.get("confidence", 0.0))
        provider = row.get("llm_provider", "fallback")
        dur_ms = row.get("pipeline_duration_ms")
        dur_str = f"{dur_ms:.0f}ms" if dur_ms else "—"

        st.markdown(f"""
        <div class="rcard">
          <div class="rcard-top">
            <span class="rmachine">⚙️ {row.get("machine_id","—")}</span>
            {severity_badge(sev)}
          </div>
          <div class="rmeta">
            🕐 {ts} &nbsp;|&nbsp;
            Fault: <strong style="color:#e2e8f0">{row.get("fault_label","—")}</strong> &nbsp;|&nbsp;
            Score: <strong style="color:#38bdf8">{score:.3f}</strong> &nbsp;|&nbsp;
            Conf: <strong style="color:#818cf8">{conf:.2%}</strong> &nbsp;|&nbsp;
            LLM: <strong style="color:#c084fc">{provider}</strong> &nbsp;|&nbsp;
            ⏱ {dur_str}
          </div>
          <div class="rsummary">{str(row.get("summary","—"))[:280]}</div>
          <ul class="ractions">{actions_html}</ul>
          <div class="rrisk">⏱ Downtime risk: {str(row.get("estimated_downtime_risk","—"))[:120]}</div>
        </div>
        """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="alert alert-info">
        ℹ️ No diagnostic reports yet. Click <strong>Run Diagnosis</strong> in the sidebar after selecting a machine.
    </div>""", unsafe_allow_html=True)

# ── Raw Data Expander ──────────────────────────────────────────────────────────
with st.expander("📊 Raw Sensor Data", expanded=False):
    if not df_s.empty:
        num_cols = [c for c in ["vibration", "temperature", "rpm", "pressure"] if c in df_s.columns]
        fmt = {c: "{:.3f}" for c in num_cols}
        st.dataframe(df_s.style.format(fmt), use_container_width=True, hide_index=True)

with st.expander("📋 Fault Stats (from /agents/stats)", expanded=False):
    st.json(stats)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:2.5rem 0 .5rem;color:#1e293b;font-size:.72rem;">
  VigilAI v3 Equipment Intelligence Platform &nbsp;·&nbsp;
  LangGraph + FAISS + Gemini &nbsp;·&nbsp;
  <a href="https://github.com/Puneetdivedi/VigilAI" style="color:#38bdf8;text-decoration:none;">GitHub</a>
</div>
""", unsafe_allow_html=True)

# ── Auto-refresh ───────────────────────────────────────────────────────────────
if auto_refresh:
    time.sleep(REFRESH_SECS)
    st.cache_data.clear()
    st.rerun()
