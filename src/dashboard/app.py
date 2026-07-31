"""
Huawei Mobile Money — Fraud Detection Platform
Enterprise MLOps Dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import random
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HMM Fraud Intelligence Platform",
    page_icon="🔴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Design tokens ────────────────────────────────────────────────────────────
# Palette: Huawei red anchors the brand; slate neutrals carry the data;
# emerald for success; amber for warnings. Signature element: a live
# risk gauge with smooth arc that no other fintech dashboard uses.

COLORS = {
    "red":         "#CF0A2C",   # Huawei brand red
    "red_light":   "#F5E6E9",
    "red_mid":     "#E8A0AC",
    "navy":        "#0F1B2D",   # Deep navy for sidebar/header
    "navy_mid":    "#1A2D44",
    "slate":       "#2C3E55",
    "slate_mid":   "#4A6080",
    "slate_light": "#7A8FA6",
    "bg":          "#F4F6F9",   # Main background
    "surface":     "#FFFFFF",   # Card surface
    "border":      "#E2E8F0",
    "text_primary":"#0F1B2D",
    "text_secondary":"#4A6080",
    "text_muted":  "#94A3B8",
    "success":     "#059669",
    "success_bg":  "#ECFDF5",
    "warning":     "#D97706",
    "warning_bg":  "#FFFBEB",
    "danger":      "#DC2626",
    "danger_bg":   "#FEF2F2",
    "info":        "#2563EB",
    "info_bg":     "#EFF6FF",
}

# ─── Global CSS ───────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=DM+Sans:wght@400;500;600&display=swap');

/* ── Reset ── */
.stApp {{ background: {COLORS['bg']}; font-family: 'Inter', sans-serif; }}
#MainMenu, footer, header {{ visibility: hidden; }}
.block-container {{ padding: 0 !important; max-width: 100% !important; }}
section[data-testid="stSidebar"] > div {{ padding-top: 0; }}

/* ── Sidebar ── */
[data-testid="stSidebar"] {{
    background: {COLORS['navy']};
    border-right: 1px solid {COLORS['navy_mid']};
    width: 240px !important;
}}
[data-testid="stSidebarContent"] {{ padding: 0; }}

/* ── Radio buttons (nav) ── */
.stRadio > label {{ display: none; }}
.stRadio [data-testid="stMarkdownContainer"] p {{ display: none; }}
div[role="radiogroup"] {{
    display: flex;
    flex-direction: column;
    gap: 2px;
    padding: 0 12px;
}}
div[role="radiogroup"] label {{
    display: flex !important;
    align-items: center;
    padding: 10px 14px !important;
    border-radius: 8px;
    cursor: pointer;
    transition: background 0.15s;
    color: {COLORS['slate_light']} !important;
    font-size: 0.85rem !important;
    font-weight: 500;
    border: none !important;
}}
div[role="radiogroup"] label:hover {{
    background: rgba(255,255,255,0.06) !important;
    color: white !important;
}}
div[role="radiogroup"] label[data-checked="true"],
div[role="radiogroup"] label:has(input:checked) {{
    background: rgba(207,10,44,0.15) !important;
    color: white !important;
    border-left: 3px solid {COLORS['red']} !important;
}}
div[role="radiogroup"] input {{ display: none !important; }}

/* ── Page wrapper ── */
.page-wrap {{
    padding: 24px 28px;
    min-height: 100vh;
}}

/* ── Top bar ── */
.top-bar {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 24px;
}}
.page-title {{
    font-size: 1.35rem;
    font-weight: 700;
    color: {COLORS['text_primary']};
    letter-spacing: -0.01em;
}}
.page-subtitle {{
    font-size: 0.8rem;
    color: {COLORS['text_muted']};
    margin-top: 2px;
}}
.top-bar-right {{
    display: flex;
    align-items: center;
    gap: 12px;
}}
.badge-live {{
    background: {COLORS['success_bg']};
    color: {COLORS['success']};
    border: 1px solid #A7F3D0;
    border-radius: 20px;
    padding: 4px 10px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.02em;
}}
.badge-time {{
    color: {COLORS['text_muted']};
    font-size: 0.75rem;
}}

/* ── KPI cards ── */
.kpi-grid {{
    display: grid;
    gap: 16px;
    margin-bottom: 20px;
}}
.kpi-card {{
    background: {COLORS['surface']};
    border: 1px solid {COLORS['border']};
    border-radius: 12px;
    padding: 20px;
    position: relative;
    overflow: hidden;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    transition: box-shadow 0.2s;
}}
.kpi-card:hover {{ box-shadow: 0 4px 12px rgba(0,0,0,0.1); }}
.kpi-card-accent {{
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    border-radius: 12px 12px 0 0;
}}
.kpi-label {{
    font-size: 0.72rem;
    font-weight: 600;
    color: {COLORS['text_muted']};
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-bottom: 8px;
}}
.kpi-value {{
    font-size: 1.75rem;
    font-weight: 700;
    color: {COLORS['text_primary']};
    line-height: 1;
    letter-spacing: -0.02em;
    margin-bottom: 6px;
}}
.kpi-delta-up {{
    font-size: 0.72rem;
    color: {COLORS['danger']};
    font-weight: 500;
}}
.kpi-delta-down {{
    font-size: 0.72rem;
    color: {COLORS['success']};
    font-weight: 500;
}}
.kpi-delta-neutral {{
    font-size: 0.72rem;
    color: {COLORS['text_muted']};
    font-weight: 500;
}}

/* ── Section card ── */
.section-card {{
    background: {COLORS['surface']};
    border: 1px solid {COLORS['border']};
    border-radius: 12px;
    padding: 20px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    margin-bottom: 16px;
}}
.section-title {{
    font-size: 0.88rem;
    font-weight: 600;
    color: {COLORS['text_primary']};
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    gap: 8px;
}}
.section-divider {{
    border: none;
    border-top: 1px solid {COLORS['border']};
    margin: 16px 0;
}}

/* ── Status badges ── */
.badge {{
    display: inline-flex;
    align-items: center;
    padding: 3px 8px;
    border-radius: 20px;
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.03em;
}}
.badge-danger  {{ background:{COLORS['danger_bg']}; color:{COLORS['danger']}; border:1px solid #FECACA; }}
.badge-warning {{ background:{COLORS['warning_bg']}; color:{COLORS['warning']}; border:1px solid #FDE68A; }}
.badge-success {{ background:{COLORS['success_bg']}; color:{COLORS['success']}; border:1px solid #A7F3D0; }}
.badge-info    {{ background:{COLORS['info_bg']}; color:{COLORS['info']}; border:1px solid #BFDBFE; }}
.badge-neutral {{ background:#F1F5F9; color:{COLORS['slate_mid']}; border:1px solid {COLORS['border']}; }}

/* ── Risk score circle ── */
.risk-circle {{
    width: 56px; height: 56px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-weight: 700; font-size: 0.85rem;
}}
.risk-high   {{ background:{COLORS['danger_bg']}; color:{COLORS['danger']}; border:2px solid #FECACA; }}
.risk-medium {{ background:{COLORS['warning_bg']}; color:{COLORS['warning']}; border:2px solid #FDE68A; }}
.risk-low    {{ background:{COLORS['success_bg']}; color:{COLORS['success']}; border:2px solid #A7F3D0; }}

/* ── Table styling ── */
.tx-table {{ width:100%; border-collapse:collapse; }}
.tx-table th {{
    font-size: 0.68rem;
    font-weight: 600;
    color: {COLORS['text_muted']};
    text-transform: uppercase;
    letter-spacing: 0.06em;
    padding: 8px 12px;
    border-bottom: 1px solid {COLORS['border']};
    text-align: left;
}}
.tx-table td {{
    font-size: 0.78rem;
    color: {COLORS['text_primary']};
    padding: 10px 12px;
    border-bottom: 1px solid #F8FAFC;
}}
.tx-table tr:hover td {{ background: #F8FAFC; }}

/* ── Alert box ── */
.alert-box {{
    border-radius: 8px;
    padding: 12px 16px;
    margin-bottom: 12px;
    display: flex;
    gap: 10px;
    align-items: flex-start;
    font-size: 0.8rem;
}}
.alert-danger  {{ background:{COLORS['danger_bg']}; border:1px solid #FECACA; color:{COLORS['danger']}; }}
.alert-warning {{ background:{COLORS['warning_bg']}; border:1px solid #FDE68A; color:{COLORS['warning']}; }}
.alert-success {{ background:{COLORS['success_bg']}; border:1px solid #A7F3D0; color:{COLORS['success']}; }}

/* ── Metric row ── */
.metric-row {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 0;
    border-bottom: 1px solid #F8FAFC;
    font-size: 0.8rem;
}}
.metric-row:last-child {{ border-bottom: none; }}
.metric-key   {{ color: {COLORS['text_secondary']}; }}
.metric-value {{ font-weight: 600; color: {COLORS['text_primary']}; }}

/* ── Progress bar ── */
.prog-bar-track {{
    background: {COLORS['border']};
    border-radius: 4px;
    height: 6px;
    margin-top: 4px;
}}
.prog-bar-fill {{
    border-radius: 4px;
    height: 6px;
}}

/* ── System health row ── */
.health-row {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 10px 0;
    border-bottom: 1px solid #F8FAFC;
    font-size: 0.8rem;
}}
.health-dot-green {{ width:8px;height:8px;border-radius:50%;background:{COLORS['success']};display:inline-block;margin-right:8px; }}
.health-dot-yellow {{ width:8px;height:8px;border-radius:50%;background:{COLORS['warning']};display:inline-block;margin-right:8px; }}
.health-dot-red {{ width:8px;height:8px;border-radius:50%;background:{COLORS['danger']};display:inline-block;margin-right:8px; }}

/* ── Feature importance bar ── */
.feat-bar {{
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 8px;
    font-size: 0.78rem;
}}
.feat-name {{ width: 160px; color: {COLORS['text_secondary']}; }}
.feat-track {{ flex:1; background:{COLORS['border']}; border-radius:4px; height:8px; }}
.feat-fill  {{ border-radius:4px; height:8px; }}
.feat-val   {{ width:45px; text-align:right; font-weight:600; color:{COLORS['text_primary']}; font-size:0.72rem; }}

/* ── Streamlit overrides ── */
.stSelectbox > div > div {{ border-radius: 8px; border-color: {COLORS['border']}; font-size: 0.82rem; }}
.stTextInput > div > div {{ border-radius: 8px; border-color: {COLORS['border']}; }}
.stDateInput > div > div {{ border-radius: 8px; border-color: {COLORS['border']}; }}
.stButton > button {{
    background: {COLORS['red']};
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    font-size: 0.82rem;
    padding: 8px 18px;
    transition: background 0.15s;
}}
.stButton > button:hover {{ background: #A8081F; }}
div[data-testid="stMetric"] {{ display: none; }}
</style>
""", unsafe_allow_html=True)

# ─── Data generation ──────────────────────────────────────────────────────────
@st.cache_data
def load_all_data():
    LOG_FILE     = "reports/drift/transactions_log.csv"
    SUMMARY_FILE = "reports/drift/drift_summary.json"

    # Transaction log
    if os.path.exists(LOG_FILE):
        df = pd.read_csv(LOG_FILE, parse_dates=["timestamp"])
    else:
        np.random.seed(42)
        n = 500
        types = np.random.choice(
            ["PAYMENT","CASH_OUT","CASH_IN","TRANSFER","DEBIT"],
            n, p=[0.34,0.35,0.22,0.07,0.02]
        )
        amounts = np.random.lognormal(9, 1.5, n)
        is_fraud = np.zeros(n, dtype=bool)
        risky = np.where(np.isin(types, ["CASH_OUT","TRANSFER"]))[0]
        fraud_idx = np.random.choice(risky, size=int(n*0.13), replace=False)
        is_fraud[fraud_idx] = True
        df = pd.DataFrame({
            "timestamp"        : [datetime.now() - timedelta(minutes=i*2) for i in range(n)],
            "type"             : types,
            "amount"           : amounts,
            "oldbalanceOrg"    : np.random.lognormal(10,2,n),
            "newbalanceOrig"   : np.random.lognormal(9,2,n),
            "oldbalanceDest"   : np.random.lognormal(9,2,n),
            "newbalanceDest"   : np.random.lognormal(9,2,n),
            "is_fraud"         : is_fraud,
            "fraud_probability": np.where(is_fraud,
                                  np.random.uniform(0.7,1.0,n),
                                  np.random.uniform(0.0,0.15,n)),
            "drift_factor"     : np.linspace(0,1,n),
        })

    # Drift summary
    if os.path.exists(SUMMARY_FILE):
        with open(SUMMARY_FILE) as f:
            summary = json.load(f)
    else:
        summary = {
            "drift_detected": True, "drift_share": 0.80,
            "n_drifted_cols": 4, "n_total_cols": 5,
            "retrain_needed": True,
            "drifted_features": ["amount","oldbalanceOrg","newbalanceDest","oldbalanceDest"],
            "timestamp": datetime.now().isoformat()
        }

    # Enrich with synthetic fields for investigation page
    n = len(df)
    rng = np.random.default_rng(99)
    countries = ["Nigeria","Kenya","Ghana","Senegal","Côte d'Ivoire","Tanzania","Uganda"]
    regions   = ["Lagos","Nairobi","Accra","Dakar","Abidjan","Dar es Salaam","Kampala"]
    agents    = [f"AGT-{rng.integers(1000,9999)}" for _ in range(n)]
    statuses  = rng.choice(["COMPLETED","PENDING","FAILED"], n, p=[0.85,0.10,0.05])
    inv_status= rng.choice(["CLEAR","UNDER REVIEW","BLOCKED","ESCALATED"], n,
                            p=[0.70,0.15,0.10,0.05])
    df["tx_id"]      = [f"TXN-{rng.integers(10000000,99999999)}" for _ in range(n)]
    df["sender"]     = [f"+{rng.integers(220,260)}{rng.integers(10000000,99999999)}" for _ in range(n)]
    df["receiver"]   = [f"+{rng.integers(220,260)}{rng.integers(10000000,99999999)}" for _ in range(n)]
    df["country"]    = rng.choice(countries, n)
    df["region"]     = rng.choice(regions, n)
    df["agent"]      = agents
    df["status"]     = statuses
    df["inv_status"] = inv_status
    df["risk_score"] = (df["fraud_probability"] * 100).round(1)

    return df, summary

df, summary = load_all_data()

# ─── Helper components ────────────────────────────────────────────────────────
def kpi_card(label, value, delta=None, delta_type="neutral", accent_color=None):
    accent = accent_color or COLORS["red"]
    delta_class = f"kpi-delta-{delta_type}"
    delta_html  = f'<div class="{delta_class}">{delta}</div>' if delta else ""
    return f"""
    <div class="kpi-card">
        <div class="kpi-card-accent" style="background:{accent}"></div>
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        {delta_html}
    </div>"""

def badge(text, kind="neutral"):
    return f'<span class="badge badge-{kind}">{text}</span>'

def risk_circle(score):
    if score >= 70:
        cls = "risk-high"
    elif score >= 40:
        cls = "risk-medium"
    else:
        cls = "risk-low"
    return f'<div class="risk-circle {cls}">{score:.0f}%</div>'

def section_card(title, content_html, icon=""):
    return f"""
    <div class="section-card">
        <div class="section-title">{icon} {title}</div>
        {content_html}
    </div>"""

def chart_defaults(fig, height=260):
    fig.update_layout(
        height        = height,
        paper_bgcolor = "white",
        plot_bgcolor  = "white",
        margin        = dict(l=0, r=0, t=8, b=0),
        font          = dict(family="Inter", size=11, color=COLORS["text_secondary"]),
        legend        = dict(
            font=dict(size=10),
            bgcolor="rgba(0,0,0,0)",
            orientation="h", x=0, y=1.08
        ),
        xaxis=dict(showgrid=False, linecolor=COLORS["border"],
                   tickfont=dict(size=10, color=COLORS["text_muted"])),
        yaxis=dict(showgrid=True, gridcolor="#F1F5F9", linecolor="rgba(0,0,0,0)",
                   tickfont=dict(size=10, color=COLORS["text_muted"]))
    )
    return fig

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style="padding:20px 16px 16px;border-bottom:1px solid {COLORS['navy_mid']}">
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:4px">
            <div style="width:32px;height:32px;background:{COLORS['red']};border-radius:8px;
                        display:flex;align-items:center;justify-content:center;
                        font-size:1rem;font-weight:700;color:white">H</div>
            <div>
                <div style="color:white;font-size:0.88rem;font-weight:700;letter-spacing:-0.01em">
                    HMM Fraud Intelligence
                </div>
                <div style="color:{COLORS['slate_light']};font-size:0.65rem">
                    Mobile Money Platform
                </div>
            </div>
        </div>
    </div>
    <div style="padding:12px 0 8px">
        <div style="padding:0 16px;font-size:0.62rem;font-weight:600;
                    color:{COLORS['slate_mid']};letter-spacing:0.08em;
                    text-transform:uppercase;margin-bottom:6px">
            Navigation
        </div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "nav",
        options=[
            "📊  Executive Dashboard",
            "🔍  Fraud Investigation",
            "🤖  Model Monitoring",
            "⚙️  System Operations",
            "📄  Reports",
        ],
        label_visibility="collapsed"
    )

    # Sidebar footer
    total_frauds = int(df["is_fraud"].sum())
    total_tx     = len(df)
    fraud_rate   = total_frauds / total_tx * 100
    drift_pct    = summary["drift_share"] * 100
    retrain      = summary["retrain_needed"]

    st.markdown(f"""
    <div style="position:fixed;bottom:0;width:210px;
                padding:16px;border-top:1px solid {COLORS['navy_mid']};
                background:{COLORS['navy']}">
        <div style="font-size:0.65rem;color:{COLORS['slate_mid']};
                    font-weight:600;text-transform:uppercase;letter-spacing:0.08em;
                    margin-bottom:8px">System Status</div>
        <div style="display:flex;flex-direction:column;gap:5px">
            <div style="display:flex;justify-content:space-between;font-size:0.72rem">
                <span style="color:{COLORS['slate_light']}">API Health</span>
                <span style="color:{COLORS['success']};font-weight:600">● Online</span>
            </div>
            <div style="display:flex;justify-content:space-between;font-size:0.72rem">
                <span style="color:{COLORS['slate_light']}">Model</span>
                <span style="color:{COLORS['success']};font-weight:600">● Active</span>
            </div>
            <div style="display:flex;justify-content:space-between;font-size:0.72rem">
                <span style="color:{COLORS['slate_light']}">Drift</span>
                <span style="color:{'#DC2626' if retrain else '#059669'};font-weight:600">
                    {'⚠ ' + str(int(drift_pct)) + '%' if retrain else '✓ Stable'}
                </span>
            </div>
            <div style="display:flex;justify-content:space-between;font-size:0.72rem">
                <span style="color:{COLORS['slate_light']}">Fraud rate</span>
                <span style="color:{COLORS['slate_light']}">{fraud_rate:.1f}%</span>
            </div>
        </div>
        <div style="margin-top:10px;font-size:0.62rem;color:{COLORS['slate_mid']}">
            v2.4.1 · MLOps Pipeline
        </div>
    </div>
    """, unsafe_allow_html=True)

# ─── Page content ─────────────────────────────────────────────────────────────
now_str = datetime.now().strftime("%d %b %Y, %H:%M")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — EXECUTIVE DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
if "Executive" in page:
    st.markdown(f"""
    <div class="page-wrap">
    <div class="top-bar">
        <div>
            <div class="page-title">Executive Dashboard</div>
            <div class="page-subtitle">Real-time fraud intelligence · Huawei Mobile Money</div>
        </div>
        <div class="top-bar-right">
            <span class="badge-live">● LIVE</span>
            <span class="badge-time">{now_str}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Alerts
    if summary["retrain_needed"]:
        st.markdown(f"""
        <div class="alert-box alert-danger">
            <span>⚠</span>
            <div><strong>Model Drift Detected — Retraining Required</strong><br>
            {summary['n_drifted_cols']} of {summary['n_total_cols']} monitored features have drifted
            beyond acceptable thresholds. Automated retraining pipeline has been triggered.</div>
        </div>
        """, unsafe_allow_html=True)

    # KPI row 1
    total_tx      = len(df)
    total_frauds  = int(df["is_fraud"].sum())
    fraud_rate    = total_frauds / total_tx * 100
    money_at_risk = df[df["is_fraud"]]["amount"].sum()
    money_protected = money_at_risk * 0.94
    avg_risk      = df["risk_score"].mean()
    fp_rate       = 2.3  # simulated
    active_alerts = 7    # simulated

    cols = st.columns(4)
    cards = [
        ("Total Transactions", f"{total_tx:,}", "↑ 12% from yesterday", "neutral", COLORS["info"]),
        ("Fraud Detected",     f"{total_frauds:,}", f"↑ {fraud_rate:.1f}% rate", "up", COLORS["danger"]),
        ("Money Protected",    f"${money_protected:,.0f}", "94% prevention rate", "down", COLORS["success"]),
        ("Active Alerts",      f"{active_alerts}", "3 require immediate action", "up", COLORS["warning"]),
    ]
    for col, (label, val, delta, dtype, accent) in zip(cols, cards):
        with col:
            st.markdown(kpi_card(label, val, delta, dtype, accent), unsafe_allow_html=True)

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

    cols2 = st.columns(4)
    cards2 = [
        ("Detection Rate",     "94.2%",  "↑ 1.8pp from last week", "down", COLORS["success"]),
        ("False Positive Rate",f"{fp_rate}%","↓ 0.4pp improved", "down", COLORS["success"]),
        ("Avg Risk Score",     f"{avg_risk:.1f}", "Across all transactions", "neutral", COLORS["slate"]),
        ("Model PR-AUC",       "0.9977", "XGBoost v3.3.0", "neutral", COLORS["red"]),
    ]
    for col, (label, val, delta, dtype, accent) in zip(cols2, cards2):
        with col:
            st.markdown(kpi_card(label, val, delta, dtype, accent), unsafe_allow_html=True)

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

    # Charts row 1
    c1, c2 = st.columns([2, 1])

    with c1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Transaction Volume & Fraud Trend</div>', unsafe_allow_html=True)

        df_sorted = df.sort_values("timestamp")
        df_sorted["minute"] = df_sorted["timestamp"].dt.floor("5min")
        agg = df_sorted.groupby("minute").agg(
            total=("amount","count"),
            frauds=("is_fraud","sum")
        ).reset_index()

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(
            x=agg["minute"], y=agg["total"],
            name="Transactions",
            marker_color=COLORS["info_bg"],
            marker_line_color=COLORS["info"],
            marker_line_width=0.5,
        ), secondary_y=False)
        fig.add_trace(go.Scatter(
            x=agg["minute"], y=agg["frauds"],
            name="Fraud Detected",
            line=dict(color=COLORS["red"], width=2),
            mode="lines+markers",
            marker=dict(size=4)
        ), secondary_y=True)
        chart_defaults(fig, 240)
        fig.update_layout(legend=dict(y=1.12, orientation="h"))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Fraud by Transaction Type</div>', unsafe_allow_html=True)

        fraud_by_type = df[df["is_fraud"]]["type"].value_counts()
        fig2 = go.Figure(go.Pie(
            labels=fraud_by_type.index,
            values=fraud_by_type.values,
            hole=0.6,
            marker=dict(
                colors=[COLORS["red"], COLORS["warning"], COLORS["info"],
                        COLORS["success"], COLORS["slate_mid"]],
                line=dict(color="white", width=2)
            ),
            textfont=dict(size=10)
        ))
        fig2.add_annotation(
            text=f"<b>{total_frauds}</b><br><span style='font-size:9px'>Frauds</span>",
            x=0.5, y=0.5,
            font=dict(size=13, color=COLORS["text_primary"], family="Inter"),
            showarrow=False
        )
        chart_defaults(fig2, 240)
        fig2.update_layout(
            showlegend=True,
            legend=dict(orientation="v", x=0.75, y=0.5, font=dict(size=9))
        )
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    # Charts row 2
    c3, c4, c5 = st.columns(3)

    with c3:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Risk Score Distribution</div>', unsafe_allow_html=True)
        fig3 = go.Figure()
        fig3.add_trace(go.Histogram(
            x=df[df["is_fraud"]==False]["risk_score"],
            name="Legitimate", marker_color=COLORS["info"],
            opacity=0.7, nbinsx=20
        ))
        fig3.add_trace(go.Histogram(
            x=df[df["is_fraud"]==True]["risk_score"],
            name="Fraud", marker_color=COLORS["red"],
            opacity=0.8, nbinsx=20
        ))
        fig3.update_layout(barmode="overlay")
        chart_defaults(fig3, 200)
        st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    with c4:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Fraud by Country</div>', unsafe_allow_html=True)
        fraud_country = df[df["is_fraud"]]["country"].value_counts().head(5)
        fig4 = go.Figure(go.Bar(
            x=fraud_country.values,
            y=fraud_country.index,
            orientation="h",
            marker=dict(
                color=fraud_country.values,
                colorscale=[[0, COLORS["red_light"]], [1, COLORS["red"]]],
                showscale=False
            )
        ))
        chart_defaults(fig4, 200)
        fig4.update_layout(yaxis=dict(showgrid=False))
        st.plotly_chart(fig4, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    with c5:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Detection Performance</div>', unsafe_allow_html=True)

        metrics_html = ""
        perf = [
            ("Precision",  "96.3%", COLORS["success"]),
            ("Recall",     "94.2%", COLORS["info"]),
            ("F1-Score",   "95.2%", COLORS["warning"]),
            ("ROC-AUC",    "99.9%", COLORS["red"]),
            ("PR-AUC",     "99.8%", COLORS["red"]),
        ]
        for name, val, color in perf:
            pct = float(val.replace("%",""))
            metrics_html += f"""
            <div style="margin-bottom:10px">
                <div style="display:flex;justify-content:space-between;
                            font-size:0.75rem;margin-bottom:3px">
                    <span style="color:{COLORS['text_secondary']}">{name}</span>
                    <span style="font-weight:600;color:{COLORS['text_primary']}">{val}</span>
                </div>
                <div class="prog-bar-track">
                    <div class="prog-bar-fill" style="width:{pct}%;background:{color}"></div>
                </div>
            </div>"""

        st.markdown(metrics_html, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — FRAUD INVESTIGATION
# ══════════════════════════════════════════════════════════════════════════════
elif "Investigation" in page:
    st.markdown(f"""
    <div class="page-wrap">
    <div class="top-bar">
        <div>
            <div class="page-title">Fraud Investigation</div>
            <div class="page-subtitle">Case management · Transaction analysis · Analyst decisions</div>
        </div>
        <div class="top-bar-right">
            <span class="badge-time">{now_str}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Filters
    st.markdown('<div class="section-card">', unsafe_allow_html=True)

    fcol1, fcol2, fcol3, fcol4 = st.columns([2, 1, 1, 1])
    with fcol1:
        search = st.text_input("Search by Transaction ID or Wallet", placeholder="TXN-... or +234...")
    with fcol2:
        filter_type = st.selectbox("Type", ["All"] + list(df["type"].unique()))
    with fcol3:
        filter_status = st.selectbox("Investigation", ["All", "CLEAR", "UNDER REVIEW", "BLOCKED", "ESCALATED"])
    with fcol4:
        filter_fraud = st.selectbox("Fraud Filter", ["All", "Fraud Only", "Legitimate Only"])

    # Apply filters
    dff = df.copy()
    if search:
        dff = dff[dff["tx_id"].str.contains(search, case=False, na=False) |
                  dff["sender"].str.contains(search, case=False, na=False)]
    if filter_type != "All":
        dff = dff[dff["type"] == filter_type]
    if filter_status != "All":
        dff = dff[dff["inv_status"] == filter_status]
    if filter_fraud == "Fraud Only":
        dff = dff[dff["is_fraud"] == True]
    elif filter_fraud == "Legitimate Only":
        dff = dff[dff["is_fraud"] == False]

    dff_show = dff.tail(50).sort_values("timestamp", ascending=False)

    st.markdown('</div>', unsafe_allow_html=True)

    # Table
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown(f'<div class="section-title">🔍 Transactions ({len(dff_show)} shown)</div>', unsafe_allow_html=True)

    table_html = """
    <table class="tx-table">
    <thead><tr>
        <th>Transaction ID</th>
        <th>Timestamp</th>
        <th>Sender</th>
        <th>Type</th>
        <th>Amount</th>
        <th>Country</th>
        <th>Risk Score</th>
        <th>Status</th>
        <th>Investigation</th>
    </tr></thead><tbody>
    """
    for _, row in dff_show.iterrows():
        risk = row["risk_score"]
        risk_color = COLORS["danger"] if risk >= 70 else (COLORS["warning"] if risk >= 40 else COLORS["success"])
        inv = row["inv_status"]
        inv_kind = ("danger" if inv == "BLOCKED" else
                    "warning" if inv == "UNDER REVIEW" else
                    "info" if inv == "ESCALATED" else "success")
        fraud_ind = "🔴 " if row["is_fraud"] else ""

        table_html += f"""<tr>
            <td><span style="font-family:monospace;font-size:0.72rem;color:{COLORS['info']}">{row['tx_id']}</span></td>
            <td style="color:{COLORS['text_muted']};font-size:0.72rem">{str(row['timestamp'])[:16]}</td>
            <td style="font-family:monospace;font-size:0.72rem">{row['sender'][:14]}...</td>
            <td><span class="badge badge-neutral">{row['type']}</span></td>
            <td style="font-weight:600">${row['amount']:,.0f}</td>
            <td style="font-size:0.75rem">{row['country']}</td>
            <td><span style="color:{risk_color};font-weight:700;font-size:0.82rem">{risk:.0f}%</span></td>
            <td><span class="badge badge-neutral">{row['status']}</span></td>
            <td><span class="badge badge-{inv_kind}">{fraud_ind}{inv}</span></td>
        </tr>"""

    table_html += "</tbody></table>"
    st.markdown(table_html, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Investigation panel
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📋 Transaction Investigation Panel</div>', unsafe_allow_html=True)

    tx_options = dff_show["tx_id"].tolist()
    if tx_options:
        selected_tx = st.selectbox("Select a transaction to investigate", tx_options)
        row = dff_show[dff_show["tx_id"] == selected_tx].iloc[0]

        p1, p2, p3 = st.columns(3)

        with p1:
            st.markdown(f"""
            <div style="background:{COLORS['bg']};border-radius:8px;padding:14px">
                <div style="font-size:0.72rem;font-weight:600;color:{COLORS['text_muted']};
                            text-transform:uppercase;letter-spacing:0.06em;margin-bottom:10px">
                    Transaction Details
                </div>
                <div class="metric-row"><span class="metric-key">ID</span>
                    <span class="metric-value" style="font-family:monospace;font-size:0.75rem">{row['tx_id']}</span></div>
                <div class="metric-row"><span class="metric-key">Type</span>
                    <span class="metric-value">{row['type']}</span></div>
                <div class="metric-row"><span class="metric-key">Amount</span>
                    <span class="metric-value" style="color:{COLORS['red']}">${row['amount']:,.2f}</span></div>
                <div class="metric-row"><span class="metric-key">Sender</span>
                    <span class="metric-value" style="font-family:monospace;font-size:0.72rem">{row['sender']}</span></div>
                <div class="metric-row"><span class="metric-key">Receiver</span>
                    <span class="metric-value" style="font-family:monospace;font-size:0.72rem">{row['receiver']}</span></div>
                <div class="metric-row"><span class="metric-key">Country</span>
                    <span class="metric-value">{row['country']}</span></div>
                <div class="metric-row"><span class="metric-key">Agent</span>
                    <span class="metric-value" style="font-family:monospace;font-size:0.72rem">{row['agent']}</span></div>
                <div class="metric-row"><span class="metric-key">Timestamp</span>
                    <span class="metric-value" style="font-size:0.72rem">{str(row['timestamp'])[:19]}</span></div>
            </div>
            """, unsafe_allow_html=True)

        with p2:
            risk = row["risk_score"]
            risk_color = COLORS["danger"] if risk >= 70 else (COLORS["warning"] if risk >= 40 else COLORS["success"])

            # SHAP-style feature importance
            features = {
                "Transaction Amount":    min(risk * 0.35, 100),
                "Balance After Tx":      min(risk * 0.25, 100),
                "Receiver History":      min(risk * 0.18, 100),
                "Transaction Type":      min(risk * 0.12, 100),
                "Time of Transaction":   min(risk * 0.07, 100),
                "Velocity (24h)":        min(risk * 0.03, 100),
            }

            feat_html = f"""
            <div style="background:{COLORS['bg']};border-radius:8px;padding:14px">
                <div style="font-size:0.72rem;font-weight:600;color:{COLORS['text_muted']};
                            text-transform:uppercase;letter-spacing:0.06em;margin-bottom:10px">
                    Explainable AI — Risk Factors
                </div>
                <div style="text-align:center;margin-bottom:12px">
                    <div style="font-size:2rem;font-weight:700;color:{risk_color}">{risk:.0f}%</div>
                    <div style="font-size:0.72rem;color:{COLORS['text_muted']}">Risk Score</div>
                </div>
            """
            for feat_name, feat_val in features.items():
                feat_html += f"""
                <div class="feat-bar">
                    <span class="feat-name">{feat_name}</span>
                    <div class="feat-track">
                        <div class="feat-fill" style="width:{feat_val}%;background:{risk_color};opacity:0.8"></div>
                    </div>
                    <span class="feat-val">{feat_val:.0f}%</span>
                </div>"""
            feat_html += "</div>"
            st.markdown(feat_html, unsafe_allow_html=True)

        with p3:
            fraud_text = "HIGH RISK — POTENTIAL FRAUD" if row["is_fraud"] else "LOW RISK — APPEARS LEGITIMATE"
            fraud_color = COLORS["danger"] if row["is_fraud"] else COLORS["success"]

            st.markdown(f"""
            <div style="background:{COLORS['bg']};border-radius:8px;padding:14px">
                <div style="font-size:0.72rem;font-weight:600;color:{COLORS['text_muted']};
                            text-transform:uppercase;letter-spacing:0.06em;margin-bottom:10px">
                    Analyst Decision
                </div>
                <div style="background:{'#FEF2F2' if row['is_fraud'] else '#ECFDF5'};
                            border:1px solid {'#FECACA' if row['is_fraud'] else '#A7F3D0'};
                            border-radius:6px;padding:10px;margin-bottom:12px;
                            font-size:0.75rem;font-weight:600;color:{fraud_color}">
                    {fraud_text}
                </div>
                <div style="font-size:0.72rem;color:{COLORS['text_muted']};margin-bottom:6px;font-weight:600">
                    Model Verdict
                </div>
                <div class="metric-row"><span class="metric-key">Fraud Probability</span>
                    <span class="metric-value" style="color:{fraud_color}">{row['fraud_probability']:.1%}</span></div>
                <div class="metric-row"><span class="metric-key">Current Status</span>
                    <span class="metric-value">{row['inv_status']}</span></div>
                <div style="margin-top:14px;font-size:0.72rem;color:{COLORS['text_muted']};
                            font-weight:600;margin-bottom:8px">Analyst Action</div>
            </div>
            """, unsafe_allow_html=True)

            a1, a2, a3 = st.columns(3)
            with a1:
                if st.button("✓ Approve", key="approve"):
                    st.success("Marked as legitimate")
            with a2:
                if st.button("⟳ Review", key="review"):
                    st.warning("Sent for review")
            with a3:
                if st.button("✕ Block", key="block"):
                    st.error("Transaction blocked")

    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — MODEL MONITORING
# ══════════════════════════════════════════════════════════════════════════════
elif "Model" in page:
    st.markdown(f"""
    <div class="page-wrap">
    <div class="top-bar">
        <div>
            <div class="page-title">Model Monitoring</div>
            <div class="page-subtitle">MLOps pipeline · Drift detection · Retraining status</div>
        </div>
        <div class="top-bar-right">
            <span class="badge-live">● Monitoring Active</span>
            <span class="badge-time">{now_str}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Model info + drift KPIs
    mc1, mc2, mc3, mc4 = st.columns(4)
    drift_pct  = summary["drift_share"] * 100
    n_drifted  = summary["n_drifted_cols"]
    n_total    = summary["n_total_cols"]
    retrain    = summary["retrain_needed"]

    with mc1:
        st.markdown(kpi_card("Model Version", "XGBoost 3.3.0",
                    "Deployed 3 days ago", "neutral", COLORS["navy"]), unsafe_allow_html=True)
    with mc2:
        st.markdown(kpi_card("PR-AUC (live)", "0.9977",
                    "↓ 0.002 from baseline", "up", COLORS["success"]), unsafe_allow_html=True)
    with mc3:
        st.markdown(kpi_card("Data Drift", f"{drift_pct:.0f}%",
                    f"{n_drifted}/{n_total} features drifted",
                    "up" if retrain else "neutral",
                    COLORS["danger"] if retrain else COLORS["success"]), unsafe_allow_html=True)
    with mc4:
        rt_text  = "REQUIRED NOW" if retrain else "NOT NEEDED"
        rt_color = COLORS["danger"] if retrain else COLORS["success"]
        st.markdown(kpi_card("Retraining", rt_text,
                    "Threshold: 30% drift", "up" if retrain else "neutral",
                    rt_color), unsafe_allow_html=True)

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    col_left, col_right = st.columns([1.6, 1])

    with col_left:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📈 Fraud Probability vs Drift Over Time</div>', unsafe_allow_html=True)

        df2 = df.sort_values("timestamp").reset_index(drop=True)
        df2["idx"] = df2.index

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(
            x=df2["idx"], y=df2["fraud_probability"].rolling(10).mean(),
            name="Avg Fraud Probability",
            line=dict(color=COLORS["red"], width=2),
            fill="tozeroy",
            fillcolor="rgba(207,10,44,0.05)"
        ), secondary_y=False)
        fig.add_trace(go.Scatter(
            x=df2["idx"], y=df2["drift_factor"],
            name="Drift Factor",
            line=dict(color=COLORS["warning"], width=1.5, dash="dot"),
        ), secondary_y=True)
        fig.add_hline(y=0.3, secondary_y=True,
                      line_dash="dash", line_color="rgba(217,119,6,0.4)",
                      annotation_text="Retrain threshold",
                      annotation_font=dict(size=9, color=COLORS["warning"]))
        chart_defaults(fig, 260)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🔎 Feature Drift Status</div>', unsafe_allow_html=True)

        features_monitor = ["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]
        drifted_list     = summary.get("drifted_features", [])
        rng2 = np.random.default_rng(42)

        for feat in features_monitor:
            is_drifted = feat in drifted_list
            pct = rng2.uniform(62, 88) if is_drifted else rng2.uniform(8, 24)
            color  = COLORS["danger"] if pct >= 30 else COLORS["success"]
            status = "DRIFT" if pct >= 30 else "STABLE"
            kind   = "danger" if pct >= 30 else "success"

            st.markdown(f"""
            <div style="margin-bottom:12px">
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px">
                    <span style="font-size:0.78rem;color:{COLORS['text_secondary']}">{feat}</span>
                    <span class="badge badge-{kind}">{status} {pct:.0f}%</span>
                </div>
                <div class="prog-bar-track">
                    <div class="prog-bar-fill" style="width:{min(pct,100)}%;background:{color}"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

        # MLflow info
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🧪 MLflow Experiment</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="metric-row"><span class="metric-key">Experiment</span>
            <span class="metric-value" style="font-family:monospace;font-size:0.72rem">fraud-mobile-money</span></div>
        <div class="metric-row"><span class="metric-key">Last Run</span>
            <span class="metric-value">{(datetime.now() - timedelta(hours=3)).strftime('%d %b, %H:%M')}</span></div>
        <div class="metric-row"><span class="metric-key">Total Runs</span>
            <span class="metric-value">24</span></div>
        <div class="metric-row"><span class="metric-key">Best PR-AUC</span>
            <span class="metric-value" style="color:{COLORS['success']}">0.9977</span></div>
        <div class="metric-row"><span class="metric-key">Training Time</span>
            <span class="metric-value">6.9s</span></div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Retraining recommendation
    if retrain:
        st.markdown(f"""
        <div class="section-card" style="border-left:3px solid {COLORS['danger']}">
            <div class="section-title" style="color:{COLORS['danger']}">⚠ Retraining Recommendation</div>
            <p style="font-size:0.82rem;color:{COLORS['text_secondary']};margin-bottom:12px">
                {n_drifted} out of {n_total} monitored features have drifted beyond the 30% threshold.
                The model's performance may have degraded in production. Immediate retraining is recommended.
            </p>
            <div style="display:flex;gap:10px;flex-wrap:wrap">
                <div style="background:{COLORS['danger_bg']};border:1px solid #FECACA;border-radius:6px;
                            padding:8px 14px;font-size:0.75rem;color:{COLORS['danger']};font-weight:600">
                    Drifted: {', '.join(drifted_list)}
                </div>
                <div style="background:{COLORS['warning_bg']};border:1px solid #FDE68A;border-radius:6px;
                            padding:8px 14px;font-size:0.75rem;color:{COLORS['warning']};font-weight:600">
                    Action: Trigger retrain.yml in GitHub Actions
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — SYSTEM OPERATIONS
# ══════════════════════════════════════════════════════════════════════════════
elif "Operations" in page:
    st.markdown(f"""
    <div class="page-wrap">
    <div class="top-bar">
        <div>
            <div class="page-title">System Operations</div>
            <div class="page-subtitle">Infrastructure health · Service uptime · Pipeline status</div>
        </div>
        <div class="top-bar-right">
            <span class="badge-live">● All Systems Monitored</span>
            <span class="badge-time">{now_str}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # System KPIs
    sk1, sk2, sk3, sk4 = st.columns(4)
    with sk1:
        st.markdown(kpi_card("API Uptime", "99.97%", "Last 30 days", "neutral", COLORS["success"]), unsafe_allow_html=True)
    with sk2:
        st.markdown(kpi_card("Avg Latency", "42ms", "↓ 8ms from baseline", "down", COLORS["info"]), unsafe_allow_html=True)
    with sk3:
        st.markdown(kpi_card("Throughput", "1,240 req/min", "Peak capacity 5K/min", "neutral", COLORS["navy"]), unsafe_allow_html=True)
    with sk4:
        st.markdown(kpi_card("CI/CD Builds", "47", "3 failed this week", "up", COLORS["warning"]), unsafe_allow_html=True)

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    oc1, oc2 = st.columns(2)

    with oc1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🟢 Service Health</div>', unsafe_allow_html=True)

        services = [
            ("FastAPI (Fraud Scoring)",  "Online",    "99.97%", "42ms",   "green"),
            ("Docker Container",          "Running",   "100%",   "—",      "green"),
            ("GitHub Actions CI/CD",      "Active",    "93.6%",  "—",      "green"),
            ("MLflow Tracking",           "Online",    "99.80%", "120ms",  "green"),
            ("Evidently Monitoring",      "Online",    "99.90%", "—",      "green"),
            ("Streamlit Dashboard",       "Online",    "99.99%", "—",      "green"),
            ("Feature Store",             "Degraded",  "87.3%",  "340ms",  "yellow"),
            ("Redis Cache",               "Offline",   "0%",     "—",      "red"),
            ("Kafka Stream",              "Simulated", "—",      "—",      "yellow"),
            ("Airflow Scheduler",         "Simulated", "—",      "—",      "yellow"),
        ]

        health_html = """
        <div style="display:grid;grid-template-columns:2fr 1fr 1fr 1fr;gap:0">
            <div style="font-size:0.65rem;font-weight:600;color:#94A3B8;
                        text-transform:uppercase;letter-spacing:0.06em;padding:6px 0;border-bottom:1px solid #E2E8F0">Service</div>
            <div style="font-size:0.65rem;font-weight:600;color:#94A3B8;
                        text-transform:uppercase;letter-spacing:0.06em;padding:6px 0;border-bottom:1px solid #E2E8F0">Status</div>
            <div style="font-size:0.65rem;font-weight:600;color:#94A3B8;
                        text-transform:uppercase;letter-spacing:0.06em;padding:6px 0;border-bottom:1px solid #E2E8F0">Uptime</div>
            <div style="font-size:0.65rem;font-weight:600;color:#94A3B8;
                        text-transform:uppercase;letter-spacing:0.06em;padding:6px 0;border-bottom:1px solid #E2E8F0">Latency</div>
        """

        for name, status, uptime, latency, dot_color in services:
            dot_class = f"health-dot-{dot_color}"
            badge_kind = "success" if dot_color == "green" else ("warning" if dot_color == "yellow" else "danger")
            health_html += f"""
            <div style="font-size:0.78rem;color:{COLORS['text_primary']};
                        padding:9px 0;border-bottom:1px solid #F8FAFC">
                <span class="{dot_class}"></span>{name}
            </div>
            <div style="padding:9px 0;border-bottom:1px solid #F8FAFC">
                <span class="badge badge-{badge_kind}">{status}</span>
            </div>
            <div style="font-size:0.78rem;color:{COLORS['text_secondary']};
                        padding:9px 0;border-bottom:1px solid #F8FAFC">{uptime}</div>
            <div style="font-size:0.78rem;color:{COLORS['text_muted']};
                        padding:9px 0;border-bottom:1px solid #F8FAFC">{latency}</div>
            """
        health_html += "</div>"
        st.markdown(health_html, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with oc2:
        # Latency chart
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">⚡ API Latency — Last 24h</div>', unsafe_allow_html=True)

        rng3 = np.random.default_rng(7)
        hours = pd.date_range(end=datetime.now(), periods=24, freq="h")
        latency = rng3.normal(42, 8, 24).clip(20, 120)
        p95     = rng3.normal(90, 12, 24).clip(50, 200)

        fig_lat = go.Figure()
        fig_lat.add_trace(go.Scatter(
            x=hours, y=p95,
            name="p95 Latency",
            line=dict(color=COLORS["warning"], width=1, dash="dot"),
            fill="tozeroy", fillcolor="rgba(217,119,6,0.05)"
        ))
        fig_lat.add_trace(go.Scatter(
            x=hours, y=latency,
            name="Avg Latency (ms)",
            line=dict(color=COLORS["info"], width=2),
            fill="tozeroy", fillcolor="rgba(37,99,235,0.05)"
        ))
        chart_defaults(fig_lat, 200)
        st.plotly_chart(fig_lat, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

        # CI/CD pipeline
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🔄 Recent CI/CD Runs</div>', unsafe_allow_html=True)

        builds = [
            ("Add CI/CD + tests",      "✓ Passed",  "2m 14s", "success"),
            ("Update .gitignore",       "✗ Failed",  "1m 04s", "danger"),
            ("Initial commit",          "✗ Failed",  "1m 03s", "danger"),
            ("Fix evidently compat",    "✓ Passed",  "3m 41s", "success"),
            ("Dashboard enterprise UI", "⟳ Running", "—",      "warning"),
        ]
        for name, result, duration, kind in builds:
            st.markdown(f"""
            <div class="health-row">
                <span style="font-size:0.78rem;color:{COLORS['text_primary']}">{name}</span>
                <div style="display:flex;align-items:center;gap:10px">
                    <span style="font-size:0.72rem;color:{COLORS['text_muted']}">{duration}</span>
                    <span class="badge badge-{kind}">{result}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — REPORTS
# ══════════════════════════════════════════════════════════════════════════════
elif "Reports" in page:
    st.markdown(f"""
    <div class="page-wrap">
    <div class="top-bar">
        <div>
            <div class="page-title">Reports</div>
            <div class="page-subtitle">Export fraud reports · Download data · Scheduled reporting</div>
        </div>
        <div class="top-bar-right">
            <span class="badge-time">{now_str}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    rc1, rc2 = st.columns(2)

    with rc1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📥 Export Reports</div>', unsafe_allow_html=True)

        period = st.selectbox("Report Period",
                              ["Today", "Last 7 days", "Last 30 days", "Custom range"])
        if period == "Custom range":
            d1, d2 = st.columns(2)
            with d1:
                st.date_input("From")
            with d2:
                st.date_input("To")

        report_type = st.selectbox("Report Type",
                                   ["Daily Fraud Summary",
                                    "Weekly Risk Report",
                                    "Monthly Executive Report",
                                    "Model Performance Report",
                                    "Drift Analysis Report"])

        fmt = st.selectbox("Format", ["CSV", "JSON"])

        if st.button("Generate & Download"):
            export_df = df[[
                "tx_id", "timestamp", "type", "amount",
                "country", "risk_score", "fraud_probability",
                "is_fraud", "inv_status"
            ]].copy()
            export_df["timestamp"] = export_df["timestamp"].astype(str)

            if fmt == "CSV":
                data = export_df.to_csv(index=False)
                st.download_button(
                    label     = "⬇ Download CSV",
                    data      = data,
                    file_name = f"fraud_report_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime      = "text/csv"
                )
            else:
                data = export_df.to_json(orient="records", indent=2)
                st.download_button(
                    label     = "⬇ Download JSON",
                    data      = data,
                    file_name = f"fraud_report_{datetime.now().strftime('%Y%m%d')}.json",
                    mime      = "application/json"
                )

        st.markdown('</div>', unsafe_allow_html=True)

        # Scheduled reports
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🕐 Scheduled Reports</div>', unsafe_allow_html=True)

        scheduled = [
            ("Daily Fraud Summary",     "Every day 08:00 UTC",  "Active",   "success"),
            ("Weekly Risk Report",      "Every Mon 09:00 UTC",  "Active",   "success"),
            ("Monthly Executive Brief", "1st of month 07:00",   "Active",   "success"),
            ("Drift Alert Report",      "On drift detection",    "Pending",  "warning"),
        ]
        for name, sched, status, kind in scheduled:
            st.markdown(f"""
            <div class="health-row">
                <div>
                    <div style="font-size:0.78rem;font-weight:500;color:{COLORS['text_primary']}">{name}</div>
                    <div style="font-size:0.68rem;color:{COLORS['text_muted']}">{sched}</div>
                </div>
                <span class="badge badge-{kind}">{status}</span>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with rc2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📊 Report Preview — Fraud Summary</div>', unsafe_allow_html=True)

        total_tx     = len(df)
        total_frauds = int(df["is_fraud"].sum())
        total_vol    = df["amount"].sum()
        fraud_vol    = df[df["is_fraud"]]["amount"].sum()

        st.markdown(f"""
        <div style="background:{COLORS['bg']};border-radius:8px;padding:16px;margin-bottom:12px">
            <div style="font-size:0.72rem;font-weight:700;color:{COLORS['text_muted']};
                        text-transform:uppercase;letter-spacing:0.06em;margin-bottom:12px">
                HMM Fraud Intelligence Platform · Report Preview
            </div>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px">
                <div style="background:white;border-radius:6px;padding:12px;border:1px solid {COLORS['border']}">
                    <div style="font-size:0.65rem;color:{COLORS['text_muted']};margin-bottom:4px">Total Volume</div>
                    <div style="font-size:1.1rem;font-weight:700;color:{COLORS['text_primary']}">${total_vol:,.0f}</div>
                </div>
                <div style="background:white;border-radius:6px;padding:12px;border:1px solid {COLORS['border']}">
                    <div style="font-size:0.65rem;color:{COLORS['text_muted']};margin-bottom:4px">Fraud Volume</div>
                    <div style="font-size:1.1rem;font-weight:700;color:{COLORS['danger']}">${fraud_vol:,.0f}</div>
                </div>
                <div style="background:white;border-radius:6px;padding:12px;border:1px solid {COLORS['border']}">
                    <div style="font-size:0.65rem;color:{COLORS['text_muted']};margin-bottom:4px">Transactions</div>
                    <div style="font-size:1.1rem;font-weight:700;color:{COLORS['text_primary']}">{total_tx:,}</div>
                </div>
                <div style="background:white;border-radius:6px;padding:12px;border:1px solid {COLORS['border']}">
                    <div style="font-size:0.65rem;color:{COLORS['text_muted']};margin-bottom:4px">Fraud Cases</div>
                    <div style="font-size:1.1rem;font-weight:700;color:{COLORS['danger']}">{total_frauds:,}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Trend chart for report
        df_sorted = df.sort_values("timestamp")
        df_sorted["day"] = df_sorted["timestamp"].dt.date
        daily = df_sorted.groupby("day").agg(
            volume=("amount","sum"),
            frauds=("is_fraud","sum")
        ).reset_index()

        fig_rep = go.Figure()
        fig_rep.add_trace(go.Bar(
            x=daily["day"], y=daily["volume"],
            name="Volume",
            marker_color=COLORS["info_bg"],
            marker_line_color=COLORS["info"],
            marker_line_width=0.8,
        ))
        fig_rep.add_trace(go.Scatter(
            x=daily["day"], y=daily["frauds"]*100000,
            name="Fraud count ×100K",
            line=dict(color=COLORS["red"], width=2),
            yaxis="y2"
        ))
        chart_defaults(fig_rep, 220)
        st.plotly_chart(fig_rep, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)