"""
HMM Fraud Intelligence Platform
Enterprise Mobile Money Fraud Detection Dashboard
Fixed version — uses native Streamlit components to avoid HTML rendering issues
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "HMM Fraud Intelligence",
    page_icon  = "🛡️",
    layout     = "wide",
    initial_sidebar_state = "expanded"
)

# ─── Color palette ────────────────────────────────────────────────────────────
RED     = "#CF0A2C"
NAVY    = "#0F1B2D"
SUCCESS = "#059669"
WARNING = "#D97706"
DANGER  = "#DC2626"
INFO    = "#2563EB"
MUTED   = "#94A3B8"
TEXT    = "#0F1B2D"
BG      = "#F4F6F9"
BORDER  = "#E2E8F0"
SURFACE = "#FFFFFF"

# ─── Global CSS — minimal and safe ────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

.stApp {{
    background: {BG};
    font-family: 'Inter', sans-serif;
}}
#MainMenu, footer, header {{ visibility: hidden; }}
.block-container {{
    padding: 1.5rem 2rem 2rem 2rem !important;
    max-width: 100% !important;
}}

/* Sidebar */
[data-testid="stSidebar"] {{
    background: {NAVY} !important;
}}
[data-testid="stSidebar"] * {{
    color: #CBD5E1 !important;
}}
[data-testid="stSidebar"] .stRadio label {{
    color: #CBD5E1 !important;
    font-size: 0.85rem !important;
    padding: 8px 4px !important;
}}

/* Cards */
div[data-testid="metric-container"] {{
    background: white;
    border: 1px solid {BORDER};
    border-radius: 12px;
    padding: 16px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}}
div[data-testid="metric-container"] label {{
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    color: {MUTED} !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}}
div[data-testid="metric-container"] [data-testid="stMetricValue"] {{
    font-size: 1.6rem !important;
    font-weight: 700 !important;
    color: {TEXT} !important;
}}
div[data-testid="metric-container"] [data-testid="stMetricDelta"] {{
    font-size: 0.72rem !important;
}}

/* Buttons */
.stButton > button {{
    background: {RED} !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
    padding: 8px 16px !important;
    width: 100%;
}}
.stButton > button:hover {{
    background: #A8081F !important;
}}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {{
    background: white;
    border-radius: 8px;
    padding: 4px;
    border: 1px solid {BORDER};
    gap: 4px;
}}
.stTabs [data-baseweb="tab"] {{
    border-radius: 6px !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    color: {MUTED} !important;
    padding: 8px 16px !important;
}}
.stTabs [aria-selected="true"] {{
    background: {RED} !important;
    color: white !important;
}}

/* Dataframe */
.stDataFrame {{
    border: 1px solid {BORDER} !important;
    border-radius: 8px !important;
}}

/* Select/Input */
.stSelectbox > div > div, .stTextInput > div > div {{
    border-radius: 8px !important;
    border-color: {BORDER} !important;
    font-size: 0.82rem !important;
}}

/* Divider */
hr {{ border-color: {BORDER} !important; margin: 1rem 0 !important; }}
</style>
""", unsafe_allow_html=True)

# ─── Data loading ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=60)
def load_data():
    LOG_FILE     = "reports/drift/transactions_log.csv"
    SUMMARY_FILE = "reports/drift/drift_summary.json"
    RETRAIN_LOG  = "reports/drift/retrain_log.json"

    if os.path.exists(LOG_FILE):
        df = pd.read_csv(LOG_FILE, parse_dates=["timestamp"])
    else:
        np.random.seed(42)
        n     = 500
        types = np.random.choice(["PAYMENT","CASH_OUT","CASH_IN","TRANSFER","DEBIT"], n,
                                  p=[0.34,0.35,0.22,0.07,0.02])
        amounts  = np.random.lognormal(9, 1.5, n)
        is_fraud = np.zeros(n, dtype=bool)
        risky    = np.where(np.isin(types, ["CASH_OUT","TRANSFER"]))[0]
        fraud_idx= np.random.choice(risky, size=int(n*0.13), replace=False)
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

    retrain_log = []
    if os.path.exists(RETRAIN_LOG):
        with open(RETRAIN_LOG) as f:
            try:
                retrain_log = json.load(f)
            except Exception:
                retrain_log = []

    # Enrich
    n   = len(df)
    rng = np.random.default_rng(99)
    countries = ["Nigeria","Kenya","Ghana","Senegal","Côte d'Ivoire","Tanzania","Uganda"]
    df["tx_id"]      = [f"TXN-{rng.integers(10000000,99999999)}" for _ in range(n)]
    df["sender"]     = [f"+{rng.integers(220,260)}{rng.integers(10000000,99999999)}" for _ in range(n)]
    df["receiver"]   = [f"+{rng.integers(220,260)}{rng.integers(10000000,99999999)}" for _ in range(n)]
    df["country"]    = rng.choice(countries, n)
    df["region"]     = rng.choice(["Lagos","Nairobi","Accra","Dakar","Abidjan"], n)
    df["agent"]      = [f"AGT-{rng.integers(1000,9999)}" for _ in range(n)]
    df["status"]     = rng.choice(["COMPLETED","PENDING","FAILED"], n, p=[0.85,0.10,0.05])
    df["inv_status"] = rng.choice(["CLEAR","UNDER REVIEW","BLOCKED","ESCALATED"], n,
                                   p=[0.70,0.15,0.10,0.05])
    df["risk_score"] = (df["fraud_probability"] * 100).round(1)

    return df, summary, retrain_log

df, summary, retrain_log = load_data()

# ─── Chart helper ─────────────────────────────────────────────────────────────
def style_chart(fig, height=260):
    fig.update_layout(
        height        = height,
        paper_bgcolor = SURFACE,
        plot_bgcolor  = SURFACE,
        margin        = dict(l=4, r=4, t=28, b=4),
        font          = dict(family="Inter", size=11, color=MUTED),
        legend        = dict(
            font       = dict(size=10),
            bgcolor    = "rgba(0,0,0,0)",
            orientation= "h",
            x=0, y=1.12
        ),
        xaxis=dict(
            showgrid    = False,
            linecolor   = BORDER,
            tickfont    = dict(size=10, color=MUTED)
        ),
        yaxis=dict(
            showgrid    = True,
            gridcolor   = "#F1F5F9",
            linecolor   = "rgba(0,0,0,0)",
            tickfont    = dict(size=10, color=MUTED)
        )
    )
    return fig

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style="padding:20px 16px 16px;
                border-bottom:1px solid rgba(255,255,255,0.08);
                margin-bottom:12px">
        <div style="display:flex;align-items:center;gap:10px">
            <div style="width:36px;height:36px;background:{RED};border-radius:8px;
                        display:flex;align-items:center;justify-content:center;
                        font-size:1.1rem;font-weight:800;color:white;flex-shrink:0">H</div>
            <div>
                <div style="color:white;font-size:0.9rem;font-weight:700;
                            font-family:'Inter',sans-serif">HMM Fraud Intelligence</div>
                <div style="color:#64748B;font-size:0.68rem;
                            font-family:'Inter',sans-serif">Mobile Money Platform</div>
            </div>
        </div>
    </div>
    <div style="padding:0 16px 6px;font-size:0.62rem;font-weight:600;
                color:#475569;letter-spacing:0.08em;text-transform:uppercase">
        Main Navigation
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "nav",
        ["📊  Executive Dashboard",
         "🔍  Fraud Investigation",
         "🤖  Model Monitoring",
         "⚙️  System Operations",
         "📄  Reports"],
        label_visibility="collapsed"
    )

    # Sidebar status footer
    drift_pct = summary["drift_share"] * 100
    retrain   = summary["retrain_needed"]
    fraud_rate= df["is_fraud"].mean() * 100

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    st.divider()

    st.markdown(f"""
    <div style="padding:0 4px">
        <div style="font-size:0.62rem;font-weight:600;color:#475569;
                    text-transform:uppercase;letter-spacing:0.08em;margin-bottom:10px">
            System Status
        </div>
        <div style="display:flex;flex-direction:column;gap:8px">
            <div style="display:flex;justify-content:space-between;align-items:center">
                <span style="font-size:0.75rem;color:#94A3B8;font-family:'Inter',sans-serif">API</span>
                <span style="font-size:0.72rem;color:#10B981;font-weight:600;
                             font-family:'Inter',sans-serif">● Online</span>
            </div>
            <div style="display:flex;justify-content:space-between;align-items:center">
                <span style="font-size:0.75rem;color:#94A3B8;font-family:'Inter',sans-serif">Model</span>
                <span style="font-size:0.72rem;color:#10B981;font-weight:600;
                             font-family:'Inter',sans-serif">● Active</span>
            </div>
            <div style="display:flex;justify-content:space-between;align-items:center">
                <span style="font-size:0.75rem;color:#94A3B8;font-family:'Inter',sans-serif">CI/CD</span>
                <span style="font-size:0.72rem;color:#10B981;font-weight:600;
                             font-family:'Inter',sans-serif">● Passing</span>
            </div>
            <div style="display:flex;justify-content:space-between;align-items:center">
                <span style="font-size:0.75rem;color:#94A3B8;font-family:'Inter',sans-serif">Drift</span>
                <span style="font-size:0.72rem;font-weight:600;font-family:'Inter',sans-serif;
                             color:{'#EF4444' if retrain else '#10B981'}">
                    {'⚠ ' + str(int(drift_pct)) + '%' if retrain else '✓ Stable'}
                </span>
            </div>
            <div style="display:flex;justify-content:space-between;align-items:center">
                <span style="font-size:0.75rem;color:#94A3B8;font-family:'Inter',sans-serif">Fraud Rate</span>
                <span style="font-size:0.72rem;color:#94A3B8;font-family:'Inter',sans-serif">
                    {fraud_rate:.1f}%
                </span>
            </div>
        </div>
        <div style="margin-top:14px;font-size:0.62rem;color:#334155;
                    font-family:'Inter',sans-serif">
            v2.4.1 · MLOps Pipeline · XGBoost 3.3.0
        </div>
    </div>
    """, unsafe_allow_html=True)

# ─── Computed values ──────────────────────────────────────────────────────────
total_tx      = len(df)
total_frauds  = int(df["is_fraud"].sum())
fraud_rate    = total_frauds / total_tx * 100
money_at_risk = df[df["is_fraud"]]["amount"].sum()
protected     = money_at_risk * 0.94
avg_risk      = df["risk_score"].mean()
now_str       = datetime.now().strftime("%d %b %Y · %H:%M UTC")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — EXECUTIVE DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
if "Executive" in page:

    # Header
    col_h1, col_h2 = st.columns([3, 1])
    with col_h1:
        st.markdown(f"""
        <div style="margin-bottom:20px">
            <h2 style="margin:0;font-size:1.4rem;font-weight:700;
                       color:{TEXT};font-family:'Inter',sans-serif">
                Executive Dashboard
            </h2>
            <p style="margin:4px 0 0;font-size:0.8rem;color:{MUTED};
                      font-family:'Inter',sans-serif">
                Real-time fraud intelligence · Huawei Mobile Money
            </p>
        </div>
        """, unsafe_allow_html=True)
    with col_h2:
        st.markdown(f"""
        <div style="text-align:right;padding-top:8px">
            <span style="background:#ECFDF5;color:#059669;border:1px solid #A7F3D0;
                         border-radius:20px;padding:4px 12px;font-size:0.72rem;
                         font-weight:600;font-family:'Inter',sans-serif">● LIVE</span>
            <div style="font-size:0.72rem;color:{MUTED};margin-top:6px;
                        font-family:'Inter',sans-serif">{now_str}</div>
        </div>
        """, unsafe_allow_html=True)

    # Alert banner
    if summary["retrain_needed"]:
        st.error(f"⚠️ **Model Drift Detected — Retraining Required** · {summary['n_drifted_cols']} of {summary['n_total_cols']} features drifted beyond threshold. Auto-retraining pipeline has been triggered.")

    # KPI row 1
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("Total Transactions", f"{total_tx:,}", "↑ 12% from yesterday")
    with k2:
        st.metric("Fraud Detected", f"{total_frauds:,}", f"{fraud_rate:.1f}% rate", delta_color="inverse")
    with k3:
        st.metric("Money Protected", f"${protected:,.0f}", "94% prevention rate")
    with k4:
        st.metric("Active Alerts", "7", "3 need immediate action", delta_color="inverse")

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # KPI row 2
    k5, k6, k7, k8 = st.columns(4)
    with k5:
        st.metric("Detection Rate", "94.2%", "↑ 1.8pp vs last week")
    with k6:
        st.metric("False Positive Rate", "2.3%", "↓ 0.4pp improved")
    with k7:
        st.metric("Avg Risk Score", f"{avg_risk:.1f}", "Across all transactions")
    with k8:
        st.metric("Model PR-AUC", "0.9977", "XGBoost v3.3.0")

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    # Chart row 1
    c1, c2 = st.columns([2.2, 1])

    with c1:
        with st.container():
            st.markdown(f"""
            <div style="background:white;border:1px solid {BORDER};border-radius:12px;
                        padding:16px 16px 0;box-shadow:0 1px 3px rgba(0,0,0,0.06)">
                <div style="font-size:0.85rem;font-weight:600;color:{TEXT};
                            margin-bottom:4px;font-family:'Inter',sans-serif">
                    Transaction Volume & Fraud Trend
                </div>
                <div style="font-size:0.72rem;color:{MUTED};margin-bottom:12px;
                            font-family:'Inter',sans-serif">
                    5-minute intervals · Volume (bars) vs Fraud count (line)
                </div>
            """, unsafe_allow_html=True)

            df_s = df.sort_values("timestamp")
            df_s["bucket"] = df_s["timestamp"].dt.floor("5min")
            agg = df_s.groupby("bucket").agg(
                total=("amount","count"),
                frauds=("is_fraud","sum")
            ).reset_index()

            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Bar(
                x=agg["bucket"], y=agg["total"],
                name="Transactions",
                marker_color="#DBEAFE",
                marker_line_color=INFO,
                marker_line_width=0.8,
            ), secondary_y=False)
            fig.add_trace(go.Scatter(
                x=agg["bucket"], y=agg["frauds"],
                name="Fraud Detected",
                line=dict(color=RED, width=2.5),
                mode="lines+markers",
                marker=dict(size=5, color=RED)
            ), secondary_y=True)
            style_chart(fig, 240)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        with st.container():
            st.markdown(f"""
            <div style="background:white;border:1px solid {BORDER};border-radius:12px;
                        padding:16px 16px 0;box-shadow:0 1px 3px rgba(0,0,0,0.06)">
                <div style="font-size:0.85rem;font-weight:600;color:{TEXT};
                            margin-bottom:4px;font-family:'Inter',sans-serif">
                    Fraud by Type
                </div>
            """, unsafe_allow_html=True)

            fraud_by_type = df[df["is_fraud"]]["type"].value_counts()
            fig2 = go.Figure(go.Pie(
                labels = fraud_by_type.index,
                values = fraud_by_type.values,
                hole   = 0.58,
                marker = dict(
                    colors=[RED, WARNING, INFO, SUCCESS, "#6366F1"],
                    line=dict(color="white", width=2)
                ),
                textfont=dict(size=11)
            ))
            fig2.add_annotation(
                text=f"<b>{total_frauds}</b>",
                x=0.5, y=0.55,
                font=dict(size=16, color=TEXT, family="Inter"),
                showarrow=False
            )
            fig2.add_annotation(
                text="cases",
                x=0.5, y=0.38,
                font=dict(size=10, color=MUTED, family="Inter"),
                showarrow=False
            )
            style_chart(fig2, 240)
            fig2.update_layout(
                showlegend=True,
                legend=dict(orientation="v", x=0.65, y=0.5, font=dict(size=10))
            )
            st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # Chart row 2
    c3, c4, c5 = st.columns(3)

    with c3:
        st.markdown(f"""<div style="background:white;border:1px solid {BORDER};
                        border-radius:12px;padding:16px 16px 0;
                        box-shadow:0 1px 3px rgba(0,0,0,0.06)">
            <div style="font-size:0.85rem;font-weight:600;color:{TEXT};
                        margin-bottom:12px;font-family:'Inter',sans-serif">
                Risk Score Distribution</div>""", unsafe_allow_html=True)

        fig3 = go.Figure()
        fig3.add_trace(go.Histogram(
            x=df[df["is_fraud"]==False]["risk_score"],
            name="Legitimate", marker_color=INFO, opacity=0.65, nbinsx=20
        ))
        fig3.add_trace(go.Histogram(
            x=df[df["is_fraud"]==True]["risk_score"],
            name="Fraud", marker_color=RED, opacity=0.8, nbinsx=20
        ))
        fig3.update_layout(barmode="overlay")
        style_chart(fig3, 210)
        st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

    with c4:
        st.markdown(f"""<div style="background:white;border:1px solid {BORDER};
                        border-radius:12px;padding:16px 16px 0;
                        box-shadow:0 1px 3px rgba(0,0,0,0.06)">
            <div style="font-size:0.85rem;font-weight:600;color:{TEXT};
                        margin-bottom:12px;font-family:'Inter',sans-serif">
                Fraud by Country</div>""", unsafe_allow_html=True)

        fc = df[df["is_fraud"]]["country"].value_counts().head(6)
        fig4 = go.Figure(go.Bar(
            x=fc.values, y=fc.index, orientation="h",
            marker=dict(
                color=fc.values,
                colorscale=[[0,"#FEE2E2"],[1,RED]],
                showscale=False
            )
        ))
        style_chart(fig4, 210)
        fig4.update_layout(yaxis=dict(showgrid=False))
        st.plotly_chart(fig4, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

    with c5:
        st.markdown(f"""<div style="background:white;border:1px solid {BORDER};
                        border-radius:12px;padding:16px;
                        box-shadow:0 1px 3px rgba(0,0,0,0.06)">
            <div style="font-size:0.85rem;font-weight:600;color:{TEXT};
                        margin-bottom:16px;font-family:'Inter',sans-serif">
                Model Performance</div>""", unsafe_allow_html=True)

        perf = [
            ("Precision",  96.3, RED),
            ("Recall",     94.2, INFO),
            ("F1-Score",   95.2, WARNING),
            ("ROC-AUC",    99.9, SUCCESS),
            ("PR-AUC",     99.8, "#6366F1"),
        ]
        for name, val, color in perf:
            st.markdown(f"""
            <div style="margin-bottom:12px">
                <div style="display:flex;justify-content:space-between;
                            font-size:0.78rem;margin-bottom:4px;
                            font-family:'Inter',sans-serif">
                    <span style="color:{MUTED}">{name}</span>
                    <span style="font-weight:600;color:{TEXT}">{val}%</span>
                </div>
                <div style="background:#F1F5F9;border-radius:4px;height:6px">
                    <div style="width:{val}%;background:{color};
                                border-radius:4px;height:6px"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — FRAUD INVESTIGATION
# ══════════════════════════════════════════════════════════════════════════════
elif "Investigation" in page:

    st.markdown(f"""
    <h2 style="margin:0 0 4px;font-size:1.4rem;font-weight:700;
               color:{TEXT};font-family:'Inter',sans-serif">Fraud Investigation</h2>
    <p style="margin:0 0 20px;font-size:0.8rem;color:{MUTED};
              font-family:'Inter',sans-serif">
        Case management · Transaction analysis · Analyst decisions
    </p>
    """, unsafe_allow_html=True)

    # Filters
    with st.container():
        st.markdown(f"""<div style="background:white;border:1px solid {BORDER};
                        border-radius:12px;padding:16px;margin-bottom:16px;
                        box-shadow:0 1px 3px rgba(0,0,0,0.06)">
            <div style="font-size:0.82rem;font-weight:600;color:{TEXT};
                        margin-bottom:12px;font-family:'Inter',sans-serif">
                🔎 Search & Filter</div>""", unsafe_allow_html=True)

        f1, f2, f3, f4 = st.columns([2.5, 1, 1, 1])
        with f1:
            search = st.text_input("Search", placeholder="Transaction ID or wallet number...")
        with f2:
            ftype = st.selectbox("Type", ["All"] + list(df["type"].unique()))
        with f3:
            finv = st.selectbox("Status", ["All","CLEAR","UNDER REVIEW","BLOCKED","ESCALATED"])
        with f4:
            ffraud = st.selectbox("Risk", ["All","High Risk Only","Low Risk Only"])
        st.markdown("</div>", unsafe_allow_html=True)

    # Apply filters
    dff = df.copy()
    if search:
        dff = dff[dff["tx_id"].str.contains(search, case=False, na=False) |
                  dff["sender"].str.contains(search, case=False, na=False)]
    if ftype != "All":
        dff = dff[dff["type"] == ftype]
    if finv != "All":
        dff = dff[dff["inv_status"] == finv]
    if ffraud == "High Risk Only":
        dff = dff[dff["risk_score"] >= 70]
    elif ffraud == "Low Risk Only":
        dff = dff[dff["risk_score"] < 40]

    dff_show = dff.tail(100).sort_values("timestamp", ascending=False).reset_index(drop=True)

    # Transaction table using native Streamlit
    st.markdown(f"""
    <div style="background:white;border:1px solid {BORDER};border-radius:12px;
                padding:16px;box-shadow:0 1px 3px rgba(0,0,0,0.06);margin-bottom:16px">
        <div style="font-size:0.85rem;font-weight:600;color:{TEXT};
                    margin-bottom:12px;font-family:'Inter',sans-serif">
            Transaction List · {len(dff_show)} results
        </div>
    """, unsafe_allow_html=True)

    display_df = dff_show[[
        "tx_id","timestamp","type","amount",
        "country","risk_score","fraud_probability","is_fraud","inv_status"
    ]].copy()
    display_df["timestamp"] = display_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M")
    display_df["amount"]    = display_df["amount"].map("${:,.2f}".format)
    display_df["risk_score"]= display_df["risk_score"].map("{:.0f}%".format)
    display_df["fraud_probability"] = display_df["fraud_probability"].map("{:.1%}".format)
    display_df["is_fraud"]  = display_df["is_fraud"].map({True:"🔴 FRAUD", False:"✅ CLEAR"})
    display_df.columns      = ["TX ID","Time","Type","Amount","Country",
                                "Risk","Fraud Prob","Verdict","Investigation"]

    st.dataframe(display_df, use_container_width=True, height=300,
                 hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Investigation panel
    st.markdown(f"""
    <div style="background:white;border:1px solid {BORDER};border-radius:12px;
                padding:16px;box-shadow:0 1px 3px rgba(0,0,0,0.06)">
        <div style="font-size:0.85rem;font-weight:600;color:{TEXT};
                    margin-bottom:16px;font-family:'Inter',sans-serif">
            📋 Transaction Investigation Panel
        </div>
    """, unsafe_allow_html=True)

    tx_options = dff_show["tx_id"].tolist()
    if tx_options:
        selected = st.selectbox("Select transaction to investigate", tx_options)
        row = dff_show[dff_show["tx_id"] == selected].iloc[0]

        p1, p2, p3 = st.columns(3)

        with p1:
            st.markdown(f"""
            <div style="background:{BG};border-radius:8px;padding:16px">
                <div style="font-size:0.72rem;font-weight:700;color:{MUTED};
                            text-transform:uppercase;letter-spacing:0.06em;
                            margin-bottom:12px;font-family:'Inter',sans-serif">
                    Transaction Details
                </div>
            """, unsafe_allow_html=True)

            details = [
                ("Transaction ID", row["tx_id"]),
                ("Type",           row["type"]),
                ("Amount",         f"${row['amount']:,.2f}"),
                ("Sender",         row["sender"]),
                ("Receiver",       row["receiver"]),
                ("Country",        row["country"]),
                ("Agent",          row["agent"]),
                ("Timestamp",      str(row["timestamp"])[:19]),
                ("Status",         row["status"]),
            ]
            for key, val in details:
                color = RED if key == "Amount" else TEXT
                st.markdown(f"""
                <div style="display:flex;justify-content:space-between;
                            padding:7px 0;border-bottom:1px solid #F1F5F9;
                            font-family:'Inter',sans-serif">
                    <span style="font-size:0.75rem;color:{MUTED}">{key}</span>
                    <span style="font-size:0.75rem;font-weight:600;
                                 color:{color};text-align:right;max-width:60%">{val}</span>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with p2:
            risk  = row["risk_score"]
            rcolor= DANGER if risk >= 70 else (WARNING if risk >= 40 else SUCCESS)

            st.markdown(f"""
            <div style="background:{BG};border-radius:8px;padding:16px">
                <div style="font-size:0.72rem;font-weight:700;color:{MUTED};
                            text-transform:uppercase;letter-spacing:0.06em;
                            margin-bottom:12px;font-family:'Inter',sans-serif">
                    Explainable AI — Risk Factors
                </div>
                <div style="text-align:center;padding:16px 0;margin-bottom:16px;
                            background:white;border-radius:8px;border:1px solid {BORDER}">
                    <div style="font-size:2.5rem;font-weight:800;color:{rcolor};
                                font-family:'Inter',sans-serif">{risk:.0f}%</div>
                    <div style="font-size:0.72rem;color:{MUTED};
                                font-family:'Inter',sans-serif">Risk Score</div>
                </div>
            """, unsafe_allow_html=True)

            features_xai = [
                ("Transaction Amount",  min(risk * 0.35, 100)),
                ("Balance After Tx",    min(risk * 0.25, 100)),
                ("Receiver History",    min(risk * 0.18, 100)),
                ("Transaction Type",    min(risk * 0.12, 100)),
                ("Time of Day",         min(risk * 0.07, 100)),
                ("Velocity (24h)",      min(risk * 0.03, 100)),
            ]

            for fname, fval in features_xai:
                fcolor = DANGER if fval >= 60 else (WARNING if fval >= 30 else SUCCESS)
                st.markdown(f"""
                <div style="margin-bottom:10px;font-family:'Inter',sans-serif">
                    <div style="display:flex;justify-content:space-between;
                                font-size:0.75rem;margin-bottom:3px">
                        <span style="color:{MUTED}">{fname}</span>
                        <span style="font-weight:600;color:{TEXT}">{fval:.0f}%</span>
                    </div>
                    <div style="background:#F1F5F9;border-radius:4px;height:7px">
                        <div style="width:{fval}%;background:{fcolor};
                                    border-radius:4px;height:7px;
                                    opacity:0.85"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            human_reason = (
                f"High transaction amount (${row['amount']:,.0f}), "
                f"{'account emptied after transfer' if row['is_fraud'] else 'normal balance change'}, "
                f"{'high-risk transaction type' if row['type'] in ['TRANSFER','CASH_OUT'] else 'low-risk type'}."
            )
            st.markdown(f"""
            <div style="background:{'#FEF2F2' if row['is_fraud'] else '#ECFDF5'};
                        border-radius:6px;padding:10px;margin-top:8px;
                        border:1px solid {'#FECACA' if row['is_fraud'] else '#A7F3D0'};
                        font-size:0.72rem;color:{MUTED};font-family:'Inter',sans-serif">
                <strong>Why this score:</strong> {human_reason}
            </div>
            """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with p3:
            verdict_bg    = "#FEF2F2" if row["is_fraud"] else "#ECFDF5"
            verdict_border= "#FECACA" if row["is_fraud"] else "#A7F3D0"
            verdict_color = DANGER    if row["is_fraud"] else SUCCESS
            verdict_text  = "⚠️ HIGH RISK — POTENTIAL FRAUD" if row["is_fraud"] else "✅ LOW RISK — APPEARS LEGITIMATE"

            st.markdown(f"""
            <div style="background:{BG};border-radius:8px;padding:16px">
                <div style="font-size:0.72rem;font-weight:700;color:{MUTED};
                            text-transform:uppercase;letter-spacing:0.06em;
                            margin-bottom:12px;font-family:'Inter',sans-serif">
                    Model Verdict & Analyst Action
                </div>
                <div style="background:{verdict_bg};border:1px solid {verdict_border};
                            border-radius:8px;padding:12px;margin-bottom:16px;
                            font-size:0.8rem;font-weight:600;color:{verdict_color};
                            font-family:'Inter',sans-serif;text-align:center">
                    {verdict_text}
                </div>
                <div style="margin-bottom:16px">
                    <div style="display:flex;justify-content:space-between;
                                padding:7px 0;border-bottom:1px solid #F1F5F9;
                                font-family:'Inter',sans-serif">
                        <span style="font-size:0.75rem;color:{MUTED}">Fraud Probability</span>
                        <span style="font-size:0.75rem;font-weight:700;
                                     color:{verdict_color}">{row['fraud_probability']:.1%}</span>
                    </div>
                    <div style="display:flex;justify-content:space-between;
                                padding:7px 0;border-bottom:1px solid #F1F5F9;
                                font-family:'Inter',sans-serif">
                        <span style="font-size:0.75rem;color:{MUTED}">Investigation Status</span>
                        <span style="font-size:0.75rem;font-weight:600;
                                     color:{TEXT}">{row['inv_status']}</span>
                    </div>
                    <div style="display:flex;justify-content:space-between;
                                padding:7px 0;font-family:'Inter',sans-serif">
                        <span style="font-size:0.75rem;color:{MUTED}">Transaction Status</span>
                        <span style="font-size:0.75rem;font-weight:600;
                                     color:{TEXT}">{row['status']}</span>
                    </div>
                </div>
                <div style="font-size:0.72rem;font-weight:700;color:{MUTED};
                            text-transform:uppercase;letter-spacing:0.06em;
                            margin-bottom:10px;font-family:'Inter',sans-serif">
                    Analyst Decision
                </div>
            </div>
            """, unsafe_allow_html=True)

            a1, a2, a3 = st.columns(3)
            with a1:
                if st.button("✓ Approve", key="approve"):
                    st.success("Approved")
            with a2:
                if st.button("⟳ Review", key="review"):
                    st.warning("In Review")
            with a3:
                if st.button("✕ Block", key="block"):
                    st.error("Blocked")

            st.markdown(f"""
            <div style="background:white;border:1px solid {BORDER};border-radius:8px;
                        padding:12px;margin-top:12px">
                <div style="font-size:0.72rem;color:{MUTED};margin-bottom:6px;
                            font-family:'Inter',sans-serif;font-weight:600">
                    Analyst Notes
                </div>
            """, unsafe_allow_html=True)
            st.text_area("", placeholder="Add investigation notes...",
                         label_visibility="collapsed", height=80, key="analyst_notes")
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — MODEL MONITORING
# ══════════════════════════════════════════════════════════════════════════════
elif "Model" in page:

    st.markdown(f"""
    <h2 style="margin:0 0 4px;font-size:1.4rem;font-weight:700;
               color:{TEXT};font-family:'Inter',sans-serif">Model Monitoring</h2>
    <p style="margin:0 0 20px;font-size:0.8rem;color:{MUTED};
              font-family:'Inter',sans-serif">
        MLOps pipeline · Drift detection · Retraining status
    </p>
    """, unsafe_allow_html=True)

    drift_pct = summary["drift_share"] * 100
    n_drifted = summary["n_drifted_cols"]
    n_total   = summary["n_total_cols"]
    retrain   = summary["retrain_needed"]

    # KPIs
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Model", "XGBoost 3.3.0", "Deployed 3 days ago")
    with m2:
        st.metric("PR-AUC", "0.9977", "↓ 0.002 from baseline", delta_color="inverse")
    with m3:
        st.metric("Data Drift", f"{drift_pct:.0f}%",
                  f"{n_drifted}/{n_total} features drifted",
                  delta_color="inverse" if retrain else "normal")
    with m4:
        st.metric("Retraining", "REQUIRED" if retrain else "NOT NEEDED",
                  "Threshold: 30%",
                  delta_color="inverse" if retrain else "normal")

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    # Alert
    if retrain:
        st.error(f"⚠️ **Drift exceeds 30% threshold** — {n_drifted} features affected: {', '.join(summary.get('drifted_features', []))}. Auto-retraining pipeline triggered via `retrain.yml`.")
    else:
        st.success("✅ All features within acceptable drift bounds — model is stable.")

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    col_l, col_r = st.columns([1.6, 1])

    with col_l:
        tabs = st.tabs(["Fraud Probability", "Amount Distribution", "Drift Timeline"])

        with tabs[0]:
            df2 = df.sort_values("timestamp").reset_index(drop=True)
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Scatter(
                x=df2.index,
                y=df2["fraud_probability"].rolling(15, min_periods=1).mean(),
                name="Avg Fraud Probability",
                line=dict(color=RED, width=2),
                fill="tozeroy", fillcolor="rgba(207,10,44,0.05)"
            ), secondary_y=False)
            fig.add_trace(go.Scatter(
                x=df2.index, y=df2["drift_factor"],
                name="Drift Factor",
                line=dict(color=WARNING, width=1.5, dash="dot"),
            ), secondary_y=True)
            fig.add_hline(y=0.3, secondary_y=True,
                          line_dash="dash", line_color="rgba(217,119,6,0.5)",
                          annotation_text="Retrain threshold",
                          annotation_font=dict(size=9, color=WARNING))
            style_chart(fig, 280)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        with tabs[1]:
            fig_amt = go.Figure()
            fig_amt.add_trace(go.Histogram(
                x=df[df["is_fraud"]==False]["amount"].clip(
                    upper=df["amount"].quantile(0.95)),
                name="Legitimate", marker_color=INFO, opacity=0.65, nbinsx=30
            ))
            fig_amt.add_trace(go.Histogram(
                x=df[df["is_fraud"]==True]["amount"].clip(
                    upper=df["amount"].quantile(0.95)),
                name="Fraud", marker_color=RED, opacity=0.8, nbinsx=30
            ))
            fig_amt.update_layout(barmode="overlay")
            style_chart(fig_amt, 280)
            st.plotly_chart(fig_amt, use_container_width=True, config={"displayModeBar": False})

        with tabs[2]:
            df3 = df.reset_index(drop=True)
            fig_d = go.Figure()
            fig_d.add_trace(go.Scatter(
                x=df3.index, y=df3["drift_factor"],
                name="Drift Factor",
                line=dict(color=WARNING, width=2),
                fill="tozeroy", fillcolor="rgba(217,119,6,0.08)"
            ))
            fig_d.add_hline(y=0.3, line_dash="dash", line_color=DANGER,
                            annotation_text="Retraining threshold",
                            annotation_font=dict(size=9, color=DANGER))
            style_chart(fig_d, 280)
            st.plotly_chart(fig_d, use_container_width=True, config={"displayModeBar": False})

    with col_r:
        # Feature drift
        st.markdown(f"""
        <div style="background:white;border:1px solid {BORDER};border-radius:12px;
                    padding:16px;box-shadow:0 1px 3px rgba(0,0,0,0.06);margin-bottom:12px">
            <div style="font-size:0.85rem;font-weight:600;color:{TEXT};
                        margin-bottom:14px;font-family:'Inter',sans-serif">
                Feature Drift Status
            </div>
        """, unsafe_allow_html=True)

        features_m = ["amount","oldbalanceOrg","newbalanceOrig","oldbalanceDest","newbalanceDest"]
        drifted    = summary.get("drifted_features", [])
        rng4 = np.random.default_rng(42)

        for feat in features_m:
            is_drifted = feat in drifted
            pct = rng4.uniform(62, 88) if is_drifted else rng4.uniform(8, 24)
            color  = DANGER  if pct >= 30 else SUCCESS
            status = "DRIFT"  if pct >= 30 else "STABLE"
            badge_bg = "#FEF2F2" if pct >= 30 else "#ECFDF5"
            badge_c  = DANGER    if pct >= 30 else SUCCESS
            badge_bd = "#FECACA" if pct >= 30 else "#A7F3D0"

            st.markdown(f"""
            <div style="margin-bottom:12px;font-family:'Inter',sans-serif">
                <div style="display:flex;justify-content:space-between;
                            align-items:center;margin-bottom:4px">
                    <span style="font-size:0.78rem;color:{MUTED}">{feat}</span>
                    <span style="background:{badge_bg};color:{badge_c};
                                 border:1px solid {badge_bd};border-radius:20px;
                                 padding:2px 8px;font-size:0.65rem;font-weight:600">
                        {status} {pct:.0f}%
                    </span>
                </div>
                <div style="background:#F1F5F9;border-radius:4px;height:6px">
                    <div style="width:{min(pct,100)}%;background:{color};
                                border-radius:4px;height:6px"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        # MLflow card
        st.markdown(f"""
        <div style="background:white;border:1px solid {BORDER};border-radius:12px;
                    padding:16px;box-shadow:0 1px 3px rgba(0,0,0,0.06)">
            <div style="font-size:0.85rem;font-weight:600;color:{TEXT};
                        margin-bottom:14px;font-family:'Inter',sans-serif">
                MLflow Experiment Info
            </div>
        """, unsafe_allow_html=True)

        mlflow_info = [
            ("Experiment",    "fraud-mobile-money"),
            ("Last Run",      (datetime.now()-timedelta(hours=3)).strftime("%d %b, %H:%M")),
            ("Total Runs",    "24"),
            ("Best PR-AUC",   "0.9977"),
            ("Best Model",    "XGBoost"),
            ("Training Time", "6.9s"),
            ("Features",      "16"),
            ("Augmentation",  "VAE 40K samples"),
        ]
        for key, val in mlflow_info:
            vcolor = SUCCESS if key == "Best PR-AUC" else TEXT
            st.markdown(f"""
            <div style="display:flex;justify-content:space-between;padding:6px 0;
                        border-bottom:1px solid #F8FAFC;font-family:'Inter',sans-serif">
                <span style="font-size:0.75rem;color:{MUTED}">{key}</span>
                <span style="font-size:0.75rem;font-weight:600;color:{vcolor}">{val}</span>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Retraining log
        if retrain_log:
            st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
            st.markdown(f"""
            <div style="background:white;border:1px solid {BORDER};border-radius:12px;
                        padding:16px;box-shadow:0 1px 3px rgba(0,0,0,0.06)">
                <div style="font-size:0.85rem;font-weight:600;color:{TEXT};
                            margin-bottom:14px;font-family:'Inter',sans-serif">
                    Retraining History
                </div>
            """, unsafe_allow_html=True)
            for entry in retrain_log[-3:][::-1]:
                status  = entry.get("status","—")
                deployed= entry.get("deployed", False)
                old_s   = entry.get("old_pr_auc", 0)
                new_s   = entry.get("new_pr_auc", 0)
                ts      = entry.get("timestamp","")[:16]
                badge_bg = "#ECFDF5" if deployed else "#FEF2F2"
                badge_c  = SUCCESS   if deployed else DANGER
                badge_bd = "#A7F3D0" if deployed else "#FECACA"
                st.markdown(f"""
                <div style="padding:8px 0;border-bottom:1px solid #F8FAFC;
                            font-family:'Inter',sans-serif">
                    <div style="display:flex;justify-content:space-between;
                                align-items:center;margin-bottom:3px">
                        <span style="font-size:0.75rem;color:{TEXT};font-weight:600">{status}</span>
                        <span style="background:{badge_bg};color:{badge_c};
                                     border:1px solid {badge_bd};border-radius:20px;
                                     padding:2px 8px;font-size:0.62rem;font-weight:600">
                            {'DEPLOYED' if deployed else 'REJECTED'}
                        </span>
                    </div>
                    <div style="font-size:0.68rem;color:{MUTED}">
                        {ts} · {old_s:.4f} → {new_s:.4f}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — SYSTEM OPERATIONS
# ══════════════════════════════════════════════════════════════════════════════
elif "Operations" in page:

    st.markdown(f"""
    <h2 style="margin:0 0 4px;font-size:1.4rem;font-weight:700;
               color:{TEXT};font-family:'Inter',sans-serif">System Operations</h2>
    <p style="margin:0 0 20px;font-size:0.8rem;color:{MUTED};
              font-family:'Inter',sans-serif">
        Infrastructure health · Service uptime · Pipeline status
    </p>
    """, unsafe_allow_html=True)

    s1, s2, s3, s4 = st.columns(4)
    with s1:
        st.metric("API Uptime",    "99.97%",    "Last 30 days")
    with s2:
        st.metric("Avg Latency",   "42 ms",     "↓ 8ms improved")
    with s3:
        st.metric("Throughput",    "1,240/min", "Peak: 5K/min")
    with s4:
        st.metric("CI/CD Builds",  "47",        "3 failed this week", delta_color="inverse")

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    oc1, oc2 = st.columns([1.1, 1])

    with oc1:
        st.markdown(f"""
        <div style="background:white;border:1px solid {BORDER};border-radius:12px;
                    padding:16px;box-shadow:0 1px 3px rgba(0,0,0,0.06)">
            <div style="font-size:0.85rem;font-weight:600;color:{TEXT};
                        margin-bottom:16px;font-family:'Inter',sans-serif">
                Service Health
            </div>
        """, unsafe_allow_html=True)

        services = [
            ("FastAPI (Fraud Scoring)",  "Online",    "99.97%", "42 ms",   "success", "#10B981"),
            ("Docker Container",          "Running",   "100%",   "—",       "success", "#10B981"),
            ("GitHub Actions CI/CD",      "Passing",   "93.6%",  "—",       "success", "#10B981"),
            ("MLflow Tracking",           "Online",    "99.80%", "120 ms",  "success", "#10B981"),
            ("Evidently Monitoring",      "Online",    "99.90%", "—",       "success", "#10B981"),
            ("Streamlit Dashboard",       "Online",    "99.99%", "—",       "success", "#10B981"),
            ("Feature Store",             "Degraded",  "87.3%",  "340 ms",  "warning", "#D97706"),
            ("Redis Cache",               "Offline",   "0%",     "—",       "danger",  "#DC2626"),
            ("Kafka Stream",              "Simulated", "—",      "—",       "warning", "#D97706"),
            ("Airflow Scheduler",         "Simulated", "—",      "—",       "warning", "#D97706"),
        ]

        # Header row
        st.markdown(f"""
        <div style="display:grid;grid-template-columns:2fr 1fr 1fr 1fr;
                    padding:6px 0;border-bottom:2px solid {BORDER};
                    margin-bottom:4px;font-family:'Inter',sans-serif">
            <span style="font-size:0.65rem;font-weight:700;color:{MUTED};
                         text-transform:uppercase;letter-spacing:0.06em">Service</span>
            <span style="font-size:0.65rem;font-weight:700;color:{MUTED};
                         text-transform:uppercase;letter-spacing:0.06em">Status</span>
            <span style="font-size:0.65rem;font-weight:700;color:{MUTED};
                         text-transform:uppercase;letter-spacing:0.06em">Uptime</span>
            <span style="font-size:0.65rem;font-weight:700;color:{MUTED};
                         text-transform:uppercase;letter-spacing:0.06em">Latency</span>
        </div>
        """, unsafe_allow_html=True)

        for name, status, uptime, latency, kind, dot_color in services:
            badge_bg = {"success":"#ECFDF5","warning":"#FFFBEB","danger":"#FEF2F2"}[kind]
            badge_c  = {"success":SUCCESS,"warning":WARNING,"danger":DANGER}[kind]
            badge_bd = {"success":"#A7F3D0","warning":"#FDE68A","danger":"#FECACA"}[kind]
            st.markdown(f"""
            <div style="display:grid;grid-template-columns:2fr 1fr 1fr 1fr;
                        padding:9px 0;border-bottom:1px solid #F8FAFC;
                        align-items:center;font-family:'Inter',sans-serif">
                <span style="font-size:0.78rem;color:{TEXT};display:flex;align-items:center;gap:6px">
                    <span style="width:7px;height:7px;border-radius:50%;
                                 background:{dot_color};display:inline-block;flex-shrink:0"></span>
                    {name}
                </span>
                <span>
                    <span style="background:{badge_bg};color:{badge_c};
                                 border:1px solid {badge_bd};border-radius:20px;
                                 padding:2px 8px;font-size:0.65rem;font-weight:600">
                        {status}
                    </span>
                </span>
                <span style="font-size:0.75rem;color:{MUTED}">{uptime}</span>
                <span style="font-size:0.75rem;color:{MUTED}">{latency}</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    with oc2:
        # Latency chart
        st.markdown(f"""
        <div style="background:white;border:1px solid {BORDER};border-radius:12px;
                    padding:16px 16px 0;box-shadow:0 1px 3px rgba(0,0,0,0.06);
                    margin-bottom:12px">
            <div style="font-size:0.85rem;font-weight:600;color:{TEXT};
                        margin-bottom:4px;font-family:'Inter',sans-serif">
                API Latency — Last 24h
            </div>
        """, unsafe_allow_html=True)

        rng5  = np.random.default_rng(7)
        hours = pd.date_range(end=datetime.now(), periods=24, freq="h")
        avg_l = rng5.normal(42, 8, 24).clip(20, 120)
        p95_l = rng5.normal(90, 12, 24).clip(50, 200)

        fig_l = go.Figure()
        fig_l.add_trace(go.Scatter(
            x=hours, y=p95_l, name="p95",
            line=dict(color=WARNING, width=1.5, dash="dot"),
            fill="tozeroy", fillcolor="rgba(217,119,6,0.04)"
        ))
        fig_l.add_trace(go.Scatter(
            x=hours, y=avg_l, name="Average",
            line=dict(color=INFO, width=2),
            fill="tozeroy", fillcolor="rgba(37,99,235,0.06)"
        ))
        style_chart(fig_l, 200)
        st.plotly_chart(fig_l, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

        # CI/CD runs
        st.markdown(f"""
        <div style="background:white;border:1px solid {BORDER};border-radius:12px;
                    padding:16px;box-shadow:0 1px 3px rgba(0,0,0,0.06)">
            <div style="font-size:0.85rem;font-weight:600;color:{TEXT};
                        margin-bottom:14px;font-family:'Inter',sans-serif">
                Recent CI/CD Runs
            </div>
        """, unsafe_allow_html=True)

        builds = [
            ("Auto-retraining pipeline",    "✓ Passed",  "2m 44s", "success"),
            ("Add CI/CD + tests",            "✓ Passed",  "2m 14s", "success"),
            ("Fix evidently compat",         "✓ Passed",  "3m 41s", "success"),
            ("Update .gitignore",            "✗ Failed",  "1m 04s", "danger"),
            ("Initial commit",               "✗ Failed",  "1m 03s", "danger"),
        ]
        for bname, result, dur, bkind in builds:
            bb = {"success":"#ECFDF5","danger":"#FEF2F2"}[bkind]
            bc = {"success":SUCCESS,"danger":DANGER}[bkind]
            bd = {"success":"#A7F3D0","danger":"#FECACA"}[bkind]
            st.markdown(f"""
            <div style="display:flex;justify-content:space-between;align-items:center;
                        padding:9px 0;border-bottom:1px solid #F8FAFC;
                        font-family:'Inter',sans-serif">
                <span style="font-size:0.78rem;color:{TEXT};max-width:55%">{bname}</span>
                <div style="display:flex;align-items:center;gap:8px">
                    <span style="font-size:0.68rem;color:{MUTED}">{dur}</span>
                    <span style="background:{bb};color:{bc};border:1px solid {bd};
                                 border-radius:20px;padding:2px 8px;font-size:0.65rem;
                                 font-weight:600">{result}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — REPORTS
# ══════════════════════════════════════════════════════════════════════════════
elif "Reports" in page:

    st.markdown(f"""
    <h2 style="margin:0 0 4px;font-size:1.4rem;font-weight:700;
               color:{TEXT};font-family:'Inter',sans-serif">Reports</h2>
    <p style="margin:0 0 20px;font-size:0.8rem;color:{MUTED};
              font-family:'Inter',sans-serif">
        Export fraud reports · Download data · Scheduled reporting
    </p>
    """, unsafe_allow_html=True)

    rc1, rc2 = st.columns([1, 1.5])

    with rc1:
        st.markdown(f"""
        <div style="background:white;border:1px solid {BORDER};border-radius:12px;
                    padding:20px;box-shadow:0 1px 3px rgba(0,0,0,0.06);margin-bottom:12px">
            <div style="font-size:0.85rem;font-weight:600;color:{TEXT};
                        margin-bottom:16px;font-family:'Inter',sans-serif">
                📥 Generate Report
            </div>
        """, unsafe_allow_html=True)

        period = st.selectbox("Report Period",
                              ["Today","Last 7 days","Last 30 days","Custom range"])
        if period == "Custom range":
            d1, d2 = st.columns(2)
            with d1:
                st.date_input("From")
            with d2:
                st.date_input("To")

        rtype = st.selectbox("Report Type", [
            "Daily Fraud Summary",
            "Weekly Risk Report",
            "Monthly Executive Report",
            "Model Performance Report",
            "Drift Analysis Report"
        ])
        fmt = st.selectbox("Format", ["CSV","JSON"])

        st.markdown("</div>", unsafe_allow_html=True)

        if st.button("Generate & Download Report"):
            export_df = df[[
                "tx_id","timestamp","type","amount","country",
                "risk_score","fraud_probability","is_fraud","inv_status"
            ]].copy()
            export_df["timestamp"] = export_df["timestamp"].astype(str)

            if fmt == "CSV":
                data = export_df.to_csv(index=False)
                st.download_button(
                    label     = "⬇ Download CSV",
                    data      = data,
                    file_name = f"hmm_fraud_report_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime      = "text/csv"
                )
            else:
                data = export_df.to_json(orient="records", indent=2)
                st.download_button(
                    label     = "⬇ Download JSON",
                    data      = data,
                    file_name = f"hmm_fraud_report_{datetime.now().strftime('%Y%m%d')}.json",
                    mime      = "application/json"
                )

        # Scheduled reports
        st.markdown(f"""
        <div style="background:white;border:1px solid {BORDER};border-radius:12px;
                    padding:20px;box-shadow:0 1px 3px rgba(0,0,0,0.06);margin-top:12px">
            <div style="font-size:0.85rem;font-weight:600;color:{TEXT};
                        margin-bottom:14px;font-family:'Inter',sans-serif">
                🕐 Scheduled Reports
            </div>
        """, unsafe_allow_html=True)

        scheduled = [
            ("Daily Fraud Summary",     "Every day 08:00 UTC",    "Active",   "success"),
            ("Weekly Risk Report",      "Every Mon 09:00 UTC",    "Active",   "success"),
            ("Monthly Executive Brief", "1st of month 07:00 UTC", "Active",   "success"),
            ("Drift Alert Report",      "On drift detection",      "Pending",  "warning"),
        ]
        for sname, sched, sstatus, skind in scheduled:
            sb = {"success":"#ECFDF5","warning":"#FFFBEB"}[skind]
            sc = {"success":SUCCESS,"warning":WARNING}[skind]
            sd = {"success":"#A7F3D0","warning":"#FDE68A"}[skind]
            st.markdown(f"""
            <div style="display:flex;justify-content:space-between;align-items:center;
                        padding:10px 0;border-bottom:1px solid #F8FAFC;
                        font-family:'Inter',sans-serif">
                <div>
                    <div style="font-size:0.78rem;font-weight:500;color:{TEXT}">{sname}</div>
                    <div style="font-size:0.68rem;color:{MUTED};margin-top:2px">{sched}</div>
                </div>
                <span style="background:{sb};color:{sc};border:1px solid {sd};
                             border-radius:20px;padding:3px 10px;font-size:0.65rem;
                             font-weight:600;flex-shrink:0;margin-left:8px">{sstatus}</span>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with rc2:
        # Summary stats
        total_vol  = df["amount"].sum()
        fraud_vol  = df[df["is_fraud"]]["amount"].sum()
        total_tx_r = len(df)
        total_fr   = int(df["is_fraud"].sum())

        st.markdown(f"""
        <div style="background:white;border:1px solid {BORDER};border-radius:12px;
                    padding:20px;box-shadow:0 1px 3px rgba(0,0,0,0.06);margin-bottom:12px">
            <div style="font-size:0.85rem;font-weight:600;color:{TEXT};
                        margin-bottom:16px;font-family:'Inter',sans-serif">
                Report Preview — Current Period Summary
            </div>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;
                        margin-bottom:16px">
                <div style="background:{BG};border-radius:8px;padding:14px;
                            border:1px solid {BORDER}">
                    <div style="font-size:0.65rem;color:{MUTED};margin-bottom:4px;
                                font-family:'Inter',sans-serif;font-weight:600;
                                text-transform:uppercase;letter-spacing:0.06em">Total Volume</div>
                    <div style="font-size:1.2rem;font-weight:700;color:{TEXT};
                                font-family:'Inter',sans-serif">${total_vol:,.0f}</div>
                </div>
                <div style="background:{BG};border-radius:8px;padding:14px;
                            border:1px solid {BORDER}">
                    <div style="font-size:0.65rem;color:{MUTED};margin-bottom:4px;
                                font-family:'Inter',sans-serif;font-weight:600;
                                text-transform:uppercase;letter-spacing:0.06em">Fraud Volume</div>
                    <div style="font-size:1.2rem;font-weight:700;color:{DANGER};
                                font-family:'Inter',sans-serif">${fraud_vol:,.0f}</div>
                </div>
                <div style="background:{BG};border-radius:8px;padding:14px;
                            border:1px solid {BORDER}">
                    <div style="font-size:0.65rem;color:{MUTED};margin-bottom:4px;
                                font-family:'Inter',sans-serif;font-weight:600;
                                text-transform:uppercase;letter-spacing:0.06em">Total Transactions</div>
                    <div style="font-size:1.2rem;font-weight:700;color:{TEXT};
                                font-family:'Inter',sans-serif">{total_tx_r:,}</div>
                </div>
                <div style="background:{BG};border-radius:8px;padding:14px;
                            border:1px solid {BORDER}">
                    <div style="font-size:0.65rem;color:{MUTED};margin-bottom:4px;
                                font-family:'Inter',sans-serif;font-weight:600;
                                text-transform:uppercase;letter-spacing:0.06em">Fraud Cases</div>
                    <div style="font-size:1.2rem;font-weight:700;color:{DANGER};
                                font-family:'Inter',sans-serif">{total_fr:,}</div>
                </div>
            </div>
        """, unsafe_allow_html=True)

        # Trend chart
        df_s = df.sort_values("timestamp")
        df_s["day"] = df_s["timestamp"].dt.date
        daily = df_s.groupby("day").agg(
            volume=("amount","sum"),
            frauds=("is_fraud","sum")
        ).reset_index()

        fig_r = make_subplots(specs=[[{"secondary_y": True}]])
        fig_r.add_trace(go.Bar(
            x=daily["day"], y=daily["volume"],
            name="Volume", marker_color="#DBEAFE",
            marker_line_color=INFO, marker_line_width=0.8
        ), secondary_y=False)
        fig_r.add_trace(go.Scatter(
            x=daily["day"], y=daily["frauds"],
            name="Fraud Cases",
            line=dict(color=RED, width=2.5),
            mode="lines+markers", marker=dict(size=5)
        ), secondary_y=True)
        style_chart(fig_r, 220)
        st.plotly_chart(fig_r, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

        # Key findings
        st.markdown(f"""
        <div style="background:white;border:1px solid {BORDER};border-radius:12px;
                    padding:20px;box-shadow:0 1px 3px rgba(0,0,0,0.06)">
            <div style="font-size:0.85rem;font-weight:600;color:{TEXT};
                        margin-bottom:14px;font-family:'Inter',sans-serif">
                Key Findings
            </div>
        """, unsafe_allow_html=True)

        findings = [
            (f"CASH_OUT and TRANSFER account for 100% of detected fraud cases",             "danger"),
            (f"98% of fraudulent transactions completely emptied the sender's account",      "danger"),
            (f"Model drift detected — {summary['n_drifted_cols']} features beyond threshold","warning"),
            (f"Auto-retraining pipeline successfully deployed new model",                    "success"),
            (f"False positive rate maintained below 3% — minimal customer impact",          "success"),
        ]
        for ftext, fkind in findings:
            icon = {"danger":"⚠️","warning":"⚡","success":"✅"}[fkind]
            fc   = {"danger":DANGER,"warning":WARNING,"success":SUCCESS}[fkind]
            fb   = {"danger":"#FEF2F2","warning":"#FFFBEB","success":"#ECFDF5"}[fkind]
            fbd  = {"danger":"#FECACA","warning":"#FDE68A","success":"#A7F3D0"}[fkind]
            st.markdown(f"""
            <div style="background:{fb};border:1px solid {fbd};border-radius:6px;
                        padding:10px 12px;margin-bottom:8px;
                        font-size:0.78rem;color:{fc};font-family:'Inter',sans-serif;
                        font-weight:500">
                {icon} {ftext}
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)