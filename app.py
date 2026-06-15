import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from utils.preprocessing import clean_sales_data
from utils.forecasting import forecast_sales, compare_forecast_models
from utils.anomaly import detect_anomalies
from utils.insights import generate_insights
from utils.genai_summary import generate_business_summary
from utils.recommendations import generate_inventory_recommendation
from utils.pdf_report import create_pdf_report

st.set_page_config(
    page_title="SalesIQ",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Page background ── */
.stApp {
    background: #f0f4f8;
}

.block-container {
    padding: 1.8rem 2.2rem 2rem !important;
    max-width: 1400px;
}

/* ── Hero banner ── */
.hero-banner {
    background: linear-gradient(135deg, #0a1628 0%, #0f2a4a 45%, #0d3d5c 100%);
    border-radius: 16px;
    padding: 32px 36px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
    border: 1px solid rgba(13, 148, 136, 0.3);
}

.hero-banner::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 280px; height: 280px;
    background: radial-gradient(circle, rgba(13,148,136,0.18) 0%, transparent 70%);
    border-radius: 50%;
}

.hero-banner::after {
    content: '';
    position: absolute;
    bottom: -80px; left: 40%;
    width: 320px; height: 320px;
    background: radial-gradient(circle, rgba(99,102,241,0.12) 0%, transparent 70%);
    border-radius: 50%;
}

.hero-title {
    font-size: 28px;
    font-weight: 800;
    color: #ffffff;
    margin: 0 0 6px 0;
    letter-spacing: -0.5px;
    position: relative;
    z-index: 1;
}

.hero-title span {
    background: linear-gradient(90deg, #0d9488, #6366f1);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.hero-subtitle {
    font-size: 14px;
    color: rgba(255,255,255,0.6);
    margin: 0;
    position: relative;
    z-index: 1;
}

.hero-badges {
    display: flex;
    gap: 8px;
    margin-top: 14px;
    flex-wrap: wrap;
    position: relative;
    z-index: 1;
}

.badge {
    background: rgba(13,148,136,0.18);
    border: 1px solid rgba(13,148,136,0.4);
    color: #5eead4;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 500;
}

.badge-purple {
    background: rgba(99,102,241,0.15);
    border: 1px solid rgba(99,102,241,0.35);
    color: #a5b4fc;
}

/* ── KPI metric cards ── */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
    margin-bottom: 24px;
}

.kpi-card {
    background: #ffffff;
    border-radius: 14px;
    padding: 20px 22px;
    border: 1px solid #e2e8f0;
    position: relative;
    overflow: hidden;
    transition: box-shadow 0.2s;
}

.kpi-card:hover {
    box-shadow: 0 8px 24px rgba(10, 22, 40, 0.1);
}

.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 4px; height: 100%;
    background: linear-gradient(180deg, #0d9488, #6366f1);
    border-radius: 4px 0 0 4px;
}

.kpi-icon {
    font-size: 22px;
    margin-bottom: 10px;
    display: block;
}

.kpi-label {
    font-size: 12px;
    font-weight: 600;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.6px;
    margin-bottom: 4px;
}

.kpi-value {
    font-size: 26px;
    font-weight: 800;
    color: #0a1628;
    line-height: 1.1;
}

.kpi-delta {
    font-size: 12px;
    color: #10b981;
    font-weight: 600;
    margin-top: 4px;
}

/* ── Section headers ── */
.section-header {
    font-size: 16px;
    font-weight: 700;
    color: #0a1628;
    margin: 24px 0 14px 0;
    display: flex;
    align-items: center;
    gap: 8px;
}

.section-header::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, #e2e8f0, transparent);
    margin-left: 8px;
}

/* ── Summary cards (exec overview) ── */
.summary-card {
    background: #ffffff;
    border-radius: 12px;
    padding: 16px 18px;
    border: 1px solid #e2e8f0;
    text-align: center;
}

.summary-card .s-label {
    font-size: 11px;
    font-weight: 600;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 6px;
}

.summary-card .s-value {
    font-size: 15px;
    font-weight: 700;
    color: #0a1628;
    word-break: break-word;
}

/* ── Info / data quality cards ── */
.info-card {
    background: #ffffff;
    border-radius: 12px;
    padding: 20px;
    border: 1px solid #e2e8f0;
    height: 100%;
}

.info-card h4 {
    font-size: 14px;
    font-weight: 700;
    color: #0a1628;
    margin: 0 0 12px 0;
}

.info-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 6px 0;
    border-bottom: 1px solid #f1f5f9;
    font-size: 13px;
    color: #475569;
}

.info-row span:last-child {
    font-weight: 600;
    color: #0a1628;
}

/* ── Welcome screen ── */
.welcome-hero {
    background: linear-gradient(135deg, #0a1628 0%, #0f2a4a 50%, #0d3d5c 100%);
    border-radius: 20px;
    padding: 48px 44px;
    text-align: center;
    position: relative;
    overflow: hidden;
    margin-bottom: 28px;
    border: 1px solid rgba(13,148,136,0.25);
}

.welcome-hero::before {
    content: '';
    position: absolute;
    top: -100px; right: -100px;
    width: 400px; height: 400px;
    background: radial-gradient(circle, rgba(13,148,136,0.15) 0%, transparent 65%);
    border-radius: 50%;
}

.welcome-hero::after {
    content: '';
    position: absolute;
    bottom: -80px; left: -60px;
    width: 340px; height: 340px;
    background: radial-gradient(circle, rgba(99,102,241,0.12) 0%, transparent 65%);
    border-radius: 50%;
}

.welcome-hero h1 {
    font-size: 42px;
    font-weight: 800;
    color: #ffffff;
    margin: 0 0 12px 0;
    letter-spacing: -1px;
    position: relative;
    z-index: 1;
}

.welcome-hero h1 span {
    background: linear-gradient(90deg, #0d9488, #6366f1);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.welcome-hero p {
    font-size: 16px;
    color: rgba(255,255,255,0.6);
    max-width: 560px;
    margin: 0 auto 20px;
    line-height: 1.7;
    position: relative;
    z-index: 1;
}

.welcome-cols {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 16px;
    margin-bottom: 20px;
}

.welcome-stat-card {
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 12px;
    padding: 18px;
    text-align: center;
    position: relative;
    z-index: 1;
}

.welcome-stat-card .ws-emoji {
    font-size: 24px;
    display: block;
    margin-bottom: 6px;
}

.welcome-stat-card .ws-title {
    font-size: 16px;
    font-weight: 700;
    color: #ffffff;
}

.welcome-stat-card .ws-sub {
    font-size: 12px;
    color: rgba(255,255,255,0.5);
    margin-top: 3px;
}

.feature-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 14px;
    margin-bottom: 24px;
}

.feature-card {
    background: #ffffff;
    border-radius: 14px;
    padding: 20px;
    border: 1px solid #e2e8f0;
    display: flex;
    gap: 14px;
    align-items: flex-start;
    transition: box-shadow 0.2s;
}

.feature-card:hover {
    box-shadow: 0 6px 20px rgba(10,22,40,0.08);
}

.feature-icon {
    font-size: 26px;
    flex-shrink: 0;
    margin-top: 2px;
}

.feature-card h4 {
    font-size: 14px;
    font-weight: 700;
    color: #0a1628;
    margin: 0 0 6px 0;
}

.feature-card p {
    font-size: 13px;
    color: #64748b;
    margin: 0;
    line-height: 1.55;
}

.columns-hint {
    background: linear-gradient(135deg, #f0fdf9, #eff6ff);
    border: 1px solid #ccfbf1;
    border-radius: 12px;
    padding: 18px 22px;
}

.columns-hint h4 {
    font-size: 13px;
    font-weight: 700;
    color: #0a1628;
    margin: 0 0 10px 0;
}

.columns-hint .col-tags {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
}

.col-tag {
    background: #ccfbf1;
    color: #0f766e;
    border: 1px solid #99f6e4;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
}

/* ── Tabs ── */
div[data-baseweb="tab-list"] {
    gap: 6px;
    background: #e8edf5 !important;
    padding: 6px !important;
    border-radius: 12px !important;
    border: none !important;
}

button[data-baseweb="tab"] {
    border-radius: 8px !important;
    padding: 8px 18px !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    color: #475569 !important;
    background: transparent !important;
    border: none !important;
    transition: all 0.2s !important;
}

button[data-baseweb="tab"]:hover {
    background: rgba(255,255,255,0.7) !important;
    color: #0a1628 !important;
}

button[data-baseweb="tab"][aria-selected="true"] {
    background: #ffffff !important;
    color: #0a1628 !important;
    box-shadow: 0 2px 8px rgba(10,22,40,0.1) !important;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a1628 0%, #0d2040 100%) !important;
    border-right: 1px solid rgba(13,148,136,0.2) !important;
}

section[data-testid="stSidebar"] * {
    color: rgba(255,255,255,0.85) !important;
}

section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stMultiSelect label,
section[data-testid="stSidebar"] .stDateInput label {
    color: rgba(255,255,255,0.6) !important;
    font-size: 11px !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.6px !important;
}

section[data-testid="stSidebar"] [data-baseweb="select"],
section[data-testid="stSidebar"] [data-baseweb="input"] {
    background: rgba(255,255,255,0.07) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 8px !important;
}

.sidebar-logo {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 6px 0 18px 0;
    border-bottom: 1px solid rgba(255,255,255,0.1);
    margin-bottom: 20px;
}

.sidebar-logo-icon {
    font-size: 26px;
}

.sidebar-logo-text {
    font-size: 20px;
    font-weight: 800;
    color: #ffffff !important;
}

.sidebar-logo-text span {
    color: #0d9488 !important;
}

.sidebar-section-label {
    font-size: 10px;
    font-weight: 700;
    color: rgba(255,255,255,0.35) !important;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin: 18px 0 8px 0;
}

/* ── Streamlit overrides ── */
[data-testid="stMetric"] {
    display: none !important;
}

div[data-testid="stDataFrame"] {
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid #e2e8f0 !important;
}

.stDownloadButton > button {
    background: linear-gradient(135deg, #0d9488, #0f766e) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    padding: 10px 20px !important;
    transition: opacity 0.2s !important;
}

.stDownloadButton > button:hover {
    opacity: 0.9 !important;
}

.stButton > button {
    background: linear-gradient(135deg, #6366f1, #4f46e5) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    padding: 10px 22px !important;
}

.stButton > button:hover {
    opacity: 0.9 !important;
}

.stAlert {
    border-radius: 10px !important;
}

.stSuccess {
    background: linear-gradient(135deg, #f0fdf4, #dcfce7) !important;
    border: 1px solid #86efac !important;
    border-radius: 10px !important;
    color: #166534 !important;
}

.stInfo {
    background: linear-gradient(135deg, #eff6ff, #dbeafe) !important;
    border: 1px solid #93c5fd !important;
    border-radius: 10px !important;
    color: #1e40af !important;
}

.stWarning {
    background: linear-gradient(135deg, #fffbeb, #fef3c7) !important;
    border: 1px solid #fcd34d !important;
    border-radius: 10px !important;
    color: #92400e !important;
}

h2, h3 {
    color: #0a1628 !important;
    font-weight: 700 !important;
}

.plotly-chart-wrapper {
    background: #ffffff;
    border-radius: 14px;
    border: 1px solid #e2e8f0;
    padding: 4px;
    margin-bottom: 16px;
}

/* ── About page ── */
.about-card {
    background: #ffffff;
    border-radius: 14px;
    padding: 24px;
    border: 1px solid #e2e8f0;
    margin-bottom: 16px;
}

.about-card h4 {
    font-size: 15px;
    font-weight: 700;
    color: #0a1628;
    margin: 0 0 10px 0;
    display: flex;
    align-items: center;
    gap: 8px;
}

.about-card p, .about-card li {
    font-size: 13px;
    color: #475569;
    line-height: 1.65;
    margin: 4px 0;
}

.tech-tags {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 8px;
}

.tech-tag {
    background: #f1f5f9;
    color: #334155;
    border: 1px solid #e2e8f0;
    padding: 4px 12px;
    border-radius: 6px;
    font-size: 12px;
    font-weight: 600;
}

/* ── Insight cards ── */
.insight-item {
    background: linear-gradient(135deg, #eff6ff 0%, #f0fdf4 100%);
    border: 1px solid #bfdbfe;
    border-left: 4px solid #6366f1;
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 12px;
    font-size: 14px;
    color: #1e3a5f;
    font-weight: 500;
}

/* ── Forecast controls ── */
.forecast-controls {
    background: #ffffff;
    border-radius: 14px;
    padding: 22px;
    border: 1px solid #e2e8f0;
    margin-bottom: 20px;
}

.forecast-controls label {
    font-size: 13px;
    font-weight: 600;
    color: #334155;
}

/* ── Anomaly badge ── */
.anomaly-count {
    display: inline-block;
    background: linear-gradient(135deg, #fef2f2, #fee2e2);
    border: 1px solid #fca5a5;
    color: #b91c1c;
    padding: 6px 16px;
    border-radius: 20px;
    font-size: 13px;
    font-weight: 700;
    margin-bottom: 16px;
}

/* ── Report section ── */
.report-meta {
    background: linear-gradient(135deg, #0a1628, #0f2a4a);
    border-radius: 14px;
    padding: 22px 26px;
    margin-bottom: 20px;
    border: 1px solid rgba(13,148,136,0.25);
}

.report-meta p {
    color: rgba(255,255,255,0.65);
    font-size: 13px;
    margin: 0;
    line-height: 1.6;
}

.report-meta strong {
    color: #5eead4;
}

.generated-report {
    background: #ffffff;
    border-radius: 14px;
    padding: 28px;
    border: 1px solid #e2e8f0;
    line-height: 1.8;
    font-size: 14px;
    color: #1e293b;
}
</style>
""", unsafe_allow_html=True)

# ── Plotly chart theme ──────────────────────────────────────────────────────────
CHART_COLORS = ["#0d9488", "#6366f1", "#f59e0b", "#ef4444", "#10b981", "#8b5cf6"]
CHART_TEMPLATE = "plotly_white"

def styled_chart(fig):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="#334155", size=12),
        title_font=dict(family="Inter", size=15, color="#0a1628"),
        margin=dict(l=16, r=16, t=44, b=16),
        legend=dict(
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="#e2e8f0",
            borderwidth=1,
            font=dict(size=11)
        ),
        xaxis=dict(gridcolor="#f1f5f9", linecolor="#e2e8f0", tickfont=dict(size=11)),
        yaxis=dict(gridcolor="#f1f5f9", linecolor="#e2e8f0", tickfont=dict(size=11)),
    )
    return fig

# ── Hero banner ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
    <div class="hero-title">📊 <span>SalesIQ</span> — AI-Powered Sales Intelligence</div>
    <div class="hero-subtitle">Upload your sales data to unlock forecasting, anomaly detection, and Gemini AI analyst reports.</div>
    <div class="hero-badges">
        <span class="badge">⚡ Real-time Analytics</span>
        <span class="badge">🤖 Gemini AI Reports</span>
        <span class="badge badge-purple">📈 ML Forecasting</span>
        <span class="badge badge-purple">🔍 Anomaly Detection</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── File uploader ────────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "📂 Drop your sales CSV here or click to browse",
    type=["csv"],
    help="Recommended columns: Order Date, Sales, Profit, Quantity, Category, Region, Product Name"
)

# ════════════════════════════════════════════════════════════════════════════════
# MAIN DASHBOARD
# ════════════════════════════════════════════════════════════════════════════════
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, encoding="latin1")
    df = clean_sales_data(df)

    required_columns = ["Order Date", "Sales"]
    missing_required_columns = [c for c in required_columns if c not in df.columns]

    if missing_required_columns:
        st.error("⚠️ This dataset must contain **Order Date** and **Sales** columns.")
        st.stop()

    filtered_df = df.copy()

    # ── Sidebar ──────────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-logo">
            <span class="sidebar-logo-icon">📊</span>
            <span class="sidebar-logo-text">Sales<span>IQ</span></span>
        </div>
        <div class="sidebar-section-label">🗓️ Time Range</div>
        """, unsafe_allow_html=True)

        if "Order Date" in df.columns:
            min_date = df["Order Date"].min().date()
            max_date = df["Order Date"].max().date()
            selected_date_range = st.date_input(
                "Date Range",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date,
                label_visibility="collapsed"
            )
            if len(selected_date_range) == 2:
                start_date, end_date = selected_date_range
                filtered_df = filtered_df[
                    (filtered_df["Order Date"].dt.date >= start_date) &
                    (filtered_df["Order Date"].dt.date <= end_date)
                ]

        st.markdown('<div class="sidebar-section-label">🗺️ Region</div>', unsafe_allow_html=True)
        if "Region" in df.columns:
            selected_regions = st.multiselect(
                "Region",
                options=df["Region"].dropna().unique(),
                default=df["Region"].dropna().unique(),
                label_visibility="collapsed"
            )
            filtered_df = filtered_df[filtered_df["Region"].isin(selected_regions)]

        st.markdown('<div class="sidebar-section-label">🏷️ Category</div>', unsafe_allow_html=True)
        if "Category" in df.columns:
            selected_categories = st.multiselect(
                "Category",
                options=df["Category"].dropna().unique(),
                default=df["Category"].dropna().unique(),
                label_visibility="collapsed"
            )
            filtered_df = filtered_df[filtered_df["Category"].isin(selected_categories)]

        st.markdown("---")
        st.markdown(f"""
        <div style="font-size:11px; color:rgba(255,255,255,0.35); line-height:1.7;">
            📋 <strong style="color:rgba(255,255,255,0.55);">{filtered_df.shape[0]:,}</strong> rows loaded<br>
            📐 <strong style="color:rgba(255,255,255,0.55);">{filtered_df.shape[1]}</strong> columns detected
        </div>
        """, unsafe_allow_html=True)

    if filtered_df.empty:
        st.warning("⚠️ No data available for the selected filters. Please adjust your selections.")
        st.stop()

    # ── Tabs ─────────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "🏠 Overview",
        "📊 Analytics",
        "📈 Forecasting",
        "🔍 Anomalies",
        "💡 Insights",
        "🤖 AI Report",
        "ℹ️ About"
    ])

    # ════════════════════════════════════════════════════════════════════════════
    # TAB 1 — OVERVIEW
    # ════════════════════════════════════════════════════════════════════════════
    with tab1:
        # ── KPI metrics ──────────────────────────────────────────────────────────
        total_sales = filtered_df["Sales"].sum()
        total_orders = len(filtered_df)
        average_order_value = filtered_df["Sales"].mean()

        if "Profit" in filtered_df.columns:
            total_profit = filtered_df["Profit"].sum()
            profit_margin = total_profit / total_sales if total_sales > 0 else 0
        else:
            total_profit = 0
            profit_margin = 0

        st.markdown(f"""
        <div class="kpi-grid">
            <div class="kpi-card">
                <span class="kpi-icon">💰</span>
                <div class="kpi-label">Total Sales</div>
                <div class="kpi-value">${total_sales:,.0f}</div>
                <div class="kpi-delta">▲ Across filtered period</div>
            </div>
            <div class="kpi-card">
                <span class="kpi-icon">📦</span>
                <div class="kpi-label">Total Orders</div>
                <div class="kpi-value">{total_orders:,}</div>
                <div class="kpi-delta">▲ Transactions recorded</div>
            </div>
            <div class="kpi-card">
                <span class="kpi-icon">🏦</span>
                <div class="kpi-label">Total Profit</div>
                <div class="kpi-value">${total_profit:,.0f}</div>
                <div class="kpi-delta" style="color:{'#10b981' if total_profit >= 0 else '#ef4444'};">
                    {'▲' if total_profit >= 0 else '▼'} {abs(profit_margin*100):.1f}% margin
                </div>
            </div>
            <div class="kpi-card">
                <span class="kpi-icon">🛒</span>
                <div class="kpi-label">Avg Order Value</div>
                <div class="kpi-value">${average_order_value:,.0f}</div>
                <div class="kpi-delta">▲ Per transaction</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Executive summary ─────────────────────────────────────────────────────
        st.markdown('<div class="section-header">🎯 Executive Summary</div>', unsafe_allow_html=True)

        if "Category" in filtered_df.columns:
            best_category = filtered_df.groupby("Category")["Sales"].sum().idxmax()
        else:
            best_category = "N/A"

        if "Region" in filtered_df.columns:
            best_region = filtered_df.groupby("Region")["Sales"].sum().idxmax()
        else:
            best_region = "N/A"

        if "Product Name" in filtered_df.columns:
            best_product = filtered_df.groupby("Product Name")["Sales"].sum().idxmax()
            best_product_display = best_product[:24] + "..." if len(best_product) > 24 else best_product
        else:
            best_product_display = "N/A"

        sc1, sc2, sc3, sc4 = st.columns(4)
        for col, label, val, icon in [
            (sc1, "Best Category", best_category, "🏆"),
            (sc2, "Best Region", best_region, "🌍"),
            (sc3, "Profit Margin", f"{profit_margin*100:.2f}%", "📉"),
            (sc4, "Top Product", best_product_display, "⭐"),
        ]:
            with col:
                st.markdown(f"""
                <div class="summary-card">
                    <div class="s-label">{icon} {label}</div>
                    <div class="s-value">{val}</div>
                </div>
                """, unsafe_allow_html=True)

        # ── Dataset preview & info ─────────────────────────────────────────────────
        st.markdown('<div class="section-header">📋 Dataset Preview</div>', unsafe_allow_html=True)
        st.dataframe(filtered_df.head(10), use_container_width=True)

        csv_data = filtered_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download Filtered Data as CSV",
            data=csv_data,
            file_name="salesiq_filtered_data.csv",
            mime="text/csv"
        )

        info_col1, info_col2 = st.columns(2)

        with info_col1:
            date_min = str(filtered_df["Order Date"].min().date()) if "Order Date" in filtered_df.columns else "—"
            date_max = str(filtered_df["Order Date"].max().date()) if "Order Date" in filtered_df.columns else "—"
            st.markdown(f"""
            <div class="info-card">
                <h4>📐 Dataset Information</h4>
                <div class="info-row"><span>Rows</span><span>{filtered_df.shape[0]:,}</span></div>
                <div class="info-row"><span>Columns</span><span>{filtered_df.shape[1]}</span></div>
                <div class="info-row"><span>Start Date</span><span>{date_min}</span></div>
                <div class="info-row"><span>End Date</span><span>{date_max}</span></div>
            </div>
            """, unsafe_allow_html=True)

        with info_col2:
            missing_values = filtered_df.isnull().sum()
            missing_values = missing_values[missing_values > 0]
            if len(missing_values) > 0:
                rows_html = "".join([
                    f'<div class="info-row"><span>{col}</span><span style="color:#ef4444;">{cnt} missing</span></div>'
                    for col, cnt in missing_values.items()
                ])
                st.markdown(f"""
                <div class="info-card">
                    <h4>⚠️ Data Quality</h4>
                    {rows_html}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="info-card">
                    <h4>✅ Data Quality</h4>
                    <div style="text-align:center; padding:24px 0;">
                        <div style="font-size:36px; margin-bottom:8px;">🎉</div>
                        <div style="font-size:14px; color:#10b981; font-weight:700;">No missing values found!</div>
                        <div style="font-size:12px; color:#64748b; margin-top:4px;">Your dataset is clean and ready.</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════════════════
    # TAB 2 — ANALYTICS
    # ════════════════════════════════════════════════════════════════════════════
    with tab2:
        st.markdown('<div class="section-header">📊 Sales Performance Analytics</div>', unsafe_allow_html=True)

        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            monthly_sales = (
                filtered_df.set_index("Order Date")
                .resample("ME")["Sales"]
                .sum()
                .reset_index()
            )
            fig = px.line(
                monthly_sales, x="Order Date", y="Sales",
                title="📅 Monthly Sales Trend",
                markers=True, template=CHART_TEMPLATE,
                color_discrete_sequence=[CHART_COLORS[0]]
            )
            fig.update_traces(line=dict(width=2.5), marker=dict(size=7))
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              font=dict(family="Inter"), title_font_size=14, margin=dict(l=10,r=10,t=44,b=10))
            fig.update_xaxes(gridcolor="#f1f5f9")
            fig.update_yaxes(gridcolor="#f1f5f9")
            st.plotly_chart(fig, use_container_width=True)

        with chart_col2:
            if "Category" in filtered_df.columns:
                category_sales = filtered_df.groupby("Category")["Sales"].sum().reset_index()
                fig = px.bar(
                    category_sales, x="Category", y="Sales",
                    title="🏷️ Sales by Category",
                    color="Category", template=CHART_TEMPLATE,
                    color_discrete_sequence=CHART_COLORS
                )
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                  font=dict(family="Inter"), title_font_size=14, margin=dict(l=10,r=10,t=44,b=10),
                                  showlegend=False)
                fig.update_xaxes(gridcolor="#f1f5f9")
                fig.update_yaxes(gridcolor="#f1f5f9")
                st.plotly_chart(fig, use_container_width=True)

        chart_col3, chart_col4 = st.columns(2)

        with chart_col3:
            if "Region" in filtered_df.columns:
                region_sales = filtered_df.groupby("Region")["Sales"].sum().reset_index()
                fig = px.pie(
                    region_sales, names="Region", values="Sales",
                    title="🌍 Sales by Region",
                    hole=0.45, color_discrete_sequence=CHART_COLORS
                )
                fig.update_traces(textposition="outside", textinfo="label+percent")
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                                  font=dict(family="Inter"), title_font_size=14, margin=dict(l=10,r=10,t=44,b=10))
                st.plotly_chart(fig, use_container_width=True)

        with chart_col4:
            if "Product Name" in filtered_df.columns:
                top_products = (
                    filtered_df.groupby("Product Name")["Sales"]
                    .sum().sort_values(ascending=True).head(10).reset_index()
                )
                fig = px.bar(
                    top_products, x="Sales", y="Product Name",
                    orientation="h", title="⭐ Top 10 Products by Sales",
                    template=CHART_TEMPLATE,
                    color_discrete_sequence=[CHART_COLORS[1]]
                )
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                  font=dict(family="Inter"), title_font_size=14, margin=dict(l=10,r=10,t=44,b=10))
                fig.update_xaxes(gridcolor="#f1f5f9")
                fig.update_yaxes(gridcolor="#f1f5f9")
                st.plotly_chart(fig, use_container_width=True)

        # ── Profit analysis if available ──────────────────────────────────────────
        if "Profit" in filtered_df.columns and "Category" in filtered_df.columns:
            st.markdown('<div class="section-header">💹 Profit Analysis</div>', unsafe_allow_html=True)
            profit_by_cat = filtered_df.groupby("Category")["Profit"].sum().reset_index()
            fig = px.bar(
                profit_by_cat, x="Category", y="Profit",
                title="💹 Profit by Category",
                color="Profit", template=CHART_TEMPLATE,
                color_continuous_scale=["#ef4444", "#f59e0b", "#10b981"]
            )
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              font=dict(family="Inter"), title_font_size=14)
            fig.update_xaxes(gridcolor="#f1f5f9")
            fig.update_yaxes(gridcolor="#f1f5f9")
            st.plotly_chart(fig, use_container_width=True)

    # ════════════════════════════════════════════════════════════════════════════
    # TAB 3 — FORECASTING
    # ════════════════════════════════════════════════════════════════════════════
    with tab3:
        st.markdown('<div class="section-header">📈 Sales Forecasting Engine</div>', unsafe_allow_html=True)

        model_results, evaluation_df = compare_forecast_models(filtered_df)

        if model_results is not None:
            st.markdown("**🔬 Model Performance Comparison**")
            st.dataframe(model_results, use_container_width=True)

            fig = px.line(
                evaluation_df, x="Date",
                y=["Actual Sales", "Linear Regression", "Random Forest"],
                title="📊 Actual vs Predicted Sales — Model Comparison",
                template=CHART_TEMPLATE,
                color_discrete_sequence=[CHART_COLORS[0], CHART_COLORS[1], CHART_COLORS[2]]
            )
            fig.update_traces(selector=dict(name="Actual Sales"), line=dict(width=2.5))
            fig.update_traces(selector=dict(name="Linear Regression"), line=dict(width=1.8, dash="dash"))
            fig.update_traces(selector=dict(name="Random Forest"), line=dict(width=1.8, dash="dot"))
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              font=dict(family="Inter"), title_font_size=14)
            fig.update_xaxes(gridcolor="#f1f5f9")
            fig.update_yaxes(gridcolor="#f1f5f9")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("⚠️ Not enough data available for model comparison. Try a wider date range.")

        st.markdown('<div class="section-header">🎛️ Forecast Configuration</div>', unsafe_allow_html=True)

        ctrl1, ctrl2, ctrl3 = st.columns(3)

        with ctrl1:
            forecast_days = st.slider("📅 Forecast Period (days)", 7, 90, 30)

        with ctrl2:
            selected_model = st.selectbox(
                "🤖 Forecasting Model",
                ["Random Forest", "Linear Regression"]
            )

        with ctrl3:
            current_stock_capacity = st.number_input(
                "📦 Current Stock / Demand Capacity ($)",
                min_value=0, value=100000
            )

        if st.button("🚀 Generate Future Forecast"):
            with st.spinner("🔄 Running forecast model..."):
                forecast_df = forecast_sales(filtered_df, forecast_days, selected_model)

            fig = px.line(
                forecast_df, x="Date", y="Predicted Sales",
                title=f"🔮 {forecast_days}-Day Sales Forecast — {selected_model}",
                markers=True, template=CHART_TEMPLATE,
                color_discrete_sequence=[CHART_COLORS[1]]
            )
            fig.update_traces(
                line=dict(width=2.5, dash="dash"),
                marker=dict(size=6, symbol="diamond")
            )
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              font=dict(family="Inter"), title_font_size=14)
            fig.update_xaxes(gridcolor="#f1f5f9")
            fig.update_yaxes(gridcolor="#f1f5f9")
            st.plotly_chart(fig, use_container_width=True)

            predicted_total = forecast_df["Predicted Sales"].sum()
            st.success(f"✅ Predicted total sales for the next **{forecast_days} days**: **${predicted_total:,.2f}**")

            recommendation = generate_inventory_recommendation(predicted_total, current_stock_capacity)
            st.info(f"📦 **Inventory Recommendation:** {recommendation}")

    # ════════════════════════════════════════════════════════════════════════════
    # TAB 4 — ANOMALIES
    # ════════════════════════════════════════════════════════════════════════════
    with tab4:
        st.markdown('<div class="section-header">🔍 Sales Anomaly Detection</div>', unsafe_allow_html=True)

        with st.spinner("🔄 Running anomaly detection..."):
            anomaly_df = detect_anomalies(filtered_df)

        anomalies = anomaly_df[anomaly_df["Anomaly"] == True]

        st.markdown(f"""
        <div class="anomaly-count">
            ⚠️ {len(anomalies)} anomalies detected out of {len(anomaly_df):,} data points
        </div>
        """, unsafe_allow_html=True)

        fig = px.scatter(
            anomaly_df, x="Order Date", y="Sales",
            color="Anomaly",
            title="🔎 Sales Anomaly Map — Normal vs Flagged",
            template=CHART_TEMPLATE,
            color_discrete_map={True: "#ef4444", False: "#0d9488"}
        )
        fig.update_traces(marker=dict(size=8, opacity=0.8))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter"), title_font_size=14
        )
        fig.update_xaxes(gridcolor="#f1f5f9")
        fig.update_yaxes(gridcolor="#f1f5f9")
        st.plotly_chart(fig, use_container_width=True)

        if not anomalies.empty:
            st.markdown('<div class="section-header">🚨 Flagged Anomaly Records</div>', unsafe_allow_html=True)
            st.dataframe(anomalies, use_container_width=True)
        else:
            st.success("✅ No anomalies detected in the selected data range. Sales patterns look healthy!")

    # ════════════════════════════════════════════════════════════════════════════
    # TAB 5 — INSIGHTS
    # ════════════════════════════════════════════════════════════════════════════
    with tab5:
        st.markdown('<div class="section-header">💡 AI-Generated Business Insights</div>', unsafe_allow_html=True)

        with st.spinner("🧠 Analyzing your data..."):
            insights = generate_insights(filtered_df)

        for i, insight in enumerate(insights, 1):
            st.markdown(f"""
            <div class="insight-item">
                <strong>#{i}</strong> &nbsp; {insight}
            </div>
            """, unsafe_allow_html=True)

  # ════════════════════════════════════════════════════════════════════════════
    # TAB 6 — AI REPORT
    # ════════════════════════════════════════════════════════════════════════════
    with tab6:
        st.markdown('<div class="section-header">🤖 Gemini AI Business Analyst Report</div>', unsafe_allow_html=True)

        st.markdown("""
        <div class="report-meta">
            <p>
                This report is generated by <strong>Google Gemini AI</strong> — a large language model that analyzes
                your filtered sales data and produces a professional business analyst-style report with
                strategic recommendations, trend observations, and actionable insights.
            </p>
        </div>
        """, unsafe_allow_html=True)

        api_key = st.secrets.get("GEMINI_API_KEY", None)

        with st.spinner("✨ Gemini AI is analyzing your data..."):
            current_metrics_payload = {
                "total_sales": f"${total_sales:,.2f}",
                "total_orders": f"{total_orders:,}",
                "total_profit": f"${total_profit:,.2f}",
                "profit_margin": f"{profit_margin*100:.2f}%",
                "aov": f"${average_order_value:,.2f}",
                "best_category": best_category,
                "best_region": best_region,
                "best_product": best_product_display
            }
            
            try:
                anomalies_data = detect_anomalies(filtered_df)
            except Exception:
                anomalies_data = None

            business_summary = generate_business_summary(current_metrics_payload, anomalies_df=anomalies_data)

        st.markdown('<div class="section-header">📄 Generated Report</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="generated-report">
            {business_summary}
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        dl1, dl2 = st.columns(2)
        with dl1:
            pdf_report = create_pdf_report(business_summary)
            st.download_button(
                label="📥 Download Report as PDF",
                data=pdf_report,
                file_name="salesiq_gemini_ai_report.pdf",
                mime="application/pdf"
            )
        with dl2:
            st.download_button(
                label="📄 Download Report as Text",
                data=business_summary,
                file_name="salesiq_gemini_ai_report.txt",
                mime="text/plain"
            )

    # ════════════════════════════════════════════════════════════════════════════
    # TAB 7 — ABOUT
    # ════════════════════════════════════════════════════════════════════════════
    with tab7:
        st.markdown('<div class="section-header">ℹ️ About SalesIQ</div>', unsafe_allow_html=True)

        st.markdown("""
        <div class="about-card">
            <h4>🎯 Problem Statement</h4>
            <p>
                Businesses often sit on mountains of raw sales data but lack intuitive tools to
                forecast demand, surface anomalies, and generate decision-ready reports. SalesIQ
                bridges that gap — turning spreadsheets into strategic intelligence in minutes.
            </p>
        </div>

        <div class="about-card">
            <h4>🛠️ Tech Stack</h4>
            <div class="tech-tags">
                <span class="tech-tag">🐍 Python</span>
                <span class="tech-tag">⚡ Streamlit</span>
                <span class="tech-tag">🐼 Pandas</span>
                <span class="tech-tag">📊 Plotly</span>
                <span class="tech-tag">🤖 Scikit-learn</span>
                <span class="tech-tag">✨ Google Gemini API</span>
            </div>
        </div>

        <div class="about-card">
            <h4>🧠 ML Models</h4>
            <ul>
                <li><strong>Linear Regression</strong> — Fast, interpretable baseline forecasting</li>
                <li><strong>Random Forest Regressor</strong> — Ensemble model capturing non-linear patterns</li>
            </ul>
        </div>

        <div class="about-card">
            <h4>🔮 GenAI Integration</h4>
            <p>
                Google Gemini AI analyzes aggregated sales metrics and generates a professional
                business analyst report with strategic insights, trend analysis, and
                inventory recommendations — downloadable as PDF or plain text.
            </p>
        </div>

        <div class="about-card">
            <h4>📦 Core Modules</h4>
            <ul>
                <li>🏠 <strong>Overview</strong> — KPI dashboard, executive summary, data quality checks</li>
                <li>📊 <strong>Analytics</strong> — Sales trends, category/region/product breakdowns</li>
                <li>📈 <strong>Forecasting</strong> — 7–90 day ML-powered sales forecasts with model comparison</li>
                <li>🔍 <strong>Anomaly Detection</strong> — Flagging unusual sales patterns automatically</li>
                <li>💡 <strong>Insights</strong> — Rule-based and AI-assisted business observations</li>
                <li>🤖 <strong>AI Report</strong> — Full Gemini-generated analyst report with PDF export</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════════
# WELCOME SCREEN (no file uploaded)
# ════════════════════════════════════════════════════════════════════════════════
else:
    st.markdown("""
    <div class="welcome-hero">
        <h1>Welcome to <span>SalesIQ</span> 👋</h1>
        <p>
            Your AI-powered command center for sales analytics, demand forecasting,
            anomaly detection, and Gemini AI business reports.
            Upload a CSV to get started in seconds.
        </p>
        <div class="welcome-cols">
            <div class="welcome-stat-card">
                <span class="ws-emoji">📊</span>
                <div class="ws-title">Live Dashboard</div>
                <div class="ws-sub">KPIs · Charts · Trends</div>
            </div>
            <div class="welcome-stat-card">
                <span class="ws-emoji">📈</span>
                <div class="ws-title">ML Forecasting</div>
                <div class="ws-sub">Linear · Random Forest</div>
            </div>
            <div class="welcome-stat-card">
                <span class="ws-emoji">🤖</span>
                <div class="ws-title">Gemini AI Reports</div>
                <div class="ws-sub">PDF · Text Export</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">⚡ What SalesIQ Will Generate For You</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="feature-grid">
        <div class="feature-card">
            <span class="feature-icon">🏠</span>
            <div>
                <h4>Business Overview Dashboard</h4>
                <p>Instant KPI cards for total sales, profit, order count, and average order value — with executive summary and data quality report.</p>
            </div>
        </div>
        <div class="feature-card">
            <span class="feature-icon">📊</span>
            <div>
                <h4>Deep Sales Analytics</h4>
                <p>Monthly trends, category breakdowns, regional pie charts, and your top 10 products ranked by revenue — all interactive.</p>
            </div>
        </div>
        <div class="feature-card">
            <span class="feature-icon">📈</span>
            <div>
                <h4>ML-Powered Forecasting</h4>
                <p>Predict 7–90 days of future sales using Linear Regression or Random Forest — with side-by-side model comparison and inventory recommendations.</p>
            </div>
        </div>
        <div class="feature-card">
            <span class="feature-icon">🔍</span>
            <div>
                <h4>Anomaly Detection</h4>
                <p>Automatically flags unusual sales spikes and drops in your data, helping you investigate outliers before they become costly problems.</p>
            </div>
        </div>
        <div class="feature-card">
            <span class="feature-icon">💡</span>
            <div>
                <h4>Business Insights Engine</h4>
                <p>Rule-based and AI-assisted observations surfaced directly from your data — so you spot opportunities without digging through rows.</p>
            </div>
        </div>
        <div class="feature-card">
            <span class="feature-icon">🤖</span>
            <div>
                <h4>Gemini AI Analyst Report</h4>
                <p>Google Gemini reads your sales metrics and writes a full business analyst report — downloadable as PDF or plain text instantly.</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="columns-hint">
        <h4>📋 Recommended CSV Columns for Best Results</h4>
        <div class="col-tags">
            <span class="col-tag">📅 Order Date</span>
            <span class="col-tag">💰 Sales</span>
            <span class="col-tag">🏦 Profit</span>
            <span class="col-tag">📦 Quantity</span>
            <span class="col-tag">🏷️ Category</span>
            <span class="col-tag">🌍 Region</span>
            <span class="col-tag">🛍️ Product Name</span>
        </div>
    </div>
    """, unsafe_allow_html=True)