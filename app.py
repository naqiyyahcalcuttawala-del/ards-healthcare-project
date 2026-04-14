"""
ARDS ICU Clinical Decision Support System
==========================================
A production-grade Streamlit application for ICU clinical decision support.
Authors: Naqiyyah Calcuttawala & Ishan Kurhekar
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────
st.set_page_config(
    page_title="ARDS CDS | ICU Intelligence",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ──────────────────────────────────────────
# GLOBAL CSS
# ──────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@300;400;500;600;700&display=swap');

/* ── Root & Reset ── */
html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

.stApp {
    background: #090D14;
    color: #E2E8F0;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0D1421 0%, #0A0F1C 100%);
    border-right: 1px solid #1E2D45;
}

[data-testid="stSidebar"] * {
    color: #94A3B8 !important;
}

[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stMultiSelect label {
    color: #64748B !important;
    font-size: 0.72rem !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

/* ── Headers ── */
h1 { color: #F1F5F9 !important; font-weight: 700 !important; letter-spacing: -0.02em; }
h2 { color: #CBD5E1 !important; font-weight: 600 !important; }
h3 { color: #94A3B8 !important; font-weight: 500 !important; }

/* ── KPI Cards ── */
.kpi-card {
    background: linear-gradient(135deg, #111827 0%, #0F172A 100%);
    border: 1px solid #1E293B;
    border-radius: 12px;
    padding: 20px 24px;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s;
}
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    border-radius: 12px 12px 0 0;
}
.kpi-card.blue::before { background: linear-gradient(90deg, #3B82F6, #60A5FA); }
.kpi-card.red::before  { background: linear-gradient(90deg, #EF4444, #F87171); }
.kpi-card.amber::before { background: linear-gradient(90deg, #F59E0B, #FCD34D); }
.kpi-card.green::before { background: linear-gradient(90deg, #10B981, #34D399); }
.kpi-value {
    font-size: 2.2rem;
    font-weight: 700;
    font-family: 'IBM Plex Mono', monospace;
    line-height: 1.1;
    margin: 4px 0;
}
.kpi-label {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #64748B;
    margin-bottom: 6px;
}
.kpi-delta {
    font-size: 0.75rem;
    font-family: 'IBM Plex Mono', monospace;
    margin-top: 6px;
}
.kpi-delta.up { color: #10B981; }
.kpi-delta.down { color: #EF4444; }

/* ── Section Header ── */
.section-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 28px 0 18px;
    padding-bottom: 10px;
    border-bottom: 1px solid #1E293B;
}
.section-header .icon {
    font-size: 1.3rem;
}
.section-header .title {
    font-size: 1.1rem;
    font-weight: 600;
    color: #CBD5E1;
    letter-spacing: -0.01em;
}
.section-header .badge {
    background: #1E293B;
    color: #64748B;
    font-size: 0.65rem;
    padding: 2px 8px;
    border-radius: 20px;
    font-family: 'IBM Plex Mono', monospace;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-left: auto;
}

/* ── Info Panel ── */
.info-panel {
    background: #0F172A;
    border: 1px solid #1E293B;
    border-left: 3px solid #3B82F6;
    border-radius: 8px;
    padding: 14px 18px;
    margin: 10px 0;
}
.info-panel.warning { border-left-color: #F59E0B; }
.info-panel.danger  { border-left-color: #EF4444; }
.info-panel.success { border-left-color: #10B981; }

/* ── Prediction Card ── */
.pred-card {
    background: linear-gradient(135deg, #111827 0%, #0F172A 100%);
    border: 1px solid #1E293B;
    border-radius: 14px;
    padding: 24px;
    text-align: center;
}
.pred-card .label {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #475569;
    margin-bottom: 8px;
}
.pred-card .value {
    font-size: 2.8rem;
    font-weight: 700;
    font-family: 'IBM Plex Mono', monospace;
    line-height: 1;
}
.pred-card .sub {
    font-size: 0.8rem;
    color: #64748B;
    margin-top: 6px;
}
.risk-high { color: #EF4444; }
.risk-med  { color: #F59E0B; }
.risk-low  { color: #10B981; }

/* ── SBAR Card ── */
.sbar-section {
    background: #0F172A;
    border: 1px solid #1E293B;
    border-radius: 10px;
    padding: 16px 20px;
    margin-bottom: 12px;
}
.sbar-label {
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #3B82F6;
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 600;
    margin-bottom: 6px;
}
.sbar-content {
    font-size: 0.9rem;
    color: #CBD5E1;
    line-height: 1.6;
}

/* ── Feature Tag ── */
.feat-tag {
    display: inline-block;
    background: #1E293B;
    color: #94A3B8;
    border-radius: 6px;
    padding: 3px 10px;
    font-size: 0.75rem;
    font-family: 'IBM Plex Mono', monospace;
    margin: 2px;
}

/* ── Navigation badge ── */
.nav-header {
    font-size: 0.6rem;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    color: #374151 !important;
    padding: 14px 0 6px;
    font-weight: 600;
}

/* ── Metric override ── */
[data-testid="metric-container"] {
    background: #111827;
    border: 1px solid #1E293B;
    border-radius: 10px;
    padding: 14px;
}
[data-testid="metric-container"] [data-testid="stMetricLabel"] {
    font-size: 0.7rem !important;
    color: #64748B !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 1.6rem !important;
    color: #F1F5F9 !important;
}

/* ── Tabs ── */
[data-testid="stTabs"] [data-baseweb="tab"] {
    font-size: 0.8rem !important;
    color: #64748B !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
[data-testid="stTabs"] [aria-selected="true"] {
    color: #3B82F6 !important;
    border-bottom: 2px solid #3B82F6 !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #1D4ED8, #2563EB) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    padding: 0.5rem 1.2rem !important;
    letter-spacing: 0.02em;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #2563EB, #3B82F6) !important;
    transform: translateY(-1px) !important;
}

/* ── Inputs ── */
.stSelectbox [data-baseweb="select"] {
    background: #111827 !important;
    border-color: #1E293B !important;
}
.stSlider [data-baseweb="slider"] {
    background: #1E293B !important;
}

/* ── Expander ── */
[data-testid="stExpander"] {
    background: #0F172A !important;
    border: 1px solid #1E293B !important;
    border-radius: 8px !important;
}

/* ── Divider ── */
hr { border-color: #1E293B !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; background: #0A0F1C; }
::-webkit-scrollbar-thumb { background: #1E293B; border-radius: 4px; }

/* ── Banner ── */
.top-banner {
    background: linear-gradient(135deg, #0F172A 0%, #111827 50%, #0F172A 100%);
    border: 1px solid #1E293B;
    border-radius: 14px;
    padding: 28px 32px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}
.top-banner::after {
    content: '🫁';
    position: absolute;
    right: 32px;
    top: 50%;
    transform: translateY(-50%);
    font-size: 4rem;
    opacity: 0.08;
}
.top-banner h1 {
    font-size: 1.8rem !important;
    margin: 0 0 6px !important;
}
.top-banner p {
    color: #475569;
    font-size: 0.85rem;
    margin: 0;
}

/* ── Cluster badge ── */
.cluster-badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
    font-family: 'IBM Plex Mono', monospace;
}
.c1 { background: #1E3A5F; color: #60A5FA; }
.c2 { background: #1F2D1F; color: #4ADE80; }
.c3 { background: #3B1F1F; color: #F87171; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────
# DATA & MODEL LOADING
# ──────────────────────────────────────────
DATA_PATH = "/mnt/user-data/uploads/ARDS_ICU_V2_15000_final_buan305.csv"

@st.cache_data(show_spinner=False)
def load_data():
    df = pd.read_csv(DATA_PATH)
    return df

@st.cache_data(show_spinner=False)
def preprocess_data(df):
    df = df.copy()
    # Encode categoricals
    df['sex_enc'] = (df['sex'] == 'Male').astype(int)
    sm_map = {'Never': 0, 'Former': 1, 'Current': 2}
    df['smoking_enc'] = df['smoking_status'].map(sm_map).fillna(0)
    vent_map = {v: i for i, v in enumerate(df['ventilation_type'].dropna().unique())}
    df['vent_enc'] = df['ventilation_type'].map(vent_map).fillna(0)
    # Impute numerics with median
    num_cols = df.select_dtypes(include='number').columns
    for c in num_cols:
        df[c] = df[c].fillna(df[c].median())
    return df

@st.cache_resource(show_spinner=False)
def train_models(df):
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_error

    feature_cols = [
        'age','bmi','sex_enc','smoking_enc',
        'hypertension','diabetes','copd','ckd','cardiovascular_disease','liver_disease',
        'comorbidity_count','high_risk_comorbidity_flag',
        'heart_rate_d0','map_d0','respiratory_rate_d0','spo2_d0',
        'heart_rate_d3','map_d3','respiratory_rate_d3','spo2_d3',
        'pao2_fio2_ratio_d0','fio2_d0','peep_d0','mean_airway_pressure_d0',
        'pao2_fio2_ratio_d3','fio2_d3','peep_d3','mean_airway_pressure_d3',
        'lactate_d0','crp_d0','albumin_d0','platelet_d0','bicarbonate_d0',
        'creatinine_d0','bilirubin_d0','wbc_d0',
        'lactate_d3','crp_d3','albumin_d3','platelet_d3','bicarbonate_d3',
        'creatinine_d3','bilirubin_d3','wbc_d3',
        'sofa_score_d0','sofa_score_d3',
        'vasopressor_use_d0','vasopressor_use_d3','vasopressor_duration',
        'delta_sofa','delta_lactate','delta_pf_ratio','delta_creatinine','delta_crp',
        'shock_index','organ_failure_count','mechanical_ventilation_days','vent_enc'
    ]
    feature_cols = [c for c in feature_cols if c in df.columns]
    X = df[feature_cols].fillna(df[feature_cols].median())
    y_mort = df['mortality_60d']
    y_los  = df['icu_los_days']

    # Risk category encoding
    risk_map = {'Low': 0, 'Medium': 1, 'High': 2}
    y_risk = df['risk_category'].map(risk_map)

    X_tr, X_te, ym_tr, ym_te = train_test_split(X, y_mort, test_size=0.2, random_state=42)
    _, _, yl_tr, yl_te       = train_test_split(X, y_los,  test_size=0.2, random_state=42)
    _, _, yr_tr, yr_te       = train_test_split(X, y_risk, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    # Mortality model
    mort_model = RandomForestClassifier(n_estimators=120, max_depth=8, random_state=42, n_jobs=-1)
    mort_model.fit(X_tr, ym_tr)
    mort_auc = roc_auc_score(ym_te, mort_model.predict_proba(X_te)[:,1])
    mort_preds = mort_model.predict(X_te)

    # LOS model
    los_model = RandomForestRegressor(n_estimators=120, max_depth=8, random_state=42, n_jobs=-1)
    los_model.fit(X_tr, yl_tr)
    los_preds = los_model.predict(X_te)
    los_rmse  = np.sqrt(mean_squared_error(yl_te, los_preds))
    los_mae   = mean_absolute_error(yl_te, los_preds)

    # Risk model
    risk_model = RandomForestClassifier(n_estimators=120, max_depth=8, random_state=42, n_jobs=-1)
    risk_model.fit(X_tr, yr_tr)

    return {
        'mort_model': mort_model,
        'los_model':  los_model,
        'risk_model': risk_model,
        'scaler': scaler,
        'feature_cols': feature_cols,
        'X_test': X_te,
        'y_mort_test': ym_te,
        'y_los_test': yl_te,
        'mort_preds': mort_preds,
        'los_preds': los_preds,
        'mort_auc': mort_auc,
        'los_rmse': los_rmse,
        'los_mae': los_mae,
    }

# ──────────────────────────────────────────
# HELPER: PLOTLY THEME
# ──────────────────────────────────────────
PLOTLY_TEMPLATE = dict(
    layout=dict(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='IBM Plex Sans', color='#94A3B8', size=11),
        xaxis=dict(gridcolor='#1E293B', zerolinecolor='#1E293B', tickfont=dict(size=10)),
        yaxis=dict(gridcolor='#1E293B', zerolinecolor='#1E293B', tickfont=dict(size=10)),
        legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(size=10)),
        margin=dict(l=10, r=10, t=30, b=10),
    )
)

CLR = {
    'blue':  '#3B82F6',
    'red':   '#EF4444',
    'amber': '#F59E0B',
    'green': '#10B981',
    'purple':'#A855F7',
    'cyan':  '#06B6D4',
    'pink':  '#EC4899',
}

def themed_fig(fig, title=None, height=320):
    fig.update_layout(
        template=None,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='IBM Plex Sans', color='#94A3B8', size=11),
        xaxis=dict(gridcolor='#1E293B', zerolinecolor='#1E293B', linecolor='#1E293B'),
        yaxis=dict(gridcolor='#1E293B', zerolinecolor='#1E293B', linecolor='#1E293B'),
        legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(size=10)),
        margin=dict(l=10, r=10, t=36, b=10),
        height=height,
    )
    if title:
        fig.update_layout(title=dict(text=title, font=dict(size=13, color='#CBD5E1'), x=0.01))
    return fig

def section(icon, title, badge=None):
    badge_html = f'<span class="badge">{badge}</span>' if badge else ''
    st.markdown(f"""
    <div class="section-header">
        <span class="icon">{icon}</span>
        <span class="title">{title}</span>
        {badge_html}
    </div>""", unsafe_allow_html=True)

def kpi(color, label, value, delta=None, delta_dir='up'):
    delta_html = f'<div class="kpi-delta {delta_dir}">{delta}</div>' if delta else ''
    st.markdown(f"""
    <div class="kpi-card {color}">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value" style="color:{'#3B82F6' if color=='blue' else '#EF4444' if color=='red' else '#F59E0B' if color=='amber' else '#10B981'}">{value}</div>
        {delta_html}
    </div>""", unsafe_allow_html=True)

# ──────────────────────────────────────────
# SIDEBAR NAVIGATION
# ──────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:16px 0 8px;">
        <div style="font-size:1.15rem;font-weight:700;color:#F1F5F9;letter-spacing:-0.01em;">🫁 ARDS·CDS</div>
        <div style="font-size:0.7rem;color:#374151;letter-spacing:0.08em;text-transform:uppercase;margin-top:2px;">ICU Intelligence v2.0</div>
    </div>
    <hr style="border-color:#1E293B;margin:10px 0;">
    """, unsafe_allow_html=True)

    PAGES = {
        "🏠  Overview": "overview",
        "🧠  Mortality Intelligence": "mortality",
        "⏱  LOS Forecasting": "los",
        "🧬  Patient Phenotyping": "phenotyping",
        "🚨  Risk Engine": "risk",
        "🧑‍⚕️  Live Patient Tool": "live",
        "🤖  Clinical Handover": "handover",
    }
    page_key = st.radio("", list(PAGES.keys()), label_visibility="collapsed")
    page = PAGES[page_key]

    st.markdown("<hr style='border-color:#1E293B;margin:16px 0 10px;'>", unsafe_allow_html=True)
    st.markdown('<div class="nav-header">Global Filters</div>', unsafe_allow_html=True)

    age_range = st.slider("Age Range", 18, 90, (18, 90), step=1)
    gender_sel = st.multiselect("Sex", ["Male", "Female"], default=["Male", "Female"])
    sepsis_only = st.checkbox("Sepsis-Comorbid Only", value=False)
    risk_filter = st.multiselect("Risk Category", ["Low", "Medium", "High"], default=["Low", "Medium", "High"])

    st.markdown("<hr style='border-color:#1E293B;margin:14px 0;'>", unsafe_allow_html=True)
    st.markdown('<div style="font-size:0.65rem;color:#374151;text-align:center;line-height:1.6;">Dataset: 15,000 ICU Patients<br>ARDS Cohort · 64 Features</div>', unsafe_allow_html=True)


# ──────────────────────────────────────────
# LOAD DATA
# ──────────────────────────────────────────
with st.spinner("Initialising clinical intelligence engine…"):
    raw_df = load_data()
    df = preprocess_data(raw_df)

# Apply global filters
fdf = df.copy()
fdf = fdf[(fdf['age'] >= age_range[0]) & (fdf['age'] <= age_range[1])]
if gender_sel:
    fdf = fdf[fdf['sex'].isin(gender_sel)]
if sepsis_only:
    fdf = fdf[fdf['high_risk_comorbidity_flag'] == 1]
if risk_filter:
    fdf = fdf[fdf['risk_category'].isin(risk_filter)]

# ══════════════════════════════════════════
# PAGE: OVERVIEW
# ══════════════════════════════════════════
if page == "overview":
    st.markdown("""
    <div class="top-banner">
        <h1>ICU Clinical Decision Support</h1>
        <p>Integrated AI Framework for ARDS · Real-time predictions · Explainable insights · Clinical-grade outputs</p>
    </div>
    """, unsafe_allow_html=True)

    # KPI Row
    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi('red',   'Mortality Rate',   f"{fdf['mortality_60d'].mean()*100:.1f}%",   '↑ 60-day endpoint', 'down')
    with c2: kpi('amber', 'Avg ICU LOS',      f"{fdf['icu_los_days'].mean():.1f}d",         '± 8.3 days', 'up')
    with c3: kpi('blue',  'High-Risk Patients', f"{(fdf['risk_category']=='High').mean()*100:.1f}%", f"{(fdf['risk_category']=='High').sum():,} pts", 'down')
    with c4: kpi('green', 'Active Cohort',    f"{len(fdf):,}",                              'Filtered patients', 'up')

    st.markdown("<br>", unsafe_allow_html=True)
    section("📊", "Cohort Overview", f"n={len(fdf):,}")

    tab1, tab2, tab3 = st.tabs(["DEMOGRAPHICS", "BIOMARKERS", "CORRELATIONS"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            fig = px.histogram(fdf, x='age', nbins=30, color_discrete_sequence=[CLR['blue']],
                               title='Age Distribution')
            fig.update_traces(marker_line_width=0.5, marker_line_color='#1E293B')
            st.plotly_chart(themed_fig(fig), use_container_width=True)

        with c2:
            sex_cnt = fdf['sex'].value_counts()
            fig = go.Figure(go.Pie(
                labels=sex_cnt.index, values=sex_cnt.values,
                hole=0.62, marker=dict(colors=[CLR['blue'], CLR['pink']],
                                       line=dict(color='#090D14', width=3)),
                textfont=dict(size=11)
            ))
            st.plotly_chart(themed_fig(fig, "Sex Distribution"), use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            risk_cnt = fdf['risk_category'].value_counts()
            clrs = [CLR['green'], CLR['amber'], CLR['red']]
            cats = ['Low', 'Medium', 'High']
            vals = [risk_cnt.get(c, 0) for c in cats]
            fig = go.Figure(go.Bar(
                x=cats, y=vals,
                marker_color=clrs,
                marker_line_width=0,
                text=vals, textposition='outside', textfont=dict(size=10)
            ))
            st.plotly_chart(themed_fig(fig, "Risk Category Distribution"), use_container_width=True)

        with c2:
            smoke_cnt = fdf['smoking_status'].value_counts()
            fig = go.Figure(go.Bar(
                x=smoke_cnt.index, y=smoke_cnt.values,
                marker_color=[CLR['green'], CLR['amber'], CLR['red']],
                marker_line_width=0,
                text=smoke_cnt.values, textposition='outside', textfont=dict(size=10)
            ))
            st.plotly_chart(themed_fig(fig, "Smoking Status"), use_container_width=True)

    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            fig = px.box(fdf, x='risk_category', y='lactate_d0',
                         color='risk_category',
                         color_discrete_map={'Low': CLR['green'], 'Medium': CLR['amber'], 'High': CLR['red']},
                         category_orders={'risk_category': ['Low', 'Medium', 'High']})
            fig.update_traces(line_color='#475569')
            st.plotly_chart(themed_fig(fig, "Lactate D0 by Risk"), use_container_width=True)

        with c2:
            fig = px.box(fdf, x='risk_category', y='sofa_score_d0',
                         color='risk_category',
                         color_discrete_map={'Low': CLR['green'], 'Medium': CLR['amber'], 'High': CLR['red']},
                         category_orders={'risk_category': ['Low', 'Medium', 'High']})
            st.plotly_chart(themed_fig(fig, "SOFA Score D0 by Risk"), use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            fig = px.scatter(fdf.sample(min(1500, len(fdf))), x='pao2_fio2_ratio_d0', y='mortality_60d',
                             color='risk_category',
                             color_discrete_map={'Low': CLR['green'], 'Medium': CLR['amber'], 'High': CLR['red']},
                             opacity=0.5, size_max=4)
            st.plotly_chart(themed_fig(fig, "PaO₂/FiO₂ vs Mortality"), use_container_width=True)

        with c2:
            fig = px.violin(fdf, x='mortality_60d', y='crp_d0', color='mortality_60d',
                            color_discrete_map={0: CLR['green'], 1: CLR['red']},
                            box=True)
            fig.update_layout(xaxis=dict(tickvals=[0, 1], ticktext=['Survived', 'Died']))
            st.plotly_chart(themed_fig(fig, "CRP D0 by Outcome"), use_container_width=True)

    with tab3:
        corr_cols = ['age', 'sofa_score_d0', 'lactate_d0', 'crp_d0', 'pao2_fio2_ratio_d0',
                     'creatinine_d0', 'albumin_d0', 'organ_failure_count', 'mortality_60d', 'icu_los_days']
        corr_cols = [c for c in corr_cols if c in fdf.columns]
        corr = fdf[corr_cols].corr()
        fig = go.Figure(go.Heatmap(
            z=corr.values, x=corr.columns, y=corr.columns,
            colorscale=[[0, '#EF4444'], [0.5, '#1E293B'], [1, '#3B82F6']],
            zmid=0, text=np.round(corr.values, 2),
            texttemplate="%{text}", textfont_size=9,
            colorbar=dict(tickfont=dict(size=9))
        ))
        st.plotly_chart(themed_fig(fig, "Clinical Feature Correlation Matrix", height=480), use_container_width=True)

    # Comorbidity prevalence
    section("🏥", "Comorbidity Prevalence")
    comorbids = ['hypertension', 'diabetes', 'copd', 'ckd', 'cardiovascular_disease', 'liver_disease']
    comorbids = [c for c in comorbids if c in fdf.columns]
    prev = {c: fdf[c].mean() * 100 for c in comorbids}
    fig = go.Figure(go.Bar(
        x=list(prev.values()), y=[c.replace('_', ' ').title() for c in prev.keys()],
        orientation='h',
        marker_color=[CLR['blue'], CLR['cyan'], CLR['amber'], CLR['purple'], CLR['red'], CLR['pink']],
        marker_line_width=0,
        text=[f"{v:.1f}%" for v in prev.values()], textposition='outside',
    ))
    st.plotly_chart(themed_fig(fig, "Comorbidity Prevalence (%)", height=280), use_container_width=True)


# ══════════════════════════════════════════
# PAGE: MORTALITY INTELLIGENCE
# ══════════════════════════════════════════
elif page == "mortality":
    st.markdown("""
    <div class="top-banner">
        <h1>🧠 Mortality Intelligence</h1>
        <p>Binary classification · 60-day endpoint · Random Forest with SHAP explainability</p>
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("Training mortality model…"):
        models = train_models(df)

    mort_model = models['mort_model']
    X_te = models['X_test']
    y_te = models['y_mort_test']
    feat_cols = models['feature_cols']

    col1, col2, col3 = st.columns(3)
    with col1: st.metric("ROC-AUC", f"{models['mort_auc']:.3f}", delta="Target ≥ 0.80")
    from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
    preds = models['mort_preds']
    with col2: st.metric("F1 Score", f"{f1_score(y_te, preds):.3f}")
    with col3: st.metric("Accuracy", f"{accuracy_score(y_te, preds)*100:.1f}%")

    section("📈", "Model Performance")
    tab1, tab2, tab3 = st.tabs(["ROC CURVE", "CONFUSION MATRIX", "FEATURE IMPORTANCE"])

    with tab1:
        from sklearn.metrics import roc_curve
        proba = mort_model.predict_proba(X_te)[:, 1]
        fpr, tpr, _ = roc_curve(y_te, proba)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC (AUC={models["mort_auc"]:.3f})',
                                 line=dict(color=CLR['blue'], width=2.5)))
        fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', name='Random',
                                 line=dict(color='#374151', dash='dash', width=1)))
        fig.update_layout(xaxis_title='False Positive Rate', yaxis_title='True Positive Rate')
        st.plotly_chart(themed_fig(fig, "ROC Curve — Mortality Prediction", height=360), use_container_width=True)

    with tab2:
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_te, preds)
        labels = ['Survived', 'Died']
        fig = go.Figure(go.Heatmap(
            z=cm, x=labels, y=labels,
            colorscale=[[0, '#0F172A'], [1, '#3B82F6']],
            text=cm, texttemplate="%{text}",
            textfont_size=22,
            showscale=False
        ))
        fig.update_layout(xaxis_title='Predicted', yaxis_title='Actual')
        st.plotly_chart(themed_fig(fig, "Confusion Matrix", height=360), use_container_width=True)

    with tab3:
        fi = pd.Series(mort_model.feature_importances_, index=feat_cols).nlargest(20)
        fig = go.Figure(go.Bar(
            x=fi.values[::-1], y=fi.index[::-1], orientation='h',
            marker=dict(
                color=fi.values[::-1],
                colorscale=[[0, '#1E3A5F'], [1, '#3B82F6']],
                line_width=0
            ),
            text=[f"{v:.3f}" for v in fi.values[::-1]], textposition='outside'
        ))
        st.plotly_chart(themed_fig(fig, "Top 20 Feature Importances", height=440), use_container_width=True)

    section("🔍", "SHAP Explainability", "Global")
    with st.expander("View SHAP Feature Impact Analysis", expanded=True):
        try:
            import shap
            sample_size = min(200, len(X_te))
            X_sample = X_te.iloc[:sample_size]
            explainer = shap.TreeExplainer(mort_model)
            shap_vals = explainer.shap_values(X_sample)
            if isinstance(shap_vals, list):
                sv = shap_vals[1]
            else:
                sv = shap_vals[:, :, 1] if shap_vals.ndim == 3 else shap_vals

            mean_abs = np.abs(sv).mean(axis=0)
            top_idx = np.argsort(mean_abs)[-15:]
            top_feats = [feat_cols[i] for i in top_idx]
            top_vals  = mean_abs[top_idx]

            fig = go.Figure(go.Bar(
                x=top_vals, y=top_feats, orientation='h',
                marker=dict(color=top_vals, colorscale=[[0, '#1E3A5F'], [1, '#EF4444']], line_width=0),
                text=[f"{v:.3f}" for v in top_vals], textposition='outside'
            ))
            st.plotly_chart(themed_fig(fig, "Mean |SHAP Value| — Global Impact", height=400), use_container_width=True)
        except Exception as e:
            st.info(f"SHAP analysis: {e}")

    section("🎛️", "What-If Simulator", "Interactive")
    st.markdown('<div class="info-panel">Adjust parameters to see real-time mortality risk changes.</div>', unsafe_allow_html=True)

    med = df[feat_cols].median()
    c1, c2, c3 = st.columns(3)
    with c1:
        sim_sofa = st.slider("SOFA Score D0", 0, 20, int(med.get('sofa_score_d0', 8)))
        sim_lactate = st.slider("Lactate D0", 0.5, 15.0, float(med.get('lactate_d0', 2.5)), 0.1)
        sim_age = st.slider("Age", 18, 90, int(med.get('age', 55)))
    with c2:
        sim_pf = st.slider("PaO₂/FiO₂ D0", 50, 400, int(med.get('pao2_fio2_ratio_d0', 180)))
        sim_crp = st.slider("CRP D0", 1.0, 300.0, float(med.get('crp_d0', 80.0)), 1.0)
        sim_creat = st.slider("Creatinine D0", 0.4, 8.0, float(med.get('creatinine_d0', 1.2)), 0.1)
    with c3:
        sim_organ = st.slider("Organ Failures", 0, 5, int(med.get('organ_failure_count', 1)))
        sim_shock = st.slider("Shock Index", 0.3, 2.5, float(med.get('shock_index', 0.8)), 0.05)
        sim_delta_sofa = st.slider("ΔSOFA (D3-D0)", -5, 10, int(med.get('delta_sofa', 1)))

    sim_input = med.copy()
    sim_input['sofa_score_d0'] = sim_sofa
    sim_input['lactate_d0']    = sim_lactate
    sim_input['age']           = sim_age
    sim_input['pao2_fio2_ratio_d0'] = sim_pf
    sim_input['crp_d0']        = sim_crp
    sim_input['creatinine_d0'] = sim_creat
    sim_input['organ_failure_count'] = sim_organ
    sim_input['shock_index']   = sim_shock
    sim_input['delta_sofa']    = sim_delta_sofa

    sim_prob = mort_model.predict_proba([sim_input.values])[0][1]
    risk_label = "HIGH" if sim_prob > 0.6 else "MEDIUM" if sim_prob > 0.3 else "LOW"
    risk_cls   = "risk-high" if sim_prob > 0.6 else "risk-med" if sim_prob > 0.3 else "risk-low"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=sim_prob * 100,
        number=dict(suffix="%", font=dict(size=36, color='#F1F5F9', family='IBM Plex Mono')),
        gauge=dict(
            axis=dict(range=[0, 100], tickfont=dict(color='#64748B', size=10)),
            bar=dict(color='#EF4444' if sim_prob > 0.6 else '#F59E0B' if sim_prob > 0.3 else '#10B981', thickness=0.3),
            bgcolor='#111827',
            bordercolor='#1E293B',
            steps=[
                dict(range=[0, 30], color='#0F2D1F'),
                dict(range=[30, 60], color='#2D1F0F'),
                dict(range=[60, 100], color='#2D0F0F'),
            ],
            threshold=dict(line=dict(color='#F1F5F9', width=2), value=sim_prob * 100)
        ),
        title=dict(text=f"Mortality Risk — {risk_label}", font=dict(size=13, color='#CBD5E1'))
    ))
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', height=240, margin=dict(t=20,b=0,l=20,r=20))
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════
# PAGE: LOS FORECASTING
# ══════════════════════════════════════════
elif page == "los":
    st.markdown("""
    <div class="top-banner">
        <h1>⏱ LOS Forecasting</h1>
        <p>Regression model · ICU length-of-stay prediction · Proactive bed management</p>
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("Training LOS model…"):
        models = train_models(df)

    los_model   = models['los_model']
    X_te        = models['X_test']
    y_los_te    = models['y_los_test']
    los_preds   = models['los_preds']
    feat_cols   = models['feature_cols']

    c1, c2, c3 = st.columns(3)
    with c1: st.metric("RMSE", f"{models['los_rmse']:.2f} days")
    with c2: st.metric("MAE",  f"{models['los_mae']:.2f} days")
    with c3: st.metric("Avg LOS", f"{df['icu_los_days'].mean():.1f} days")

    section("📈", "Model Performance")
    tab1, tab2, tab3 = st.tabs(["ACTUAL vs PREDICTED", "RESIDUALS", "FEATURE IMPORTANCE"])

    with tab1:
        sample = min(500, len(y_los_te))
        idx = np.random.choice(len(y_los_te), sample, replace=False)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=y_los_te.values[idx], y=los_preds[idx],
            mode='markers', name='Patients',
            marker=dict(color=CLR['blue'], size=4, opacity=0.6)
        ))
        mx = max(y_los_te.max(), los_preds.max())
        fig.add_trace(go.Scatter(x=[0, mx], y=[0, mx], mode='lines', name='Perfect',
                                 line=dict(color='#374151', dash='dash', width=1.5)))
        fig.update_layout(xaxis_title='Actual LOS (days)', yaxis_title='Predicted LOS (days)')
        st.plotly_chart(themed_fig(fig, "Actual vs Predicted ICU LOS", height=380), use_container_width=True)

    with tab2:
        residuals = y_los_te.values - los_preds
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=los_preds, y=residuals,
                                 mode='markers',
                                 marker=dict(color=CLR['cyan'], size=3, opacity=0.5)))
        fig.add_hline(y=0, line_color='#374151', line_dash='dash')
        fig.update_layout(xaxis_title='Predicted LOS', yaxis_title='Residual')
        st.plotly_chart(themed_fig(fig, "Residual Plot", height=300), use_container_width=True)

    with tab3:
        fi = pd.Series(los_model.feature_importances_, index=feat_cols).nlargest(20)
        fig = go.Figure(go.Bar(
            x=fi.values[::-1], y=fi.index[::-1], orientation='h',
            marker=dict(color=fi.values[::-1],
                        colorscale=[[0, '#1A2E40'], [1, '#06B6D4']], line_width=0),
            text=[f"{v:.3f}" for v in fi.values[::-1]], textposition='outside'
        ))
        st.plotly_chart(themed_fig(fig, "Top 20 Features — LOS Model", height=440), use_container_width=True)

    section("🔬", "Scenario Simulation")
    st.markdown('<div class="info-panel warning">Modify parameters to forecast how LOS changes with different clinical profiles.</div>', unsafe_allow_html=True)

    med = df[feat_cols].median()
    c1, c2 = st.columns(2)
    with c1:
        sc_sofa = st.slider("SOFA D0", 0, 20, int(med.get('sofa_score_d0', 8)), key='los_sofa')
        sc_mech = st.slider("Mech. Vent. Days", 0, 29, int(med.get('mechanical_ventilation_days', 10)), key='los_mv')
        sc_pf   = st.slider("PaO₂/FiO₂ D0", 50, 400, int(med.get('pao2_fio2_ratio_d0', 180)), key='los_pf')
    with c2:
        sc_organ = st.slider("Organ Failures", 0, 5, int(med.get('organ_failure_count', 1)), key='los_organ')
        sc_lact  = st.slider("Lactate D0", 0.5, 15.0, float(med.get('lactate_d0', 2.5)), 0.1, key='los_lact')
        sc_alb   = st.slider("Albumin D0", 1.5, 5.0, float(med.get('albumin_d0', 3.2)), 0.1, key='los_alb')

    sc_input = med.copy()
    sc_input['sofa_score_d0'] = sc_sofa
    sc_input['mechanical_ventilation_days'] = sc_mech
    sc_input['pao2_fio2_ratio_d0'] = sc_pf
    sc_input['organ_failure_count'] = sc_organ
    sc_input['lactate_d0'] = sc_lact
    sc_input['albumin_d0'] = sc_alb

    pred_los = los_model.predict([sc_input.values])[0]
    pred_los = max(1, min(29, pred_los))

    c1, c2 = st.columns([1, 2])
    with c1:
        tier = "EXTENDED" if pred_los > 21 else "MODERATE" if pred_los > 10 else "SHORT"
        clr  = CLR['red'] if pred_los > 21 else CLR['amber'] if pred_los > 10 else CLR['green']
        st.markdown(f"""
        <div class="pred-card">
            <div class="label">Predicted ICU LOS</div>
            <div class="value" style="color:{clr}">{pred_los:.1f}</div>
            <div class="sub">days · {tier} stay</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        sofa_vals = list(range(1, 21))
        preds_curve = []
        for sv in sofa_vals:
            inp = sc_input.copy()
            inp['sofa_score_d0'] = sv
            preds_curve.append(max(1, min(29, los_model.predict([inp.values])[0])))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=sofa_vals, y=preds_curve, mode='lines+markers',
                                 line=dict(color=CLR['cyan'], width=2),
                                 marker=dict(size=5),
                                 name='Predicted LOS'))
        fig.add_vline(x=sc_sofa, line_color='#F59E0B', line_dash='dash',
                      annotation_text=f"Current: {sc_sofa}", annotation_font_size=10)
        fig.update_layout(xaxis_title='SOFA Score D0', yaxis_title='Predicted LOS (days)')
        st.plotly_chart(themed_fig(fig, "LOS vs SOFA Score", height=220), use_container_width=True)


# ══════════════════════════════════════════
# PAGE: PATIENT PHENOTYPING
# ══════════════════════════════════════════
elif page == "phenotyping":
    st.markdown("""
    <div class="top-banner">
        <h1>🧬 Patient Phenotyping</h1>
        <p>Unsupervised clustering · ARDS subgroup identification · Personalised treatment targeting</p>
    </div>
    """, unsafe_allow_html=True)

    @st.cache_data(show_spinner=False)
    def run_clustering(df):
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        feat = ['sofa_score_d0','lactate_d0','crp_d0','albumin_d0','pao2_fio2_ratio_d0',
                'organ_failure_count','age','creatinine_d0','wbc_d0','platelet_d0']
        feat = [f for f in feat if f in df.columns]
        X = df[feat].fillna(df[feat].median())
        sc = StandardScaler()
        Xs = sc.fit_transform(X)
        pca = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(Xs)
        return coords, df['phenotype_cluster'].values, pca.explained_variance_ratio_, feat

    coords, clusters, evr, feat = run_clustering(df)

    cmap = {'Cluster_1': CLR['blue'], 'Cluster_2': CLR['green'], 'Cluster_3': CLR['red']}
    plot_df = pd.DataFrame({'PC1': coords[:,0], 'PC2': coords[:,1],
                            'Cluster': clusters,
                            'mortality': df['mortality_60d'].astype(str)})

    c1, c2 = st.columns(3)
    for i, (cl, cnt) in enumerate(df['phenotype_cluster'].value_counts().items()):
        cols = [c1, c2, st.columns(3)[2]] if i < 3 else []
        pct = cnt / len(df) * 100
    for cl, cnt in df['phenotype_cluster'].value_counts().items():
        pct = cnt / len(df) * 100

    col1, col2, col3 = st.columns(3)
    cnt = df['phenotype_cluster'].value_counts()
    with col1: st.metric("Cluster 1 (Moderate)", f"{cnt.get('Cluster_1',0):,}", f"{cnt.get('Cluster_1',0)/len(df)*100:.1f}%")
    with col2: st.metric("Cluster 2 (Mild)", f"{cnt.get('Cluster_2',0):,}", f"{cnt.get('Cluster_2',0)/len(df)*100:.1f}%")
    with col3: st.metric("Cluster 3 (Severe)", f"{cnt.get('Cluster_3',0):,}", f"{cnt.get('Cluster_3',0)/len(df)*100:.1f}%")

    section("🔵", "PCA Cluster Visualisation")

    sample_n = min(2000, len(plot_df))
    sample_df = plot_df.sample(sample_n, random_state=42)
    fig = px.scatter(sample_df, x='PC1', y='PC2', color='Cluster',
                     color_discrete_map=cmap, opacity=0.65, size_max=5)
    fig.update_traces(marker_size=4)
    fig.update_layout(
        xaxis_title=f"PC1 ({evr[0]*100:.1f}% variance)",
        yaxis_title=f"PC2 ({evr[1]*100:.1f}% variance)",
    )
    st.plotly_chart(themed_fig(fig, "PCA — ARDS Patient Clusters", height=420), use_container_width=True)

    section("📊", "Cluster Profiles")
    profile_feats = ['sofa_score_d0','lactate_d0','crp_d0','albumin_d0',
                     'pao2_fio2_ratio_d0','organ_failure_count','mortality_60d','icu_los_days']
    profile_feats = [f for f in profile_feats if f in df.columns]
    profile = df.groupby('phenotype_cluster')[profile_feats].mean().round(2)

    fig = go.Figure()
    clrs_list = [CLR['blue'], CLR['green'], CLR['red']]
    for i, cl in enumerate(profile.index):
        fig.add_trace(go.Bar(
            name=cl, x=[c.replace('_', ' ').replace(' d0', ' D0').title() for c in profile_feats],
            y=profile.loc[cl].values,
            marker_color=clrs_list[i % 3],
            marker_line_width=0
        ))
    fig.update_layout(barmode='group')
    st.plotly_chart(themed_fig(fig, "Mean Feature Values by Cluster", height=380), use_container_width=True)

    section("💡", "Cluster Interpretation")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class="info-panel">
            <div style="color:#60A5FA;font-size:0.7rem;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:6px;">🔵 Cluster 1 — Moderate Inflammatory</div>
            <div style="font-size:0.82rem;color:#CBD5E1;line-height:1.6;">
            Intermediate SOFA scores, moderate CRP elevation, moderate PaO₂/FiO₂ impairment. Represents the largest subgroup. Intermediate mortality risk — benefits from early intervention.
            </div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="info-panel success">
            <div style="color:#4ADE80;font-size:0.7rem;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:6px;">🟢 Cluster 2 — Mild / Recovering</div>
            <div style="font-size:0.82rem;color:#CBD5E1;line-height:1.6;">
            Lower lactate and CRP, better albumin levels, higher PaO₂/FiO₂ ratios. Favourable outcomes, shorter ICU stays. Candidates for early weaning protocols.
            </div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="info-panel danger">
            <div style="color:#F87171;font-size:0.7rem;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:6px;">🔴 Cluster 3 — Severe / Multi-organ</div>
            <div style="font-size:0.82rem;color:#CBD5E1;line-height:1.6;">
            Highest SOFA scores, severe hyperlactataemia, worst oxygenation. Multi-organ dysfunction dominant. Requires aggressive management and escalation planning.
            </div>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════
# PAGE: RISK ENGINE
# ══════════════════════════════════════════
elif page == "risk":
    st.markdown("""
    <div class="top-banner">
        <h1>🚨 Risk Engine</h1>
        <p>Sepsis-comorbid ARDS · Multi-class stratification · Biomarker-driven triage</p>
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("Loading risk models…"):
        models = train_models(df)

    risk_model = models['risk_model']
    X_te = models['X_test']
    feat_cols = models['feature_cols']

    section("⚠️", "Risk Distribution")
    risk_cnt = fdf['risk_category'].value_counts()

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure(go.Pie(
            labels=['Low', 'Medium', 'High'],
            values=[risk_cnt.get('Low', 0), risk_cnt.get('Medium', 0), risk_cnt.get('High', 0)],
            hole=0.55,
            marker=dict(colors=[CLR['green'], CLR['amber'], CLR['red']],
                        line=dict(color='#090D14', width=3)),
        ))
        st.plotly_chart(themed_fig(fig, "Risk Tier Distribution", height=300), use_container_width=True)

    with c2:
        fig = px.box(fdf, x='risk_category', y='sofa_score_d0',
                     color='risk_category',
                     color_discrete_map={'Low': CLR['green'], 'Medium': CLR['amber'], 'High': CLR['red']},
                     category_orders={'risk_category': ['Low', 'Medium', 'High']},
                     points='outliers')
        st.plotly_chart(themed_fig(fig, "SOFA Score by Risk Tier", height=300), use_container_width=True)

    section("🔥", "Key Biomarker Analysis")
    biomarkers = ['crp_d0', 'lactate_d0', 'creatinine_d0', 'albumin_d0', 'wbc_d0', 'platelet_d0']
    biomarkers = [b for b in biomarkers if b in fdf.columns]

    bio_agg = fdf.groupby('risk_category')[biomarkers].mean()
    bio_norm = (bio_agg - bio_agg.min()) / (bio_agg.max() - bio_agg.min() + 1e-9)

    fig = go.Figure(go.Heatmap(
        z=bio_norm.values,
        x=[b.replace('_d0','').upper() for b in biomarkers],
        y=bio_norm.index,
        colorscale=[[0, '#0F2D1F'], [0.5, '#1E293B'], [1, '#EF4444']],
        text=np.round(bio_agg.values, 2),
        texttemplate="%{text}",
        textfont_size=10,
        colorbar=dict(tickfont=dict(size=9))
    ))
    st.plotly_chart(themed_fig(fig, "Biomarker Profile by Risk Tier (Normalised)", height=200), use_container_width=True)

    section("📉", "Trend Analysis — D0 → D3")
    trend_feats = [('sofa_score_d0', 'sofa_score_d3', 'SOFA Score'),
                   ('lactate_d0', 'lactate_d3', 'Lactate'),
                   ('crp_d0', 'crp_d3', 'CRP')]

    for d0, d3, name in trend_feats:
        if d0 in fdf.columns and d3 in fdf.columns:
            pass

    c1, c2, c3 = st.columns(3)
    for col, (d0, d3, name) in zip([c1, c2, c3], trend_feats):
        if d0 in fdf.columns and d3 in fdf.columns:
            agg_d0 = fdf.groupby('risk_category')[d0].mean()
            agg_d3 = fdf.groupby('risk_category')[d3].mean()
            with col:
                fig = go.Figure()
                for rc, clr in [('Low', CLR['green']), ('Medium', CLR['amber']), ('High', CLR['red'])]:
                    if rc in agg_d0.index:
                        fig.add_trace(go.Scatter(
                            x=['Day 0', 'Day 3'],
                            y=[agg_d0[rc], agg_d3[rc]],
                            mode='lines+markers', name=rc,
                            line=dict(color=clr, width=2),
                            marker=dict(size=8)
                        ))
                st.plotly_chart(themed_fig(fig, f"{name} D0→D3", height=240), use_container_width=True)

    section("🔬", "Risk Feature Drivers")
    fi = pd.Series(risk_model.feature_importances_, index=feat_cols).nlargest(15)
    fig = go.Figure(go.Bar(
        x=fi.values[::-1], y=fi.index[::-1], orientation='h',
        marker=dict(color=fi.values[::-1],
                    colorscale=[[0, '#2D0F0F'], [1, '#EF4444']], line_width=0),
        text=[f"{v:.3f}" for v in fi.values[::-1]], textposition='outside'
    ))
    st.plotly_chart(themed_fig(fig, "Top Feature Drivers — Risk Model", height=380), use_container_width=True)


# ══════════════════════════════════════════
# PAGE: LIVE PATIENT TOOL
# ══════════════════════════════════════════
elif page == "live":
    st.markdown("""
    <div class="top-banner">
        <h1>🧑‍⚕️ Live Patient Assessment</h1>
        <p>Enter patient data for real-time AI predictions across all clinical dimensions</p>
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("Loading models…"):
        models = train_models(df)

    mort_model  = models['mort_model']
    los_model   = models['los_model']
    risk_model  = models['risk_model']
    feat_cols   = models['feature_cols']
    med         = df[feat_cols].median()

    st.markdown("### Patient Demographics & History")
    c1, c2, c3, c4 = st.columns(4)
    with c1: p_age  = st.number_input("Age (yrs)", 18, 90, 55)
    with c2: p_sex  = st.selectbox("Sex", ["Male", "Female"])
    with c3: p_bmi  = st.number_input("BMI", 15.0, 60.0, 25.0, 0.1)
    with c4: p_smk  = st.selectbox("Smoking", ["Never", "Former", "Current"])

    st.markdown("### Comorbidities")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1: p_htn  = int(st.checkbox("Hypertension"))
    with c2: p_dm   = int(st.checkbox("Diabetes"))
    with c3: p_copd = int(st.checkbox("COPD"))
    with c4: p_ckd  = int(st.checkbox("CKD"))
    with c5: p_cvd  = int(st.checkbox("CVD"))
    with c6: p_liver= int(st.checkbox("Liver Dis."))

    st.markdown("### Day 0 Vitals & Labs")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        p_hr0   = st.number_input("Heart Rate D0", 40, 180, 90)
        p_map0  = st.number_input("MAP D0 (mmHg)", 40, 130, 72)
        p_rr0   = st.number_input("RR D0", 8, 50, 22)
        p_spo0  = st.number_input("SpO₂ D0 (%)", 70, 100, 92)
    with c2:
        p_pf0   = st.number_input("PaO₂/FiO₂ D0", 50, 400, 180)
        p_peep0 = st.number_input("PEEP D0", 0, 25, 8)
        p_fio0  = st.number_input("FiO₂ D0", 0.21, 1.0, 0.6, 0.01)
        p_sofa0 = st.number_input("SOFA D0", 0, 20, 8)
    with c3:
        p_lact0 = st.number_input("Lactate D0", 0.5, 15.0, 2.0, 0.1)
        p_crp0  = st.number_input("CRP D0 (mg/L)", 1.0, 400.0, 80.0, 1.0)
        p_creat0= st.number_input("Creatinine D0", 0.4, 10.0, 1.2, 0.1)
        p_alb0  = st.number_input("Albumin D0 (g/dL)", 1.5, 5.0, 3.2, 0.1)
    with c4:
        p_plt0  = st.number_input("Platelets D0 (×10³)", 20, 600, 200)
        p_wbc0  = st.number_input("WBC D0 (×10³)", 1.0, 30.0, 10.0, 0.5)
        p_bili0 = st.number_input("Bilirubin D0", 0.1, 20.0, 1.2, 0.1)
        p_bicarb0= st.number_input("Bicarbonate D0", 10.0, 35.0, 22.0, 0.5)

    st.markdown("### Day 3 & Derived Features")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        p_sofa3 = st.number_input("SOFA D3", 0, 20, 7)
        p_lact3 = st.number_input("Lactate D3", 0.5, 15.0, 1.8, 0.1)
    with c2:
        p_pf3   = st.number_input("PaO₂/FiO₂ D3", 50, 400, 200)
        p_creat3= st.number_input("Creatinine D3", 0.4, 10.0, 1.1, 0.1)
    with c3:
        p_organ = st.number_input("Organ Failures", 0, 5, 1)
        p_shock = st.number_input("Shock Index", 0.3, 3.0, 0.8, 0.05)
    with c4:
        p_mv    = st.number_input("Mech. Vent. Days", 0, 29, 5)
        p_vaso  = int(st.checkbox("Vasopressor Use D0"))

    if st.button("🚀  Run AI Assessment", use_container_width=True):
        sm_map = {'Never': 0, 'Former': 1, 'Current': 2}
        comorbid_cnt = p_htn + p_dm + p_copd + p_ckd + p_cvd + p_liver
        high_risk_flag = int(comorbid_cnt >= 3)

        patient_vals = {
            'age': p_age, 'bmi': p_bmi, 'sex_enc': int(p_sex == 'Male'),
            'smoking_enc': sm_map[p_smk],
            'hypertension': p_htn, 'diabetes': p_dm, 'copd': p_copd,
            'ckd': p_ckd, 'cardiovascular_disease': p_cvd, 'liver_disease': p_liver,
            'comorbidity_count': comorbid_cnt, 'high_risk_comorbidity_flag': high_risk_flag,
            'heart_rate_d0': p_hr0, 'map_d0': p_map0, 'respiratory_rate_d0': p_rr0, 'spo2_d0': p_spo0,
            'pao2_fio2_ratio_d0': p_pf0, 'fio2_d0': p_fio0, 'peep_d0': p_peep0,
            'mean_airway_pressure_d0': med.get('mean_airway_pressure_d0', 12),
            'lactate_d0': p_lact0, 'crp_d0': p_crp0, 'albumin_d0': p_alb0,
            'platelet_d0': p_plt0, 'bicarbonate_d0': p_bicarb0, 'creatinine_d0': p_creat0,
            'bilirubin_d0': p_bili0, 'wbc_d0': p_wbc0,
            'sofa_score_d0': p_sofa0, 'sofa_score_d3': p_sofa3,
            'delta_sofa': p_sofa3 - p_sofa0, 'delta_lactate': p_lact3 - p_lact0,
            'delta_pf_ratio': p_pf3 - p_pf0, 'delta_creatinine': p_creat3 - p_creat0,
            'delta_crp': med.get('delta_crp', 0),
            'shock_index': p_shock, 'organ_failure_count': p_organ,
            'mechanical_ventilation_days': p_mv, 'vasopressor_use_d0': p_vaso,
            'vasopressor_use_d3': int(p_vaso), 'vasopressor_duration': med.get('vasopressor_duration', 0),
            'vent_enc': 0, 'minute_ventilation_d3': med.get('minute_ventilation_d3', 8),
        }
        # Fill remaining features with median
        inp_vec = []
        for fc in feat_cols:
            inp_vec.append(patient_vals.get(fc, float(med.get(fc, 0))))

        mort_prob  = mort_model.predict_proba([inp_vec])[0][1]
        pred_los   = max(1, min(29, los_model.predict([inp_vec])[0]))
        risk_proba = risk_model.predict_proba([inp_vec])[0]
        risk_cls_idx = np.argmax(risk_proba)
        risk_labels = ['Low', 'Medium', 'High']
        pred_risk  = risk_labels[risk_cls_idx]

        # Cluster assignment based on SOFA/lactate heuristic
        if p_sofa0 >= 10 or p_lact0 >= 4:
            pred_cluster = "Cluster 3 — Severe"
            cl_cls = "c3"
        elif p_sofa0 >= 5 or p_lact0 >= 2:
            pred_cluster = "Cluster 1 — Moderate"
            cl_cls = "c1"
        else:
            pred_cluster = "Cluster 2 — Mild"
            cl_cls = "c2"

        st.markdown("---")
        st.markdown("## 🧾 Assessment Results")

        c1, c2, c3, c4 = st.columns(4)
        mort_color = CLR['red'] if mort_prob > 0.6 else CLR['amber'] if mort_prob > 0.3 else CLR['green']
        los_color  = CLR['red'] if pred_los > 21 else CLR['amber'] if pred_los > 10 else CLR['green']
        risk_color = CLR['red'] if pred_risk == 'High' else CLR['amber'] if pred_risk == 'Medium' else CLR['green']

        with c1:
            st.markdown(f"""<div class="pred-card"><div class="label">🚨 Mortality Risk</div>
            <div class="value" style="color:{mort_color}">{mort_prob*100:.1f}%</div>
            <div class="sub">60-day probability</div></div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""<div class="pred-card"><div class="label">⏱ Predicted LOS</div>
            <div class="value" style="color:{los_color}">{pred_los:.1f}d</div>
            <div class="sub">ICU length of stay</div></div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""<div class="pred-card"><div class="label">⚠️ Risk Tier</div>
            <div class="value" style="color:{risk_color}">{pred_risk}</div>
            <div class="sub">Sepsis-ARDS composite</div></div>""", unsafe_allow_html=True)
        with c4:
            st.markdown(f"""<div class="pred-card"><div class="label">🧬 Phenotype</div>
            <div class="value" style="font-size:1.3rem;color:#94A3B8">{pred_cluster.split('—')[0].strip()}</div>
            <div class="sub">{pred_cluster.split('—')[1].strip() if '—' in pred_cluster else ''}</div></div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        c1, c2 = st.columns([1, 2])
        with c1:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=mort_prob * 100,
                number=dict(suffix="%", font=dict(size=32, family='IBM Plex Mono', color='#F1F5F9')),
                gauge=dict(
                    axis=dict(range=[0, 100], tickfont=dict(size=9, color='#64748B')),
                    bar=dict(color=mort_color, thickness=0.28),
                    bgcolor='#111827', bordercolor='#1E293B',
                    steps=[
                        dict(range=[0, 30], color='#0F2D1F'),
                        dict(range=[30, 60], color='#2D1F0F'),
                        dict(range=[60, 100], color='#2D0F0F'),
                    ],
                ),
                title=dict(text="Mortality Risk", font=dict(size=12, color='#CBD5E1'))
            ))
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', height=220, margin=dict(t=20,b=0,l=20,r=20))
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            fi = pd.Series(mort_model.feature_importances_, index=feat_cols)
            top_fi = fi.nlargest(8)
            fig = go.Figure(go.Bar(
                x=top_fi.values[::-1], y=top_fi.index[::-1],
                orientation='h',
                marker=dict(color=[CLR['blue'] if v < 0.06 else CLR['amber'] if v < 0.10 else CLR['red']
                                   for v in top_fi.values[::-1]], line_width=0),
            ))
            fig.update_layout(xaxis_title='Feature Importance')
            st.plotly_chart(themed_fig(fig, "🔥 Top Contributing Factors", height=220), use_container_width=True)

        # Clinical alerts
        alerts = []
        if mort_prob > 0.6:
            alerts.append(("danger", "🚨 HIGH MORTALITY RISK — Immediate escalation of care recommended"))
        if pred_los > 21:
            alerts.append(("warning", "⏱ EXTENDED STAY PREDICTED — Initiate resource planning & family communication"))
        if p_sofa0 >= 10:
            alerts.append(("danger", "⚠️ SEVERE SOFA SCORE — Consider ICU senior review and organ support"))
        if p_lact0 >= 4:
            alerts.append(("danger", "🔬 HYPERLACTATAEMIA — Assess perfusion status and resuscitation adequacy"))
        if p_pf0 < 100:
            alerts.append(("warning", "🫁 SEVERE ARDS — PaO₂/FiO₂ < 100 · Consider prone positioning"))

        if alerts:
            section("🚨", "Clinical Alerts")
            for atype, msg in alerts:
                st.markdown(f'<div class="info-panel {atype}">{msg}</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════
# PAGE: CLINICAL HANDOVER
# ══════════════════════════════════════════
elif page == "handover":
    st.markdown("""
    <div class="top-banner">
        <h1>🤖 AI Clinical Handover</h1>
        <p>SBAR / I-PASS structured documentation · AI-generated clinical summaries</p>
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("Loading models…"):
        models = train_models(df)

    mort_model = models['mort_model']
    los_model  = models['los_model']
    risk_model = models['risk_model']
    feat_cols  = models['feature_cols']
    med        = df[feat_cols].median()

    st.markdown("### Patient Snapshot")
    c1, c2, c3 = st.columns(3)
    with c1:
        h_name   = st.text_input("Patient ID / Name", "ARDS-12345")
        h_age    = st.number_input("Age", 18, 90, 58, key='h_age')
        h_sex    = st.selectbox("Sex", ["Male", "Female"], key='h_sex')
    with c2:
        h_sofa   = st.number_input("SOFA D0", 0, 20, 9, key='h_sofa')
        h_lact   = st.number_input("Lactate D0", 0.5, 15.0, 3.2, 0.1, key='h_lact')
        h_pf     = st.number_input("PaO₂/FiO₂", 50, 400, 140, key='h_pf')
    with c3:
        h_risk   = st.selectbox("Known Risk Category", ["Low", "Medium", "High"], index=2, key='h_risk')
        h_diag   = st.text_area("Primary Diagnosis", "Severe ARDS secondary to community-acquired pneumonia", key='h_diag')

    c1, c2 = st.columns(2)
    with c1: h_htn = st.checkbox("Hypertension", key='h_htn')
    with c2: h_dm  = st.checkbox("Diabetes", key='h_dm')

    if st.button("🤖  Generate SBAR Handover Note", use_container_width=True):
        # Build quick prediction
        inp = med.copy()
        inp['age']                = h_age
        inp['sex_enc']            = int(h_sex == 'Male')
        inp['sofa_score_d0']      = h_sofa
        inp['lactate_d0']         = h_lact
        inp['pao2_fio2_ratio_d0'] = h_pf
        inp['hypertension']       = int(h_htn)
        inp['diabetes']           = int(h_dm)

        inp_vec = [float(inp.get(fc, 0)) for fc in feat_cols]
        mort_prob = mort_model.predict_proba([inp_vec])[0][1]
        pred_los  = max(1, min(29, los_model.predict([inp_vec])[0]))

        risk_color = {'Low': '#10B981', 'Medium': '#F59E0B', 'High': '#EF4444'}[h_risk]
        mort_tier  = "HIGH" if mort_prob > 0.6 else "MODERATE" if mort_prob > 0.3 else "LOW"

        comorbids_str = ', '.join([x for x, v in [("hypertension", h_htn), ("diabetes", h_dm)] if v]) or "none documented"

        # SBAR Content
        situation = (
            f"Patient {h_name}, {h_age}-year-old {h_sex.lower()}, currently admitted to the ICU with "
            f"<b>{h_diag}</b>. Risk category: <b style='color:{risk_color}'>{h_risk}</b>. "
            f"AI-predicted 60-day mortality: <b>{mort_prob*100:.1f}%</b> ({mort_tier} tier)."
        )

        background = (
            f"Known comorbidities: {comorbids_str}. Presenting SOFA score: <b>{h_sofa}</b>. "
            f"Lactate on admission: <b>{h_lact} mmol/L</b>. "
            f"PaO₂/FiO₂ ratio: <b>{h_pf}</b> (ARDS severity: "
            f"{'Severe' if h_pf < 100 else 'Moderate' if h_pf < 200 else 'Mild'}). "
            f"Patient phenotype consistent with "
            f"{'severe inflammatory / multi-organ failure' if h_sofa >= 10 or h_lact >= 4 else 'moderate inflammatory' if h_sofa >= 5 else 'mild / recovering'} cluster."
        )

        assessment = (
            f"AI framework assessment: Predicted ICU LOS <b>{pred_los:.1f} days</b>. "
            f"Key risk drivers include SOFA score, lactate trajectory, and PaO₂/FiO₂ impairment. "
            f"{'Hyperlactataemia warrants urgent assessment of tissue perfusion. ' if h_lact >= 4 else ''}"
            f"{'Severe oxygenation impairment (PaO₂/FiO₂ < 100): prone positioning should be considered. ' if h_pf < 100 else ''}"
            f"Patient falls into {h_risk}-risk tier for sepsis-comorbid ARDS composite scoring."
        )

        recommendation = (
            f"{'1. Immediate senior ICU physician review. ' if mort_prob > 0.6 else '1. Continue current management with daily senior review. '}"
            f"2. Repeat lactate in 4–6 hours to assess trajectory. "
            f"3. {'Initiate prone positioning protocol per local guidelines. ' if h_pf < 150 else 'Optimise ventilator settings for lung-protective strategy. '}"
            f"4. {'Goals-of-care discussion with family recommended given high mortality prediction. ' if mort_prob > 0.6 else ''}"
            f"5. Arrange bed management planning for predicted {pred_los:.0f}-day ICU stay. "
            f"6. Reassess SOFA and lactate at Day 3 for trajectory-based risk re-stratification."
        )

        st.markdown("---")
        st.markdown("## 📋 SBAR Handover Note")

        for label, content, color in [
            ("S — Situation", situation, "#3B82F6"),
            ("B — Background", background, "#A855F7"),
            ("A — Assessment", assessment, "#F59E0B"),
            ("R — Recommendation", recommendation, "#10B981"),
        ]:
            st.markdown(f"""
            <div class="sbar-section">
                <div class="sbar-label" style="color:{color}">{label}</div>
                <div class="sbar-content">{content}</div>
            </div>""", unsafe_allow_html=True)

        # I-PASS summary
        st.markdown("## 📌 I-PASS Summary")
        ipass = {
            "I — Illness Severity": f"{h_risk} Risk · Mortality: {mort_prob*100:.1f}%",
            "P — Patient Summary": f"{h_age}y {h_sex} · {h_diag}",
            "A — Action List": f"Repeat lactate · Ventilator optimisation · {'Prone positioning' if h_pf < 150 else 'Lung protective ventilation'}",
            "S — Situation Awareness": f"SOFA {h_sofa} · Lactate {h_lact} · PF {h_pf} · Predicted LOS {pred_los:.1f}d",
            "S — Synthesis by Receiver": "Confirm understanding · Clarify outstanding investigations · Escalation trigger: SOFA worsening or lactate rise",
        }
        for key, val in ipass.items():
            st.markdown(f"""
            <div style="display:flex;align-items:flex-start;gap:16px;padding:10px 0;border-bottom:1px solid #1E293B;">
                <div style="font-size:0.72rem;font-family:'IBM Plex Mono';color:#3B82F6;min-width:200px;padding-top:2px;text-transform:uppercase;letter-spacing:0.06em;">{key}</div>
                <div style="font-size:0.88rem;color:#CBD5E1;line-height:1.5;">{val}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class="info-panel" style="background:#0A1628;border-left-color:#1E293B;">
            <div style="font-size:0.65rem;color:#374151;font-family:'IBM Plex Mono';">
            ⚠️ AI-GENERATED CLINICAL SUMMARY — FOR DECISION SUPPORT ONLY. All AI predictions should be interpreted in clinical context by a qualified physician. 
            Not a substitute for professional medical judgment. Generated: {pd.Timestamp.now().strftime('%d %b %Y %H:%M')} UTC
            </div>
        </div>""", unsafe_allow_html=True)
