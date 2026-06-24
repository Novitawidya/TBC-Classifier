
import streamlit as st
import pandas as pd
import numpy as np
import os
import glob
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_curve, roc_auc_score
)

DEFAULT_FILE = "Tuberculosis_Trends 2000-2023.xlsx"
REQUIRED_COLUMNS = ["Country", "Region", "Income_Level", "Year", "TB_Treatment_Success_Rate"]


def validate_columns(dataframe):
    return [c for c in REQUIRED_COLUMNS if c not in dataframe.columns]


# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="TBC Classifier Dashboard",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

:root {
    --primary:      #14E0B4;
    --primary-dark: #0B8F73;
    --accent:       #FF7A7A;
    --accent2:      #FFD166;
    --bg-app:       #0A0E14;
    --bg-card:      #161B26;
    --bg-card2:     #1F2633;
    --bg-input:     #20283A;
    --text-heading: #FFFFFF;
    --text-body:    #DCE3F0;
    --text-muted:   #97A3BD;
    --text-on-primary: #062A22;
    --success:      #2ECC71;
    --warning:      #FFC542;
    --danger:       #FF5C5C;
    --border:       rgba(20,224,180,0.30);
    --border-soft:  rgba(255,255,255,0.08);
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    color: var(--text-body) !important;
    font-size: 16px;
}
.stApp { background: var(--bg-app); }
p, span, div, label, li, td, th { color: var(--text-body); }
h1, h2, h3, h4, h5, h6 { color: var(--text-heading) !important; }

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0E141F 0%, #0A0E14 100%);
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * { color: var(--text-body) !important; }
[data-testid="stSidebar"] .stRadio label {
    color: var(--text-body) !important;
    font-size: 1.02rem;
    font-weight: 600;
    padding: 10px 4px;
    cursor: pointer;
    transition: color .2s;
}
[data-testid="stSidebar"] .stRadio label:hover { color: var(--primary) !important; }
[data-testid="stSidebar"] .stRadio [data-baseweb="radio"] div:first-child {
    border-color: var(--primary) !important;
}

#MainMenu, footer { visibility: hidden; }
header { visibility: visible; }
[data-testid="collapsedControl"] { visibility: visible !important; display: flex !important; }

.hero-banner {
    background: linear-gradient(135deg, #0B2B26, #0F3F38, #11544A);
    border-radius: 18px;
    padding: 52px 44px;
    margin-bottom: 30px;
    border: 1px solid var(--border);
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: "🫁";
    position: absolute;
    right: 36px; top: 16px;
    font-size: 110px;
    opacity: .14;
}
.hero-title {
    font-size: 3rem;
    font-weight: 900;
    color: var(--text-heading);
    margin: 0 0 10px;
    letter-spacing: -0.5px;
}
.hero-sub {
    font-size: 1.18rem;
    color: #C7F5EA;
    margin: 0;
    font-weight: 500;
}

.card {
    background: var(--bg-card);
    border: 1px solid var(--border-soft);
    border-radius: 16px;
    padding: 28px 30px;
    margin-bottom: 22px;
}
.card p, .card li, .card span { color: var(--text-body); }
.card-title {
    font-size: 1.25rem;
    font-weight: 800;
    color: var(--primary);
    margin: 0 0 16px;
    display: flex;
    align-items: center;
    gap: 10px;
}

.metric-grid { display: flex; gap: 18px; flex-wrap: wrap; margin-bottom: 22px; }
.metric-box {
    flex: 1; min-width: 165px;
    background: var(--bg-card2);
    border-left: 5px solid var(--primary);
    border-radius: 12px;
    padding: 22px 22px;
    text-align: center;
}
.metric-box.danger  { border-left-color: var(--danger); }
.metric-box.warning { border-left-color: var(--warning); }
.metric-box.success { border-left-color: var(--success); }
.metric-val {
    font-size: 2.1rem;
    font-weight: 900;
    color: var(--primary);
    line-height: 1.15;
}
.metric-box.danger  .metric-val { color: var(--danger); }
.metric-box.warning .metric-val { color: var(--warning); }
.metric-box.success .metric-val { color: var(--success); }
.metric-lbl {
    font-size: 0.92rem;
    color: var(--text-body);
    margin-top: 6px;
    font-weight: 600;
}

.sec-header {
    font-size: 1.6rem;
    font-weight: 800;
    color: var(--text-heading);
    border-bottom: 3px solid var(--primary);
    padding-bottom: 10px;
    margin: 34px 0 22px;
    display: inline-block;
}

.info-box {
    background: rgba(20,224,180,.12);
    border-left: 5px solid var(--primary);
    border-radius: 10px;
    padding: 18px 22px;
    margin: 14px 0;
    font-size: 1rem;
    line-height: 1.7;
    color: var(--text-body);
}
.warn-box {
    background: rgba(255,92,92,.12);
    border-left: 5px solid var(--accent);
    border-radius: 10px;
    padding: 18px 22px;
    margin: 14px 0;
    font-size: 1rem;
    line-height: 1.7;
    color: var(--text-body);
}
.warn-box b, .info-box b { color: var(--text-heading); }

.synth-badge {
    background: rgba(255,209,102,.15);
    border: 1.5px solid #FFD166;
    border-radius: 8px;
    padding: 10px 18px;
    font-size: .88rem;
    color: #FFD166;
    font-weight: 700;
    display: inline-block;
    margin-bottom: 12px;
}

.tag-grid { display: flex; flex-wrap: wrap; gap: 12px; margin: 16px 0; }
.tag {
    background: rgba(20,224,180,.18);
    border: 1.5px solid var(--border);
    border-radius: 22px;
    padding: 8px 18px;
    font-size: .92rem;
    color: var(--primary);
    font-weight: 700;
}

.result-high {
    background: linear-gradient(135deg, rgba(46,204,113,.20), rgba(20,224,180,.12));
    border: 2.5px solid var(--success);
    border-radius: 18px;
    padding: 32px 36px;
    text-align: center;
    margin: 22px 0;
}
.result-low {
    background: linear-gradient(135deg, rgba(255,92,92,.20), rgba(255,122,122,.12));
    border: 2.5px solid var(--danger);
    border-radius: 18px;
    padding: 32px 36px;
    text-align: center;
    margin: 22px 0;
}
.result-label { font-size: 1.2rem; color: var(--text-body); margin-bottom: 8px; font-weight: 600; }
.result-val { font-size: 3.2rem; font-weight: 900; }
.result-val.high { color: var(--success); }
.result-val.low  { color: var(--danger); }

.watermark {
    position: fixed;
    bottom: 14px; right: 18px;
    background: rgba(20,224,180,.16);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 7px 16px;
    font-size: .78rem;
    color: var(--text-body);
    backdrop-filter: blur(6px);
    z-index: 9999;
    pointer-events: none;
    font-weight: 600;
}

.sidebar-logo {
    text-align: center;
    padding: 22px 0 30px;
}
.sidebar-logo .icon { font-size: 3.4rem; }
.sidebar-logo .name {
    font-size: 1.25rem;
    font-weight: 900;
    color: var(--primary);
    margin-top: 8px;
}
.sidebar-logo .version {
    font-size: .8rem;
    color: var(--text-muted);
    font-weight: 600;
}

.step-title { font-weight: 800; font-size: 1.02rem; color: var(--text-heading); }
.step-desc  { font-size: .92rem; color: var(--text-body); margin-top: 3px; line-height: 1.5; }

.stDataFrame { border-radius: 12px; overflow: hidden; }
.js-plotly-plot .plotly { border-radius: 14px; }
.stProgress > div > div { background: var(--primary); }

.stSelectbox label, .stSlider label, .stNumberInput label, .stTextInput label, .stRadio label {
    color: var(--text-body) !important;
    font-weight: 600 !important;
    font-size: 0.98rem !important;
}
.stSelectbox div[data-baseweb="select"] > div {
    background: var(--bg-input) !important;
    color: var(--text-heading) !important;
    border-color: var(--border) !important;
}
.stTextInput input, .stNumberInput input {
    background: var(--bg-input) !important;
    color: var(--text-heading) !important;
    border-color: var(--border) !important;
}
.stSlider [data-baseweb="slider"] { color: var(--primary) !important; }

[data-testid="stMetricValue"] { color: var(--primary) !important; font-weight: 900 !important; font-size: 1.9rem !important; }
[data-testid="stMetricLabel"] { color: var(--text-body) !important; font-weight: 600 !important; }

.stTabs [data-baseweb="tab"] {
    color: var(--text-muted) !important;
    font-weight: 700 !important;
    font-size: 1.02rem !important;
}
.stTabs [aria-selected="true"] { color: var(--primary) !important; }

.stButton button, .stFormSubmitButton button {
    background: var(--primary) !important;
    color: var(--text-on-primary) !important;
    font-weight: 800 !important;
    font-size: 1.05rem !important;
    border-radius: 10px !important;
    border: none !important;
    padding: 12px 0 !important;
}
.stButton button:hover, .stFormSubmitButton button:hover {
    background: var(--primary-dark) !important;
    color: #FFFFFF !important;
}

[data-testid="stAlert"] p { color: var(--text-heading) !important; font-weight: 600; }
[data-testid="stDataFrame"] * { color: var(--text-body); }
</style>

<div class="watermark">🫁 TBC Classifier &nbsp;|&nbsp; Citra Putri Ramadhani  |  Novita Widya Asmara &nbsp;|&nbsp; © 2026</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# DATA LOADING — terpusat via st.session_state
# ─────────────────────────────────────────────
if "df" not in st.session_state:
    st.session_state["df"] = None
    st.session_state["data_source"] = None
    st.session_state["load_error"] = None

    if os.path.exists(DEFAULT_FILE):
        try:
            _df0 = pd.read_excel(DEFAULT_FILE, engine="openpyxl")
            _missing0 = validate_columns(_df0)
            if not _missing0:
                st.session_state["df"] = _df0
                st.session_state["data_source"] = DEFAULT_FILE
            else:
                st.session_state["load_error"] = (
                    f"File {DEFAULT_FILE} ditemukan, tapi kolom wajib hilang: {', '.join(_missing0)}"
                )
        except Exception as e:
            st.session_state["load_error"] = f"Gagal membaca {DEFAULT_FILE}: {e}"

if st.session_state["df"] is None:
    st.markdown("""
    <div class="hero-banner" style="text-align:center">
        <p class="hero-title">🫁 TBC Classifier</p>
        <p class="hero-sub">Dataset tidak ditemukan</p>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state["load_error"]:
        st.error(f"❌ {st.session_state['load_error']}")
    else:
        st.error(f"❌ File **{DEFAULT_FILE}** tidak ditemukan. Pastikan file ada di folder yang sama dengan `app.py`.")

    st.markdown(f"""
    <div class="info-box">
    📌 Letakkan file <b>{DEFAULT_FILE}</b> di folder yang sama dengan <b>app.py</b>, lalu restart dashboard.<br>
    Atau gunakan menu <b>📤 Ganti Dataset</b> di bawah untuk upload manual.
    </div>
    """, unsafe_allow_html=True)

    gate_file = st.file_uploader("📂 Upload Dataset (.xlsx)", type=["xlsx"], key="gate_upload")
    if gate_file is not None:
        try:
            gate_df = pd.read_excel(gate_file, engine="openpyxl")
            missing = validate_columns(gate_df)
            if missing:
                st.error(f"❌ Kolom wajib tidak ditemukan: **{', '.join(missing)}**")
            else:
                st.session_state["df"] = gate_df
                st.session_state["data_source"] = gate_file.name
                st.session_state["load_error"] = None
                st.cache_resource.clear()
                st.success(f"✅ Dataset **{gate_file.name}** berhasil dimuat.")
                st.rerun()
        except Exception as e:
            st.error(f"❌ Gagal membaca file: {e}")
    st.stop()


@st.cache_resource(show_spinner=False)
def train_models(df_hash):
    """Train all 3 models dan return results."""
    df = st.session_state["df"]
    median_success = df["TB_Treatment_Success_Rate"].median()
    df = df.copy()
    df["Target_Success"] = (df["TB_Treatment_Success_Rate"] >= median_success).astype(int)

    df_model = df.drop(columns=["Country", "TB_Treatment_Success_Rate"])
    df_encoded = pd.get_dummies(df_model, columns=["Region", "Income_Level"], drop_first=True)

    X = df_encoded.drop(columns=["Target_Success"])
    y = df_encoded["Target_Success"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_s, y_train)
    yp_lr  = lr.predict(X_test_s)
    ypr_lr = lr.predict_proba(X_test_s)[:, 1]

    rf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
    rf.fit(X_train, y_train)
    yp_rf  = rf.predict(X_test)
    ypr_rf = rf.predict_proba(X_test)[:, 1]

    nb = GaussianNB()
    nb.fit(X_train_s, y_train)
    yp_nb  = nb.predict(X_test_s)
    ypr_nb = nb.predict_proba(X_test_s)[:, 1]

    def metrics(yt, yp, ypr):
        return {
            "Accuracy":  round(accuracy_score(yt, yp), 4),
            "Precision": round(precision_score(yt, yp), 4),
            "Recall":    round(recall_score(yt, yp), 4),
            "F1-Score":  round(f1_score(yt, yp), 4),
            "AUC":       round(roc_auc_score(yt, ypr), 4),
        }

    results = {
        "Logistic Regression": metrics(y_test, yp_lr, ypr_lr),
        "Random Forest":       metrics(y_test, yp_rf, ypr_rf),
        "Naive Bayes":         metrics(y_test, yp_nb, ypr_nb),
    }

    cms = {
        "Logistic Regression": confusion_matrix(y_test, yp_lr),
        "Random Forest":       confusion_matrix(y_test, yp_rf),
        "Naive Bayes":         confusion_matrix(y_test, yp_nb),
    }

    roc_data = {}
    for name, yt_prob in [
        ("Logistic Regression", ypr_lr),
        ("Random Forest",       ypr_rf),
        ("Naive Bayes",         ypr_nb),
    ]:
        fpr, tpr, _ = roc_curve(y_test, yt_prob)
        roc_data[name] = (fpr, tpr)

    feat_imp = {
        "LR": pd.DataFrame({"Feature": X.columns, "Value": abs(lr.coef_[0])})
                .sort_values("Value", ascending=False).head(15),
        "RF": pd.DataFrame({"Feature": X.columns, "Value": rf.feature_importances_})
                .sort_values("Value", ascending=False).head(15),
    }

    return {
        "models": {"LR": lr, "RF": rf, "NB": nb},
        "scaler": scaler,
        "feature_cols": list(X.columns),
        "median_success": median_success,
        "results": results,
        "cms": cms,
        "roc_data": roc_data,
        "feat_imp": feat_imp,
        "X": X,
        "y": y,
        "df_full": df,
    }


# ─────────────────────────────────────────────
# Dataset dijamin tersedia — langsung ke dashboard
# ─────────────────────────────────────────────
df = st.session_state["df"]

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <div class="icon">🫁</div>
        <div class="name">TBC Classifier</div>
        <div class="version">Dashboard @045|063</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    page = st.radio(
        "📍 Navigasi",
        ["🏠 Home", "📊 Analisis", "🔮 Prediksi", "📤 Ganti Dataset"],
        label_visibility="collapsed",
    )
    st.markdown("---")

    st.markdown(f"""
    <div style='font-size:.85rem;color:#DCE3F0;line-height:1.9'>
    <b style='color:#14E0B4'>Dataset Aktif</b><br>
    {st.session_state['data_source']}<br>
    {df.shape[0]} baris × {df.shape[1]} kolom<br><br>
    <b style='color:#14E0B4'>Model</b><br>
    ✦ Logistic Regression<br>
    ✦ Random Forest<br>
    ✦ Naive Bayes<br><br>
    <b style='color:#14E0B4'>Target</b><br>
    Treatment Success Classification
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    with st.expander("📖 Keterangan Variabel", expanded=False):
        st.markdown("""
        <div style='font-size:.82rem;color:#DCE3F0;line-height:1.85'>

        <b style='color:#14E0B4'>🌍 Identitas</b><br>
        <b>Country</b> — Nama negara<br>
        <b>Region</b> — Wilayah WHO (Africa, Americas, dll.)<br>
        <b>Income_Level</b> — Klasifikasi pendapatan negara (Low / Lower-Middle / Upper-Middle / High)<br>
        <b>Year</b> — Tahun pengamatan (2000–2023)<br><br>

        <b style='color:#14E0B4'>📊 Epidemiologi</b><br>
        <b>TB_Cases</b> — Jumlah kasus TBC baru per tahun<br>
        <b>TB_Deaths</b> — Jumlah kematian akibat TBC per tahun<br>
        <b>TB_Incidence_Rate</b> — Kasus baru per 100.000 penduduk<br>
        <b>TB_Mortality_Rate</b> — Kematian TBC per 100.000 penduduk<br>
        <b>TB_Treatment_Success_Rate</b> — % pasien TBC yang berhasil menyelesaikan pengobatan <i>(variabel target)</i><br><br>

        <b style='color:#14E0B4'>🦠 Resistensi & Koinfeksi</b><br>
        <b>Drug_Resistant_TB_Cases</b> — Jumlah kasus TBC resisten obat (MDR/XDR-TB)<br>
        <b>HIV_CoInfected_TB_Cases</b> — Jumlah pasien TBC dengan koinfeksi HIV<br>
        <b>HIV_Testing_Coverage</b> — % pasien TBC yang menjalani tes HIV (%)<br><br>

        <b style='color:#14E0B4'>👥 Populasi & Ekonomi</b><br>
        <b>Population</b> — Jumlah penduduk negara<br>
        <b>GDP_Per_Capita</b> — PDB per kapita (USD)<br>
        <b>Health_Expenditure_Per_Capita</b> — Pengeluaran kesehatan per kapita (USD)<br>
        <b>Urban_Population_Percentage</b> — % penduduk yang tinggal di perkotaan<br><br>

        <b style='color:#14E0B4'>🍽️ Sosial & Kesehatan</b><br>
        <b>Malnutrition_Prevalence</b> — Prevalensi malnutrisi di populasi (%)<br>
        <b>Smoking_Prevalence</b> — Prevalensi perokok di populasi (%)<br>
        <b>Access_To_Health_Services</b> — % penduduk dengan akses layanan kesehatan<br>
        <b>BCG_Vaccination_Coverage</b> — % cakupan vaksinasi BCG pada bayi<br><br>

        <b style='color:#14E0B4'>🏥 Kapasitas Layanan</b><br>
        <b>TB_Doctors_Per_100K</b> — Jumlah dokter TBC per 100.000 penduduk<br>
        <b>TB_Hospitals_Per_Million</b> — Jumlah fasilitas kesehatan TBC per 1 juta penduduk<br><br>

        <b style='color:#14E0B4'>🎯 Target Klasifikasi</b><br>
        <b>High (1)</b> — Success rate ≥ median dataset<br>
        <b>Low (0)</b> — Success rate &lt; median dataset

        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div class="info-box" style="font-size:.85rem;padding:14px 16px;">
    💡 Ingin pakai dataset lain? Buka menu <b>📤 Ganti Dataset</b> — setelah dikonfirmasi,
    seluruh dashboard otomatis memakai data baru dan model dilatih ulang.
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# TRAIN MODELS
# ─────────────────────────────────────────────
df_hash = str(st.session_state["data_source"]) + str(df.shape) + str(df.columns.tolist())
with st.spinner("Melatih model ML..."):
    cache = train_models(df_hash)


# ═══════════════════════════════════════════════════
# PAGE: HOME
# ═══════════════════════════════════════════════════
if page == "🏠 Home":

    st.markdown("""
    <div class="hero-banner">
        <p class="hero-title">Tuberkulosis (TBC)</p>
        <p class="hero-sub">Memahami penyakit lama yang masih jadi ancaman global hingga hari ini</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<p class="sec-header">📌 TBC dalam Angka Global</p>', unsafe_allow_html=True)
    st.markdown("""
    <div class="metric-grid">
        <div class="metric-box danger">
            <div class="metric-val">10.8 Jt</div>
            <div class="metric-lbl">Kasus Baru / Tahun</div>
        </div>
        <div class="metric-box danger">
            <div class="metric-val">1.25 Jt</div>
            <div class="metric-lbl">Kematian / Tahun</div>
        </div>
        <div class="metric-box warning">
            <div class="metric-val">30</div>
            <div class="metric-lbl">Negara Beban Tinggi</div>
        </div>
        <div class="metric-box warning">
            <div class="metric-val">#2</div>
            <div class="metric-lbl">Penyakit Menular Pembunuh</div>
        </div>
        <div class="metric-box success">
            <div class="metric-val">87%</div>
            <div class="metric-lbl">Tingkat Keberhasilan Rata-rata</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown('<p class="sec-header">🔬 Apa itu Tuberkulosis?</p>', unsafe_allow_html=True)
        st.markdown("""
        <div class="card">
            <p style="line-height:1.85;font-size:1.05rem;">
            <b style="color:#14E0B4">Tuberkulosis (TBC atau TB)</b> adalah penyakit menular yang disebabkan oleh
            bakteri <i>Mycobacterium tuberculosis</i>. Meskipun sebagian besar menyerang paru-paru
            (<b>TB Paru</b>), bakteri ini dapat menyerang organ tubuh lain seperti ginjal, tulang belakang,
            otak, dan kelenjar getah bening (<b>TB Ekstra Paru</b>).
            </p>
            <p style="line-height:1.85;font-size:1.05rem;">
            TBC menyebar melalui <b>udara</b> — ketika penderita batuk, bersin, atau berbicara, percikan
            droplet yang mengandung bakteri TB terhirup oleh orang di sekitarnya. Tidak semua yang
            terinfeksi akan langsung sakit; sebagian mengalami <b>TB Laten</b> (tidak menular, tidak bergejala)
            yang dapat aktif jika sistem imun melemah.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<p class="sec-header">⚠️ Bahaya & Dampak TBC</p>', unsafe_allow_html=True)
        st.markdown("""
        <div class="warn-box">
        <b>🦠 Resistensi Obat (MDR-TB / XDR-TB)</b><br>
        Varian TBC yang kebal terhadap obat lini pertama semakin meningkat dan jauh lebih sulit diobati.
        </div>
        <div class="warn-box">
        <b>🫀 Komplikasi Organ</b><br>
        TBC yang tidak diobati dapat merusak paru secara permanen, menyebabkan gagal napas, meningitis TB,
        dan TB milier yang menyebar ke seluruh tubuh.
        </div>
        <div class="warn-box">
        <b>💔 Koinfeksi HIV</b><br>
        Penderita HIV 20× lebih berisiko mengembangkan TB aktif. Keduanya memperburuk kondisi satu sama lain.
        </div>
        <div class="warn-box">
        <b>💸 Beban Ekonomi</b><br>
        Satu kasus TB mengakibatkan hilangnya rata-rata 3–4 bulan kerja produktif, menghantam keluarga
        miskin paling keras.
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown('<p class="sec-header">🩺 Gejala & Tanda TBC</p>', unsafe_allow_html=True)
        st.markdown("""
        <div class="card">
        <p class="card-title">🔴 Gejala Utama</p>
        <div class="tag-grid">
            <span class="tag">Batuk ≥ 2 minggu</span>
            <span class="tag">Batuk berdarah</span>
            <span class="tag">Nyeri dada</span>
            <span class="tag">Sesak napas</span>
        </div>
        <p class="card-title" style="margin-top:20px">🟡 Gejala Sistemik</p>
        <div class="tag-grid">
            <span class="tag">Demam malam hari</span>
            <span class="tag">Keringat malam</span>
            <span class="tag">Berat badan turun</span>
            <span class="tag">Lemas & lesu</span>
            <span class="tag">Nafsu makan hilang</span>
        </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<p class="sec-header">💊 Penanganan TBC</p>', unsafe_allow_html=True)
        steps = [
            ("1", "Deteksi Dini", "Tes dahak (sputum smear), foto Rontgen, TCM (Xpert MTB/RIF)"),
            ("2", "Pengobatan OAT", "6 bulan: 2 bulan fase intensif (RHZE) + 4 bulan fase lanjutan (RH)"),
            ("3", "DOTS Strategy", "Pengawasan langsung menelan obat oleh PMO (Pengawas Menelan Obat)"),
            ("4", "Follow-up", "Kontrol rutin, cek dahak bulan ke-2, 5, dan 6"),
            ("5", "MDR-TB", "Pengobatan 18–24 bulan dengan obat lini kedua yang lebih kompleks"),
        ]
        for num, title, desc in steps:
            st.markdown(f"""
            <div style="display:flex;gap:14px;margin-bottom:18px;align-items:flex-start">
                <div style="background:#14E0B4;color:#062A22;width:32px;height:32px;border-radius:50%;
                            display:flex;align-items:center;justify-content:center;
                            font-weight:900;font-size:.95rem;flex-shrink:0">{num}</div>
                <div>
                    <div class="step-title">{title}</div>
                    <div class="step-desc">{desc}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<p class="sec-header">📈 Tren TBC Global dari Dataset (2000–2023)</p>', unsafe_allow_html=True)

    trend = df.groupby("Year")[["TB_Incidence_Rate", "TB_Treatment_Success_Rate", "TB_Mortality_Rate"]].mean().reset_index()

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(
        x=trend["Year"], y=trend["TB_Incidence_Rate"],
        name="Incidence Rate", line=dict(color="#FF7A7A", width=3),
        mode="lines+markers", marker=dict(size=6)
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=trend["Year"], y=trend["TB_Mortality_Rate"],
        name="Mortality Rate", line=dict(color="#FFD166", width=3, dash="dot"),
        mode="lines+markers", marker=dict(size=6)
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=trend["Year"], y=trend["TB_Treatment_Success_Rate"],
        name="Treatment Success Rate", line=dict(color="#14E0B4", width=4),
        mode="lines+markers", marker=dict(size=7), fill="tozeroy",
        fillcolor="rgba(20,224,180,0.08)"
    ), secondary_y=True)

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#DCE3F0", size=13),
        legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", y=1.12, font=dict(size=13)),
        hovermode="x unified",
        height=420,
        margin=dict(l=0, r=0, t=30, b=0),
    )
    fig.update_xaxes(showgrid=False, linecolor="#2a2f3e")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.07)", linecolor="#2a2f3e")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<p class="sec-header">🎬 Video Edukasi TBC</p>', unsafe_allow_html=True)
    yt1, yt2 = st.columns(2)
    with yt1:
        st.markdown("**🎯 Apa Itu Tuberkulosis?**")
        st.video("https://youtu.be/G4142KR9A8I?si=PcI3XzeAXkWEgR6K")
    with yt2:
        st.markdown("**🌍 TBC Global — End TB Strategy**")
        st.video("https://youtu.be/51RDclab5nc?si=sE-ioNhv51D7uy7k")


# ═══════════════════════════════════════════════════
# PAGE: ANALISIS
# ═══════════════════════════════════════════════════
elif page == "📊 Analisis":

    st.markdown("""
    <div class="hero-banner">
        <p class="hero-title">Analisis Data & Model ML</p>
        <p class="hero-sub">Eksplorasi dataset, visualisasi, dan perbandingan tiga metode klasifikasi</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="info-box">
    📊 Halaman ini menggunakan dataset aktif: <b>{st.session_state['data_source']}</b> agar hasil analisis dan
    model konsisten dengan halaman Prediksi. Untuk mengganti dataset, gunakan menu <b>📤 Ganti Dataset</b> di sidebar.
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["📋 Karakteristik Data", "📊 Visualisasi EDA", "⚔️ Perbandingan Model", "🎯 Output Model"])

    with tab1:
        st.markdown('<p class="sec-header">📋 Informasi Dataset</p>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Baris", f"{df.shape[0]:,}")
        c2.metric("Total Kolom", f"{df.shape[1]:,}")
        c3.metric("Negara Unik", f"{df['Country'].nunique()}")
        c4.metric("Rentang Tahun", f"{int(df['Year'].min())}–{int(df['Year'].max())}")

        st.markdown("**📊 Statistik Deskriptif**")
        num_desc = df.describe().T.round(2)
        st.dataframe(num_desc, use_container_width=True, height=380)

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**🌍 Distribusi Region**")
            reg_cnt = df["Region"].value_counts().reset_index()
            reg_cnt.columns = ["Region", "Count"]
            fig_reg = px.pie(reg_cnt, names="Region", values="Count",
                             color_discrete_sequence=px.colors.sequential.Teal, hole=0.45)
            fig_reg.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#DCE3F0", size=13),
                                   showlegend=True, height=320, margin=dict(t=10,b=10))
            st.plotly_chart(fig_reg, use_container_width=True)

        with col_b:
            st.markdown("**💰 Distribusi Income Level**")
            inc_cnt = df["Income_Level"].value_counts().reset_index()
            inc_cnt.columns = ["Income_Level", "Count"]
            fig_inc = px.bar(inc_cnt, x="Income_Level", y="Count",
                             color="Count", color_continuous_scale="teal", text="Count")
            fig_inc.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                   font=dict(color="#DCE3F0", size=13), height=320, margin=dict(t=10,b=0),
                                   coloraxis_showscale=False)
            fig_inc.update_traces(textposition="outside", textfont=dict(color="#FFFFFF"))
            st.plotly_chart(fig_inc, use_container_width=True)

        st.markdown("**📂 Sample Dataset (10 baris pertama)**")
        st.dataframe(df.head(10), use_container_width=True)

        st.markdown("**❓ Missing Values**")
        mv = df.isnull().sum()
        mv = mv[mv > 0]
        if mv.empty:
            st.success("✅ Tidak ada missing value dalam dataset!")
        else:
            st.dataframe(mv.rename("Missing Count"), use_container_width=True)

    with tab2:
        st.markdown('<p class="sec-header">📊 Exploratory Data Analysis</p>', unsafe_allow_html=True)

        num_cols = ["TB_Incidence_Rate", "TB_Mortality_Rate", "TB_Treatment_Success_Rate",
                    "Health_Expenditure_Per_Capita", "Access_To_Health_Services", "BCG_Vaccination_Coverage"]

        st.markdown("**📈 Distribusi Variabel Kunci**")
        for i in range(0, len(num_cols), 3):
            cols = st.columns(3)
            for j, col_name in enumerate(num_cols[i:i+3]):
                with cols[j]:
                    fig_h = px.histogram(df, x=col_name, nbins=30,
                                          color_discrete_sequence=["#14E0B4"],
                                          marginal="box", opacity=0.85)
                    fig_h.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#DCE3F0", size=12), height=280,
                        margin=dict(l=0,r=0,t=30,b=0),
                        title=dict(text=col_name.replace("_"," "), font=dict(size=13, color="#FFFFFF"))
                    )
                    fig_h.update_xaxes(showgrid=False)
                    fig_h.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.07)")
                    st.plotly_chart(fig_h, use_container_width=True)

        st.markdown("**📦 Treatment Success Rate per Region & Income Level**")
        col_b1, col_b2 = st.columns(2)
        with col_b1:
            fig_bx = px.box(df, x="Region", y="TB_Treatment_Success_Rate",
                             color="Region", color_discrete_sequence=px.colors.qualitative.Vivid)
            fig_bx.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                  font=dict(color="#DCE3F0", size=12), height=370, showlegend=False,
                                  margin=dict(t=10,b=0))
            st.plotly_chart(fig_bx, use_container_width=True)
        with col_b2:
            order = ["Low","Lower-Middle","Upper-Middle","High"]
            fig_bx2 = px.box(df[df["Income_Level"].isin(order)],
                              x="Income_Level", y="TB_Treatment_Success_Rate",
                              category_orders={"Income_Level": order},
                              color="Income_Level",
                              color_discrete_sequence=px.colors.sequential.Teal_r)
            fig_bx2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                   font=dict(color="#DCE3F0", size=12), height=370, showlegend=False,
                                   margin=dict(t=10,b=0))
            st.plotly_chart(fig_bx2, use_container_width=True)

        st.markdown("**🔥 Heatmap Korelasi Variabel Numerik**")
        numeric_df = df.select_dtypes(include=[np.number])
        corr = numeric_df.corr()
        fig_heat = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                              zmin=-1, zmax=1, aspect="auto")
        fig_heat.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#DCE3F0", size=11),
                                height=560, margin=dict(t=10,b=0))
        st.plotly_chart(fig_heat, use_container_width=True)

        st.markdown("**🎯 Distribusi Kelas Target**")
        median_s = cache["median_success"]
        df_vis = df.copy()
        df_vis["Target"] = (df_vis["TB_Treatment_Success_Rate"] >= median_s).map({True:"High (1)", False:"Low (0)"})
        fig_tgt = px.pie(df_vis["Target"].value_counts().reset_index(),
                          names="Target", values="count",
                          color_discrete_map={"High (1)":"#14E0B4","Low (0)":"#FF7A7A"}, hole=0.5)
        fig_tgt.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#DCE3F0", size=13),
                               height=320, margin=dict(t=10,b=10))
        st.plotly_chart(fig_tgt, use_container_width=True)
        st.info(f"📌 Median Treatment Success Rate: **{median_s:.2f}%**  |  Threshold klasifikasi binary: High ≥ {median_s:.2f}%")

    with tab3:
        st.markdown('<p class="sec-header">⚔️ Perbandingan Ketiga Model</p>', unsafe_allow_html=True)

        results = cache["results"]
        res_df = pd.DataFrame(results).T.reset_index().rename(columns={"index":"Model"})

        best_model = res_df.loc[res_df["F1-Score"].idxmax(), "Model"]
        best_acc   = res_df["Accuracy"].max()
        best_auc   = res_df["AUC"].max()

        st.markdown(f"""
        <div class="metric-grid">
            <div class="metric-box success">
                <div class="metric-val">🏆</div>
                <div class="metric-lbl">Best Model: {best_model}</div>
            </div>
            <div class="metric-box">
                <div class="metric-val">{best_acc:.1%}</div>
                <div class="metric-lbl">Best Accuracy</div>
            </div>
            <div class="metric-box">
                <div class="metric-val">{best_auc:.3f}</div>
                <div class="metric-lbl">Best AUC</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        melted = res_df.melt(id_vars="Model", var_name="Metric", value_name="Score")
        colors = {"Logistic Regression":"#14E0B4","Random Forest":"#FF7A7A","Naive Bayes":"#FFD166"}
        fig_bar = px.bar(melted, x="Metric", y="Score", color="Model",
                          barmode="group", color_discrete_map=colors,
                          text_auto=".3f", range_y=[0, 1.1])
        fig_bar.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#DCE3F0", size=13), height=440,
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=13)),
            margin=dict(t=10,b=0)
        )
        fig_bar.update_traces(textfont=dict(color="#FFFFFF", size=11))
        fig_bar.update_xaxes(showgrid=False)
        fig_bar.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.07)")
        st.plotly_chart(fig_bar, use_container_width=True)

        st.markdown("**📉 ROC Curves — Semua Model**")
        fig_roc = go.Figure()
        roc_colors = {"Logistic Regression":"#14E0B4","Random Forest":"#FF7A7A","Naive Bayes":"#FFD166"}
        for name, (fpr, tpr) in cache["roc_data"].items():
            auc_val = results[name]["AUC"]
            fig_roc.add_trace(go.Scatter(
                x=fpr, y=tpr,
                name=f"{name} (AUC={auc_val:.3f})",
                line=dict(color=roc_colors[name], width=3)
            ))
        fig_roc.add_trace(go.Scatter(
            x=[0,1], y=[0,1], mode="lines",
            line=dict(dash="dash", color="#7A859C", width=1.5),
            name="Random Classifier", showlegend=True
        ))
        fig_roc.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#DCE3F0", size=13), height=440,
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=12)),
            xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
            margin=dict(t=10,b=0)
        )
        fig_roc.update_xaxes(showgrid=False)
        fig_roc.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.07)")
        st.plotly_chart(fig_roc, use_container_width=True)

        st.markdown("**📋 Tabel Perbandingan Lengkap**")
        styled = res_df.set_index("Model").style.background_gradient(cmap="YlGn", axis=0).format("{:.4f}")
        st.dataframe(styled, use_container_width=True)

    with tab4:
        st.markdown('<p class="sec-header">🎯 Output Detail per Model</p>', unsafe_allow_html=True)

        model_sel = st.selectbox("Pilih Model:", ["Logistic Regression", "Random Forest", "Naive Bayes"])

        cm = cache["cms"][model_sel]
        fig_cm = px.imshow(
            cm,
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=["Low (0)","High (1)"], y=["Low (0)","High (1)"],
            text_auto=True, color_continuous_scale="Teal",
        )
        fig_cm.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#DCE3F0", size=13),
            title=dict(text=f"Confusion Matrix — {model_sel}", font=dict(color="#FFFFFF", size=15)),
            height=360, margin=dict(t=50,b=0)
        )

        col_cm, col_met = st.columns([1,1])
        with col_cm:
            st.plotly_chart(fig_cm, use_container_width=True)
        with col_met:
            m = cache["results"][model_sel]
            st.markdown(f"<br>", unsafe_allow_html=True)
            for metric, val in m.items():
                pct = val * 100
                color = "#14E0B4" if pct >= 80 else ("#FFD166" if pct >= 65 else "#FF7A7A")
                st.markdown(f"""
                <div style="margin-bottom:16px">
                    <div style="display:flex;justify-content:space-between;margin-bottom:6px">
                        <span style="font-size:1rem;font-weight:700;color:#FFFFFF">{metric}</span>
                        <span style="color:{color};font-weight:800;font-size:1rem">{val:.4f}</span>
                    </div>
                    <div style="background:#1F2633;border-radius:7px;height:10px;overflow:hidden">
                        <div style="width:{pct:.1f}%;background:{color};height:100%;border-radius:7px;
                                    transition:width .8s ease"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        if model_sel in ["Logistic Regression", "Random Forest"]:
            key = "LR" if model_sel == "Logistic Regression" else "RF"
            fi = cache["feat_imp"][key]
            label = "Koefisien (|nilai|)" if key=="LR" else "Importance"
            st.markdown(f"**🔑 Top 15 Feature Importance — {model_sel}**")
            fig_fi = px.bar(fi, x="Value", y="Feature", orientation="h",
                             color="Value", color_continuous_scale="Teal",
                             labels={"Value": label})
            fig_fi.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#DCE3F0", size=12), height=480,
                coloraxis_showscale=False, margin=dict(t=10,b=0),
                yaxis=dict(autorange="reversed")
            )
            fig_fi.update_xaxes(showgrid=False)
            fig_fi.update_yaxes(showgrid=False)
            st.plotly_chart(fig_fi, use_container_width=True)

        st.markdown("**🌍 Prediksi Keberhasilan per Negara (dari Test Set)**")
        df_full = cache["df_full"]
        df_country = df_full.groupby("Country")[["TB_Treatment_Success_Rate","Year"]].last().reset_index()
        median_s = cache["median_success"]
        df_country["Status"] = df_country["TB_Treatment_Success_Rate"].apply(
            lambda x: "✅ High" if x >= median_s else "❌ Low"
        )
        df_country = df_country.sort_values("TB_Treatment_Success_Rate", ascending=False)

        fig_country = px.bar(
            df_country, x="Country", y="TB_Treatment_Success_Rate",
            color="Status",
            color_discrete_map={"✅ High":"#14E0B4","❌ Low":"#FF7A7A"},
            title=f"Treatment Success Rate per Negara (Threshold: {median_s:.1f}%)",
            text="TB_Treatment_Success_Rate",
        )
        fig_country.add_hline(y=median_s, line_dash="dash", line_color="#FFD166",
                               annotation_text=f"Median ({median_s:.1f}%)",
                               annotation_font_color="#FFFFFF")
        fig_country.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#DCE3F0", size=12), height=440,
            title_font=dict(color="#FFFFFF", size=14),
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=12)),
            margin=dict(t=50,b=0)
        )
        fig_country.update_traces(texttemplate="%{text:.1f}", textposition="outside",
                                   textfont=dict(color="#FFFFFF", size=10))
        fig_country.update_xaxes(showgrid=False)
        fig_country.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.07)")
        st.plotly_chart(fig_country, use_container_width=True)


# ═══════════════════════════════════════════════════
# PAGE: PREDIKSI
# ═══════════════════════════════════════════════════
elif page == "🔮 Prediksi":

    st.markdown("""
    <div class="hero-banner">
        <p class="hero-title">Prediksi Keberhasilan Pengobatan TBC</p>
        <p class="hero-sub">Masukkan variabel epidemiologis negara untuk memprediksi tingkat keberhasilan</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<p class="sec-header">📝 Input Variabel Negara</p>', unsafe_allow_html=True)

    with st.form("prediction_form"):
        col_f1, col_f2, col_f3 = st.columns(3)

        with col_f1:
            st.markdown("**🗺️ Identitas**")
            country_name = st.text_input("Nama Negara (opsional)", placeholder="e.g. Indonesia")
            year_inp = st.slider("Tahun", 2000, 2025, 2023)
            region_inp = st.selectbox("Region", df["Region"].unique())
            income_inp = st.selectbox("Income Level", ["Low","Lower-Middle","Upper-Middle","High"])

            st.markdown("**📊 Epidemiologi**")
            tb_inc  = st.number_input("TB Incidence Rate (/100k)", 0.0, 1000.0, 150.0, step=1.0)
            tb_mort = st.number_input("TB Mortality Rate (/100k)", 0.0, 100.0, 20.0, step=0.5)
            tb_cases = st.number_input("TB Cases", 0, 500000, 10000, step=100)
            tb_deaths= st.number_input("TB Deaths", 0, 50000, 2000, step=50)

        with col_f2:
            st.markdown("**🦠 Resistensi & Koinfeksi**")
            dr_tb   = st.number_input("Drug-Resistant TB Cases", 0, 20000, 1000, step=50)
            hiv_tb  = st.number_input("HIV Co-Infected TB Cases", 0, 20000, 2000, step=100)
            hiv_cov = st.slider("HIV Testing Coverage (%)", 0.0, 100.0, 60.0)

            st.markdown("**👥 Populasi & Ekonomi**")
            population = st.number_input("Population", 1_000_000, 2_000_000_000, 100_000_000, step=1_000_000)
            gdp        = st.number_input("GDP per Capita (USD)", 0, 120_000, 15_000, step=500)
            health_exp = st.number_input("Health Expenditure per Capita (USD)", 0, 20_000, 3_000, step=100)
            urban_pct  = st.slider("Urban Population (%)", 0.0, 100.0, 55.0)

        with col_f3:
            st.markdown("**🍽️ Sosial & Kesehatan**")
            malnut    = st.slider("Malnutrition Prevalence (%)", 0.0, 60.0, 20.0)
            smoking   = st.slider("Smoking Prevalence (%)", 0.0, 60.0, 25.0)
            access_h  = st.slider("Access to Health Services (%)", 0.0, 100.0, 65.0)
            bcg_cov   = st.slider("BCG Vaccination Coverage (%)", 0.0, 100.0, 80.0)

            st.markdown("**🏥 Kapasitas Layanan**")
            tb_doctors= st.number_input("TB Doctors per 100k", 0.0, 20.0, 5.0, step=0.1)
            tb_hosp   = st.number_input("TB Hospitals per Million", 0.0, 30.0, 8.0, step=0.5)

            st.markdown("**🤖 Pilih Model**")
            model_choice = st.selectbox("Model Prediksi",
                ["Logistic Regression", "Random Forest", "Naive Bayes"])

        submitted = st.form_submit_button("🚀 Prediksi Sekarang", use_container_width=True)

    if submitted:
        feature_cols = cache["feature_cols"]
        row = {col: 0.0 for col in feature_cols}

        row["Year"]                      = year_inp
        row["TB_Cases"]                  = tb_cases
        row["TB_Deaths"]                 = tb_deaths
        row["TB_Incidence_Rate"]         = tb_inc
        row["TB_Mortality_Rate"]         = tb_mort
        row["Drug_Resistant_TB_Cases"]   = dr_tb
        row["HIV_CoInfected_TB_Cases"]   = hiv_tb
        row["Population"]                = population
        row["GDP_Per_Capita"]            = gdp
        row["Health_Expenditure_Per_Capita"] = health_exp
        row["Urban_Population_Percentage"]   = urban_pct
        row["Malnutrition_Prevalence"]   = malnut
        row["Smoking_Prevalence"]        = smoking
        row["TB_Doctors_Per_100K"]       = tb_doctors
        row["TB_Hospitals_Per_Million"]  = tb_hosp
        row["Access_To_Health_Services"] = access_h
        row["BCG_Vaccination_Coverage"]  = bcg_cov
        row["HIV_Testing_Coverage"]      = hiv_cov

        all_regions = sorted(df["Region"].unique())
        all_incomes = ["Low","Lower-Middle","Upper-Middle","High"]

        for reg in all_regions[1:]:
            key = f"Region_{reg}"
            if key in feature_cols:
                row[key] = 1.0 if region_inp == reg else 0.0

        for inc in all_incomes[1:]:
            key = f"Income_Level_{inc}"
            if key in feature_cols:
                row[key] = 1.0 if income_inp == inc else 0.0

        input_df = pd.DataFrame([row])[feature_cols]

        scaler = cache["scaler"]
        models = cache["models"]

        key_map = {"Logistic Regression":"LR","Random Forest":"RF","Naive Bayes":"NB"}
        mk = key_map[model_choice]
        model = models[mk]

        if mk == "RF":
            pred_proba = model.predict_proba(input_df)[0]
        else:
            pred_proba = model.predict_proba(scaler.transform(input_df))[0]

        pred_class = int(np.argmax(pred_proba))
        prob_high  = pred_proba[1] * 100
        prob_low   = pred_proba[0] * 100

        label_name = country_name if country_name else "Negara Input"
        if pred_class == 1:
            st.markdown(f"""
            <div class="result-high">
                <div class="result-label">🌍 {label_name} — Prediksi Menggunakan {model_choice}</div>
                <div class="result-val high">✅ HIGH SUCCESS</div>
                <div style="font-size:1.2rem;color:#DCE3F0;margin-top:10px">
                    Peluang Keberhasilan Tinggi: <b style="color:#2ECC71">{prob_high:.1f}%</b>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-low">
                <div class="result-label">🌍 {label_name} — Prediksi Menggunakan {model_choice}</div>
                <div class="result-val low">❌ LOW SUCCESS</div>
                <div style="font-size:1.2rem;color:#DCE3F0;margin-top:10px">
                    Peluang Keberhasilan Rendah: <b style="color:#FF5C5C">{prob_low:.1f}%</b>
                </div>
            </div>
            """, unsafe_allow_html=True)

        col_g1, col_g2 = st.columns(2)
        with col_g1:
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=prob_high,
                title={"text": "Probabilitas High Success (%)", "font": {"color": "#FFFFFF", "size": 16}},
                number={"font": {"color": "#14E0B4", "size": 36}},
                gauge={
                    "axis": {"range": [0, 100], "tickfont": {"color": "#DCE3F0"}},
                    "bar": {"color": "#14E0B4"},
                    "bgcolor": "#1F2633",
                    "steps": [
                        {"range":[0,50],  "color":"rgba(255,92,92,.25)"},
                        {"range":[50,75], "color":"rgba(255,209,102,.25)"},
                        {"range":[75,100],"color":"rgba(46,204,113,.25)"},
                    ],
                    "threshold": {"line":{"color":"white","width":3},"thickness":.75,"value":50}
                },
                delta={"reference": 50, "increasing": {"color":"#14E0B4"}, "font": {"color": "#DCE3F0"}}
            ))
            fig_g.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#DCE3F0"), height=320,
                margin=dict(t=40,b=0)
            )
            st.plotly_chart(fig_g, use_container_width=True)

        with col_g2:
            fig_pie = go.Figure(go.Pie(
                labels=["High Success","Low Success"],
                values=[prob_high, prob_low],
                marker_colors=["#14E0B4","#FF7A7A"],
                hole=0.55,
                textinfo="label+percent",
                textfont=dict(color="#062A22", size=13)
            ))
            fig_pie.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#DCE3F0", size=13),
                height=320, margin=dict(t=10,b=10),
                legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=12))
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        st.markdown("**⚔️ Perbandingan Prediksi Antar Model untuk Input Ini**")
        cross_data = []
        for mname, mk2 in key_map.items():
            m2 = models[mk2]
            if mk2 == "RF":
                pp = m2.predict_proba(input_df)[0]
            else:
                pp = m2.predict_proba(scaler.transform(input_df))[0]
            cross_data.append({
                "Model": mname,
                "P(High) %": round(pp[1]*100, 2),
                "P(Low) %":  round(pp[0]*100, 2),
                "Prediksi":  "✅ HIGH" if pp[1] >= 0.5 else "❌ LOW"
            })
        cross_df = pd.DataFrame(cross_data)
        st.dataframe(cross_df.set_index("Model"), use_container_width=True)


# ═══════════════════════════════════════════════════
# PAGE: GANTI DATASET
# ═══════════════════════════════════════════════════
elif page == "📤 Ganti Dataset":

    st.markdown("""
    <div class="hero-banner">
        <p class="hero-title">Ganti Dataset Aktif</p>
        <p class="hero-sub">Ganti dataset yang dipakai di seluruh dashboard — Home, Analisis, dan Prediksi</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="info-box">
    📌 Dataset aktif saat ini: <b>{st.session_state['data_source']}</b>
    — {df.shape[0]} baris, {df.shape[1]} kolom. Upload dataset baru di bawah untuk mengganti.
    Kolom wajib: <b>{', '.join(REQUIRED_COLUMNS)}</b> beserta kolom numerik epidemiologis lainnya.
    Dataset baru baru benar-benar aktif setelah kamu <b>konfirmasi</b>, dan model akan dilatih ulang otomatis.
    </div>
    """, unsafe_allow_html=True)

    detected_files = sorted(
        f for f in glob.glob("*.xlsx")
        if os.path.basename(f) != str(st.session_state.get("data_source"))
    )

    pending_df   = None
    pending_name = None

    st.markdown('<p class="sec-header">📂 Pilih Sumber Dataset Baru</p>', unsafe_allow_html=True)

    if detected_files:
        col_pick, col_refresh = st.columns([4, 1])
        with col_pick:
            chosen_file = st.selectbox(
                "File .xlsx terdeteksi di Colab",
                ["-- Pilih file --"] + detected_files,
                key="ganti_detected_pick",
            )
        with col_refresh:
            st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
            if st.button("🔄 Refresh", use_container_width=True, key="ganti_refresh_btn"):
                st.rerun()

        if chosen_file != "-- Pilih file --":
            try:
                pending_df   = pd.read_excel(chosen_file, engine="openpyxl")
                pending_name = chosen_file
            except Exception as e:
                st.error(f"❌ Gagal membaca {chosen_file}: {e}")
    else:
        st.caption("Tidak ada file .xlsx lain yang terdeteksi otomatis di folder ini.")

    with st.expander("➕ Upload file dari komputer"):
        manual_file = st.file_uploader("📂 Upload File Excel", type=["xlsx"], key="ganti_manual_upload")
        if manual_file is not None:
            try:
                pending_df   = pd.read_excel(manual_file, engine="openpyxl")
                pending_name = manual_file.name
            except Exception as e:
                st.error(f"❌ Gagal membaca file: {e}")

    if pending_df is not None:
        missing = validate_columns(pending_df)

        st.markdown('<p class="sec-header">📋 Preview Dataset Baru</p>', unsafe_allow_html=True)
        st.dataframe(pending_df.head(10), use_container_width=True)

        if missing:
            st.error(
                f"❌ Kolom wajib tidak ditemukan: **{', '.join(missing)}**. "
                "Dataset ini tidak bisa dipakai sampai kolom tersebut tersedia."
            )
        else:
            new_median = pending_df["TB_Treatment_Success_Rate"].median()
            c1n, c2n, c3n = st.columns(3)
            c1n.metric("Jumlah Baris", pending_df.shape[0])
            c2n.metric("Median Success Rate", f"{new_median:.2f}%")
            c3n.metric("Kolom", pending_df.shape[1])

            st.markdown("""
            <div class="warn-box">
            ⚠️ Konfirmasi akan <b>mengganti dataset aktif</b> untuk seluruh dashboard (Home, Analisis,
            Prediksi) dan melatih ulang ketiga model dari awal. Dataset lama tidak otomatis tersimpan.
            </div>
            """, unsafe_allow_html=True)

            col_confirm, col_cancel = st.columns(2)
            with col_confirm:
                if st.button("✅ Ya, gunakan dataset ini", type="primary", use_container_width=True, key="ganti_confirm_btn"):
                    st.session_state["df"] = pending_df
                    st.session_state["data_source"] = pending_name
                    st.session_state["load_error"] = None
                    st.cache_resource.clear()
                    st.success(f"✅ Dataset aktif diganti ke **{pending_name}**. Memuat ulang dashboard...")
                    st.rerun()
            with col_cancel:
                st.button("❌ Batal", use_container_width=True, key="ganti_cancel_btn")
    else:
        st.markdown("""
        <div class="card" style="text-align:center;padding:70px 20px;">
            <div style="font-size:4.5rem;margin-bottom:18px">📤</div>
            <div style="font-size:1.3rem;font-weight:800;color:#FFFFFF">Pilih atau upload dataset baru</div>
            <div style="font-size:.98rem;margin-top:10px;color:#97A3BD">
                Dashboard tetap berjalan dengan dataset saat ini sampai kamu konfirmasi penggantian
            </div>
        </div>
        """, unsafe_allow_html=True)
