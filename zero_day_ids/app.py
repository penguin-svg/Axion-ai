"""
app.py
======
Streamlit Dashboard — Zero-Day Anomaly Detection IDS
Run:  streamlit run app.py
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(__file__))

# ──────────────────────────────────────────────────────────────────────────────
# Page config (MUST be first Streamlit call)
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ZeroDay IDS — Anomaly Detection Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Background */
    .main { background-color: #0E1117; }
    section[data-testid="stSidebar"] { background-color: #111827; }

    /* Metric cards */
    [data-testid="metric-container"] {
        background: #1A1F2E;
        border: 1px solid #2D3748;
        border-radius: 10px;
        padding: 12px 18px;
    }
    [data-testid="stMetricLabel"]  { font-size: 13px !important; color: #9CA3AF !important; }
    [data-testid="stMetricValue"]  { font-size: 28px !important; color: #F9FAFB !important; font-weight: 700; }
    [data-testid="stMetricDelta"]  { font-size: 12px !important; }

    /* Section headers */
    .section-header {
        font-size: 20px; font-weight: 700; color: #F9FAFB;
        border-left: 4px solid #2196F3;
        padding-left: 12px; margin: 28px 0 16px 0;
    }

    /* Badge */
    .badge-normal  { background:#1565C0; color:#fff; padding:3px 10px; border-radius:12px; font-size:12px; }
    .badge-anomaly { background:#C62828; color:#fff; padding:3px 10px; border-radius:12px; font-size:12px; }
    .badge-warn    { background:#E65100; color:#fff; padding:3px 10px; border-radius:12px; font-size:12px; }

    /* Divider */
    hr { border-color: #2D3748; }

    /* Scrollable table */
    .stDataFrame { border-radius: 8px; }

    /* Tab styling */
    button[data-baseweb="tab"] { font-size: 14px; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
ARTIFACT_PATH = "models/pipeline_artifacts.pkl"

@st.cache_resource(show_spinner="Running pipeline …")
def get_artifacts(contamination, n_estimators, n_samples):
    """Run (or reload) the pipeline and cache the result."""
    from pipeline import run_pipeline
    return run_pipeline(
        contamination=contamination,
        n_estimators=n_estimators,
        n_samples=n_samples,
    )


def metric_card(col, label, value, delta=None, delta_color="normal"):
    col.metric(label, value, delta=delta, delta_color=delta_color)


def section(title: str):
    st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# Sidebar — controls
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/security-shield-green.png", width=72)
    st.markdown("## 🛡️ ZeroDay IDS")
    st.markdown("*Unsupervised Anomaly Detection*")
    st.divider()

    st.markdown("### ⚙️ Model Parameters")
    contamination = st.slider("Contamination", 0.05, 0.40, 0.20, 0.01,
                               help="Expected fraction of anomalies in training data")
    n_estimators  = st.select_slider("n_estimators (trees)",
                                      options=[50, 100, 150, 200, 300], value=200)
    n_samples     = st.select_slider("Synthetic Records",
                                      options=[2000, 4000, 6000, 8000, 10000], value=8000)

    st.divider()
    retrain = st.button("🔄 Retrain Model", use_container_width=True, type="primary")
    st.markdown("### 📂 Custom Dataset")
    uploaded = st.file_uploader("Upload NSL-KDD CSV", type=["csv"])
    st.divider()
    st.markdown("### ℹ️ About")
    st.markdown("""
**Algorithm**: Isolation Forest  
**Dataset**: NSL-KDD / CICIDS2017 (synthetic)  
**Framework**: Scikit-learn + Streamlit  
**Academic Year**: 2025–2026  
    """)


# ──────────────────────────────────────────────────────────────────────────────
# Load / train artifacts
# ──────────────────────────────────────────────────────────────────────────────
if retrain:
    st.cache_resource.clear()

artifacts = get_artifacts(contamination, n_estimators, n_samples)

# Unpack
df               = artifacts["df"]
stats            = artifacts["stats"]
X_test_pca       = artifacts["X_test_pca"]
X_train_pca      = artifacts["X_train_pca"]
y_test           = artifacts["y_test"]
y_pred           = artifacts["y_pred"]
scores_norm      = artifacts["scores_norm"]
raw_scores       = artifacts["raw_scores"]
attack_types_test= artifacts["attack_types_test"]
importance_df    = artifacts["importance_df"]
pca_summary      = artifacts["pca_summary"]
metrics          = artifacts["metrics"]
cm_df            = artifacts["cm_df"]
breakdown        = artifacts["breakdown"]
roc_data         = artifacts["roc_data"]
pr_data          = artifacts["pr_data"]
model_info_dict  = artifacts["model_info"]
report_txt       = artifacts["report_txt"]

from src.visualizations import (
    traffic_overview_donut, attack_distribution_bar, anomaly_score_histogram,
    confusion_matrix_heatmap, feature_importance_bar, pca_scatter,
    roc_curve_plot, pr_curve_plot, detection_rate_by_attack,
    pca_variance_plot, protocol_bar, anomaly_timeline,
)


# ──────────────────────────────────────────────────────────────────────────────
# ★ Header
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center; padding: 10px 0 6px 0;'>
  <span style='font-size:40px; font-weight:800; color:#F9FAFB;'>
    🛡️ Zero-Day Anomaly Detection IDS
  </span><br>
  <span style='font-size:15px; color:#9CA3AF;'>
    Unsupervised Intrusion Detection using Isolation Forest &nbsp;|&nbsp;
    Python · Scikit-learn · Streamlit
  </span>
</div>
""", unsafe_allow_html=True)
st.divider()


# ──────────────────────────────────────────────────────────────────────────────
# ★ KPI Row
# ──────────────────────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5, c6 = st.columns(6)
metric_card(c1, "🗃️ Total Records",   f"{stats['total']:,}")
metric_card(c2, "✅ Normal Traffic",   f"{stats['normal']:,}")
metric_card(c3, "🚨 Attack Records",  f"{stats['attacks']:,}")
metric_card(c4, "🎯 Accuracy",        f"{metrics['accuracy']} %")
metric_card(c5, "📊 F1-Score",        f"{metrics['f1_score']} %")
metric_card(c6, "🔍 ROC-AUC",         f"{metrics.get('roc_auc','N/A')} %")

st.divider()


# ──────────────────────────────────────────────────────────────────────────────
# ★ Main Tabs
# ──────────────────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "📊 Overview",
    "🧪 Model Performance",
    "🔍 Anomaly Analysis",
    "📈 Feature Insights",
    "🗂️ Data Explorer",
    "📋 Report",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    section("Dataset Overview")

    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.plotly_chart(
            traffic_overview_donut(stats["normal"], stats["attacks"]),
            use_container_width=True,
        )
    with col_b:
        st.plotly_chart(
            attack_distribution_bar(stats["attack_dist"]),
            use_container_width=True,
        )

    col_c, col_d = st.columns([1, 1])
    with col_c:
        st.plotly_chart(
            protocol_bar(stats["protocol_dist"]) if stats["protocol_dist"]
            else st.empty(),
            use_container_width=True,
        )
    with col_d:
        section("Dataset Statistics")
        kpi = {
            "Total Records":    stats["total"],
            "Normal Traffic":   stats["normal"],
            "Attack Records":   stats["attacks"],
            "Anomaly %":        f"{stats['anomaly_pct']} %",
            "Features":         stats["n_features"],
            "PCA Components":   pca_summary["n_components"],
            "Variance Retained":f"{pca_summary['total_variance']} %",
        }
        kpi_df = pd.DataFrame(list(kpi.items()), columns=["Metric", "Value"])
        st.dataframe(kpi_df, hide_index=True, use_container_width=True)

    section("System Architecture Flow")
    st.markdown("""
    <div style='background:#1A1F2E; border-radius:12px; padding:22px 30px;
                border:1px solid #2D3748; font-size:14px; line-height:2.2;'>
    &nbsp;📥 <b>Network Traffic Dataset</b> (NSL-KDD / CICIDS2017)<br>
    &nbsp;&nbsp;&nbsp;&nbsp;↓<br>
    &nbsp;🔧 <b>Data Preprocessing</b> — Cleaning · Encoding · Normalization<br>
    &nbsp;&nbsp;&nbsp;&nbsp;↓<br>
    &nbsp;⚙️ <b>Feature Engineering</b> — Selection · PCA Dimensionality Reduction<br>
    &nbsp;&nbsp;&nbsp;&nbsp;↓<br>
    &nbsp;🌲 <b>Isolation Forest Training</b> — Learns Normal Traffic Baseline<br>
    &nbsp;&nbsp;&nbsp;&nbsp;↓<br>
    &nbsp;🎯 <b>Anomaly Scoring &amp; Detection</b> — Flags Statistical Deviations<br>
    &nbsp;&nbsp;&nbsp;&nbsp;↓<br>
    &nbsp;📊 <b>Performance Evaluation</b> — Accuracy · F1 · Confusion Matrix · ROC<br>
    &nbsp;&nbsp;&nbsp;&nbsp;↓<br>
    &nbsp;🖥️ <b>Streamlit Dashboard &amp; Alert Generation</b>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    section("Classification Metrics")

    m1, m2, m3, m4 = st.columns(4)
    metric_card(m1, "Accuracy",  f"{metrics['accuracy']} %")
    metric_card(m2, "Precision", f"{metrics['precision']} %")
    metric_card(m3, "Recall",    f"{metrics['recall']} %")
    metric_card(m4, "F1-Score",  f"{metrics['f1_score']} %")

    m5, m6, m7, m8 = st.columns(4)
    metric_card(m5, "True Positives",  f"{metrics['true_positives']:,}")
    metric_card(m6, "True Negatives",  f"{metrics['true_negatives']:,}")
    metric_card(m7, "False Positives", f"{metrics['false_positives']:,}")
    metric_card(m8, "False Negatives", f"{metrics['false_negatives']:,}")

    st.divider()
    section("Confusion Matrix & ROC / PR Curves")

    r1c1, r1c2 = st.columns([1, 1])
    with r1c1:
        st.plotly_chart(confusion_matrix_heatmap(cm_df), use_container_width=True)
    with r1c2:
        st.plotly_chart(roc_curve_plot(roc_data), use_container_width=True)

    _, r2c2 = st.columns([1, 1])
    with r2c2:
        st.plotly_chart(pr_curve_plot(pr_data), use_container_width=True)

    st.divider()
    section("Per-Attack-Type Detection Rate")
    st.plotly_chart(detection_rate_by_attack(breakdown), use_container_width=True)
    st.dataframe(breakdown, hide_index=True, use_container_width=True)

    st.divider()
    section("Model Hyper-Parameters")
    st.json(model_info_dict)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — ANOMALY ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    section("Anomaly Score Distribution")
    st.plotly_chart(
        anomaly_score_histogram(scores_norm, y_test),
        use_container_width=True,
    )

    section("Anomaly Score Timeline (Test Set Sample)")
    st.plotly_chart(
        anomaly_timeline(scores_norm, y_pred),
        use_container_width=True,
    )

    section("PCA Traffic Clusters")
    st.plotly_chart(
        pca_scatter(X_test_pca, y_test, attack_types_test),
        use_container_width=True,
    )

    section("Top Detected Anomalies")
    df_test_view = df.copy().tail(len(y_test)).reset_index(drop=True)
    df_test_view["anomaly_score"]  = scores_norm
    df_test_view["predicted"]      = y_pred
    df_test_view["attack_type"]    = attack_types_test
    df_test_view["status"] = df_test_view["predicted"].map(
        {1: "🚨 ANOMALY", 0: "✅ Normal"}
    )

    top_anomalies = (
        df_test_view[df_test_view["predicted"] == 1]
        .sort_values("anomaly_score", ascending=False)
        .head(50)
        [["status", "attack_type", "anomaly_score", "src_bytes", "dst_bytes",
          "duration", "protocol_type", "flag"]]
        .reset_index(drop=True)
    )
    st.dataframe(top_anomalies, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — FEATURE INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    section("Feature Importance (Mutual Information)")
    top_n = st.slider("Show top N features", 5, 20, 15)
    st.plotly_chart(feature_importance_bar(importance_df, top_n=top_n),
                    use_container_width=True)

    st.divider()
    section("PCA Explained Variance")
    st.plotly_chart(
        pca_variance_plot(pca_summary["cumulative_variance"]),
        use_container_width=True,
    )
    col_pca1, col_pca2 = st.columns(2)
    col_pca1.metric("PCA Components Retained", pca_summary["n_components"])
    col_pca2.metric("Total Variance Explained", f"{pca_summary['total_variance']} %")

    st.divider()
    section("Feature Importance Table")
    st.dataframe(importance_df, hide_index=True, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — DATA EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    section("Raw Dataset Sample")

    col_f1, col_f2 = st.columns(2)
    label_filter = col_f1.multiselect(
        "Filter by Label",
        options=df["label"].unique().tolist(),
        default=df["label"].unique().tolist(),
    )
    n_rows = col_f2.slider("Rows to display", 50, 500, 100, 50)

    filtered_df = df[df["label"].isin(label_filter)].head(n_rows)
    st.dataframe(filtered_df, use_container_width=True, height=400)
    st.caption(f"Showing {len(filtered_df):,} of {len(df):,} records")

    st.divider()
    section("Descriptive Statistics")
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    st.dataframe(df[numeric_cols].describe().T, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — REPORT
# ══════════════════════════════════════════════════════════════════════════════
with tabs[5]:
    section("Full Evaluation Report")

    col_r1, col_r2 = st.columns([2, 1])
    with col_r1:
        st.code(report_txt, language="text")
    with col_r2:
        section("Quick Summary")
        summary_data = {
            "Algorithm":          "Isolation Forest",
            "Dataset":            "NSL-KDD (Synthetic)",
            "Total Records":      stats["total"],
            "Training Split":     "70 %",
            "Test Split":         "30 %",
            "Accuracy":           f"{metrics['accuracy']} %",
            "Precision":          f"{metrics['precision']} %",
            "Recall":             f"{metrics['recall']} %",
            "F1-Score":           f"{metrics['f1_score']} %",
            "ROC-AUC":            f"{metrics.get('roc_auc','N/A')} %",
            "False Positive Rate":f"{metrics['false_positive_rate']} %",
            "False Negative Rate":f"{metrics['false_negative_rate']} %",
            "Detection Rate":     f"{metrics['detection_rate']} %",
        }
        st.table(pd.DataFrame(
            list(summary_data.items()), columns=["Metric", "Value"]
        ))

    st.divider()
    section("References & Dataset Sources")
    st.markdown("""
| Reference | Link |
|-----------|------|
| NSL-KDD Dataset | [University of New Brunswick](https://www.unb.ca/cic/datasets/nsl.html) |
| CICIDS 2017 | [Canadian Institute for Cybersecurity](https://www.unb.ca/cic/datasets/ids-2017.html) |
| Isolation Forest Paper | Liu, Ting & Zhou (2008) — IEEE ICDM |
| Scikit-learn Docs | [sklearn.ensemble.IsolationForest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html) |
| Streamlit Docs | [streamlit.io/docs](https://docs.streamlit.io) |
    """)

# ──────────────────────────────────────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────────────────────────────────────
st.divider()
st.markdown("""
<div style='text-align:center; color:#6B7280; font-size:12px; padding:8px 0;'>
  Zero-Day Anomaly Detection IDS &nbsp;·&nbsp;
  Domain: Cybersecurity / AI &nbsp;·&nbsp;
  Tech: Python · Scikit-learn · Streamlit &nbsp;·&nbsp;
  Academic Year 2025–2026
</div>
""", unsafe_allow_html=True)
