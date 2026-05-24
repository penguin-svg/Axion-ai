"""
app.py  —  Zero-Day Anomaly Detection IDS  —  Streamlit Dashboard
Run:  streamlit run app.py
"""
import os, sys, pickle, tempfile
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(__file__))

# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ZeroDay IDS — Anomaly Detection Dashboard",
    page_icon="🛡️", layout="wide", initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  .main{background-color:#0E1117;}
  section[data-testid="stSidebar"]{background-color:#111827;}
  [data-testid="metric-container"]{
    background:#1A1F2E;border:1px solid #2D3748;
    border-radius:10px;padding:12px 18px;}
  [data-testid="stMetricLabel"] {font-size:13px!important;color:#9CA3AF!important;}
  [data-testid="stMetricValue"] {font-size:28px!important;color:#F9FAFB!important;font-weight:700;}
  .section-header{font-size:20px;font-weight:700;color:#F9FAFB;
    border-left:4px solid #2196F3;padding-left:12px;margin:28px 0 16px 0;}
  hr{border-color:#2D3748;}
  button[data-baseweb="tab"]{font-size:14px;font-weight:600;}
</style>""", unsafe_allow_html=True)


def section(t): st.markdown(f'<div class="section-header">{t}</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/security-shield-green.png", width=72)
    st.markdown("## 🛡️ ZeroDay IDS")
    st.markdown("*Unsupervised Anomaly Detection*")
    st.divider()

    # ── Dataset mode ──────────────────────────────────────────────────────────
    st.markdown("### 📂 Dataset")
    data_mode = st.radio(
        "Data Source",
        ["📁 NSL-KDD Files (Real)", "🔬 Synthetic Data"],
        index=0,
        help="Choose real NSL-KDD dataset files or generate synthetic data",
    )

    train_path = None
    test_path  = None

    if data_mode == "📁 NSL-KDD Files (Real)":
        st.markdown("**Upload Training File**")
        train_file = st.file_uploader(
            "KDDTrain+.txt or KDDTrain+20Percent.txt",
            type=["txt", "csv"], key="train_upload",
            help="NSL-KDD training file (no header, comma-separated)",
        )
        st.markdown("**Upload Test File** *(optional)*")
        test_file = st.file_uploader(
            "KDDTest+.txt or KDDTest-21.txt",
            type=["txt", "csv"], key="test_upload",
            help="Leave empty to auto-split training file 70/30",
        )

        # Save uploaded files to temp paths so pipeline can read them
        if train_file is not None:
            tmp_train = tempfile.NamedTemporaryFile(
                delete=False, suffix=".txt", mode="wb")
            tmp_train.write(train_file.read())
            tmp_train.flush()
            train_path = tmp_train.name
            st.success(f"✅ Train: {train_file.name}  ({train_file.size//1024:,} KB)")
        else:
            st.info("⬆️ Upload a training file to use real data")

        if test_file is not None:
            tmp_test = tempfile.NamedTemporaryFile(
                delete=False, suffix=".txt", mode="wb")
            tmp_test.write(test_file.read())
            tmp_test.flush()
            test_path = tmp_test.name
            st.success(f"✅ Test:  {test_file.name}  ({test_file.size//1024:,} KB)")
        else:
            if train_path:
                st.caption("No test file → 70/30 auto-split used")

        # Quick-load from local data/ folder if files exist there already
        st.markdown("**— or use files already in `data/` folder —**")
        local_options = {"(none)": (None, None)}
        for combo in [
            ("KDDTrain+ / KDDTest+",
             "data/KDDTrain+.txt", "data/KDDTest+.txt"),
            ("KDDTrain+20% / KDDTest+",
             "data/KDDTrain+20Percent.txt", "data/KDDTest+.txt"),
            ("KDDTrain+ / KDDTest-21",
             "data/KDDTrain+.txt", "data/KDDTest-21.txt"),
            ("KDDTrain+ only (70/30 split)",
             "data/KDDTrain+.txt", None),
        ]:
            label, tr, te = combo
            if tr and os.path.exists(tr):
                local_options[label] = (tr, te)

        if len(local_options) > 1:
            chosen = st.selectbox("Local file combo", list(local_options.keys()))
            if chosen != "(none)" and train_path is None:
                train_path, test_path = local_options[chosen]
                if train_path:
                    st.success(f"✅ Using local: {chosen}")
        else:
            st.caption("(No files found in `data/` folder)")

        n_samples = 8000   # unused in real-data mode

    else:  # Synthetic
        n_samples = st.select_slider(
            "Synthetic Records",
            options=[2000, 4000, 6000, 8000, 10000, 15000], value=8000,
        )

    st.divider()

    # ── Model parameters ──────────────────────────────────────────────────────
    st.markdown("### ⚙️ Model Parameters")
    auto_contam = st.checkbox("Auto-detect contamination", value=True,
                               help="Estimate from training label ratio")
    contamination = None
    if not auto_contam:
        contamination = st.slider("Contamination", 0.05, 0.50, 0.20, 0.01)

    n_estimators = st.select_slider(
        "n_estimators (trees)", options=[50, 100, 150, 200, 300], value=200)

    st.divider()
    retrain = st.button("🚀 Train / Reload Model",
                         use_container_width=True, type="primary")

    st.markdown("### ℹ️ About")
    st.markdown("""
**Algorithm**: Isolation Forest  
**Dataset**: NSL-KDD (KDDTrain+/KDDTest+)  
**Framework**: Scikit-learn + Streamlit  
**Academic Year**: 2025–2026
    """)


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE CACHE
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="⏳ Running pipeline on your dataset …")
def get_artifacts(_train_path, _test_path, _contamination, _n_estimators, _n_samples, _ts):
    """Cache key includes all params + a timestamp so Retrain button busts it."""
    from pipeline import run_pipeline
    return run_pipeline(
        train_path=_train_path,
        test_path=_test_path,
        contamination=_contamination,
        n_estimators=_n_estimators,
        n_samples=_n_samples,
    )


import time
if "cache_ts" not in st.session_state:
    st.session_state["cache_ts"] = 0
if retrain:
    st.cache_resource.clear()
    st.session_state["cache_ts"] = time.time()

# Show helpful message if no data chosen yet
if data_mode == "📁 NSL-KDD Files (Real)" and train_path is None:
    st.markdown("""
    <div style='text-align:center;padding:80px 0;'>
      <div style='font-size:60px;'>🛡️</div>
      <div style='font-size:26px;font-weight:700;color:#F9FAFB;margin:16px 0 8px;'>
        Zero-Day IDS Dashboard
      </div>
      <div style='font-size:15px;color:#9CA3AF;max-width:520px;margin:auto;'>
        Upload your <b>NSL-KDD training file</b> in the sidebar to get started.<br><br>
        Supported files: <code>KDDTrain+.txt</code>, <code>KDDTrain+20Percent.txt</code><br>
        Optional test file: <code>KDDTest+.txt</code>, <code>KDDTest-21.txt</code>
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# Run pipeline
with st.spinner("Loading / training … this may take 30–90 seconds for full NSL-KDD"):
    artifacts = get_artifacts(
        train_path, test_path, contamination, n_estimators, n_samples,
        st.session_state["cache_ts"],
    )

# ─────────────────────────────────────────────────────────────────────────────
# UNPACK ARTIFACTS
# ─────────────────────────────────────────────────────────────────────────────
df               = artifacts["df"]
df_train         = artifacts.get("df_train", df)
df_test          = artifacts.get("df_test",  df)
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

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
# Dataset info banner
if train_path:
    tr_name = os.path.basename(artifacts.get("train_path") or "")
    te_name = os.path.basename(artifacts.get("test_path") or "auto-split")
    st.markdown(f"""
    <div style='background:#1A2744;border:1px solid #2D3748;border-radius:8px;
                padding:10px 20px;margin-bottom:12px;font-size:13px;color:#9CA3AF;'>
      📁 <b style='color:#F9FAFB;'>Training:</b> {tr_name} &nbsp;({stats['total']:,} records)
      &nbsp;&nbsp;|&nbsp;&nbsp;
      📋 <b style='color:#F9FAFB;'>Test:</b> {te_name} &nbsp;({stats.get('test_total', len(df_test)):,} records)
    </div>""", unsafe_allow_html=True)

st.markdown("""
<div style='text-align:center;padding:6px 0 4px;'>
  <span style='font-size:38px;font-weight:800;color:#F9FAFB;'>
    🛡️ Zero-Day Anomaly Detection IDS
  </span><br>
  <span style='font-size:14px;color:#9CA3AF;'>
    Unsupervised Intrusion Detection · Isolation Forest · NSL-KDD
  </span>
</div>""", unsafe_allow_html=True)
st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# KPI ROW
# ─────────────────────────────────────────────────────────────────────────────
c1,c2,c3,c4,c5,c6,c7 = st.columns(7)
c1.metric("🗃️ Train Records", f"{stats['total']:,}")
c2.metric("📋 Test Records",  f"{stats.get('test_total', len(df_test)):,}")
c3.metric("✅ Normal",        f"{stats['normal']:,}")
c4.metric("🚨 Attacks",       f"{stats['attacks']:,}")
c5.metric("🎯 Accuracy",      f"{metrics['accuracy']} %")
c6.metric("📊 F1-Score",      f"{metrics['f1_score']} %")
c7.metric("🔍 ROC-AUC",       f"{metrics.get('roc_auc','N/A')} %")
st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "📊 Overview", "🧪 Model Performance", "🔍 Anomaly Analysis",
    "📈 Feature Insights", "🗂️ Data Explorer", "🔴 Live Predict", "📋 Report",
])

# ══ TAB 1 — OVERVIEW ═════════════════════════════════════════════════════════
with tabs[0]:
    section("Traffic Distribution")
    c_a, c_b = st.columns(2)
    with c_a:
        st.plotly_chart(
            traffic_overview_donut(stats["normal"], stats["attacks"]),
            use_container_width=True)
    with c_b:
        if stats.get("attack_dist"):
            st.plotly_chart(
                attack_distribution_bar(stats["attack_dist"]),
                use_container_width=True)
        else:
            st.info("Attack distribution not available for this dataset mode.")

    c_c, c_d = st.columns(2)
    with c_c:
        if stats.get("protocol_dist"):
            st.plotly_chart(protocol_bar(stats["protocol_dist"]),
                            use_container_width=True)
    with c_d:
        section("Dataset Summary")
        src = "NSL-KDD (Real)" if train_path else "Synthetic"
        summary = {
            "Data Source":          src,
            "Training Records":     f"{stats['total']:,}",
            "Test Records":         f"{stats.get('test_total', len(df_test)):,}",
            "Normal Traffic":       f"{stats['normal']:,}",
            "Attack Records":       f"{stats['attacks']:,}",
            "Anomaly %":            f"{stats['anomaly_pct']} %",
            "Features Used":        stats["n_features"],
            "PCA Components":       pca_summary["n_components"],
            "Variance Retained":    f"{pca_summary['total_variance']} %",
            "Contamination Used":   model_info_dict["contamination"],
        }
        st.dataframe(
            pd.DataFrame(list(summary.items()), columns=["Item", "Value"]),
            hide_index=True, use_container_width=True)

    section("Pipeline Flow")
    st.markdown("""
    <div style='background:#1A1F2E;border-radius:10px;padding:20px 28px;
                border:1px solid #2D3748;font-size:14px;line-height:2.3;'>
    📥 <b>NSL-KDD Dataset</b> (KDDTrain+ · KDDTest+ · KDDTest-21)<br>&nbsp;&nbsp;&nbsp;↓<br>
    🔧 <b>Preprocessing</b> — Dedup · Label Encoding · 99.5% Clipping · Min-Max Scale<br>&nbsp;&nbsp;&nbsp;↓<br>
    ⚙️ <b>Feature Engineering</b> — Mutual Information Top-20 · PCA 95% Variance<br>&nbsp;&nbsp;&nbsp;↓<br>
    🌲 <b>Isolation Forest Training</b> — Learns Normal Traffic Baseline (Unsupervised)<br>&nbsp;&nbsp;&nbsp;↓<br>
    🎯 <b>Anomaly Scoring</b> — Normalised Score [0,1] · Threshold Decision<br>&nbsp;&nbsp;&nbsp;↓<br>
    📊 <b>Evaluation</b> — Accuracy · F1 · Confusion Matrix · ROC-AUC · Zero-Day Rate
    </div>""", unsafe_allow_html=True)


# ══ TAB 2 — MODEL PERFORMANCE ═════════════════════════════════════════════════
with tabs[1]:
    section("Classification Metrics")
    m1,m2,m3,m4 = st.columns(4)
    m1.metric("Accuracy",  f"{metrics['accuracy']} %")
    m2.metric("Precision", f"{metrics['precision']} %")
    m3.metric("Recall",    f"{metrics['recall']} %")
    m4.metric("F1-Score",  f"{metrics['f1_score']} %")

    m5,m6,m7,m8 = st.columns(4)
    m5.metric("True Positives",  f"{metrics['true_positives']:,}")
    m6.metric("True Negatives",  f"{metrics['true_negatives']:,}")
    m7.metric("False Positives", f"{metrics['false_positives']:,}")
    m8.metric("False Negatives", f"{metrics['false_negatives']:,}")

    st.divider()
    section("Confusion Matrix  &  ROC Curve")
    r1, r2 = st.columns(2)
    with r1:
        st.plotly_chart(confusion_matrix_heatmap(cm_df), use_container_width=True)
    with r2:
        st.plotly_chart(roc_curve_plot(roc_data), use_container_width=True)

    st.plotly_chart(pr_curve_plot(pr_data), use_container_width=True)

    st.divider()
    section("Detection Rate by Attack Category")
    st.plotly_chart(detection_rate_by_attack(breakdown), use_container_width=True)
    st.dataframe(breakdown, hide_index=True, use_container_width=True)

    st.divider()
    section("Model Configuration")
    st.json(model_info_dict)


# ══ TAB 3 — ANOMALY ANALYSIS ══════════════════════════════════════════════════
with tabs[2]:
    section("Anomaly Score Distribution")
    st.plotly_chart(
        anomaly_score_histogram(scores_norm, y_test), use_container_width=True)

    section("Score Timeline — First 600 Test Records")
    st.plotly_chart(
        anomaly_timeline(scores_norm, y_pred), use_container_width=True)

    section("PCA Traffic Clusters")
    st.plotly_chart(
        pca_scatter(X_test_pca, y_test, attack_types_test),
        use_container_width=True)

    section("Top 50 Highest-Scoring Anomalies")
    df_res = df_test.copy().reset_index(drop=True)
    df_res["anomaly_score"] = scores_norm
    df_res["predicted"]     = y_pred
    df_res["status"]        = df_res["predicted"].map({1:"🚨 ANOMALY", 0:"✅ Normal"})

    show_cols = ["status","label","category","anomaly_score"]
    for c in ["src_bytes","dst_bytes","duration","protocol_type","flag"]:
        if c in df_res.columns:
            show_cols.append(c)

    top_a = (df_res[df_res["predicted"]==1]
             .sort_values("anomaly_score", ascending=False)
             .head(50)[show_cols]
             .reset_index(drop=True))
    st.dataframe(top_a, use_container_width=True)


# ══ TAB 4 — FEATURE INSIGHTS ══════════════════════════════════════════════════
with tabs[3]:
    section("Feature Importance (Mutual Information)")
    top_n = st.slider("Show top N features", 5, min(30, len(importance_df)), 15)
    st.plotly_chart(feature_importance_bar(importance_df, top_n=top_n),
                    use_container_width=True)

    st.divider()
    section("PCA Explained Variance")
    st.plotly_chart(
        pca_variance_plot(pca_summary["cumulative_variance"]),
        use_container_width=True)
    col1, col2 = st.columns(2)
    col1.metric("PCA Components", pca_summary["n_components"])
    col2.metric("Variance Retained", f"{pca_summary['total_variance']} %")

    st.divider()
    section("Full Feature Importance Table")
    st.dataframe(importance_df, hide_index=True, use_container_width=True)


# ══ TAB 5 — DATA EXPLORER ═════════════════════════════════════════════════════
with tabs[4]:
    section("Training Data Explorer")
    all_labels = df_train["label"].unique().tolist() if "label" in df_train.columns else []
    f1c, f2c = st.columns(2)
    label_filter = f1c.multiselect("Filter by Label", all_labels, default=all_labels[:10])
    n_rows = f2c.slider("Rows to display", 50, 500, 100, 50)

    filt = df_train[df_train["label"].isin(label_filter)].head(n_rows) \
           if label_filter else df_train.head(n_rows)
    st.dataframe(filt, use_container_width=True, height=380)
    st.caption(f"Showing {len(filt):,} of {len(df_train):,} training records")

    st.divider()
    section("Descriptive Statistics (Numeric Features)")
    num_cols = df_train.select_dtypes(include=np.number).columns.tolist()
    st.dataframe(df_train[num_cols].describe().T.round(4), use_container_width=True)

    st.divider()
    section("Attack Label Counts")
    if "label" in df_train.columns:
        lbl_vc = df_train["label"].value_counts().reset_index()
        lbl_vc.columns = ["Label", "Count"]
        lbl_vc["Category"] = lbl_vc["Label"].map(
            {k:v for k,v in
             __import__("pipeline").ATTACK_MAP.items()}).fillna("Unknown")
        st.dataframe(lbl_vc, hide_index=True, use_container_width=True)


# ══ TAB 6 — LIVE PREDICT ══════════════════════════════════════════════════════
with tabs[5]:
    section("🔴 Live Single-Record Anomaly Prediction")
    st.markdown("Fill in feature values below to predict whether a network connection is anomalous.")

    with st.form("predict_form"):
        c1p,c2p,c3p = st.columns(3)
        proto   = c1p.selectbox("protocol_type", ["tcp","udp","icmp"])
        service = c2p.selectbox("service",
                    ["http","ftp","smtp","ssh","dns","private","telnet","other"])
        flag    = c3p.selectbox("flag",
                    ["SF","S0","REJ","RSTO","SH","RSTR","OTH","S1","S2","S3"])

        c4p,c5p,c6p = st.columns(3)
        src_bytes = c4p.number_input("src_bytes", 0, 10_000_000, 500)
        dst_bytes = c5p.number_input("dst_bytes", 0, 10_000_000, 8000)
        duration  = c6p.number_input("duration",  0, 60000, 0)

        c7p,c8p,c9p = st.columns(3)
        count        = c7p.number_input("count",        0, 512, 5)
        srv_count    = c8p.number_input("srv_count",    0, 512, 5)
        serror_rate  = c9p.slider("serror_rate",  0.0, 1.0, 0.0, 0.01)

        c10p,c11p,c12p = st.columns(3)
        same_srv_rate = c10p.slider("same_srv_rate",  0.0, 1.0, 1.0, 0.01)
        diff_srv_rate = c11p.slider("diff_srv_rate",  0.0, 1.0, 0.0, 0.01)
        logged_in     = c12p.selectbox("logged_in", [1, 0])

        c13p,c14p,c15p = st.columns(3)
        root_shell       = c13p.selectbox("root_shell",       [0, 1])
        num_compromised  = c14p.number_input("num_compromised", 0, 10, 0)
        num_failed_logins= c15p.number_input("num_failed_logins", 0, 5, 0)

        submitted = st.form_submit_button("🔍 Predict", use_container_width=True)

    if submitted:
        import pickle as _pk
        model_bundle_path = "models/isolation_forest.pkl"
        if not os.path.exists(model_bundle_path):
            st.error("Model not found. Please train first.")
        else:
            bundle = _pk.load(open(model_bundle_path,"rb"))
            _clf   = bundle["model"]
            _sc    = bundle["scaler"]
            _pca   = bundle["pca"]
            _enc   = bundle["encoders"]
            _tidx  = bundle["top_indices"]
            _fc    = bundle["feat_cols"]
            _caps  = bundle["caps"]

            # Build feature row
            row = {c: 0.0 for c in _fc}
            row.update({
                "protocol_type": proto, "service": service, "flag": flag,
                "src_bytes": src_bytes, "dst_bytes": dst_bytes,
                "duration": duration, "count": count, "srv_count": srv_count,
                "serror_rate": serror_rate, "same_srv_rate": same_srv_rate,
                "diff_srv_rate": diff_srv_rate, "logged_in": logged_in,
                "root_shell": root_shell, "num_compromised": num_compromised,
                "num_failed_logins": num_failed_logins,
            })

            # Encode categoricals
            for col in ["protocol_type","service","flag"]:
                if col in _enc:
                    le = _enc[col]
                    v  = str(row[col])
                    row[col] = float(le.transform([v])[0]) if v in le.classes_ else 0.0

            # Clip & scale
            for col, cap in _caps.items():
                if col in row:
                    row[col] = min(float(row[col]), cap)

            X_single = np.array([[row[c] for c in _fc]], dtype=float)
            X_single = _sc.transform(X_single)
            X_single = X_single[:, _tidx]
            X_single = _pca.transform(X_single)

            is_anom = (_clf.predict(X_single)[0] == -1)
            raw_s   = _clf.score_samples(X_single)[0]
            norm_s  = float(np.clip(
                1.0 - (raw_s - artifacts["raw_scores"].min()) /
                      (artifacts["raw_scores"].max() - artifacts["raw_scores"].min() + 1e-9),
                0, 1))

            if is_anom:
                st.error(f"🚨 **ANOMALY DETECTED** — Score: `{norm_s:.4f}`")
            else:
                st.success(f"✅ **Normal Traffic** — Score: `{norm_s:.4f}`")

            col_r1, col_r2, col_r3 = st.columns(3)
            col_r1.metric("Anomaly Score", f"{norm_s:.4f}")
            col_r2.metric("Classification", "ANOMALY" if is_anom else "Normal")
            col_r3.metric("Confidence", f"{'High' if abs(norm_s - 0.5) > 0.25 else 'Medium'}")


# ══ TAB 7 — REPORT ════════════════════════════════════════════════════════════
with tabs[6]:
    section("Full Evaluation Report")
    col_r1, col_r2 = st.columns([2,1])
    with col_r1:
        st.code(report_txt, language="text")
    with col_r2:
        section("Quick Reference")
        ref = {
            "Algorithm":          "Isolation Forest",
            "Data Source":        "NSL-KDD (Real)" if train_path else "Synthetic",
            "Training Records":   f"{stats['total']:,}",
            "Test Records":       f"{stats.get('test_total', len(df_test)):,}",
            "Accuracy":           f"{metrics['accuracy']} %",
            "Precision":          f"{metrics['precision']} %",
            "Recall":             f"{metrics['recall']} %",
            "F1-Score":           f"{metrics['f1_score']} %",
            "ROC-AUC":            f"{metrics.get('roc_auc','N/A')} %",
            "False Positive Rate":f"{metrics['false_positive_rate']} %",
            "False Negative Rate":f"{metrics['false_negative_rate']} %",
            "Detection Rate":     f"{metrics['detection_rate']} %",
        }
        st.table(pd.DataFrame(list(ref.items()), columns=["Metric","Value"]))

    st.divider()
    section("References")
    st.markdown("""
| Reference | Link |
|-----------|------|
| NSL-KDD Dataset | [University of New Brunswick](https://www.unb.ca/cic/datasets/nsl.html) |
| Isolation Forest (Liu et al. 2008) | IEEE ICDM |
| Scikit-learn | [IsolationForest docs](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html) |
| Streamlit | [docs.streamlit.io](https://docs.streamlit.io) |
    """)

st.divider()
st.markdown("""
<div style='text-align:center;color:#6B7280;font-size:12px;padding:6px 0;'>
  Zero-Day Anomaly Detection IDS · NSL-KDD · Isolation Forest · Python · Scikit-learn · Streamlit · 2025–2026
</div>""", unsafe_allow_html=True)
