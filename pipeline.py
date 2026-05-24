"""
pipeline.py
===========
End-to-end training pipeline.
Run this once before launching the Streamlit dashboard.

Usage:
    python pipeline.py                   # uses synthetic data
    python pipeline.py --data path.csv   # uses your NSL-KDD CSV
"""

import os
import sys
import pickle
import argparse
import numpy as np

# Allow running from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from src.data_loader        import load_dataset, preprocess, get_statistics
from src.feature_engineering import (
    select_top_features, apply_pca,
    compute_feature_importance, get_pca_summary,
)
from src.model              import train_model, predict, anomaly_score_normalised, save_model, model_info
from src.evaluation         import (
    compute_metrics, confusion_matrix_df,
    per_class_breakdown, roc_curve_data, pr_curve_data, generate_report,
)


MODELS_DIR   = "models"
ARTIFACT_PATH = os.path.join(MODELS_DIR, "pipeline_artifacts.pkl")


def run_pipeline(data_path: str | None = None,
                 contamination: float = 0.20,
                 n_estimators: int = 200,
                 n_samples: int = 8000) -> dict:
    """
    Execute the full IDS pipeline and return all artifacts as a dict.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    print("[1/7] Loading dataset …")
    df = load_dataset(data_path, n_samples=n_samples)
    stats = get_statistics(df)
    print(f"      Total: {stats['total']:,}  |  Normal: {stats['normal']:,}  "
          f"|  Attacks: {stats['attacks']:,}  ({stats['anomaly_pct']}%)")

    # ── 2. Preprocess ─────────────────────────────────────────────────────────
    print("[2/7] Preprocessing …")
    (X_train, X_test, y_train, y_test,
     feature_names, scaler, encoders, df_proc) = preprocess(df)
    print(f"      Train: {X_train.shape}  |  Test: {X_test.shape}")

    # ── 3. Feature selection ──────────────────────────────────────────────────
    print("[3/7] Feature engineering …")
    X_train_sel, sel_features = select_top_features(X_train, feature_names, top_n=20)
    X_test_sel, _             = select_top_features(X_test,  feature_names, top_n=20)
    importance_df             = compute_feature_importance(X_train_sel, y_train, sel_features)

    # ── 4. PCA ────────────────────────────────────────────────────────────────
    print("[4/7] Applying PCA …")
    X_train_pca, X_test_pca, pca_obj, evr = apply_pca(X_train_sel, X_test_sel)
    pca_summary = get_pca_summary(pca_obj)
    print(f"      PCA components: {pca_summary['n_components']}  "
          f"|  Variance retained: {pca_summary['total_variance']}%")

    # ── 5. Train model ────────────────────────────────────────────────────────
    print("[5/7] Training Isolation Forest …")
    clf = train_model(X_train_pca, contamination=contamination,
                       n_estimators=n_estimators)
    info = model_info(clf)
    save_model(clf, os.path.join(MODELS_DIR, "isolation_forest.pkl"))
    print(f"      n_estimators={info['n_estimators']}  "
          f"contamination={info['contamination']}")

    # ── 6. Predict & score ────────────────────────────────────────────────────
    print("[6/7] Detecting anomalies …")
    y_pred, raw_scores        = predict(clf, X_test_pca)
    scores_norm               = anomaly_score_normalised(raw_scores)
    y_pred_train, raw_tr      = predict(clf, X_train_pca)

    # Attack types for test set (align with split)
    from sklearn.model_selection import train_test_split
    _, df_test = train_test_split(df_proc, test_size=0.30,
                                   random_state=42, stratify=df_proc["is_anomaly"])
    attack_types_test = df_test["attack_type"].values

    # ── 7. Evaluate ───────────────────────────────────────────────────────────
    print("[7/7] Evaluating …")
    metrics    = compute_metrics(y_test, y_pred, scores_norm)
    cm_df      = confusion_matrix_df(y_test, y_pred)
    breakdown  = per_class_breakdown(y_pred, attack_types_test)
    roc_data   = roc_curve_data(y_test, scores_norm)
    pr_data    = pr_curve_data(y_test, scores_norm)
    report_txt = generate_report(metrics, info)

    print("\n" + report_txt)

    # ── Bundle everything ─────────────────────────────────────────────────────
    artifacts = {
        "df":               df,
        "df_proc":          df_proc,
        "stats":            stats,
        "X_train_pca":      X_train_pca,
        "X_test_pca":       X_test_pca,
        "y_train":          y_train,
        "y_test":           y_test,
        "y_pred":           y_pred,
        "scores_norm":      scores_norm,
        "raw_scores":       raw_scores,
        "attack_types_test": attack_types_test,
        "feature_names":    feature_names,
        "sel_features":     sel_features,
        "importance_df":    importance_df,
        "pca_summary":      pca_summary,
        "metrics":          metrics,
        "cm_df":            cm_df,
        "breakdown":        breakdown,
        "roc_data":         roc_data,
        "pr_data":          pr_data,
        "model_info":       info,
        "report_txt":       report_txt,
    }

    with open(ARTIFACT_PATH, "wb") as fh:
        pickle.dump(artifacts, fh)
    print(f"\nArtifacts saved → {ARTIFACT_PATH}")
    return artifacts


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Zero-Day IDS Pipeline")
    parser.add_argument("--data",          default=None,  help="Path to NSL-KDD CSV")
    parser.add_argument("--contamination", default=0.20,  type=float)
    parser.add_argument("--n-estimators",  default=200,   type=int)
    parser.add_argument("--n-samples",     default=8000,  type=int,
                        help="Synthetic sample count (ignored if --data is set)")
    args = parser.parse_args()

    run_pipeline(
        data_path=args.data,
        contamination=args.contamination,
        n_estimators=args.n_estimators,
        n_samples=args.n_samples,
    )
