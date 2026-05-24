"""
evaluation.py
=============
All performance metrics, confusion-matrix helpers, and report generation
for the Isolation Forest IDS.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve,
)


# ──────────────────────────────────────────────────────────────────────────────
# Core metrics
# ──────────────────────────────────────────────────────────────────────────────
def compute_metrics(y_true: np.ndarray,
                    y_pred: np.ndarray,
                    scores: np.ndarray | None = None) -> dict:
    """
    Compute and return a comprehensive performance metrics dictionary.

    Parameters
    ----------
    y_true  : ground-truth binary labels (1 = anomaly)
    y_pred  : predicted binary labels    (1 = anomaly)
    scores  : continuous anomaly scores for AUC computation (optional)
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    metrics = {
        # Core classification metrics
        "accuracy":         round(accuracy_score(y_true, y_pred) * 100, 2),
        "precision":        round(precision_score(y_true, y_pred, zero_division=0) * 100, 2),
        "recall":           round(recall_score(y_true, y_pred, zero_division=0) * 100, 2),
        "f1_score":         round(f1_score(y_true, y_pred, zero_division=0) * 100, 2),
        # Confusion matrix elements
        "true_positives":   int(tp),
        "true_negatives":   int(tn),
        "false_positives":  int(fp),
        "false_negatives":  int(fn),
        # Derived rates
        "false_positive_rate": round(fp / (fp + tn + 1e-9) * 100, 2),
        "false_negative_rate": round(fn / (fn + tp + 1e-9) * 100, 2),
        "specificity":         round(tn / (tn + fp + 1e-9) * 100, 2),
        "detection_rate":      round(tp / (tp + fn + 1e-9) * 100, 2),
    }

    if scores is not None:
        try:
            metrics["roc_auc"]  = round(roc_auc_score(y_true, scores) * 100, 2)
            metrics["avg_prec"] = round(average_precision_score(y_true, scores) * 100, 2)
        except Exception:
            metrics["roc_auc"]  = None
            metrics["avg_prec"] = None

    return metrics


# ──────────────────────────────────────────────────────────────────────────────
# Curve data (for Plotly charts)
# ──────────────────────────────────────────────────────────────────────────────
def roc_curve_data(y_true: np.ndarray,
                   scores: np.ndarray) -> dict:
    """Return FPR / TPR arrays for an ROC curve plot."""
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    auc = roc_auc_score(y_true, scores)
    return {"fpr": fpr.tolist(), "tpr": tpr.tolist(),
            "thresholds": thresholds.tolist(), "auc": round(auc, 4)}


def pr_curve_data(y_true: np.ndarray,
                  scores: np.ndarray) -> dict:
    """Return Precision / Recall arrays for a PR curve plot."""
    prec, rec, thresholds = precision_recall_curve(y_true, scores)
    ap = average_precision_score(y_true, scores)
    return {"precision": prec.tolist(), "recall": rec.tolist(),
            "thresholds": thresholds.tolist(), "average_precision": round(ap, 4)}


# ──────────────────────────────────────────────────────────────────────────────
# Confusion matrix
# ──────────────────────────────────────────────────────────────────────────────
def confusion_matrix_df(y_true: np.ndarray,
                         y_pred: np.ndarray) -> pd.DataFrame:
    """Return the confusion matrix as a labelled DataFrame."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return pd.DataFrame(
        cm,
        index=["Actual Normal", "Actual Anomaly"],
        columns=["Predicted Normal", "Predicted Anomaly"],
    )


# ──────────────────────────────────────────────────────────────────────────────
# Per-attack-type breakdown
# ──────────────────────────────────────────────────────────────────────────────
def per_class_breakdown(y_pred: np.ndarray,
                         attack_types: np.ndarray | pd.Series) -> pd.DataFrame:
    """
    For each attack type (dos, probe, r2l, u2r, normal), compute how many
    records were correctly flagged as anomalous.
    """
    rows = []
    for atk in np.unique(attack_types):
        mask   = attack_types == atk
        total  = mask.sum()
        if total == 0:
            continue
        if atk == "normal":
            detected  = int((y_pred[mask] == 0).sum())
            label_str = "Normal (correctly kept)"
        else:
            detected  = int((y_pred[mask] == 1).sum())
            label_str = atk.upper()

        rows.append({
            "Attack Type":     label_str,
            "Total Records":   int(total),
            "Correctly Flagged": detected,
            "Detection Rate %":  round(detected / total * 100, 1),
        })

    return pd.DataFrame(rows).sort_values("Detection Rate %", ascending=False)


# ──────────────────────────────────────────────────────────────────────────────
# Printable text report
# ──────────────────────────────────────────────────────────────────────────────
def generate_report(metrics: dict, model_info: dict | None = None) -> str:
    """Build a formatted plain-text evaluation report."""
    lines = [
        "=" * 60,
        "  ZERO-DAY IDS — EVALUATION REPORT",
        "=" * 60,
        "",
        "  CLASSIFICATION METRICS",
        "  ─────────────────────────────────────────",
        f"  Accuracy         : {metrics['accuracy']} %",
        f"  Precision        : {metrics['precision']} %",
        f"  Recall           : {metrics['recall']} %",
        f"  F1-Score         : {metrics['f1_score']} %",
        f"  ROC-AUC          : {metrics.get('roc_auc', 'N/A')} %",
        "",
        "  CONFUSION MATRIX",
        "  ─────────────────────────────────────────",
        f"  True  Positives  : {metrics['true_positives']}",
        f"  True  Negatives  : {metrics['true_negatives']}",
        f"  False Positives  : {metrics['false_positives']}",
        f"  False Negatives  : {metrics['false_negatives']}",
        "",
        "  DETECTION RATES",
        "  ─────────────────────────────────────────",
        f"  Detection Rate   : {metrics['detection_rate']} %",
        f"  Specificity      : {metrics['specificity']} %",
        f"  False Pos. Rate  : {metrics['false_positive_rate']} %",
        f"  False Neg. Rate  : {metrics['false_negative_rate']} %",
    ]

    if model_info:
        lines += [
            "",
            "  MODEL HYPER-PARAMETERS",
            "  ─────────────────────────────────────────",
        ]
        for k, v in model_info.items():
            lines.append(f"  {k:<22}: {v}")

    lines += ["", "=" * 60]
    return "\n".join(lines)
