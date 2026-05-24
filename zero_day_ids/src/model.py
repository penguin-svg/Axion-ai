"""
model.py
========
Isolation Forest training, scoring, threshold tuning,
model persistence (save / load), and prediction helpers.
"""

import os
import pickle
import numpy as np
from sklearn.ensemble import IsolationForest


# ──────────────────────────────────────────────────────────────────────────────
# Default hyper-parameters
# ──────────────────────────────────────────────────────────────────────────────
DEFAULT_PARAMS = {
    "n_estimators":  200,
    "contamination": 0.20,
    "max_features":  1.0,
    "max_samples":   "auto",
    "bootstrap":     False,
    "random_state":  42,
    "n_jobs":        -1,
}


# ──────────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────────
def train_model(X_train: np.ndarray,
                contamination: float = 0.20,
                n_estimators:  int   = 200,
                random_state:  int   = 42) -> IsolationForest:
    """
    Train Isolation Forest on *X_train* (normal + mixed data).

    The model is trained on ALL training data; the contamination parameter
    guides the decision threshold internally.
    """
    params = {**DEFAULT_PARAMS,
              "contamination": contamination,
              "n_estimators":  n_estimators,
              "random_state":  random_state}
    clf = IsolationForest(**params)
    clf.fit(X_train)
    return clf


# ──────────────────────────────────────────────────────────────────────────────
# Prediction & scoring
# ──────────────────────────────────────────────────────────────────────────────
def predict(clf: IsolationForest, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Run inference.

    Returns
    -------
    predictions : np.ndarray of int  — 1 = anomaly, 0 = normal
    scores      : np.ndarray of float — raw anomaly score in [-1, +1]
                  (more negative → more anomalous)
    """
    raw_preds = clf.predict(X)           # sklearn: -1 = anomaly, 1 = normal
    predictions = (raw_preds == -1).astype(int)  # convert to 1 = anomaly
    scores = clf.score_samples(X)        # lower value → more anomalous
    return predictions, scores


def anomaly_score_normalised(scores: np.ndarray) -> np.ndarray:
    """
    Normalise raw decision scores to [0, 1] so that values close to 1
    represent the MOST anomalous records (mirrors the paper's formulation).
    """
    # score_samples returns negative of the raw anomaly score;
    # flip and min-max scale
    flipped = -scores
    mn, mx = flipped.min(), flipped.max()
    if mx == mn:
        return np.zeros_like(flipped)
    return (flipped - mn) / (mx - mn)


# ──────────────────────────────────────────────────────────────────────────────
# Threshold tuning (optional – uses labelled test set)
# ──────────────────────────────────────────────────────────────────────────────
def tune_threshold(scores: np.ndarray,
                   y_true: np.ndarray,
                   n_thresholds: int = 100) -> tuple[float, dict]:
    """
    Sweep percentile thresholds on normalised anomaly scores to find
    the one that maximises F1-score on the test labels.

    Returns the best raw-score threshold and a dict of metrics per threshold.
    """
    from sklearn.metrics import f1_score

    norm_scores = anomaly_score_normalised(scores)
    results = []

    for pct in np.linspace(50, 99, n_thresholds):
        thresh = np.percentile(norm_scores, pct)
        preds  = (norm_scores >= thresh).astype(int)
        f1     = f1_score(y_true, preds, zero_division=0)
        results.append({"percentile": pct, "threshold": thresh, "f1": f1})

    best = max(results, key=lambda r: r["f1"])
    return best["threshold"], results


# ──────────────────────────────────────────────────────────────────────────────
# Persistence
# ──────────────────────────────────────────────────────────────────────────────
def save_model(clf: IsolationForest, path: str) -> None:
    """Pickle the trained model to *path*."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(clf, fh)


def load_model(path: str) -> IsolationForest:
    """Load a pickled model from *path*."""
    with open(path, "rb") as fh:
        return pickle.load(fh)


# ──────────────────────────────────────────────────────────────────────────────
# Model information helper
# ──────────────────────────────────────────────────────────────────────────────
def model_info(clf: IsolationForest) -> dict:
    """Return a summary dict of model hyper-parameters for the dashboard."""
    return {
        "n_estimators":  clf.n_estimators,
        "contamination": clf.contamination,
        "max_features":  clf.max_features,
        "max_samples":   clf.max_samples,
        "bootstrap":     clf.bootstrap,
        "random_state":  clf.random_state,
        "n_features_in": getattr(clf, "n_features_in_", "N/A"),
    }
