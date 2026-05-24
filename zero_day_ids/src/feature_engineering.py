"""
feature_engineering.py
=======================
Feature selection, PCA dimensionality reduction, and feature-importance
utilities used by the dashboard and model pipeline.
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler


# ──────────────────────────────────────────────────────────────────────────────
# Top features identified from NSL-KDD domain knowledge + MI analysis
# ──────────────────────────────────────────────────────────────────────────────
TOP_FEATURES = [
    "serror_rate", "srv_serror_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "rerror_rate", "srv_rerror_rate",
    "same_srv_rate", "diff_srv_rate", "count", "srv_count",
    "dst_host_count", "dst_host_srv_count", "src_bytes", "dst_bytes",
    "logged_in", "root_shell", "num_compromised", "num_failed_logins",
    "wrong_fragment", "flag",
]


def select_top_features(X: np.ndarray,
                         feature_names: list[str],
                         top_n: int = 20) -> tuple[np.ndarray, list[str]]:
    """
    Return the *top_n* columns of X by matching against the curated
    TOP_FEATURES list.  Falls back to all columns if fewer are available.
    """
    indices = []
    for f in TOP_FEATURES:
        if f in feature_names:
            indices.append(feature_names.index(f))
        if len(indices) == top_n:
            break

    # Pad with remaining columns if needed
    if len(indices) < top_n:
        for i in range(len(feature_names)):
            if i not in indices:
                indices.append(i)
            if len(indices) == top_n:
                break

    selected_names = [feature_names[i] for i in indices]
    return X[:, indices], selected_names


def apply_pca(X_train: np.ndarray,
              X_test:  np.ndarray,
              variance_threshold: float = 0.95
              ) -> tuple[np.ndarray, np.ndarray, PCA, np.ndarray]:
    """
    Fit PCA on *X_train*, transform both arrays.

    Returns
    -------
    X_train_pca, X_test_pca, fitted PCA object, explained variance ratios
    """
    pca = PCA(n_components=variance_threshold, svd_solver="full")
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca  = pca.transform(X_test)
    return X_train_pca, X_test_pca, pca, pca.explained_variance_ratio_


def compute_feature_importance(X: np.ndarray,
                                y: np.ndarray,
                                feature_names: list[str]) -> pd.DataFrame:
    """
    Estimate feature importance using Mutual Information (works with
    any supervised target, here the binary anomaly label).
    """
    mi_scores = mutual_info_classif(X, y, random_state=42)
    importance_df = pd.DataFrame({
        "feature":    feature_names,
        "importance": mi_scores,
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    importance_df["rank"] = importance_df.index + 1
    return importance_df


def correlation_matrix(df: pd.DataFrame,
                        feature_names: list[str]) -> pd.DataFrame:
    """Return the Pearson correlation matrix for numeric features."""
    cols = [c for c in feature_names if c in df.columns]
    return df[cols].corr()


def get_pca_summary(pca: PCA) -> dict:
    """Return a tidy summary of PCA results for the dashboard."""
    evr = pca.explained_variance_ratio_
    return {
        "n_components":          pca.n_components_,
        "total_variance":        round(float(evr.sum()) * 100, 2),
        "cumulative_variance":   np.cumsum(evr).tolist(),
        "per_component_variance": evr.tolist(),
    }
