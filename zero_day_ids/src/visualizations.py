"""
visualizations.py
=================
All Plotly chart builders used by the Streamlit dashboard.
Each function returns a plotly.graph_objects.Figure.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


# ── Colour palette ────────────────────────────────────────────────────────────
NORMAL_COLOR  = "#2196F3"
ATTACK_COLOR  = "#F44336"
ACCENT_COLOR  = "#FF9800"
BG_COLOR      = "#0E1117"
GRID_COLOR    = "#2D2D2D"
TEXT_COLOR    = "#FAFAFA"

ATTACK_PALETTE = {
    "normal": NORMAL_COLOR,
    "dos":    "#F44336",
    "probe":  "#FF9800",
    "r2l":    "#9C27B0",
    "u2r":    "#E91E63",
}

_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor ="rgba(0,0,0,0)",
    font=dict(color=TEXT_COLOR, family="Inter, sans-serif"),
    xaxis=dict(gridcolor=GRID_COLOR, linecolor=GRID_COLOR),
    yaxis=dict(gridcolor=GRID_COLOR, linecolor=GRID_COLOR),
    margin=dict(l=40, r=20, t=50, b=40),
)


def _apply_layout(fig: go.Figure, title: str = "", **kwargs) -> go.Figure:
    fig.update_layout(title=dict(text=title, font=dict(size=15)),
                      **_LAYOUT, **kwargs)
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# 1. Traffic overview – donut
# ──────────────────────────────────────────────────────────────────────────────
def traffic_overview_donut(n_normal: int, n_anomaly: int) -> go.Figure:
    fig = go.Figure(go.Pie(
        labels=["Normal Traffic", "Anomalous Traffic"],
        values=[n_normal, n_anomaly],
        hole=0.55,
        marker=dict(colors=[NORMAL_COLOR, ATTACK_COLOR],
                    line=dict(color=BG_COLOR, width=2)),
        textinfo="percent+label",
        textfont=dict(size=13),
    ))
    total = n_normal + n_anomaly
    fig.add_annotation(text=f"<b>{total:,}</b><br>records",
                        x=0.5, y=0.5, showarrow=False,
                        font=dict(size=14, color=TEXT_COLOR))
    return _apply_layout(fig, "Traffic Distribution", showlegend=False)


# ──────────────────────────────────────────────────────────────────────────────
# 2. Attack type breakdown – horizontal bar
# ──────────────────────────────────────────────────────────────────────────────
def attack_distribution_bar(attack_dist: dict) -> go.Figure:
    labels = list(attack_dist.keys())
    values = list(attack_dist.values())
    colors = [ATTACK_PALETTE.get(l, ACCENT_COLOR) for l in labels]

    fig = go.Figure(go.Bar(
        x=values, y=[l.upper() for l in labels],
        orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        text=values, textposition="outside",
        textfont=dict(size=12),
    ))
    return _apply_layout(fig, "Attack Type Distribution",
                          xaxis_title="Count", yaxis_title="")


# ──────────────────────────────────────────────────────────────────────────────
# 3. Anomaly score histogram
# ──────────────────────────────────────────────────────────────────────────────
def anomaly_score_histogram(scores: np.ndarray,
                             y_true: np.ndarray) -> go.Figure:
    normal_scores  = scores[y_true == 0]
    anomaly_scores = scores[y_true == 1]

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=normal_scores, name="Normal",
        marker_color=NORMAL_COLOR, opacity=0.75,
        nbinsx=60, histnorm="probability density",
    ))
    fig.add_trace(go.Histogram(
        x=anomaly_scores, name="Anomaly",
        marker_color=ATTACK_COLOR, opacity=0.75,
        nbinsx=60, histnorm="probability density",
    ))
    fig.update_layout(barmode="overlay")
    return _apply_layout(fig, "Anomaly Score Distribution",
                          xaxis_title="Normalised Anomaly Score",
                          yaxis_title="Density",
                          legend=dict(bgcolor="rgba(0,0,0,0)"))


# ──────────────────────────────────────────────────────────────────────────────
# 4. Confusion matrix heatmap
# ──────────────────────────────────────────────────────────────────────────────
def confusion_matrix_heatmap(cm_df: pd.DataFrame) -> go.Figure:
    z    = cm_df.values
    text = [[f"<b>{v:,}</b>" for v in row] for row in z]

    fig = go.Figure(go.Heatmap(
        z=z, x=cm_df.columns.tolist(), y=cm_df.index.tolist(),
        text=text, texttemplate="%{text}",
        colorscale=[[0, "#1A237E"], [0.5, "#1976D2"], [1.0, ATTACK_COLOR]],
        showscale=True,
        textfont=dict(size=18),
    ))
    return _apply_layout(fig, "Confusion Matrix",
                          xaxis_title="Predicted", yaxis_title="Actual")


# ──────────────────────────────────────────────────────────────────────────────
# 5. Feature importance bar chart
# ──────────────────────────────────────────────────────────────────────────────
def feature_importance_bar(importance_df: pd.DataFrame,
                            top_n: int = 15) -> go.Figure:
    df = importance_df.head(top_n).sort_values("importance")
    fig = go.Figure(go.Bar(
        x=df["importance"],
        y=df["feature"],
        orientation="h",
        marker=dict(
            color=df["importance"],
            colorscale="Viridis",
            line=dict(width=0),
        ),
        text=[f"{v:.4f}" for v in df["importance"]],
        textposition="outside",
    ))
    return _apply_layout(fig, f"Top {top_n} Feature Importances (Mutual Info)",
                          xaxis_title="MI Score", yaxis_title="")


# ──────────────────────────────────────────────────────────────────────────────
# 6. Scatter: first two PCA components
# ──────────────────────────────────────────────────────────────────────────────
def pca_scatter(X_pca: np.ndarray,
                y_true: np.ndarray,
                attack_types: np.ndarray | None = None,
                n_sample: int = 2000) -> go.Figure:
    idx = np.random.choice(len(X_pca), min(n_sample, len(X_pca)), replace=False)
    X_s = X_pca[idx]
    y_s = y_true[idx]
    labels = attack_types[idx] if attack_types is not None else \
             np.where(y_s == 0, "normal", "attack")

    df_plot = pd.DataFrame({
        "PC1": X_s[:, 0],
        "PC2": X_s[:, 1] if X_s.shape[1] > 1 else np.zeros(len(X_s)),
        "label": labels,
    })

    color_map = {k: v for k, v in ATTACK_PALETTE.items()}
    color_map["attack"] = ATTACK_COLOR

    fig = px.scatter(
        df_plot, x="PC1", y="PC2", color="label",
        color_discrete_map=color_map,
        opacity=0.65, size_max=6,
    )
    fig.update_traces(marker=dict(size=5))
    return _apply_layout(fig, "PCA — Traffic Clusters (PC1 vs PC2)",
                          xaxis_title="Principal Component 1",
                          yaxis_title="Principal Component 2",
                          legend=dict(bgcolor="rgba(0,0,0,0)"))


# ──────────────────────────────────────────────────────────────────────────────
# 7. ROC curve
# ──────────────────────────────────────────────────────────────────────────────
def roc_curve_plot(roc_data: dict) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=roc_data["fpr"], y=roc_data["tpr"],
        mode="lines", name=f"ROC (AUC={roc_data['auc']:.3f})",
        line=dict(color=ACCENT_COLOR, width=2.5),
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines", name="Random",
        line=dict(color="#666", dash="dash", width=1.5),
    ))
    return _apply_layout(fig, "ROC Curve",
                          xaxis_title="False Positive Rate",
                          yaxis_title="True Positive Rate",
                          legend=dict(bgcolor="rgba(0,0,0,0)"))


# ──────────────────────────────────────────────────────────────────────────────
# 8. Precision-Recall curve
# ──────────────────────────────────────────────────────────────────────────────
def pr_curve_plot(pr_data: dict) -> go.Figure:
    fig = go.Figure(go.Scatter(
        x=pr_data["recall"], y=pr_data["precision"],
        mode="lines", fill="tozeroy",
        line=dict(color=NORMAL_COLOR, width=2.5),
        fillcolor="rgba(33,150,243,0.15)",
        name=f"AP={pr_data['average_precision']:.3f}",
    ))
    return _apply_layout(fig, "Precision-Recall Curve",
                          xaxis_title="Recall",
                          yaxis_title="Precision",
                          legend=dict(bgcolor="rgba(0,0,0,0)"))


# ──────────────────────────────────────────────────────────────────────────────
# 9. Per-attack detection rates
# ──────────────────────────────────────────────────────────────────────────────
def detection_rate_by_attack(breakdown_df: pd.DataFrame) -> go.Figure:
    df = breakdown_df.copy()
    colors = [NORMAL_COLOR if r >= 80 else ACCENT_COLOR if r >= 50 else ATTACK_COLOR
              for r in df["Detection Rate %"]]

    fig = go.Figure(go.Bar(
        x=df["Attack Type"],
        y=df["Detection Rate %"],
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{r}%" for r in df["Detection Rate %"]],
        textposition="outside",
    ))
    fig.add_hline(y=80, line_dash="dash", line_color="#666",
                   annotation_text="80 % target", annotation_position="bottom right")
    return _apply_layout(fig, "Detection Rate by Attack Type",
                          xaxis_title="Attack Type",
                          yaxis_title="Detection Rate (%)",
                          yaxis_range=[0, 110])


# ──────────────────────────────────────────────────────────────────────────────
# 10. PCA explained variance
# ──────────────────────────────────────────────────────────────────────────────
def pca_variance_plot(cumulative_variance: list[float]) -> go.Figure:
    x = list(range(1, len(cumulative_variance) + 1))
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=x,
        y=[c - (cumulative_variance[i-1] if i > 0 else 0)
           for i, c in enumerate(cumulative_variance)],
        name="Per-component",
        marker_color=NORMAL_COLOR, opacity=0.65,
    ))
    fig.add_trace(go.Scatter(
        x=x, y=cumulative_variance,
        mode="lines+markers", name="Cumulative",
        line=dict(color=ACCENT_COLOR, width=2.5),
        marker=dict(size=6),
    ))
    fig.add_hline(y=0.95, line_dash="dash", line_color="#888",
                   annotation_text="95 % variance")
    return _apply_layout(fig, "PCA Explained Variance",
                          xaxis_title="Component",
                          yaxis_title="Variance Explained",
                          legend=dict(bgcolor="rgba(0,0,0,0)"))


# ──────────────────────────────────────────────────────────────────────────────
# 11. Protocol distribution
# ──────────────────────────────────────────────────────────────────────────────
def protocol_bar(protocol_dist: dict) -> go.Figure:
    fig = go.Figure(go.Bar(
        x=list(protocol_dist.keys()),
        y=list(protocol_dist.values()),
        marker=dict(color=[NORMAL_COLOR, ACCENT_COLOR, ATTACK_COLOR,
                           "#9C27B0", "#4CAF50"][:len(protocol_dist)],
                    line=dict(width=0)),
        text=list(protocol_dist.values()),
        textposition="outside",
    ))
    return _apply_layout(fig, "Protocol Type Distribution",
                          xaxis_title="Protocol", yaxis_title="Count")


# ──────────────────────────────────────────────────────────────────────────────
# 12. Time-series style anomaly timeline (simulated)
# ──────────────────────────────────────────────────────────────────────────────
def anomaly_timeline(scores_norm: np.ndarray,
                     y_pred: np.ndarray,
                     n_show: int = 400) -> go.Figure:
    idx = np.linspace(0, len(scores_norm) - 1, min(n_show, len(scores_norm)),
                      dtype=int)
    x   = np.arange(len(idx))
    sc  = scores_norm[idx]
    pr  = y_pred[idx]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=sc, mode="lines",
        line=dict(color="#444", width=1),
        name="Anomaly Score",
    ))
    # Overlay detected anomalies as red dots
    mask = pr == 1
    fig.add_trace(go.Scatter(
        x=x[mask], y=sc[mask],
        mode="markers", name="Detected Anomaly",
        marker=dict(color=ATTACK_COLOR, size=6, symbol="x"),
    ))
    fig.add_hline(y=0.5, line_dash="dot", line_color=ACCENT_COLOR,
                   annotation_text="Decision Threshold ≈ 0.5")
    return _apply_layout(fig, "Anomaly Score Timeline (Sample)",
                          xaxis_title="Record Index",
                          yaxis_title="Normalised Anomaly Score",
                          legend=dict(bgcolor="rgba(0,0,0,0)"))
