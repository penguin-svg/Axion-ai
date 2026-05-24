"""
data_loader.py
==============
Handles dataset loading, synthetic data generation (NSL-KDD-like),
and all preprocessing steps: cleaning, encoding, normalization, splitting.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import os


# ──────────────────────────────────────────────────────────────────────────────
# NSL-KDD column definitions
# ──────────────────────────────────────────────────────────────────────────────
CATEGORICAL_FEATURES = ["protocol_type", "service", "flag"]

NUMERICAL_FEATURES = [
    "duration", "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent",
    "hot", "num_failed_logins", "logged_in", "num_compromised", "root_shell",
    "su_attempted", "num_root", "num_file_creations", "num_shells",
    "num_access_files", "count", "srv_count", "serror_rate", "srv_serror_rate",
    "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
]

ALL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES


# ──────────────────────────────────────────────────────────────────────────────
# Synthetic data generation
# ──────────────────────────────────────────────────────────────────────────────
def generate_synthetic_data(n_samples: int = 8000, anomaly_ratio: float = 0.22,
                             random_state: int = 42) -> pd.DataFrame:
    """
    Generate a synthetic NSL-KDD-like network traffic dataset.

    Normal records model benign TCP/UDP/ICMP sessions.
    Attack records model DoS, Probe, R2L, and U2R patterns with distinct
    statistical signatures so the Isolation Forest can differentiate them.
    """
    rng = np.random.default_rng(random_state)

    n_anomaly = int(n_samples * anomaly_ratio)
    n_normal  = n_samples - n_anomaly

    def _normal_block(n):
        return {
            "duration":                    rng.exponential(12, n),
            "protocol_type":               rng.choice(["tcp","udp","icmp"], n, p=[0.60,0.30,0.10]),
            "service":                     rng.choice(["http","ftp","smtp","ssh","dns"], n,
                                                       p=[0.45,0.15,0.15,0.15,0.10]),
            "flag":                        rng.choice(["SF","S0","REJ","RSTO"], n, p=[0.82,0.08,0.06,0.04]),
            "src_bytes":                   rng.lognormal(8.0, 1.8, n),
            "dst_bytes":                   rng.lognormal(7.2, 1.8, n),
            "land":                        np.zeros(n),
            "wrong_fragment":              rng.choice([0,1,2], n, p=[0.95,0.03,0.02]),
            "urgent":                      np.zeros(n),
            "hot":                         rng.integers(0, 10, n),
            "num_failed_logins":           np.zeros(n),
            "logged_in":                   np.ones(n),
            "num_compromised":             np.zeros(n),
            "root_shell":                  np.zeros(n),
            "su_attempted":                np.zeros(n),
            "num_root":                    np.zeros(n),
            "num_file_creations":          rng.integers(0, 3, n),
            "num_shells":                  np.zeros(n),
            "num_access_files":            rng.integers(0, 2, n),
            "count":                       rng.integers(1, 55, n).astype(float),
            "srv_count":                   rng.integers(1, 35, n).astype(float),
            "serror_rate":                 rng.uniform(0.00, 0.08, n),
            "srv_serror_rate":             rng.uniform(0.00, 0.08, n),
            "rerror_rate":                 rng.uniform(0.00, 0.08, n),
            "srv_rerror_rate":             rng.uniform(0.00, 0.08, n),
            "same_srv_rate":               rng.uniform(0.70, 1.00, n),
            "diff_srv_rate":               rng.uniform(0.00, 0.25, n),
            "srv_diff_host_rate":          rng.uniform(0.00, 0.15, n),
            "dst_host_count":              rng.integers(1, 255, n).astype(float),
            "dst_host_srv_count":          rng.integers(1, 255, n).astype(float),
            "dst_host_same_srv_rate":      rng.uniform(0.55, 1.00, n),
            "dst_host_diff_srv_rate":      rng.uniform(0.00, 0.25, n),
            "dst_host_same_src_port_rate": rng.uniform(0.00, 0.45, n),
            "dst_host_srv_diff_host_rate": rng.uniform(0.00, 0.18, n),
            "dst_host_serror_rate":        rng.uniform(0.00, 0.08, n),
            "dst_host_srv_serror_rate":    rng.uniform(0.00, 0.08, n),
            "dst_host_rerror_rate":        rng.uniform(0.00, 0.08, n),
            "dst_host_srv_rerror_rate":    rng.uniform(0.00, 0.08, n),
            "label":                       ["normal"] * n,
        }

    attack_labels = rng.choice(["dos","probe","r2l","u2r"], n_anomaly,
                                p=[0.55, 0.25, 0.12, 0.08])

    def _attack_block(n, labels):
        dos_mask   = labels == "dos"
        probe_mask = labels == "probe"
        r2l_mask   = labels == "r2l"
        u2r_mask   = labels == "u2r"

        src_bytes = np.where(
            dos_mask,   rng.integers(0, 10, n).astype(float),
            np.where(
            probe_mask, rng.lognormal(4, 2, n),
            np.where(
            r2l_mask,   rng.lognormal(7, 3, n),
                        rng.lognormal(6, 2, n)
            )))

        count = np.where(
            dos_mask,   rng.integers(200, 512, n).astype(float),
            np.where(
            probe_mask, rng.integers(1, 50, n).astype(float),
                        rng.integers(1, 30, n).astype(float)
            ))

        return {
            "duration":                    np.where(dos_mask, 0.0, rng.exponential(40, n)),
            "protocol_type":               rng.choice(["tcp","udp","icmp"], n, p=[0.38,0.22,0.40]),
            "service":                     rng.choice(["http","ftp","smtp","ssh","dns","private"], n),
            "flag":                        rng.choice(["S0","REJ","SF","RSTO","SH"], n,
                                                       p=[0.38,0.30,0.12,0.12,0.08]),
            "src_bytes":                   src_bytes,
            "dst_bytes":                   np.where(
                                               dos_mask, np.zeros(n),
                                               rng.lognormal(5, 3, n)
                                           ),
            "land":                        rng.choice([0,1], n, p=[0.88,0.12]).astype(float),
            "wrong_fragment":              rng.choice([0,1,2,3], n, p=[0.55,0.22,0.13,0.10]),
            "urgent":                      rng.choice([0,1], n, p=[0.75,0.25]).astype(float),
            "hot":                         rng.integers(0, 35, n),
            "num_failed_logins":           rng.choice([0,1,2,3,4,5], n,
                                                       p=[0.48,0.22,0.12,0.10,0.05,0.03]).astype(float),
            "logged_in":                   rng.choice([0,1], n, p=[0.60,0.40]).astype(float),
            "num_compromised":             np.where(u2r_mask,
                                               rng.integers(1,10,n).astype(float),
                                               rng.choice([0,1,2],n,p=[0.55,0.25,0.20]).astype(float)),
            "root_shell":                  np.where(u2r_mask,
                                               rng.choice([0,1],n,p=[0.30,0.70]).astype(float),
                                               np.zeros(n)),
            "su_attempted":                np.where(u2r_mask,
                                               rng.choice([0,1],n,p=[0.50,0.50]).astype(float),
                                               np.zeros(n)),
            "num_root":                    np.where(u2r_mask,
                                               rng.integers(1,8,n).astype(float),
                                               np.zeros(n)),
            "num_file_creations":          rng.integers(0, 12, n),
            "num_shells":                  np.where(u2r_mask,
                                               rng.choice([1,2],n,p=[0.60,0.40]).astype(float),
                                               rng.choice([0,1],n,p=[0.65,0.35]).astype(float)),
            "num_access_files":            rng.integers(0, 10, n),
            "count":                       count,
            "srv_count":                   rng.integers(1, 512, n).astype(float),
            "serror_rate":                 np.where(dos_mask, rng.uniform(0.80,1.00,n),
                                               rng.uniform(0.30,0.90,n)),
            "srv_serror_rate":             np.where(dos_mask, rng.uniform(0.80,1.00,n),
                                               rng.uniform(0.30,0.90,n)),
            "rerror_rate":                 rng.uniform(0.10, 0.90, n),
            "srv_rerror_rate":             rng.uniform(0.10, 0.90, n),
            "same_srv_rate":               rng.uniform(0.00, 0.45, n),
            "diff_srv_rate":               rng.uniform(0.50, 1.00, n),
            "srv_diff_host_rate":          rng.uniform(0.25, 1.00, n),
            "dst_host_count":              rng.integers(1, 255, n).astype(float),
            "dst_host_srv_count":          rng.integers(1, 255, n).astype(float),
            "dst_host_same_srv_rate":      rng.uniform(0.00, 0.50, n),
            "dst_host_diff_srv_rate":      rng.uniform(0.50, 1.00, n),
            "dst_host_same_src_port_rate": rng.uniform(0.50, 1.00, n),
            "dst_host_srv_diff_host_rate": rng.uniform(0.25, 1.00, n),
            "dst_host_serror_rate":        np.where(dos_mask, rng.uniform(0.80,1.00,n),
                                               rng.uniform(0.20,0.90,n)),
            "dst_host_srv_serror_rate":    np.where(dos_mask, rng.uniform(0.80,1.00,n),
                                               rng.uniform(0.20,0.90,n)),
            "dst_host_rerror_rate":        rng.uniform(0.10, 0.90, n),
            "dst_host_srv_rerror_rate":    rng.uniform(0.10, 0.90, n),
            "label":                       list(labels),
        }

    df_normal = pd.DataFrame(_normal_block(n_normal))
    df_attack = pd.DataFrame(_attack_block(n_anomaly, attack_labels))

    df = pd.concat([df_normal, df_attack], ignore_index=True)
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────
def load_dataset(filepath: str | None = None,
                 n_samples: int = 8000,
                 anomaly_ratio: float = 0.22) -> pd.DataFrame:
    """
    Load dataset from *filepath* (CSV) or generate a synthetic one.
    Accepts NSL-KDD CSV with optional header.
    """
    if filepath and os.path.exists(filepath):
        df = pd.read_csv(filepath)
        # Basic column-count heuristic: NSL-KDD has 42 columns
        if df.shape[1] >= 41:
            col_names = NUMERICAL_FEATURES + CATEGORICAL_FEATURES + ["label", "difficulty"]
            if df.shape[1] == 43:
                df.columns = col_names + ["extra"]
            elif df.shape[1] == 42:
                df.columns = col_names
            # Trim difficulty / extra columns
            df = df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES + ["label"]]
    else:
        df = generate_synthetic_data(n_samples, anomaly_ratio)
    return df


def preprocess(df: pd.DataFrame,
               test_size: float = 0.30,
               random_state: int = 42):
    """
    Full preprocessing pipeline.

    Returns
    -------
    X_train, X_test, y_train, y_test  : arrays / Series
    feature_names                       : list[str]
    scaler                              : fitted MinMaxScaler
    encoders                            : dict[col -> LabelEncoder]
    df_processed                        : fully processed DataFrame
    """
    df = df.copy()

    # ── 1. Drop duplicates & reset index ──
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)

    # ── 2. Binary label ──
    df["is_anomaly"] = (df["label"] != "normal").astype(int)
    df["attack_type"] = df["label"]

    # ── 3. Encode categoricals ──
    encoders = {}
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le

    # ── 4. Clip extreme values (99.5th percentile) ──
    for col in NUMERICAL_FEATURES:
        if col in df.columns:
            cap = df[col].quantile(0.995)
            df[col] = df[col].clip(upper=cap)

    # ── 5. Fill any remaining NaN ──
    df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES] = (
        df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES].fillna(0)
    )

    # ── 6. Feature matrix & labels ──
    feature_cols = [c for c in (NUMERICAL_FEATURES + CATEGORICAL_FEATURES)
                    if c in df.columns]
    X = df[feature_cols].values
    y = df["is_anomaly"].values

    # ── 7. Normalise ──
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # ── 8. Train / test split ──
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size,
        random_state=random_state, stratify=y
    )

    df_processed = df.copy()
    df_processed[feature_cols] = X_scaled

    return X_train, X_test, y_train, y_test, feature_cols, scaler, encoders, df_processed


def get_statistics(df: pd.DataFrame) -> dict:
    """Return summary statistics for the dashboard overview."""
    total    = len(df)
    normal   = int((df["label"] == "normal").sum())
    attacks  = total - normal
    proto    = df["protocol_type"].value_counts().to_dict() if "protocol_type" in df.columns else {}
    atk_dist = df.loc[df["label"] != "normal", "label"].value_counts().to_dict()

    return {
        "total":      total,
        "normal":     normal,
        "attacks":    attacks,
        "anomaly_pct": round(attacks / total * 100, 2),
        "protocol_dist": proto,
        "attack_dist":   atk_dist,
        "n_features":    len(NUMERICAL_FEATURES + CATEGORICAL_FEATURES),
    }
