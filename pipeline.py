"""
pipeline.py
===========
End-to-end IDS pipeline — supports both NSL-KDD real files and synthetic data.

Usage (command line):
    python pipeline.py                                      # synthetic 8 000 records
    python pipeline.py --train data/KDDTrain+.txt           # real NSL-KDD (auto test split)
    python pipeline.py --train data/KDDTrain+.txt \
                       --test  data/KDDTest+.txt            # real NSL-KDD (separate test file)
    python pipeline.py --train data/KDDTrain+20Percent.txt \
                       --test  data/KDDTest+.txt

Can also be called from app.py via run_pipeline().
"""

import os, sys, pickle, argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from sklearn.preprocessing   import LabelEncoder, MinMaxScaler
from sklearn.decomposition   import PCA
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble        import IsolationForest
from sklearn.metrics         import (accuracy_score, precision_score, recall_score,
                                     f1_score, confusion_matrix, roc_auc_score,
                                     roc_curve, precision_recall_curve,
                                     average_precision_score)

from src.evaluation import generate_report

MODELS_DIR    = "models"
ARTIFACT_PATH = os.path.join(MODELS_DIR, "pipeline_artifacts.pkl")

# ── NSL-KDD column schema ─────────────────────────────────────────────────────
COLUMNS = [
    'duration','protocol_type','service','flag','src_bytes','dst_bytes',
    'land','wrong_fragment','urgent','hot','num_failed_logins','logged_in',
    'num_compromised','root_shell','su_attempted','num_root',
    'num_file_creations','num_shells','num_access_files','num_outbound_cmds',
    'is_host_login','is_guest_login','count','srv_count','serror_rate',
    'srv_serror_rate','rerror_rate','srv_rerror_rate','same_srv_rate',
    'diff_srv_rate','srv_diff_host_rate','dst_host_count','dst_host_srv_count',
    'dst_host_same_srv_rate','dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate','dst_host_srv_diff_host_rate',
    'dst_host_serror_rate','dst_host_srv_serror_rate',
    'dst_host_rerror_rate','dst_host_srv_rerror_rate',
    'label','difficulty'
]
CATEGORICAL_COLS = ['protocol_type', 'service', 'flag']
NUMERIC_COLS     = [c for c in COLUMNS
                    if c not in CATEGORICAL_COLS + ['label', 'difficulty']]
FEATURE_COLS     = NUMERIC_COLS + CATEGORICAL_COLS

ATTACK_MAP = {
    'normal':'Normal',
    'back':'DoS','land':'DoS','neptune':'DoS','pod':'DoS','smurf':'DoS',
    'teardrop':'DoS','apache2':'DoS','udpstorm':'DoS','processtable':'DoS',
    'mailbomb':'DoS','worm':'DoS',
    'ipsweep':'Probe','nmap':'Probe','portsweep':'Probe','satan':'Probe',
    'mscan':'Probe','saint':'Probe',
    'ftp_write':'R2L','guess_passwd':'R2L','imap':'R2L','multihop':'R2L',
    'phf':'R2L','spy':'R2L','warezclient':'R2L','warezmaster':'R2L',
    'snmpgetattack':'R2L','named':'R2L','xlock':'R2L','xsnoop':'R2L',
    'sendmail':'R2L','httptunnel':'R2L','snmpguess':'R2L',
    'buffer_overflow':'U2R','loadmodule':'U2R','perl':'U2R','rootkit':'U2R',
    'ps':'U2R','sqlattack':'U2R','xterm':'U2R',
}


# ── File loaders ──────────────────────────────────────────────────────────────
def _load_kdd_file(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, header=None, names=COLUMNS)
    df['label']      = df['label'].str.strip().str.lower()
    df['difficulty'] = pd.to_numeric(df['difficulty'], errors='coerce')
    df['category']   = df['label'].map(ATTACK_MAP).fillna('Unknown')
    df['is_anomaly'] = (df['label'] != 'normal').astype(int)
    return df


def _load_synthetic(n_samples: int, anomaly_ratio: float = 0.22) -> pd.DataFrame:
    """Minimal synthetic fallback — only used when no real files are given."""
    from src.data_loader import generate_synthetic_data, get_statistics
    return generate_synthetic_data(n_samples, anomaly_ratio)


# ── Statistics helper ─────────────────────────────────────────────────────────
def _get_stats(df: pd.DataFrame) -> dict:
    total   = len(df)
    normal  = int((df['label'] == 'normal').sum()) if 'label' in df.columns else 0
    attacks = total - normal
    proto   = {}
    if 'protocol_type' in df.columns:
        raw = df['protocol_type']
        if raw.dtype == object:
            proto = raw.value_counts().to_dict()
    atk_dist = {}
    if 'category' in df.columns:
        atk_dist = (df[df['label'] != 'normal']['category'].value_counts().to_dict()
                    if 'label' in df.columns else {})
    return {
        'total': total, 'normal': normal, 'attacks': attacks,
        'anomaly_pct': round(attacks / max(total,1) * 100, 2),
        'protocol_dist': proto, 'attack_dist': atk_dist,
        'n_features': len(FEATURE_COLS),
    }


# ── Main pipeline ─────────────────────────────────────────────────────────────
def run_pipeline(
    train_path:    str | None = None,
    test_path:     str | None = None,
    contamination: float      = None,   # None → auto from training labels
    n_estimators:  int        = 200,
    n_samples:     int        = 8000,   # only used when no real files given
) -> dict:

    os.makedirs(MODELS_DIR, exist_ok=True)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    print("[1/7] Loading dataset …")
    separate_test = False

    if train_path and os.path.exists(train_path):
        df_train = _load_kdd_file(train_path)
        print(f"      Train file : {train_path}  ({len(df_train):,} records)")

        if test_path and os.path.exists(test_path):
            df_test       = _load_kdd_file(test_path)
            separate_test = True
            print(f"      Test  file : {test_path}  ({len(df_test):,} records)")
        else:
            # 70 / 30 split of training file
            df_train, df_test = train_test_split(
                df_train, test_size=0.30, random_state=42,
                stratify=df_train['is_anomaly']
            )
            df_train = df_train.reset_index(drop=True)
            df_test  = df_test.reset_index(drop=True)
            print(f"      No test file — using 70/30 split  "
                  f"(train {len(df_train):,} / test {len(df_test):,})")

        df_all = pd.concat([df_train, df_test], ignore_index=True)
    else:
        print(f"      No real data — generating {n_samples:,} synthetic records …")
        from src.data_loader import generate_synthetic_data
        df_all = generate_synthetic_data(n_samples)
        df_all['category']   = df_all['label'].map(ATTACK_MAP).fillna('Unknown')
        df_all['is_anomaly'] = (df_all['label'] != 'normal').astype(int)
        df_train, df_test = train_test_split(
            df_all, test_size=0.30, random_state=42, stratify=df_all['is_anomaly']
        )
        df_train = df_train.reset_index(drop=True)
        df_test  = df_test.reset_index(drop=True)
        df_all   = pd.concat([df_train, df_test], ignore_index=True)

    stats_train = _get_stats(df_train)
    stats_test  = _get_stats(df_test)
    stats = {**stats_train,
             'test_total':   len(df_test),
             'test_attacks': stats_test['attacks'],
             'test_anomaly_pct': stats_test['anomaly_pct'],
             'separate_test': separate_test}

    # ── 2. Preprocess ─────────────────────────────────────────────────────────
    print("[2/7] Preprocessing …")
    df_tr = df_train.copy()
    df_te = df_test.copy()

    # Encode — fit on combined vocab
    encoders = {}
    for col in CATEGORICAL_COLS:
        if col not in df_tr.columns:
            continue
        le = LabelEncoder()
        combined = pd.concat([df_tr[col], df_te[col]]).astype(str)
        le.fit(combined)
        df_tr[col] = le.transform(df_tr[col].astype(str))
        df_te[col] = le.transform(df_te[col].astype(str))
        encoders[col] = le

    # Clip outliers (train stats)
    caps = {}
    for col in NUMERIC_COLS:
        if col not in df_tr.columns:
            continue
        cap = df_tr[col].quantile(0.995)
        caps[col] = cap
        df_tr[col] = df_tr[col].clip(upper=cap)
        df_te[col] = df_te[col].clip(upper=cap)

    feat_cols = [c for c in FEATURE_COLS if c in df_tr.columns]

    # Normalise
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(df_tr[feat_cols].fillna(0).values)
    X_test  = scaler.transform(df_te[feat_cols].fillna(0).values)
    y_train = df_tr['is_anomaly'].values
    y_test  = df_te['is_anomaly'].values
    attack_types_test = df_te['category'].values if 'category' in df_te.columns \
                        else np.array(['Unknown'] * len(df_te))

    print(f"      X_train {X_train.shape}  |  X_test {X_test.shape}")

    # ── 3. Feature importance (MI) ────────────────────────────────────────────
    print("[3/7] Feature importance (Mutual Information) …")
    mi = mutual_info_classif(X_train, y_train, random_state=42, n_jobs=-1)
    importance_df = pd.DataFrame({'feature': feat_cols, 'importance': mi}) \
                      .sort_values('importance', ascending=False).reset_index(drop=True)
    importance_df['rank'] = importance_df.index + 1
    top_features  = importance_df.head(20)['feature'].tolist()
    top_indices   = [feat_cols.index(f) for f in top_features]
    X_train_sel   = X_train[:, top_indices]
    X_test_sel    = X_test[:, top_indices]

    # ── 4. PCA ────────────────────────────────────────────────────────────────
    print("[4/7] PCA …")
    pca = PCA(n_components=0.95, svd_solver='full', random_state=42)
    X_train_pca = pca.fit_transform(X_train_sel)
    X_test_pca  = pca.transform(X_test_sel)
    evr = pca.explained_variance_ratio_
    pca_summary = {
        'n_components':           pca.n_components_,
        'total_variance':         round(float(evr.sum()) * 100, 2),
        'cumulative_variance':    np.cumsum(evr).tolist(),
        'per_component_variance': evr.tolist(),
    }
    print(f"      Components: {pca.n_components_}  |  Variance: {pca_summary['total_variance']}%")

    # ── 5. Train Isolation Forest ─────────────────────────────────────────────
    print("[5/7] Training Isolation Forest …")
    if contamination is None:
        contamination = float(np.clip(round(y_train.mean(), 2), 0.01, 0.49))
        print(f"      Auto contamination from labels: {contamination:.2f}")

    clf = IsolationForest(n_estimators=n_estimators, contamination=contamination,
                          max_features=1.0, max_samples='auto',
                          bootstrap=False, random_state=42, n_jobs=-1)
    clf.fit(X_train_pca)
    info = {
        'n_estimators':  clf.n_estimators,
        'contamination': clf.contamination,
        'max_features':  clf.max_features,
        'max_samples':   clf.max_samples,
        'bootstrap':     clf.bootstrap,
        'random_state':  clf.random_state,
        'n_features_in': clf.n_features_in_,
    }

    # Save model
    with open(os.path.join(MODELS_DIR, 'isolation_forest.pkl'), 'wb') as fh:
        pickle.dump({'model': clf, 'scaler': scaler, 'pca': pca,
                     'encoders': encoders, 'top_features': top_features,
                     'top_indices': top_indices, 'feat_cols': feat_cols,
                     'caps': caps}, fh)

    # ── 6. Predict & score ────────────────────────────────────────────────────
    print("[6/7] Detecting anomalies …")
    raw_preds   = clf.predict(X_test_pca)
    y_pred      = (raw_preds == -1).astype(int)
    raw_scores  = clf.score_samples(X_test_pca)
    scores_norm = 1.0 - (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min() + 1e-9)

    # ── 7. Evaluate ───────────────────────────────────────────────────────────
    print("[7/7] Evaluating …")
    tn, fp, fn, tp_v = confusion_matrix(y_test, y_pred, labels=[0,1]).ravel()
    acc   = accuracy_score(y_test, y_pred) * 100
    prec  = precision_score(y_test, y_pred, zero_division=0) * 100
    rec   = recall_score(y_test, y_pred, zero_division=0) * 100
    f1    = f1_score(y_test, y_pred, zero_division=0) * 100
    auc   = roc_auc_score(y_test, scores_norm) * 100
    ap    = average_precision_score(y_test, scores_norm) * 100

    metrics = {
        'accuracy': round(acc, 2), 'precision': round(prec, 2),
        'recall': round(rec, 2),   'f1_score': round(f1, 2),
        'roc_auc': round(auc, 2),  'avg_prec': round(ap, 2),
        'true_positives': int(tp_v), 'true_negatives': int(tn),
        'false_positives': int(fp),  'false_negatives': int(fn),
        'false_positive_rate': round(fp/(fp+tn+1e-9)*100, 2),
        'false_negative_rate': round(fn/(fn+tp_v+1e-9)*100, 2),
        'specificity': round(tn/(tn+fp+1e-9)*100, 2),
        'detection_rate': round(tp_v/(tp_v+fn+1e-9)*100, 2),
    }

    # Confusion matrix df
    cm_arr = confusion_matrix(y_test, y_pred, labels=[0,1])
    cm_df  = pd.DataFrame(cm_arr,
                           index=['Actual Normal','Actual Anomaly'],
                           columns=['Predicted Normal','Predicted Anomaly'])

    # Per-category breakdown
    rows = []
    for cat in np.unique(attack_types_test):
        mask  = attack_types_test == cat
        total = mask.sum()
        if cat == 'Normal':
            det = int((y_pred[mask] == 0).sum())
        else:
            det = int((y_pred[mask] == 1).sum())
        rows.append({'Attack Type': cat, 'Total Records': int(total),
                     'Correctly Flagged': det,
                     'Detection Rate %': round(det/max(total,1)*100, 1)})
    breakdown = pd.DataFrame(rows).sort_values('Detection Rate %', ascending=False)

    # ROC / PR data
    fpr_a, tpr_a, _ = roc_curve(y_test, scores_norm)
    pr_a, rc_a,   _ = precision_recall_curve(y_test, scores_norm)
    roc_data = {'fpr': fpr_a.tolist(), 'tpr': tpr_a.tolist(), 'auc': round(auc/100, 4)}
    pr_data  = {'precision': pr_a.tolist(), 'recall': rc_a.tolist(),
                'average_precision': round(ap/100, 4)}

    report_txt = generate_report(metrics, info)
    print("\n" + report_txt)

    artifacts = {
        'df': df_all, 'df_train': df_train, 'df_test': df_test,
        'stats': stats,
        'X_train_pca': X_train_pca, 'X_test_pca': X_test_pca,
        'y_train': y_train, 'y_test': y_test,
        'y_pred': y_pred, 'scores_norm': scores_norm, 'raw_scores': raw_scores,
        'attack_types_test': attack_types_test,
        'feature_names': feat_cols, 'sel_features': top_features,
        'importance_df': importance_df, 'pca_summary': pca_summary,
        'metrics': metrics, 'cm_df': cm_df, 'breakdown': breakdown,
        'roc_data': roc_data, 'pr_data': pr_data,
        'model_info': info, 'report_txt': report_txt,
        'train_path': train_path, 'test_path': test_path,
    }

    with open(ARTIFACT_PATH, 'wb') as fh:
        pickle.dump(artifacts, fh)
    print(f"Artifacts saved → {ARTIFACT_PATH}")
    return artifacts


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Zero-Day IDS Pipeline')
    p.add_argument('--train',         default=None, help='Path to KDDTrain+.txt')
    p.add_argument('--test',          default=None, help='Path to KDDTest+.txt')
    p.add_argument('--contamination', default=None, type=float,
                   help='Contamination ratio (default: auto from labels)')
    p.add_argument('--n-estimators',  default=200,  type=int)
    p.add_argument('--n-samples',     default=8000, type=int)
    args = p.parse_args()

    run_pipeline(
        train_path=args.train, test_path=args.test,
        contamination=args.contamination,
        n_estimators=args.n_estimators,
        n_samples=args.n_samples,
    )
