# 🛡️ Zero-Day Anomaly Detection IDS
### Unsupervised Intrusion Detection using Isolation Forest

> **Domain**: Cybersecurity / Artificial Intelligence  
> **Technology**: Python · Scikit-learn · Streamlit · Plotly  
> **Algorithm**: Isolation Forest (Unsupervised ML)  
> **Academic Year**: 2025–2026

---

## 📋 Table of Contents
1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Project Structure](#project-structure)
4. [Quick Start](#quick-start)
5. [Pipeline Stages](#pipeline-stages)
6. [Algorithm: Isolation Forest](#algorithm)
7. [Evaluation Results](#evaluation-results)
8. [Dashboard Screenshots](#dashboard)
9. [Dataset Sources](#datasets)
10. [Future Enhancements](#future-enhancements)

---

## 🔍 Project Overview

Traditional signature-based IDS solutions are blind to **zero-day attacks** — novel exploits not yet catalogued in signature databases. This project builds an **Anomaly-Based IDS (AB-IDS)** that:

- Learns a statistical model of **normal network behaviour** without labelled attack data
- Flags **any significant deviation** from that baseline as a potential intrusion
- Detects **DoS, Probe, R2L, U2R**, and previously unseen attack patterns
- Provides a rich **Streamlit dashboard** for monitoring and analysis

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🤖 **Unsupervised ML** | No labelled attack data needed — fully self-learning |
| 🌲 **Isolation Forest** | O(n log n) scalable, ensemble-based anomaly detection |
| 📉 **PCA Reduction** | Retains 95%+ variance, reduces noise and computation |
| 📊 **Rich Dashboard** | 12+ interactive Plotly charts across 6 dashboard tabs |
| 🎯 **Zero-Day Ready** | Detects novel attacks without signature updates |
| ⚡ **Fast Retraining** | Full pipeline runs in < 30 seconds on 8,000 records |
| 📂 **CSV Support** | Drop in your own NSL-KDD CSV for real evaluation |

---

## 📁 Project Structure

```
zero_day_ids/
├── app.py                    # Streamlit dashboard (main entry point)
├── pipeline.py               # End-to-end training pipeline
├── requirements.txt          # Python dependencies
├── README.md
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py        # Dataset loading, synthetic generation, preprocessing
│   ├── feature_engineering.py# Feature selection, PCA, mutual information
│   ├── model.py              # Isolation Forest: train, predict, save/load
│   ├── evaluation.py         # Metrics, confusion matrix, ROC/PR curves, reports
│   └── visualizations.py     # All 12 Plotly chart builders
│
├── models/
│   ├── isolation_forest.pkl  # Saved model (auto-generated)
│   └── pipeline_artifacts.pkl# All pipeline outputs (auto-generated)
│
└── data/                     # Place NSL-KDD / CICIDS2017 CSV files here
```

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Pipeline (trains model + saves artifacts)
```bash
python pipeline.py
# Optional flags:
python pipeline.py --contamination 0.15 --n-estimators 300 --n-samples 10000
# With real NSL-KDD dataset:
python pipeline.py --data data/KDDTrain+.txt
```

### 3. Launch the Dashboard
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🔧 Pipeline Stages

```
Stage 1: Dataset Loading
  └─ Synthetic NSL-KDD-like data OR real CSV
  └─ 8,000 records; 22% attack ratio (4 attack types)

Stage 2: Preprocessing
  └─ Label encoding for categoricals (protocol, service, flag)
  └─ 99.5th percentile clipping for outlier control
  └─ Min-Max normalization across all 38 features
  └─ 70/30 stratified train/test split

Stage 3: Feature Engineering
  └─ Domain-knowledge driven top-20 feature selection
  └─ Mutual Information scoring for importance ranking

Stage 4: PCA
  └─ Retains 95% variance threshold
  └─ Reduces 20 → ~12 principal components
  └─ Dramatically reduces iTree path computation

Stage 5: Model Training
  └─ IsolationForest(n_estimators=200, contamination=0.20)
  └─ Fits on FULL training set (normal + mixed)
  └─ Model pickled to models/isolation_forest.pkl

Stage 6: Anomaly Detection
  └─ score_samples() → raw decision score
  └─ Normalised to [0,1] — higher = more anomalous
  └─ Binary prediction: score > threshold → ANOMALY

Stage 7: Evaluation
  └─ Accuracy, Precision, Recall, F1, ROC-AUC
  └─ Confusion Matrix, per-attack breakdown
  └─ ROC curve, Precision-Recall curve
```

---

## 🌲 Algorithm

### Isolation Forest (Liu, Ting & Zhou, 2008)

**Key Insight**: Anomalies are *few* and *different*, so they are isolated
in fewer random partitioning steps than normal points.

```
For each iTree:
  1. Randomly select feature f and split value v ∈ [min(f), max(f)]
  2. Recursively partition until each point is isolated
  3. Record path length h(x) for each point x

Anomaly Score:
  s(x, n) = 2^( -E[h(x)] / c(n) )
  
  s → 1.0 : highly anomalous (short path = easily isolated)
  s → 0.5 : normal (longer path = harder to isolate)
  s → 0.0 : definitely normal
```

**Advantages over density/distance methods:**
- ✅ Linear time complexity O(n log n)
- ✅ No distance/density computations
- ✅ Naturally handles high-dimensional data
- ✅ No assumption about data distribution
- ✅ Works without ANY labelled data

---

## 📊 Evaluation Results

| Metric | Value |
|--------|-------|
| **Accuracy** | 97.83 % |
| **Precision** | 100.00 % |
| **Recall** | 90.15 % |
| **F1-Score** | 94.82 % |
| **ROC-AUC** | 100.00 % |
| **False Positive Rate** | 0.00 % |
| **False Negative Rate** | 9.85 % |
| **Specificity** | 100.00 % |

> Results on 2,400 test records (8,000 total, 70/30 split)

---

## 🖥️ Dashboard Tabs

| Tab | Contents |
|-----|----------|
| 📊 **Overview** | Traffic donut, attack distribution, protocol bar, architecture flow |
| 🧪 **Model Performance** | Metrics grid, confusion matrix, ROC curve, PR curve, per-attack breakdown |
| 🔍 **Anomaly Analysis** | Score histogram, timeline chart, PCA scatter, top anomaly table |
| 📈 **Feature Insights** | Feature importance bar, PCA variance explained, importance table |
| 🗂️ **Data Explorer** | Filterable raw dataset, descriptive statistics |
| 📋 **Report** | Full text evaluation report, summary table, references |

---

## 📂 Datasets

| Dataset | Source | Attack Types |
|---------|--------|-------------|
| **NSL-KDD** | [UNB](https://www.unb.ca/cic/datasets/nsl.html) | DoS, Probe, R2L, U2R |
| **CICIDS 2017** | [CIC](https://www.unb.ca/cic/datasets/ids-2017.html) | Brute Force, DDoS, Web Attacks, Botnet |

Download `KDDTrain+.txt` and `KDDTest+.txt` from the NSL-KDD link above,
place in `data/`, then run:
```bash
python pipeline.py --data data/KDDTrain+.txt
```

---

## 🚀 Future Enhancements

- [ ] **Live packet capture** via Scapy / PyShark
- [ ] **Autoencoder / LSTM** deep learning anomaly detection
- [ ] **Hybrid IDS**: combine Isolation Forest + supervised classifier
- [ ] **Attack-type classification** (multi-class, not just binary)
- [ ] **Docker containerisation** for cloud deployment
- [ ] **SIEM integration** (Splunk / IBM QRadar API)
- [ ] **Federated learning** for privacy-preserving distributed training

---

## 📚 References

1. Liu, F.T., Ting, K.M., & Zhou, Z.H. (2008). *Isolation Forest.* IEEE ICDM.
2. Tavallaee et al. (2009). *A Detailed Analysis of the KDD CUP 99 Data Set.*
3. Sharafaldin et al. (2018). *Toward Generating a New Intrusion Detection Dataset.* CICIDS2017.
4. Scikit-learn Documentation — [IsolationForest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html)

---

*Developed as part of Final Year Project — Academic Year 2025–2026*
