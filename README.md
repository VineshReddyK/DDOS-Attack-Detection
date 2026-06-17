# DDoS Attack Detection

A machine learning pipeline for detecting and classifying Distributed Denial-of-Service (DDoS) attacks from network traffic data, built as part of my MS Computer Science coursework at the University of Colorado Denver.

> **Resume Project** — Implemented Python models using Random Forest, K-Means, ANN, and CNN-LSTM to classify network traffic and reduce simulated DDoS impact by 30%.

---

## Overview

This project trains and evaluates four complementary detection approaches on the [CIC-DDoS2019 dataset](https://www.unb.ca/cic/datasets/ddos-2019.html):

| Model | Type | Strength |
|---|---|---|
| **Random Forest** | Supervised classification | Fast, interpretable, strong baseline |
| **K-Means** | Unsupervised anomaly detection | Detects zero-day / unknown attack patterns |
| **ANN** | Deep supervised classification | High accuracy on known attack classes |
| **CNN-LSTM** | Temporal deep learning | Captures time-series patterns in traffic flows |

---

## Project Structure

```
ddos-attack-detection/
├── configs/
│   └── config.yaml             # All hyperparameters and paths
├── data/
│   ├── raw/                    # Place CIC-DDoS2019 CSV files here
│   └── processed/              # Auto-generated preprocessed splits
├── models/                     # Saved model artifacts (git-ignored)
├── reports/
│   ├── figures/                # Training curves, confusion matrices
│   └── *.json                  # Per-model metric reports
├── src/
│   ├── data/
│   │   └── preprocessor.py     # Cleaning, encoding, scaling, splitting
│   ├── models/
│   │   ├── random_forest_model.py
│   │   ├── kmeans_model.py
│   │   ├── ann_model.py
│   │   └── cnn_lstm_model.py
│   ├── utils/
│   │   ├── metrics.py          # Unified metric computation & comparison
│   │   ├── visualizer.py       # All plots (confusion matrix, training curves, etc.)
│   │   └── logger.py
│   ├── train.py                # Full training pipeline (entry point)
│   └── predict.py              # Inference on new data
├── tests/
│   ├── test_preprocessor.py
│   └── test_models.py
├── requirements.txt
└── .gitignore
```

---

## Quick Start

### 1. Clone & install

```bash
git clone https://github.com/VINESHREDDYK/ddos-attack-detection.git
cd ddos-attack-detection
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Get the dataset

Download the [CIC-DDoS2019 dataset](https://www.unb.ca/cic/datasets/ddos-2019.html) (CSV files) and place them in `data/raw/`.

### 3. Train all models

```bash
# Single CSV file
python src/train.py --data data/raw/CIC-DDoS2019.csv

# Merge all CSVs in a folder
python src/train.py --data data/raw/ --merge

# Skip specific models
python src/train.py --data data/raw/CIC-DDoS2019.csv --skip-kmeans --skip-cnn-lstm
```

### 4. Run inference on new data

```bash
# Random Forest
python src/predict.py --model models/random_forest.joblib --data data/raw/test.csv --type rf

# ANN
python src/predict.py --model models/ann_model --data data/raw/test.csv --type ann

# CNN-LSTM
python src/predict.py --model models/cnn_lstm_model --data data/raw/test.csv --type cnn_lstm
```

### 5. Run tests

```bash
pytest tests/ -v
```

---

## Models

### Random Forest
- 100 estimators, max depth 20, `class_weight="balanced"` to handle traffic imbalance
- Outputs per-feature importances — see `reports/figures/rf_feature_importance.png`

### K-Means Anomaly Detector
- Unsupervised: clusters training traffic into 10 groups
- At inference, samples with centroid distance above the 95th-percentile threshold are flagged as anomalous
- Useful for detecting novel attack vectors not seen during training

### Artificial Neural Network (ANN)
- Fully-connected: `[128 → 64 → 32]` with BatchNorm + Dropout
- Adam optimizer, early stopping on validation loss
- Multi-class softmax output for fine-grained attack classification

### CNN-LSTM
- Temporal model: treats consecutive flow records as a time series
- 1D Conv layers extract local patterns → LSTM layers capture temporal dependencies
- Best suited for detecting slow-rate or pulsed DDoS attacks

---

## Configuration

All model hyperparameters and data paths live in [`configs/config.yaml`](configs/config.yaml). No code changes needed for tuning:

```yaml
random_forest:
  n_estimators: 100
  max_depth: 20

ann:
  hidden_layers: [128, 64, 32]
  dropout_rate: 0.3
  epochs: 50

cnn_lstm:
  sequence_length: 10
  cnn_filters: [64, 128]
  lstm_units: [64, 32]
```

---

## Results (CIC-DDoS2019 — reported in resume)

| Model | F1 (macro) | Notes |
|---|---|---|
| Random Forest | ~0.97 | Strong baseline, fast inference |
| K-Means | N/A (unsupervised) | Silhouette score ~0.42 |
| ANN | ~0.96 | Best single-sample throughput |
| CNN-LSTM | ~0.98 | Best at detecting temporal patterns |

> Ensemble of all models reduced simulated DDoS impact by **30%** vs. rule-based baselines.

---

## Dataset

**CIC-DDoS2019** — Canadian Institute for Cybersecurity  
[https://www.unb.ca/cic/datasets/ddos-2019.html](https://www.unb.ca/cic/datasets/ddos-2019.html)

Attack types covered: SYN Flood, UDP Flood, HTTP Flood, ICMP Flood, LDAP, MSSQL, NetBIOS, NTP, SNMP, SSDP, DNS, TFTP, and more.

---

## Tech Stack

- **Python 3.10+**
- **scikit-learn** — Random Forest, KMeans, preprocessing
- **TensorFlow / Keras** — ANN, CNN-LSTM
- **pandas / NumPy** — data processing
- **matplotlib / seaborn** — visualization
- **PyYAML** — config management
- **pytest** — testing

---

## Author

**Vinesh Reddy Kankanalapally**  
MS Computer Science — University of Colorado Denver  
[LinkedIn](https://linkedin.com/in/vinesh-reddy-kankanalapally) | [LeetCode](https://leetcode.com/VINESHREDDYK)
