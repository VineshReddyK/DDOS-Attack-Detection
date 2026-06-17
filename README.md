# DDoS Attack Detection

[![CI](https://github.com/VineshReddyK/DDOS-Attack-Detection/actions/workflows/ci.yml/badge.svg)](https://github.com/VineshReddyK/DDOS-Attack-Detection/actions/workflows/ci.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Docker](https://img.shields.io/badge/docker-ready-blue?logo=docker)](Dockerfile)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi)](src/api/main.py)

A production-ready machine learning pipeline for **real-time DDoS attack detection** from network traffic data, deployable as a REST API service.

> Built as part of my MS Computer Science program at the University of Colorado Denver. Reduced simulated DDoS impact by **30%** vs. rule-based baselines using an ensemble of four ML models.

---

## Features

- **4 complementary ML models** — supervised, unsupervised, deep learning, and temporal
- **FastAPI REST service** — `/predict`, `/predict/batch`, `/health`, `/info` endpoints with Swagger UI
- **Docker + docker-compose** — one-command deploy
- **GitHub Actions CI** — lint, type check, pytest with coverage, Docker build, API smoke test
- **Configurable** — all hyperparameters in `configs/config.yaml`, no code changes to tune
- **Pre-commit hooks** — black, isort, flake8 enforced on every commit

---

## Architecture

```
Network Traffic Data
        │
        ▼
┌─────────────────┐
│  Preprocessor   │  Clean → Encode → Scale → Split
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
Supervised  Unsupervised
    │             │
    ├─ Random Forest    K-Means Anomaly
    ├─ ANN             (zero-day detection)
    └─ CNN-LSTM
    (temporal patterns)
         │
         ▼
   FastAPI REST API
   /api/v1/predict
   /api/v1/predict/batch
```

| Model | Type | Strength |
|---|---|---|
| **Random Forest** | Supervised | Fast, interpretable, top feature importance |
| **K-Means** | Unsupervised | Detects novel / zero-day attack patterns |
| **ANN** | Deep learning | High accuracy on known attack classes |
| **CNN-LSTM** | Temporal deep learning | Captures time-series patterns in traffic flows |

---

## Project Structure

```
ddos-attack-detection/
├── .github/workflows/ci.yml    # GitHub Actions: lint → test → docker → smoke test
├── configs/config.yaml          # All hyperparameters and paths
├── data/
│   ├── raw/                     # Place CIC-DDoS2019 CSV files here
│   └── processed/
├── models/                      # Saved model artifacts (git-ignored)
├── reports/figures/             # Training curves, confusion matrices
├── src/
│   ├── api/
│   │   ├── main.py              # FastAPI app with lifespan model loading
│   │   ├── routes.py            # /predict, /predict/batch, /health, /info
│   │   ├── schemas.py           # Pydantic request/response models
│   │   └── model_registry.py    # Singleton model loader
│   ├── data/preprocessor.py
│   ├── models/
│   │   ├── random_forest_model.py
│   │   ├── kmeans_model.py
│   │   ├── ann_model.py
│   │   └── cnn_lstm_model.py
│   ├── utils/
│   │   ├── metrics.py
│   │   ├── visualizer.py
│   │   └── logger.py
│   ├── train.py                 # Training pipeline entry point
│   └── predict.py               # CLI inference script
├── tests/
│   ├── test_preprocessor.py
│   ├── test_models.py
│   └── test_api.py              # FastAPI integration tests with mocked models
├── Dockerfile
├── docker-compose.yml
├── Makefile                     # make train / serve / test / docker-up
├── pyproject.toml
├── requirements.txt
├── requirements-dev.txt
└── .pre-commit-config.yaml
```

---

## Quick Start

### Option A — Local

```bash
git clone https://github.com/VineshReddyK/DDOS-Attack-Detection.git
cd DDOS-Attack-Detection

python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

make install
```

**Train models:**
```bash
# Single CSV
make train DATA=data/raw/CIC-DDoS2019.csv

# Merge all CSVs in folder
make train-merge
```

**Start API server:**
```bash
make serve
# Open: http://localhost:8000/docs
```

### Option B — Docker

```bash
# Build & start API
make docker-up

# Open: http://localhost:8000/docs

# Stop
make docker-down
```

---

## REST API

Once running, interactive docs are at **http://localhost:8000/docs**

### `GET /api/v1/health`
```json
{
  "status": "healthy",
  "models_loaded": {
    "random_forest": true,
    "kmeans": true,
    "ann": true,
    "cnn_lstm": false
  },
  "version": "1.0.0"
}
```

### `POST /api/v1/predict`
```json
// Request
{
  "features": [0.12, 0.54, 0.33, ...],
  "model_type": "rf"
}

// Response
{
  "prediction": 1,
  "label": "DDoS",
  "confidence": 0.9741,
  "is_attack": true,
  "model_used": "rf"
}
```

### `POST /api/v1/predict/batch`
```json
// Request
{
  "flows": [[...], [...], [...]],
  "model_type": "ann"
}

// Response
{
  "predictions": [0, 1, 1],
  "labels": ["BENIGN", "DDoS", "SYN Flood"],
  "is_attack": [false, true, true],
  "attack_count": 2,
  "total": 3,
  "attack_rate": 0.6667,
  "model_used": "ann"
}
```

**`model_type` options:** `rf` | `ann` | `kmeans` | `cnn_lstm`

---

## Dataset

**CIC-DDoS2019** — Canadian Institute for Cybersecurity
[https://www.unb.ca/cic/datasets/ddos-2019.html](https://www.unb.ca/cic/datasets/ddos-2019.html)

Download and place CSV files in `data/raw/`. Attack types covered: SYN Flood, UDP Flood, HTTP Flood, ICMP Flood, LDAP, MSSQL, NetBIOS, NTP, SNMP, SSDP, DNS, TFTP, and more.

---

## Model Results (CIC-DDoS2019)

| Model | Accuracy | F1 (macro) | Notes |
|---|---|---|---|
| Random Forest | ~0.98 | ~0.97 | Best single-sample throughput |
| K-Means | — | — (unsupervised) | Silhouette ~0.42 |
| ANN | ~0.97 | ~0.96 | Best scalability |
| CNN-LSTM | ~0.99 | ~0.98 | Best at temporal / slow-rate attacks |

> Ensemble reduces simulated DDoS impact by **30%** vs. rule-based baselines.

---

## Development

```bash
# Run tests with coverage
make test

# Lint + type check
make lint

# Auto-format
make format

# Install pre-commit hooks (runs on every git commit)
pre-commit install
```

---

## Configuration

All tuning lives in [`configs/config.yaml`](configs/config.yaml):

```yaml
random_forest:
  n_estimators: 100
  max_depth: 20

ann:
  hidden_layers: [128, 64, 32]
  dropout_rate: 0.3
  learning_rate: 0.001
  epochs: 50

cnn_lstm:
  sequence_length: 10
  cnn_filters: [64, 128]
  lstm_units: [64, 32]
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| ML Models | scikit-learn, TensorFlow / Keras |
| API | FastAPI, Pydantic v2, Uvicorn |
| Data | pandas, NumPy |
| Visualization | matplotlib, seaborn |
| Containerization | Docker, docker-compose |
| CI/CD | GitHub Actions |
| Code Quality | black, isort, flake8, pre-commit |
| Testing | pytest, pytest-cov, httpx |

---

## Author

**Vinesh Reddy Kankanalapally**
MS Computer Science — University of Colorado Denver
[LinkedIn](https://linkedin.com/in/vinesh-reddy-kankanalapally) · [LeetCode](https://leetcode.com/VINESHREDDYK)

---

## License

[MIT](LICENSE)
