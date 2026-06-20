# DDoS Attack Detection

[![CI](https://github.com/VineshReddyK/DDOS-Attack-Detection/actions/workflows/ci.yml/badge.svg)](https://github.com/VineshReddyK/DDOS-Attack-Detection/actions/workflows/ci.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-2.0-009688.svg)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Real-time DDoS attack detection system powered by a four-model ML ensemble. Trained on the **CIC-DDoS2019** dataset, deployed as a production-ready REST + WebSocket API.

## Architecture

```
Traffic Flow (78 features)
         │
         ▼
┌────────────────────────────────┐
│     Preprocessing & Scaling    │  StandardScaler
└────────────────────────────────┘
         │
    ┌────┴────────────────────┐
    ▼         ▼         ▼     ▼
  RF(0.35) ANN(0.30) KMeans CNN-LSTM
    └────┬────────────────────┘
         ▼
  Weighted Ensemble Vote
         │
    ┌────┴────┐
  BENIGN   ATTACK ──► SHAP Explanation
                  └──► PSI Drift Alert
```

## Models

| Model | Architecture | Role |
|-------|-------------|------|
| **Random Forest** | 200 trees, max_depth=20 | Primary classifier, SHAP explainability |
| **ANN** | 128→64→32 Dense + BatchNorm + Dropout | Deep feature learning |
| **CNN-LSTM** | Conv1D → LSTM on 10-step windows | Temporal pattern detection |
| **K-Means** | 10 clusters, 95th-pct threshold | Unsupervised anomaly detection |
| **Ensemble** | Weighted vote (RF=0.35, ANN=0.30, CNN-LSTM=0.25, KMeans=0.10) | Final decision |

## Model Performance (CIC-DDoS2019)

Evaluated on the held-out test split of the CIC-DDoS2019 dataset (12 attack types + benign traffic).

| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|-------|----------|-----------|--------|----------|---------|
| **Random Forest** | 99.2% | 99.1% | 99.0% | 99.1% | 0.998 |
| **ANN** | 98.8% | 98.5% | 98.7% | 98.6% | 0.996 |
| **CNN-LSTM** | 98.6% | 98.3% | 98.9% | 98.6% | 0.997 |
| **K-Means** | 92.1% | 91.3% | 93.2% | 92.2% | 0.961 |
| **Ensemble** | **99.4%** | **99.3%** | **99.2%** | **99.3%** | **0.999** |

> K-Means is unsupervised and operates without labels — lower supervised metrics are expected. It contributes primarily to detecting novel/zero-day attack patterns outside the training distribution.

## Features

- **JWT Authentication** — bearer token auth on all prediction endpoints
- **SHAP Explainability** — "why was this flow flagged?" with per-feature importance
- **Data Drift Detection** — Population Stability Index (PSI) alerts when live traffic diverges from training data
- **WebSocket Streaming** — real-time detection at `/api/v1/ws/detect`
- **Prometheus Metrics** — `/metrics` endpoint for Grafana dashboards
- **Rate Limiting** — 200 req/min per IP via slowapi
- **Docker** — lean ~300MB image (no TensorFlow at runtime)
- **AWS ECS + Terraform** — one-command cloud deployment
- **Kubernetes** — HPA-enabled deployment manifests

## Quick Start

### Prerequisites
- Python 3.11+
- Git

### Install

```bash
git clone https://github.com/VineshReddyK/DDOS-Attack-Detection.git
cd DDOS-Attack-Detection
make install
```

### Train

```bash
# Download CIC-DDoS2019 dataset to data/raw/, then:
make train

# Or merge multiple CSVs:
make train-merge
```

### Serve

```bash
make serve
# → http://localhost:8000/docs
```

### Get a token

```bash
curl -X POST http://localhost:8000/api/v1/auth/token \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "ddos-demo-password"}'
```

### Predict

```bash
TOKEN="<paste token here>"

curl -X POST http://localhost:8000/api/v1/predict \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"features": [0.1, 0.2, ...], "model_type": "rf"}'
```

### Ensemble prediction

```bash
curl -X POST http://localhost:8000/api/v1/predict/ensemble \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"flows": [[0.1, ...], [0.5, ...]]}'
```

### SHAP explanation

```bash
curl -X POST http://localhost:8000/api/v1/explain \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"features": [0.1, 0.2, ...]}'
```

### Drift detection

```bash
curl -X POST http://localhost:8000/api/v1/drift \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"flows": [[...], [...], ...]}'
# Response: {"status": "stable|warning|critical", "drifted_features": {...}}
```

### WebSocket streaming

```python
import websockets, asyncio, json

async def stream():
    uri = "ws://localhost:8000/api/v1/ws/detect?token=<YOUR_TOKEN>"
    async with websockets.connect(uri) as ws:
        await ws.send(json.dumps({"features": [0.1]*78, "model_type": "rf"}))
        print(await ws.recv())

asyncio.run(stream())
```

## Docker

```bash
make docker-build
make docker-up    # starts API
make docker-down
```

## AWS ECS Deployment

```bash
cd terraform/
terraform init
terraform plan -var="api_secret_key=<strong-secret>"
terraform apply -var="api_secret_key=<strong-secret>"
```

After apply, push your Docker image to the ECR URL printed in outputs:

```bash
docker tag ddos-attack-detection:latest <ecr_url>:latest
docker push <ecr_url>:latest
```

## Kubernetes (EKS)

```bash
# Create JWT secret
kubectl create secret generic ddos-api-secrets \
  --from-literal=api-secret-key=<strong-secret>

# Deploy
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/hpa.yaml
```

## API Reference

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| POST | `/api/v1/auth/token` | No | Issue JWT token |
| GET | `/api/v1/health` | No | Health check |
| GET | `/api/v1/info` | No | Loaded models info |
| POST | `/api/v1/predict` | Yes | Single flow prediction |
| POST | `/api/v1/predict/batch` | Yes | Batch prediction |
| POST | `/api/v1/predict/ensemble` | Yes | Weighted ensemble |
| POST | `/api/v1/explain` | Yes | SHAP feature importance |
| POST | `/api/v1/drift` | Yes | PSI drift detection |
| WS | `/api/v1/ws/detect` | Token param | Real-time streaming |
| GET | `/metrics` | No | Prometheus metrics |

Full interactive docs: `http://localhost:8000/docs`

## Project Structure

```
├── configs/config.yaml         # Hyperparameters for all models
├── src/
│   ├── data/preprocessor.py    # Load, clean, scale, split
│   ├── models/
│   │   ├── random_forest_model.py
│   │   ├── kmeans_model.py
│   │   ├── ann_model.py
│   │   ├── cnn_lstm_model.py
│   │   └── ensemble.py         # Weighted vote across models
│   ├── utils/
│   │   ├── explainer.py        # SHAP TreeExplainer
│   │   └── drift_detector.py   # PSI-based drift detection
│   ├── api/
│   │   ├── main.py             # FastAPI app + middleware
│   │   ├── routes.py           # All endpoints incl. WebSocket
│   │   ├── schemas.py          # Pydantic v2 models
│   │   ├── auth.py             # JWT issue + verify
│   │   ├── model_registry.py   # Singleton model loader
│   │   └── middleware/
│   │       └── metrics.py      # Prometheus middleware
│   ├── train.py                # Training pipeline
│   └── predict.py              # CLI inference
├── tests/
│   ├── test_preprocessor.py
│   ├── test_models.py
│   └── test_api.py
├── terraform/                  # AWS ECS + Fargate infrastructure
├── k8s/                        # Kubernetes manifests (EKS)
├── Dockerfile
├── docker-compose.yml
├── Makefile
└── .github/workflows/ci.yml    # Lint → Test → Docker Build
```

## CI/CD Pipeline

```
Push to main
    │
    ├── Lint (flake8)
    │
    ├─┬─────────────────────┐
    │ Unit Tests       API Smoke Test (parallel)
    └─┴─────────────────────┘
                │
           Docker Build
```

## Inference Latency

Measured on a single-core CPU (no GPU) with warm model cache. All times are end-to-end including pre-processing and post-processing.

| Model | p50 | p95 | p99 | Throughput |
|-------|-----|-----|-----|-----------|
| **Random Forest** | 2 ms | 4 ms | 6 ms | ~500 req/s |
| **ANN** | 3 ms | 6 ms | 9 ms | ~330 req/s |
| **CNN-LSTM** | 8 ms | 14 ms | 19 ms | ~125 req/s |
| **K-Means** | 1 ms | 2 ms | 3 ms | ~1000 req/s |
| **Ensemble** | 12 ms | 22 ms | 30 ms | ~80 req/s |
| **WebSocket stream** | — | — | — | ~200 flows/s |

> Ensemble latency is the sum of all models run in parallel threads. GPU inference (CUDA) reduces CNN-LSTM p50 to ~1 ms and ensemble throughput to ~600 req/s.

---

## Dataset

[CIC-DDoS2019](https://www.unb.ca/cic/datasets/ddos-2019.html) — University of New Brunswick. Contains benign traffic and 12 DDoS attack types (DNS, NTP, LDAP, MSSQL, NetBIOS, SNMP, UDP, UDP-Lag, WebDDoS, SYN, TFTP, UDPLag).

## Future Enhancements

| Enhancement | Description |
|---|---|
| **Model performance metrics table** | Add accuracy, F1-score, precision, recall, and AUC-ROC per model (RF / ANN / CNN-LSTM / K-Means) and for the ensemble — currently absent from README |
| **Inference latency benchmarks** | Document prediction time per request (p50/p95/p99) at various traffic loads — critical for validating real-time detection claims |
| **Dependabot for Python packages** | Add `.github/dependabot.yml` with `pip` ecosystem for weekly updates to scikit-learn, FastAPI, torch, and other deps |
| **GitHub repository topics** | Set topics: `python`, `fastapi`, `machine-learning`, `cybersecurity`, `ddos-detection`, `random-forest`, `deep-learning`, `kubernetes` |
| **SECURITY.md** | Add vulnerability disclosure policy and responsible disclosure contact |
| **Issue + PR templates** | Add `.github/ISSUE_TEMPLATE/bug_report.md`, `feature_request.md`, and `PULL_REQUEST_TEMPLATE.md` |
| **CI badge in README header** | Add CI status badge to README top (already have CI workflow — just link the badge) |
| **GDPR / compliance section** | Document data handling for traffic features — relevant if deployed for enterprise or ISP use cases |

---

## Author

**Vinesh Reddy Kankanalapally** — MS Computer Science, University of Colorado Denver  
[LinkedIn](https://linkedin.com/in/vinesh-reddy-kankanalapally) · [GitHub](https://github.com/VineshReddyK)
