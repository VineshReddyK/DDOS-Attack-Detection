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

## Dataset

[CIC-DDoS2019](https://www.unb.ca/cic/datasets/ddos-2019.html) — University of New Brunswick. Contains benign traffic and 12 DDoS attack types (DNS, NTP, LDAP, MSSQL, NetBIOS, SNMP, UDP, UDP-Lag, WebDDoS, SYN, TFTP, UDPLag).

## Author

**Vinesh Reddy Kankanalapally** — MS Computer Science, University of Colorado Denver  
[LinkedIn](https://linkedin.com/in/vinesh-reddy-kankanalapally) · [GitHub](https://github.com/VineshReddyK)
