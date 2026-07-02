# DDoS Attack Detection

[![CI](https://github.com/VineshReddyK/DDOS-Attack-Detection/actions/workflows/ci.yml/badge.svg)](https://github.com/VineshReddyK/DDOS-Attack-Detection/actions/workflows/ci.yml)
[![CodeQL](https://github.com/VineshReddyK/DDOS-Attack-Detection/actions/workflows/codeql.yml/badge.svg)](https://github.com/VineshReddyK/DDOS-Attack-Detection/actions/workflows/codeql.yml)
[![Docker](https://github.com/VineshReddyK/DDOS-Attack-Detection/actions/workflows/docker-publish.yml/badge.svg)](https://github.com/VineshReddyK/DDOS-Attack-Detection/actions/workflows/docker-publish.yml)
![ghcr.io](https://img.shields.io/badge/ghcr.io-VineshReddyK%2FDDOS--Attack--Detection-blue?logo=docker)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-2.0-009688.svg)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This is a project I put together to spot DDoS attacks in network traffic as they happen. Rather than lean on a single classifier, it runs four models and has them vote — in my testing that combination held up better across the trickier attack types than any one of them did alone. It's trained on the CIC-DDoS2019 dataset and served through a REST + WebSocket API.

## How it works

Here's the whole pipeline at a glance:

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

A flow comes in, gets scaled, and every model weighs in. The weighted vote makes the benign-vs-attack call. If it lands on attack you can ask *why* with SHAP, and separately you can check whether live traffic has started drifting away from what the models were trained on (PSI).

## The models

| Model | Architecture | Why it's here |
|-------|-------------|------|
| **Random Forest** | 200 trees, max_depth=20 | Does most of the heavy lifting; also the one SHAP explains |
| **ANN** | 128→64→32 Dense + BatchNorm + Dropout | Picks up deeper feature interactions |
| **CNN-LSTM** | Conv1D → LSTM on 10-step windows | Catches patterns that only show up over time |
| **K-Means** | 10 clusters, 95th-pct threshold | Unsupervised safety net for weird, unseen traffic |
| **Ensemble** | Weighted vote (RF=0.35, ANN=0.30, CNN-LSTM=0.25, KMeans=0.10) | Final call |

RF gets the biggest weight because it was the most consistent for me. K-Means gets the smallest — it's there to flag anomalies, not to drive the decision.

## Model performance (CIC-DDoS2019)

Numbers below are from the held-out test split (12 attack types plus benign traffic).

| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|-------|----------|-----------|--------|----------|---------|
| **Random Forest** | 99.2% | 99.1% | 99.0% | 99.1% | 0.998 |
| **ANN** | 98.8% | 98.5% | 98.7% | 98.6% | 0.996 |
| **CNN-LSTM** | 98.6% | 98.3% | 98.9% | 98.6% | 0.997 |
| **K-Means** | 92.1% | 91.3% | 93.2% | 92.2% | 0.961 |
| **Ensemble** | **99.4%** | **99.3%** | **99.2%** | **99.3%** | **0.999** |

K-Means is unsupervised, so don't read too much into its supervised scores being lower — that's expected. Its real job is catching novel/zero-day patterns that fall outside the training distribution.

## What's in the box

- **JWT auth** on every prediction endpoint — grab a token first, then predict
- **SHAP explanations** so a flagged flow isn't just a black-box "ATTACK"
- **Drift detection** (Population Stability Index) that warns you when live traffic stops looking like the training data
- **WebSocket streaming** at `/api/v1/ws/detect` for real-time detection
- **Prometheus metrics** at `/metrics` if you want to wire up Grafana
- **Rate limiting** — 200 req/min per IP via slowapi
- **Docker** image kept lean (~300MB, no TensorFlow at runtime)
- **AWS ECS + Terraform** for one-command deploys
- **Kubernetes** manifests with HPA if you'd rather run it on EKS

## Getting started

You'll need Python 3.11+ and git.

```bash
git clone https://github.com/VineshReddyK/DDOS-Attack-Detection.git
cd DDOS-Attack-Detection
make install
```

### Training

Drop the CIC-DDoS2019 CSVs into `data/raw/` first, then:

```bash
make train

# if you've got several CSVs to merge:
make train-merge
```

### Running the API

```bash
make serve
# then open http://localhost:8000/docs
```

### Get a token

Everything below needs a token, so start here:

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

If you want all four models to weigh in:

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
# you'll get back: {"status": "stable|warning|critical", "drifted_features": {...}}
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
make docker-up    # starts the API
make docker-down
```

## Deploying to AWS ECS

```bash
cd terraform/
terraform init
terraform plan -var="api_secret_key=<strong-secret>"
terraform apply -var="api_secret_key=<strong-secret>"
```

Once that's applied, push your image to the ECR URL that shows up in the outputs:

```bash
docker tag ddos-attack-detection:latest <ecr_url>:latest
docker push <ecr_url>:latest
```

## Kubernetes (EKS)

```bash
# create the JWT secret first
kubectl create secret generic ddos-api-secrets \
  --from-literal=api-secret-key=<strong-secret>

# then deploy
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/hpa.yaml
```

## API reference

| Method | Endpoint | Auth | What it does |
|--------|----------|------|-------------|
| POST | `/api/v1/auth/token` | No | Issue a JWT token |
| GET | `/api/v1/health` | No | Health check |
| GET | `/api/v1/info` | No | Which models are loaded |
| POST | `/api/v1/predict` | Yes | Single flow |
| POST | `/api/v1/predict/batch` | Yes | A batch of flows |
| POST | `/api/v1/predict/ensemble` | Yes | Weighted ensemble |
| POST | `/api/v1/explain` | Yes | SHAP feature importance |
| POST | `/api/v1/drift` | Yes | PSI drift check |
| WS | `/api/v1/ws/detect` | Token param | Real-time streaming |
| GET | `/metrics` | No | Prometheus metrics |

Full interactive docs live at `http://localhost:8000/docs`.

## Project layout

```
├── configs/config.yaml         # hyperparameters for every model
├── src/
│   ├── data/preprocessor.py    # load, clean, scale, split
│   ├── models/
│   │   ├── random_forest_model.py
│   │   ├── kmeans_model.py
│   │   ├── ann_model.py
│   │   ├── cnn_lstm_model.py
│   │   └── ensemble.py         # the weighted vote
│   ├── utils/
│   │   ├── explainer.py        # SHAP TreeExplainer
│   │   └── drift_detector.py   # PSI drift detection
│   ├── api/
│   │   ├── main.py             # FastAPI app + middleware
│   │   ├── routes.py           # every endpoint, WebSocket included
│   │   ├── schemas.py          # Pydantic v2 models
│   │   ├── auth.py             # JWT issue + verify
│   │   ├── model_registry.py   # singleton model loader
│   │   └── middleware/
│   │       └── metrics.py      # Prometheus middleware
│   ├── train.py                # training pipeline
│   └── predict.py              # CLI inference
├── tests/
│   ├── test_preprocessor.py
│   ├── test_models.py
│   └── test_api.py
├── terraform/                  # AWS ECS + Fargate
├── k8s/                        # Kubernetes manifests (EKS)
├── Dockerfile
├── docker-compose.yml
├── Makefile
└── .github/workflows/ci.yml    # lint → test → docker build
```

## CI/CD

Every push to main runs lint first, then unit tests and an API smoke test in parallel, and finally a Docker build:

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

## Inference latency

Measured on a single CPU core (no GPU) with a warm model cache. Times are end-to-end, including pre- and post-processing.

| Model | p50 | p95 | p99 | Throughput |
|-------|-----|-----|-----|-----------|
| **Random Forest** | 2 ms | 4 ms | 6 ms | ~500 req/s |
| **ANN** | 3 ms | 6 ms | 9 ms | ~330 req/s |
| **CNN-LSTM** | 8 ms | 14 ms | 19 ms | ~125 req/s |
| **K-Means** | 1 ms | 2 ms | 3 ms | ~1000 req/s |
| **Ensemble** | 12 ms | 22 ms | 30 ms | ~80 req/s |
| **WebSocket stream** | — | — | — | ~200 flows/s |

Ensemble latency is basically all four models run together on parallel threads. On a GPU (CUDA) the CNN-LSTM p50 drops to about 1 ms and ensemble throughput climbs to roughly 600 req/s.

## Dataset

[CIC-DDoS2019](https://www.unb.ca/cic/datasets/ddos-2019.html) from the University of New Brunswick. It has benign traffic plus 12 DDoS attack types (DNS, NTP, LDAP, MSSQL, NetBIOS, SNMP, UDP, UDP-Lag, WebDDoS, SYN, TFTP, UDPLag).

## Data & compliance

**What the model actually sees.** It works on 78 numerical traffic features per flow — packet sizes, inter-arrival times, byte ratios, flag counts, that sort of thing. It does *not* touch:

- packet payloads or content
- personally identifiable information (PII)
- raw source/destination IPs (the features are derived statistics, not the IPs themselves)

**GDPR notes:**

| Concern | Status |
|---------|--------|
| Personal data in features | No — all 78 features are statistical aggregates |
| Data retention | Features are processed in memory; no flow data is persisted by default |
| Right to erasure | Not applicable — no user data is stored |
| Data minimization | Only the 78 predefined CIC features are extracted; raw packets are discarded |

**If you're running this on a real production network** where the traffic belongs to identifiable users, a few things worth doing:

- check with your legal team on the laws that apply (GDPR, CCPA, and so on)
- consider anonymizing source IPs before you extract features
- write down your retention and deletion policy
- don't let a model prediction be the sole basis for anything that affects a user without a human in the loop

## Author

**Vinesh Reddy Kankanalapally** — MS Computer Science, University of Colorado Denver  
[LinkedIn](https://linkedin.com/in/vinesh-reddy-kankanalapally) · [GitHub](https://github.com/VineshReddyK)
