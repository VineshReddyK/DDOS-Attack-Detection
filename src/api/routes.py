import asyncio
import json
import logging

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect

from .auth import create_access_token, verify_token
from .model_registry import registry
from .schemas import (
    BatchPredictionResponse, BatchTrafficRequest,
    DriftRequest, DriftResponse,
    EnsemblePredictionResponse, EnsembleRequest,
    ExplainRequest, ExplainResponse,
    HealthResponse, ModelInfoResponse,
    PredictionResponse, TokenRequest, TokenResponse,
    TrafficFlowRequest,
)

router = APIRouter()
logger = logging.getLogger(__name__)

VERSION = "2.0.0"

# ── Auth ──────────────────────────────────────────────────────────────────────

DEMO_USERS = {"admin": "ddos-demo-password", "user": "readonly123"}


@router.post("/auth/token", response_model=TokenResponse, tags=["Auth"])
def issue_token(body: TokenRequest):
    expected = DEMO_USERS.get(body.username)
    if not expected or expected != body.password:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = create_access_token(body.username)
    from .auth import ACCESS_TOKEN_EXPIRE_MINUTES
    return TokenResponse(access_token=token, expires_in_minutes=ACCESS_TOKEN_EXPIRE_MINUTES)


# ── System ────────────────────────────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    return HealthResponse(
        status="healthy",
        models_loaded=registry.loaded_models,
        version=VERSION,
    )


@router.get("/info", response_model=ModelInfoResponse, tags=["System"])
def model_info():
    return ModelInfoResponse(
        available_models=[k for k, v in registry.loaded_models.items() if v],
        class_names=registry.class_names,
        feature_count=registry.feature_count,
    )


# ── Prediction ────────────────────────────────────────────────────────────────

@router.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict_single(request: TrafficFlowRequest, _: str = Depends(verify_token)):
    if request.model_type == "ensemble":
        raise HTTPException(status_code=422, detail="Use /predict/ensemble for ensemble predictions")
    model = _get_model_or_404(request.model_type)
    X = np.array(request.features, dtype="float32").reshape(1, -1)
    pred_idx, confidence = _predict_with_confidence(model, request.model_type, X)
    label = _label_for(pred_idx)
    return PredictionResponse(
        prediction=int(pred_idx),
        label=label,
        confidence=confidence,
        is_attack=label.upper() != "BENIGN",
        model_used=request.model_type,
    )


@router.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
def predict_batch(request: BatchTrafficRequest, _: str = Depends(verify_token)):
    if request.model_type == "ensemble":
        raise HTTPException(status_code=422, detail="Use /predict/ensemble for ensemble predictions")
    model = _get_model_or_404(request.model_type)
    X = np.array(request.flows, dtype="float32")
    preds = _batch_predict(model, request.model_type, X)
    labels = [_label_for(p) for p in preds]
    is_attack = [lbl.upper() != "BENIGN" for lbl in labels]
    attack_count = sum(is_attack)
    return BatchPredictionResponse(
        predictions=[int(p) for p in preds],
        labels=labels,
        is_attack=is_attack,
        attack_count=attack_count,
        total=len(preds),
        attack_rate=round(attack_count / len(preds), 4),
        model_used=request.model_type,
    )


@router.post("/predict/ensemble", response_model=EnsemblePredictionResponse, tags=["Prediction"])
def predict_ensemble(request: EnsembleRequest, _: str = Depends(verify_token)):
    if registry.ensemble is None:
        raise HTTPException(status_code=404, detail="Ensemble not available. Train at least one model first.")
    from models.ensemble import EnsemblePredictor
    ens = EnsemblePredictor(registry, weights=request.weights) if request.weights else registry.ensemble
    X = np.array(request.flows, dtype="float32")
    preds = ens.predict(X)
    probas = ens.predict_proba(X)
    labels = [_label_for(p) for p in preds]
    is_attack = [lbl.upper() != "BENIGN" for lbl in labels]
    attack_count = sum(is_attack)
    return EnsemblePredictionResponse(
        predictions=[int(p) for p in preds],
        labels=labels,
        is_attack=is_attack,
        attack_proba=[round(float(p[1]), 4) for p in probas],
        attack_count=attack_count,
        total=len(preds),
        attack_rate=round(attack_count / len(preds), 4),
        weights_used=ens.weights,
    )


# ── Explainability ────────────────────────────────────────────────────────────

@router.post("/explain", response_model=ExplainResponse, tags=["Explainability"])
def explain_prediction(request: ExplainRequest, _: str = Depends(verify_token)):
    if registry.rf is None:
        raise HTTPException(status_code=404, detail="Random Forest model not loaded (required for SHAP).")
    try:
        from utils.explainer import SHAPExplainer
        explainer = SHAPExplainer(registry.rf, registry.feature_names or [f"f{i}" for i in range(len(request.features))])
        X = np.array(request.features, dtype="float32").reshape(1, -1)
        result = explainer.explain(X)
        return ExplainResponse(
            global_importance=result["global_importance"],
            sample_top5=result["per_sample_top5"][0],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"SHAP explanation failed: {e}")


# ── Drift Detection ───────────────────────────────────────────────────────────

@router.post("/drift", response_model=DriftResponse, tags=["Monitoring"])
def detect_drift(request: DriftRequest, _: str = Depends(verify_token)):
    if registry.drift_detector is None:
        raise HTTPException(status_code=404, detail="DriftDetector not fitted. Run training pipeline first.")
    X = np.array(request.flows, dtype="float32")
    result = registry.drift_detector.detect(X)
    return DriftResponse(**result)


# ── WebSocket streaming ───────────────────────────────────────────────────────

@router.websocket("/ws/detect")
async def websocket_detect(websocket: WebSocket):
    """
    Real-time DDoS detection over WebSocket.

    Send JSON: {"features": [...], "model_type": "rf"}
    Receive JSON: {"prediction": 1, "label": "ATTACK", "is_attack": true, "confidence": 0.97}

    Auth: pass token as query param ?token=<jwt>
    """
    token = websocket.query_params.get("token")
    if not token:
        await websocket.close(code=1008, reason="Missing token")
        return

    from .auth import verify_token as _verify
    from fastapi.security import HTTPAuthorizationCredentials
    try:
        _verify(HTTPAuthorizationCredentials(scheme="Bearer", credentials=token))
    except Exception:
        await websocket.close(code=1008, reason="Invalid token")
        return

    await websocket.accept()
    logger.info("WebSocket client connected: %s", websocket.client)

    try:
        from .middleware.metrics import get_metrics
        m = get_metrics()
        if m:
            m["active_connections"].inc()

        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=60.0)
            except asyncio.TimeoutError:
                await websocket.send_text(json.dumps({"ping": "keepalive"}))
                continue

            try:
                payload = json.loads(data)
                features = payload.get("features", [])
                model_type = payload.get("model_type", "rf")
                if not features:
                    await websocket.send_text(json.dumps({"error": "features required"}))
                    continue

                model = registry.get_model(model_type)
                X = np.array(features, dtype="float32").reshape(1, -1)
                pred_idx, confidence = _predict_with_confidence(model, model_type, X)
                label = _label_for(pred_idx)
                response = {
                    "prediction": int(pred_idx),
                    "label": label,
                    "is_attack": label.upper() != "BENIGN",
                    "confidence": confidence,
                    "model_used": model_type,
                }
                await websocket.send_text(json.dumps(response))

            except (ValueError, KeyError) as e:
                await websocket.send_text(json.dumps({"error": str(e)}))
            except Exception as e:
                logger.exception("WS prediction error")
                await websocket.send_text(json.dumps({"error": f"Internal error: {e}"}))

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    finally:
        m = get_metrics()
        if m:
            m["active_connections"].dec()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_model_or_404(model_type: str):
    try:
        return registry.get_model(model_type)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


def _predict_with_confidence(model, model_type: str, X: np.ndarray) -> tuple[int, float | None]:
    if model_type == "kmeans":
        pred = int(model.predict(X)[0])
        score = float(model.anomaly_scores(X)[0])
        return pred, round(score, 6)
    if model_type in ("rf",):
        pred = int(model.predict(X)[0])
        proba = model.predict_proba(X)[0]
        return pred, round(float(proba.max()), 4)
    pred = int(model.predict(X)[0])
    try:
        proba = model.predict_proba(X)[0]
        return pred, round(float(proba.max()), 4)
    except Exception:
        return pred, None


def _batch_predict(model, model_type: str, X: np.ndarray) -> np.ndarray:
    if model_type == "cnn_lstm":
        y_dummy = np.zeros(len(X))
        return model.predict(X, y_dummy)
    return model.predict(X)


def _label_for(pred_idx: int) -> str:
    if registry.class_names and pred_idx < len(registry.class_names):
        return registry.class_names[pred_idx]
    return "ATTACK" if pred_idx == 1 else "BENIGN"
