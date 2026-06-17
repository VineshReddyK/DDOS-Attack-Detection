import numpy as np
from fastapi import APIRouter, HTTPException
from .schemas import (
    TrafficFlowRequest, BatchTrafficRequest,
    PredictionResponse, BatchPredictionResponse,
    HealthResponse, ModelInfoResponse,
)
from .model_registry import registry

router = APIRouter()

VERSION = "1.0.0"


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


@router.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict_single(request: TrafficFlowRequest):
    model = _get_model_or_404(request.model_type)
    X = np.array(request.features, dtype="float32").reshape(1, -1)

    pred_idx, confidence = _predict_with_confidence(model, request.model_type, X)
    label = _label_for(pred_idx)
    is_attack = label.upper() != "BENIGN"

    return PredictionResponse(
        prediction=int(pred_idx),
        label=label,
        confidence=confidence,
        is_attack=is_attack,
        model_used=request.model_type,
    )


@router.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
def predict_batch(request: BatchTrafficRequest):
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


# ── helpers ──────────────────────────────────────────────────────────────────

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
