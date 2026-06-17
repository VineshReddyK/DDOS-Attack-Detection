from typing import Optional

from pydantic import BaseModel, Field, field_validator

_ALLOWED_MODELS = {"rf", "ann", "kmeans", "cnn_lstm", "ensemble"}


class TrafficFlowRequest(BaseModel):
    """Single network flow feature vector for prediction."""
    features: list[float] = Field(..., min_length=1, description="Scaled numeric feature vector")
    model_type: str = Field(default="rf", description="Model to use: rf | ann | kmeans | cnn_lstm | ensemble")

    @field_validator("model_type")
    @classmethod
    def validate_model_type(cls, v: str) -> str:
        if v not in _ALLOWED_MODELS:
            raise ValueError(f"model_type must be one of {_ALLOWED_MODELS}")
        return v


class BatchTrafficRequest(BaseModel):
    """Batch of network flows for prediction."""
    flows: list[list[float]] = Field(..., min_length=1, max_length=10000)
    model_type: str = Field(default="rf")

    @field_validator("model_type")
    @classmethod
    def validate_model_type(cls, v: str) -> str:
        if v not in _ALLOWED_MODELS:
            raise ValueError(f"model_type must be one of {_ALLOWED_MODELS}")
        return v


class EnsembleRequest(BaseModel):
    """Batch request specifically for the ensemble predictor."""
    flows: list[list[float]] = Field(..., min_length=1, max_length=10000)
    weights: Optional[dict[str, float]] = Field(default=None, description="Override default ensemble weights")


class ExplainRequest(BaseModel):
    """SHAP explainability request."""
    features: list[float] = Field(..., min_length=1)
    background_size: int = Field(default=100, ge=10, le=1000)


class DriftRequest(BaseModel):
    """Data drift detection request."""
    flows: list[list[float]] = Field(..., min_length=50, description="At least 50 samples required for PSI")


class TokenRequest(BaseModel):
    username: str
    password: str


class PredictionResponse(BaseModel):
    prediction: int
    label: str
    confidence: Optional[float] = None
    is_attack: bool
    model_used: str


class BatchPredictionResponse(BaseModel):
    predictions: list[int]
    labels: list[str]
    is_attack: list[bool]
    attack_count: int
    total: int
    attack_rate: float
    model_used: str


class EnsemblePredictionResponse(BaseModel):
    predictions: list[int]
    labels: list[str]
    is_attack: list[bool]
    attack_proba: list[float]
    attack_count: int
    total: int
    attack_rate: float
    weights_used: dict[str, float]


class ExplainResponse(BaseModel):
    global_importance: dict[str, float]
    sample_top5: dict[str, float]


class DriftResponse(BaseModel):
    status: str
    mean_psi: float
    max_psi: float
    drifted_features: dict[str, float]
    thresholds: dict[str, float]


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in_minutes: int


class HealthResponse(BaseModel):
    status: str
    models_loaded: dict[str, bool]
    version: str


class ModelInfoResponse(BaseModel):
    available_models: list[str]
    class_names: list[str]
    feature_count: Optional[int]
