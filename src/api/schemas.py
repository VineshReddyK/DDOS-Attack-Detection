from pydantic import BaseModel, Field, field_validator
from typing import Optional
import numpy as np


class TrafficFlowRequest(BaseModel):
    """Single network flow feature vector for prediction."""
    features: list[float] = Field(..., min_length=1, description="Scaled numeric feature vector")
    model_type: str = Field(default="rf", description="Model to use: rf | ann | kmeans")

    @field_validator("model_type")
    @classmethod
    def validate_model_type(cls, v: str) -> str:
        allowed = {"rf", "ann", "kmeans", "cnn_lstm"}
        if v not in allowed:
            raise ValueError(f"model_type must be one of {allowed}")
        return v


class BatchTrafficRequest(BaseModel):
    """Batch of network flows for prediction."""
    flows: list[list[float]] = Field(..., min_length=1, max_length=10000)
    model_type: str = Field(default="rf")

    @field_validator("model_type")
    @classmethod
    def validate_model_type(cls, v: str) -> str:
        allowed = {"rf", "ann", "kmeans", "cnn_lstm"}
        if v not in allowed:
            raise ValueError(f"model_type must be one of {allowed}")
        return v


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


class HealthResponse(BaseModel):
    status: str
    models_loaded: dict[str, bool]
    version: str


class ModelInfoResponse(BaseModel):
    available_models: list[str]
    class_names: list[str]
    feature_count: Optional[int]
