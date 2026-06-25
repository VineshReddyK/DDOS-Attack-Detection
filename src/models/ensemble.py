import numpy as np
import logging
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_WEIGHTS = {
    "rf": 0.35,
    "ann": 0.30,
    "cnn_lstm": 0.25,
    "kmeans": 0.10,
}


class EnsemblePredictor:
    def __init__(self, registry, weights: Optional[dict] = None):
        self.registry = registry
        self.weights = weights or DEFAULT_WEIGHTS

    def _compute_votes(self, X: np.ndarray, y_dummy: Optional[np.ndarray] = None):
        """Accumulate weighted attack-probability votes from all loaded models."""
        votes = np.zeros(len(X), dtype=float)
        total_weight = 0.0

        if self.registry.rf is not None:
            proba = self.registry.rf.predict_proba(X)
            attack_proba = 1 - proba[:, 0] if proba.shape[1] > 1 else proba[:, 0]
            votes += self.weights["rf"] * attack_proba
            total_weight += self.weights["rf"]

        if self.registry.ann is not None:
            raw = self.registry.ann.predict_proba(X)
            attack_proba = 1 - raw[:, 0] if raw.shape[1] > 1 else raw.squeeze()
            votes += self.weights["ann"] * attack_proba
            total_weight += self.weights["ann"]

        if self.registry.kmeans is not None:
            scores = self.registry.kmeans.anomaly_scores(X)
            if self.registry.kmeans.threshold and self.registry.kmeans.threshold > 0:
                normalized = np.clip(scores / self.registry.kmeans.threshold, 0, 2) / 2
            else:
                normalized = scores / (scores.max() + 1e-9)
            votes += self.weights["kmeans"] * normalized
            total_weight += self.weights["kmeans"]

        if self.registry.cnn_lstm is not None and y_dummy is not None:
            seq_len = self.registry.cnn_lstm.seq_len
            if len(X) > seq_len:
                preds = self.registry.cnn_lstm.predict(X, y_dummy)
                padded = np.zeros(len(X))
                padded[seq_len:] = preds
                votes += self.weights["cnn_lstm"] * padded
                total_weight += self.weights["cnn_lstm"]

        if total_weight == 0:
            raise RuntimeError("No models loaded in registry.")

        return votes, total_weight

    def predict(self, X: np.ndarray, y_dummy: Optional[np.ndarray] = None) -> np.ndarray:
        votes, total_weight = self._compute_votes(X, y_dummy)
        return (votes / total_weight > 0.5).astype(int)

    def predict_proba(self, X: np.ndarray, y_dummy: Optional[np.ndarray] = None) -> np.ndarray:
        votes, total_weight = self._compute_votes(X, y_dummy)
        score = votes / total_weight
        return np.column_stack([1 - score, score])
