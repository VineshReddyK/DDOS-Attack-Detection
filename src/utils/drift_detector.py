"""
Data drift detection using Population Stability Index (PSI).
Alerts when live traffic features drift from training distribution.
"""
import numpy as np
import logging
from typing import Optional
import joblib
from pathlib import Path

logger = logging.getLogger(__name__)

PSI_THRESHOLD_WARNING = 0.10
PSI_THRESHOLD_CRITICAL = 0.25
N_BINS = 10


def _psi(expected: np.ndarray, actual: np.ndarray, n_bins: int = N_BINS) -> float:
    bins = np.linspace(
        min(expected.min(), actual.min()),
        max(expected.max(), actual.max()) + 1e-9,
        n_bins + 1,
    )
    exp_counts, _ = np.histogram(expected, bins=bins)
    act_counts, _ = np.histogram(actual, bins=bins)

    exp_pct = (exp_counts + 1e-9) / len(expected)
    act_pct = (act_counts + 1e-9) / len(actual)

    psi = np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct))
    return float(psi)


class DriftDetector:
    def __init__(self, feature_names: Optional[list[str]] = None):
        self.feature_names = feature_names
        self.reference_stats: Optional[dict] = None

    def fit(self, X_train: np.ndarray) -> None:
        self.reference_stats = {
            "mean": X_train.mean(axis=0).tolist(),
            "std": X_train.std(axis=0).tolist(),
            "min": X_train.min(axis=0).tolist(),
            "max": X_train.max(axis=0).tolist(),
            "data": X_train,
        }
        logger.info("DriftDetector fitted on %d training samples.", len(X_train))

    def detect(self, X_live: np.ndarray) -> dict:
        if self.reference_stats is None:
            raise RuntimeError("DriftDetector not fitted. Call fit() first.")

        X_ref = self.reference_stats["data"]
        n_features = X_ref.shape[1]
        names = self.feature_names or [f"feature_{i}" for i in range(n_features)]

        psi_scores = {}
        for i in range(n_features):
            psi_scores[names[i]] = round(_psi(X_ref[:, i], X_live[:, i]), 6)

        mean_psi = float(np.mean(list(psi_scores.values())))
        max_psi = float(max(psi_scores.values()))
        drifted = {k: v for k, v in psi_scores.items() if v > PSI_THRESHOLD_WARNING}

        if max_psi > PSI_THRESHOLD_CRITICAL:
            status = "critical"
        elif max_psi > PSI_THRESHOLD_WARNING:
            status = "warning"
        else:
            status = "stable"

        return {
            "status": status,
            "mean_psi": round(mean_psi, 6),
            "max_psi": round(max_psi, 6),
            "drifted_features": drifted,
            "thresholds": {
                "warning": PSI_THRESHOLD_WARNING,
                "critical": PSI_THRESHOLD_CRITICAL,
            },
        }

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"reference_stats": self.reference_stats, "feature_names": self.feature_names}, path)

    def load(self, path: str) -> None:
        data = joblib.load(path)
        self.reference_stats = data["reference_stats"]
        self.feature_names = data["feature_names"]
