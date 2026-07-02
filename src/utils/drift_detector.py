"""
Drift detection with the Population Stability Index (PSI).

The plan is simple: remember the feature distributions the models trained on,
and when live traffic shows up, measure how far each feature has moved. If a
feature drifts too far the predictions on it stop being trustworthy, so we'd
rather raise a flag than let things quietly rot in production.
"""
import logging
from pathlib import Path
from typing import Optional

import joblib
import numpy as np

logger = logging.getLogger(__name__)

# the usual PSI cutoffs. under 0.1 is basically noise, 0.1-0.25 is "keep an eye
# on it", and anything past 0.25 means the feature has genuinely shifted.
PSI_THRESHOLD_WARNING = 0.10
PSI_THRESHOLD_CRITICAL = 0.25
N_BINS = 10


def _psi(expected, actual, n_bins=N_BINS):
    # bin edges cover both arrays so nothing lands outside the histogram. the
    # tiny epsilon on the top edge stops the max value falling on the last edge.
    bins = np.linspace(
        min(expected.min(), actual.min()),
        max(expected.max(), actual.max()) + 1e-9,
        n_bins + 1,
    )
    exp_counts, _ = np.histogram(expected, bins=bins)
    act_counts, _ = np.histogram(actual, bins=bins)

    # turn counts into proportions. the +1e-9 keeps empty bins from blowing up
    # the division / log below.
    exp_pct = (exp_counts + 1e-9) / len(expected)
    act_pct = (act_counts + 1e-9) / len(actual)

    psi = np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct))
    return float(psi)


class DriftDetector:
    def __init__(self, feature_names: Optional[list[str]] = None):
        self.feature_names = feature_names
        self.reference_stats = None

    def fit(self, X_train):
        # stash the training distribution. we hang on to the raw data too because
        # PSI needs the actual reference values, not just the summary stats.
        self.reference_stats = {
            "mean": X_train.mean(axis=0).tolist(),
            "std": X_train.std(axis=0).tolist(),
            "min": X_train.min(axis=0).tolist(),
            "max": X_train.max(axis=0).tolist(),
            "data": X_train,
        }
        logger.info("drift detector fitted on %d training samples", len(X_train))

    def detect(self, X_live):
        if self.reference_stats is None:
            raise RuntimeError("DriftDetector not fitted. Call fit() first.")

        X_ref = self.reference_stats["data"]
        n_features = X_ref.shape[1]
        names = self.feature_names or [f"feature_{i}" for i in range(n_features)]

        # score every feature one at a time
        psi_scores = {}
        for i in range(n_features):
            psi_scores[names[i]] = round(_psi(X_ref[:, i], X_live[:, i]), 6)

        mean_psi = float(np.mean(list(psi_scores.values())))
        max_psi = float(max(psi_scores.values()))

        # surface anything past the warning line so the caller can look at it
        drifted = {k: v for k, v in psi_scores.items() if v > PSI_THRESHOLD_WARNING}

        # overall status follows the single worst feature, not the average — one
        # badly drifted feature is enough to hurt the model.
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

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"reference_stats": self.reference_stats, "feature_names": self.feature_names},
            path,
        )

    def load(self, path):
        data = joblib.load(path)
        self.reference_stats = data["reference_stats"]
        self.feature_names = data["feature_names"]
