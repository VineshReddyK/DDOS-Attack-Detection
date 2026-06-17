"""
SHAP-based explainability for the Random Forest model.
Answers: "Why was this network flow flagged as an attack?"
"""
import numpy as np
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class SHAPExplainer:
    def __init__(self, rf_model, feature_names: list[str]):
        self.rf_model = rf_model
        self.feature_names = feature_names
        self._explainer = None

    def _build(self, X_background: np.ndarray) -> None:
        try:
            import shap
            self._explainer = shap.TreeExplainer(self.rf_model.model)
            logger.info("SHAP TreeExplainer initialized.")
        except ImportError:
            raise RuntimeError("shap not installed. Run: pip install shap")

    def explain(self, X: np.ndarray, X_background: Optional[np.ndarray] = None) -> dict:
        if self._explainer is None:
            bg = X_background if X_background is not None else X[:100]
            self._build(bg)

        import shap
        shap_values = self._explainer.shap_values(X)

        if isinstance(shap_values, list):
            values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        else:
            values = shap_values

        mean_abs = np.abs(values).mean(axis=0)
        ranked = sorted(zip(self.feature_names, mean_abs.tolist()), key=lambda x: x[1], reverse=True)

        per_sample = []
        for i, row in enumerate(values):
            top = sorted(zip(self.feature_names, row.tolist()), key=lambda x: abs(x[1]), reverse=True)[:5]
            per_sample.append({feat: round(val, 6) for feat, val in top})

        return {
            "global_importance": {feat: round(val, 6) for feat, val in ranked[:20]},
            "per_sample_top5": per_sample,
        }

    def explain_single(self, x: np.ndarray) -> dict:
        result = self.explain(x.reshape(1, -1))
        return result["per_sample_top5"][0]

    def plot_summary(self, X: np.ndarray, save_path: Optional[str] = None) -> None:
        import shap
        import matplotlib.pyplot as plt

        if self._explainer is None:
            self._build(X)

        shap_values = self._explainer.shap_values(X)
        values = shap_values[1] if isinstance(shap_values, list) and len(shap_values) > 1 else shap_values

        shap.summary_plot(values, X, feature_names=self.feature_names, show=False)
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info("SHAP summary plot saved to %s", save_path)
        plt.show()
