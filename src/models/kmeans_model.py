import numpy as np
import joblib
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import yaml
import logging

logger = logging.getLogger(__name__)


class KMeansAnomalyDetector:
    # flags samples far from their cluster centroid as anomalous

    def __init__(self, config_path: str = "configs/config.yaml"):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        km_cfg = cfg["kmeans"]
        train_cfg = cfg["training"]
        self.model = KMeans(
            n_clusters=km_cfg["n_clusters"],
            max_iter=km_cfg["max_iter"],
            n_init=km_cfg["n_init"],
            random_state=train_cfg["random_state"],
        )
        self.threshold = None

    def train(self, X_train: np.ndarray, threshold_percentile: float = 95.0) -> None:
        logger.info("Fitting KMeans on %d samples...", len(X_train))
        self.model.fit(X_train)
        distances = self._distances_to_centroid(X_train)
        self.threshold = float(np.percentile(distances, threshold_percentile))
        logger.info("KMeans threshold (p%.0f): %.4f", threshold_percentile, self.threshold)

    def _distances_to_centroid(self, X: np.ndarray) -> np.ndarray:
        labels = self.model.predict(X)
        centroids = self.model.cluster_centers_[labels]
        return np.linalg.norm(X - centroids, axis=1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.threshold is None:
            raise RuntimeError("Model not trained. Call train() first.")
        distances = self._distances_to_centroid(X)
        return (distances > self.threshold).astype(int)

    def anomaly_scores(self, X: np.ndarray) -> np.ndarray:
        return self._distances_to_centroid(X)

    def evaluate(self, X: np.ndarray) -> dict:
        labels = self.model.predict(X)
        metrics = {}
        if len(np.unique(labels)) > 1:
            metrics["silhouette_score"] = float(silhouette_score(X, labels, sample_size=min(5000, len(X))))
            metrics["davies_bouldin_score"] = float(davies_bouldin_score(X, labels))
        metrics["inertia"] = float(self.model.inertia_)
        logger.info("KMeans metrics: %s", metrics)
        return metrics

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "threshold": self.threshold}, path)
        logger.info("Saved KMeans model to %s", path)

    def load(self, path: str) -> None:
        data = joblib.load(path)
        self.model = data["model"]
        self.threshold = data["threshold"]
        logger.info("Loaded KMeans model from %s", path)
