"""
Singleton model registry — loads trained models once at startup and
serves them to API route handlers.
"""
import os
from pathlib import Path
from typing import Optional
import joblib
import logging

logger = logging.getLogger(__name__)

CONFIG_PATH = os.getenv("CONFIG_PATH", "configs/config.yaml")
MODELS_DIR = Path(os.getenv("MODELS_DIR", "models"))


class ModelRegistry:
    _instance: Optional["ModelRegistry"] = None

    def __new__(cls) -> "ModelRegistry":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self.rf = None
        self.kmeans = None
        self.ann = None
        self.cnn_lstm = None
        self.ensemble = None
        self.drift_detector = None
        self.class_names: list[str] = []
        self.feature_names: list[str] = []
        self.feature_count: Optional[int] = None
        self._initialized = True

    def load_all(self) -> None:
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))

        from models.random_forest_model import RandomForestModel
        from models.kmeans_model import KMeansAnomalyDetector
        from models.ann_model import ANNModel
        from models.cnn_lstm_model import CNNLSTMModel

        rf_path = MODELS_DIR / "random_forest.joblib"
        if rf_path.exists():
            try:
                self.rf = RandomForestModel(CONFIG_PATH)
                self.rf.load(str(rf_path))
                logger.info("Loaded Random Forest")
            except Exception as e:
                logger.warning("Could not load RF: %s", e)

        km_path = MODELS_DIR / "kmeans.joblib"
        if km_path.exists():
            try:
                self.kmeans = KMeansAnomalyDetector(CONFIG_PATH)
                self.kmeans.load(str(km_path))
                logger.info("Loaded KMeans")
            except Exception as e:
                logger.warning("Could not load KMeans: %s", e)

        ann_path = MODELS_DIR / "ann_model"
        if ann_path.exists():
            try:
                self.ann = ANNModel(CONFIG_PATH)
                self.ann.load(str(ann_path))
                logger.info("Loaded ANN")
            except Exception as e:
                logger.warning("Could not load ANN: %s", e)

        cl_path = MODELS_DIR / "cnn_lstm_model"
        if cl_path.exists():
            try:
                self.cnn_lstm = CNNLSTMModel(CONFIG_PATH)
                self.cnn_lstm.load(str(cl_path))
                logger.info("Loaded CNN-LSTM")
            except Exception as e:
                logger.warning("Could not load CNN-LSTM: %s", e)

        meta_path = MODELS_DIR / "metadata.joblib"
        if meta_path.exists():
            meta = joblib.load(str(meta_path))
            self.class_names = meta.get("class_names", [])
            self.feature_count = meta.get("feature_count")
            self.feature_names = meta.get("feature_names", [f"feature_{i}" for i in range(self.feature_count or 0)])

        # Ensemble (requires at least one model loaded)
        if any([self.rf, self.ann, self.kmeans, self.cnn_lstm]):
            try:
                from models.ensemble import EnsemblePredictor
                self.ensemble = EnsemblePredictor(self)
                logger.info("Ensemble predictor ready")
            except Exception as e:
                logger.warning("Could not init ensemble: %s", e)

        # Drift detector
        drift_path = MODELS_DIR / "drift_detector.joblib"
        if drift_path.exists():
            try:
                from utils.drift_detector import DriftDetector
                self.drift_detector = DriftDetector(self.feature_names or None)
                self.drift_detector.load(str(drift_path))
                logger.info("Loaded DriftDetector")
            except Exception as e:
                logger.warning("Could not load DriftDetector: %s", e)

    @property
    def loaded_models(self) -> dict[str, bool]:
        return {
            "random_forest": self.rf is not None,
            "kmeans": self.kmeans is not None,
            "ann": self.ann is not None,
            "cnn_lstm": self.cnn_lstm is not None,
            "ensemble": self.ensemble is not None,
        }

    def get_model(self, model_type: str):
        mapping = {
            "rf": self.rf,
            "kmeans": self.kmeans,
            "ann": self.ann,
            "cnn_lstm": self.cnn_lstm,
            "ensemble": self.ensemble,
        }
        model = mapping.get(model_type)
        if model is None:
            raise ValueError(f"Model '{model_type}' is not loaded. Train it first.")
        return model


registry = ModelRegistry()
