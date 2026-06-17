import numpy as np
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import yaml
import logging

logger = logging.getLogger(__name__)


class RandomForestModel:
    def __init__(self, config_path: str = "configs/config.yaml"):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        rf_cfg = cfg["random_forest"]
        train_cfg = cfg["training"]
        self.model = RandomForestClassifier(
            n_estimators=rf_cfg["n_estimators"],
            max_depth=rf_cfg["max_depth"],
            min_samples_split=rf_cfg["min_samples_split"],
            min_samples_leaf=rf_cfg["min_samples_leaf"],
            class_weight=rf_cfg["class_weight"],
            random_state=train_cfg["random_state"],
            n_jobs=train_cfg["n_jobs"],
        )

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        logger.info("Training Random Forest on %d samples...", len(X_train))
        self.model.fit(X_train, y_train)
        logger.info("Training complete.")

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray, class_names: list[str]) -> dict:
        y_pred = self.predict(X_test)
        report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        logger.info("\n%s", classification_report(y_test, y_pred, target_names=class_names))
        return {"classification_report": report, "confusion_matrix": cm.tolist()}

    def feature_importances(self, feature_names: list[str]) -> dict:
        importances = self.model.feature_importances_
        return dict(sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True))

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)
        logger.info("Saved Random Forest model to %s", path)

    def load(self, path: str) -> None:
        self.model = joblib.load(path)
        logger.info("Loaded Random Forest model from %s", path)
