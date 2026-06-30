import numpy as np
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import yaml
import logging

logger = logging.getLogger(__name__)


class RandomForestModel:
    def __init__(self, config_path="configs/config.yaml"):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        rf_cfg = cfg["random_forest"]
        train_cfg = cfg["training"]

        self.model = RandomForestClassifier(
            n_estimators=rf_cfg["n_estimators"],
            max_depth=rf_cfg["max_depth"],
            min_samples_split=rf_cfg["min_samples_split"],
            min_samples_leaf=rf_cfg["min_samples_leaf"],
            class_weight=rf_cfg["class_weight"],  # "balanced" handles benign/attack imbalance
            random_state=train_cfg["random_state"],
            n_jobs=train_cfg["n_jobs"],
        )

    def train(self, X_train, y_train):
        logger.info("fitting RF on %d samples...", len(X_train))
        self.model.fit(X_train, y_train)
        logger.info("RF training done")

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def evaluate(self, X_test, y_test, class_names):
        y_pred = self.predict(X_test)
        report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        logger.info("\n%s", classification_report(y_test, y_pred, target_names=class_names))
        return {"classification_report": report, "confusion_matrix": cm.tolist()}

    def feature_importances(self, feature_names):
        importances = self.model.feature_importances_
        # sort descending so the most useful features are first
        ranked = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
        return dict(ranked)

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)
        logger.info("saved RF model → %s", path)

    def load(self, path):
        self.model = joblib.load(path)
        logger.info("loaded RF model from %s", path)
