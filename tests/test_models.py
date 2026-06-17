import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.random_forest_model import RandomForestModel
from models.kmeans_model import KMeansAnomalyDetector


@pytest.fixture
def binary_data():
    np.random.seed(0)
    X_train = np.random.rand(300, 10).astype("float32")
    y_train = np.random.randint(0, 2, 300)
    X_test = np.random.rand(100, 10).astype("float32")
    y_test = np.random.randint(0, 2, 100)
    return X_train, y_train, X_test, y_test


def test_random_forest_train_predict(binary_data):
    X_train, y_train, X_test, y_test = binary_data
    model = RandomForestModel()
    model.train(X_train, y_train)
    preds = model.predict(X_test)
    assert len(preds) == len(X_test)
    assert set(preds).issubset({0, 1})


def test_random_forest_evaluate(binary_data):
    X_train, y_train, X_test, y_test = binary_data
    model = RandomForestModel()
    model.train(X_train, y_train)
    results = model.evaluate(X_test, y_test, ["BENIGN", "DDoS"])
    assert "classification_report" in results
    assert "confusion_matrix" in results


def test_kmeans_train_predict(binary_data):
    X_train, y_train, X_test, _ = binary_data
    model = KMeansAnomalyDetector()
    model.train(X_train)
    preds = model.predict(X_test)
    assert len(preds) == len(X_test)
    assert set(preds).issubset({0, 1})


def test_kmeans_threshold_set(binary_data):
    X_train, *_ = binary_data
    model = KMeansAnomalyDetector()
    model.train(X_train, threshold_percentile=90)
    assert model.threshold is not None
    assert model.threshold > 0


def test_kmeans_save_load(binary_data, tmp_path):
    X_train, y_train, X_test, _ = binary_data
    model = KMeansAnomalyDetector()
    model.train(X_train)
    save_path = str(tmp_path / "kmeans.joblib")
    model.save(save_path)

    model2 = KMeansAnomalyDetector()
    model2.load(save_path)
    preds = model2.predict(X_test)
    assert len(preds) == len(X_test)
