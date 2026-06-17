import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.preprocessor import DataPreprocessor  # noqa: E402


@pytest.fixture
def sample_df():
    np.random.seed(42)
    n = 500
    return pd.DataFrame({
        "Flow ID": [f"flow_{i}" for i in range(n)],
        "Source IP": ["192.168.1.1"] * n,
        "Destination IP": ["10.0.0.1"] * n,
        "Timestamp": pd.date_range("2023-01-01", periods=n, freq="s"),
        "Flow Duration": np.random.randint(1, 10000, n),
        "Total Fwd Packets": np.random.randint(1, 100, n),
        "Total Backward Packets": np.random.randint(1, 100, n),
        "Flow Bytes/s": np.random.uniform(0, 1e6, n),
        "Flow Packets/s": np.random.uniform(0, 1000, n),
        "Label": np.random.choice(["BENIGN", "DDoS", "DoS GoldenEye"], n),
    })


@pytest.fixture
def preprocessor():
    return DataPreprocessor("configs/config.yaml")


def test_clean_drops_id_columns(preprocessor, sample_df):
    cleaned = preprocessor.clean(sample_df.copy())
    assert "Flow ID" not in cleaned.columns
    assert "Source IP" not in cleaned.columns


def test_clean_removes_inf(preprocessor, sample_df):
    sample_df.loc[0, "Flow Bytes/s"] = np.inf
    cleaned = preprocessor.clean(sample_df.copy())
    assert not np.isinf(cleaned.select_dtypes(include=[np.number]).values).any()


def test_encode_labels(preprocessor, sample_df):
    cleaned = preprocessor.clean(sample_df.copy())
    df_encoded, y = preprocessor.encode_labels(cleaned)
    assert "Label" not in df_encoded.columns
    assert len(y) == len(df_encoded)
    assert set(y).issubset(set(range(len(preprocessor.label_encoder.classes_))))


def test_split_sizes(preprocessor):
    X = np.random.rand(1000, 10).astype("float32")
    y = np.random.randint(0, 3, 1000)
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split(X, y)
    assert len(X_train) + len(X_val) + len(X_test) == 1000
    assert abs(len(X_test) / 1000 - 0.1) < 0.02


def test_scale_features_no_data_leakage(preprocessor):
    X_train = np.random.rand(800, 10).astype("float32")
    X_val = np.random.rand(100, 10).astype("float32")
    X_test = np.random.rand(100, 10).astype("float32")
    Xt, Xv, Xs = preprocessor.scale_features(X_train, X_val, X_test)
    # Train set should be approximately zero-mean after StandardScaler
    assert abs(Xt.mean()) < 0.1
