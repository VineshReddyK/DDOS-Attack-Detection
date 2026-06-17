import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
import yaml
import logging

logger = logging.getLogger(__name__)


class DataPreprocessor:
    def __init__(self, config_path: str = "configs/config.yaml"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        self.scaler = None
        self.label_encoder = LabelEncoder()
        self.feature_columns = None

    def load_dataset(self, filepath: str) -> pd.DataFrame:
        path = Path(filepath)
        if path.suffix == ".csv":
            df = pd.read_csv(filepath)
        elif path.suffix == ".parquet":
            df = pd.read_parquet(filepath)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
        logger.info("Loaded dataset: %s rows, %s columns", len(df), len(df.columns))
        return df

    def load_and_merge(self, data_dir: str) -> pd.DataFrame:
        data_path = Path(data_dir)
        csv_files = list(data_path.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {data_dir}")
        frames = [pd.read_csv(f) for f in csv_files]
        df = pd.concat(frames, ignore_index=True)
        logger.info("Merged %d files: %d total rows", len(csv_files), len(df))
        return df

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        drop_cols = [c for c in self.config["features"]["drop_columns"] if c in df.columns]
        df = df.drop(columns=drop_cols)

        df.columns = df.columns.str.strip()
        df = df.replace([np.inf, -np.inf], np.nan)

        strategy = self.config["features"]["fill_na_strategy"]
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if strategy == "median":
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        elif strategy == "mean":
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        else:
            df[numeric_cols] = df[numeric_cols].fillna(0)

        initial = len(df)
        df = df.drop_duplicates()
        logger.info("Removed %d duplicate rows", initial - len(df))
        return df

    def encode_labels(self, df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
        target_col = self.config["data"]["target_column"]
        y = self.label_encoder.fit_transform(df[target_col].str.strip())
        df = df.drop(columns=[target_col])
        return df, y

    def scale_features(self, X_train, X_val, X_test):
        method = self.config["features"]["scale_method"]
        self.scaler = StandardScaler() if method == "standard" else MinMaxScaler()
        X_train = self.scaler.fit_transform(X_train)
        X_val = self.scaler.transform(X_val)
        X_test = self.scaler.transform(X_test)
        return X_train, X_val, X_test

    def split(self, X, y):
        cfg = self.config["data"]
        rs = cfg["random_state"]
        test_size = cfg["test_split"]
        val_size = cfg["val_split"] / (1 - test_size)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=rs, stratify=y)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size, random_state=rs, stratify=y_train)
        return X_train, X_val, X_test, y_train, y_val, y_test

    def prepare(self, filepath: str):
        df = self.load_dataset(filepath)
        df = self.clean(df)
        df, y = self.encode_labels(df)

        self.feature_columns = list(df.columns)
        X = df.values.astype(np.float32)

        X_train, X_val, X_test, y_train, y_val, y_test = self.split(X, y)
        X_train, X_val, X_test = self.scale_features(X_train, X_val, X_test)

        logger.info(
            "Split sizes — train: %d, val: %d, test: %d",
            len(X_train), len(X_val), len(X_test),
        )
        return X_train, X_val, X_test, y_train, y_val, y_test

    def get_class_names(self) -> list[str]:
        return list(self.label_encoder.classes_)
