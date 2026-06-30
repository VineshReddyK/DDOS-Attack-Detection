import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
import yaml
import logging

logger = logging.getLogger(__name__)


class DataPreprocessor:
    def __init__(self, config_path="configs/config.yaml"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        self.scaler = None
        self.label_encoder = LabelEncoder()
        self.feature_columns = None

    def load_dataset(self, filepath):
        path = Path(filepath)
        if path.suffix == ".csv":
            df = pd.read_csv(filepath)
        elif path.suffix == ".parquet":
            df = pd.read_parquet(filepath)
        else:
            raise ValueError(f"unsupported file format: {path.suffix}")
        logger.info("loaded %d rows, %d columns from %s", len(df), len(df.columns), filepath)
        return df

    def load_and_merge(self, data_dir):
        # merge all CSVs in a folder - useful when CIC-DDoS2019 is split by attack type
        data_path = Path(data_dir)
        csv_files = list(data_path.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"no CSV files found in {data_dir}")
        frames = [pd.read_csv(f) for f in csv_files]
        df = pd.concat(frames, ignore_index=True)
        logger.info("merged %d files → %d rows total", len(csv_files), len(df))
        return df

    def clean(self, df):
        # drop columns we don't use (IPs, timestamps, etc.)
        drop_cols = [c for c in self.config["features"]["drop_columns"] if c in df.columns]
        df = df.drop(columns=drop_cols)

        # CIC dataset column names sometimes have leading spaces
        df.columns = df.columns.str.strip()

        # replace inf values that come from division-by-zero in CICFlowMeter
        df = df.replace([np.inf, -np.inf], np.nan)

        strategy = self.config["features"]["fill_na_strategy"]
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if strategy == "median":
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        elif strategy == "mean":
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        else:
            df[numeric_cols] = df[numeric_cols].fillna(0)

        before = len(df)
        df = df.drop_duplicates()
        logger.info("dropped %d duplicate rows", before - len(df))
        return df

    def encode_labels(self, df):
        target_col = self.config["data"]["target_column"]
        # strip whitespace from labels - CIC dataset has some inconsistency here
        y = self.label_encoder.fit_transform(df[target_col].str.strip())
        df = df.drop(columns=[target_col])
        return df, y

    def scale_features(self, X_train, X_val, X_test):
        method = self.config["features"]["scale_method"]
        # fit only on train, transform val/test separately to avoid leakage
        self.scaler = StandardScaler() if method == "standard" else MinMaxScaler()
        X_train = self.scaler.fit_transform(X_train)
        X_val = self.scaler.transform(X_val)
        X_test = self.scaler.transform(X_test)
        return X_train, X_val, X_test

    def split(self, X, y):
        cfg = self.config["data"]
        rs = cfg["random_state"]
        test_size = cfg["test_split"]
        # val_split is a fraction of the original dataset, so we need to adjust
        # for the fact that test is already taken out
        val_frac = cfg["val_split"] / (1 - test_size)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=rs, stratify=y
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=val_frac, random_state=rs, stratify=y_train
        )
        return X_train, X_val, X_test, y_train, y_val, y_test

    def prepare(self, filepath):
        df = self.load_dataset(filepath)
        df = self.clean(df)
        df, y = self.encode_labels(df)

        self.feature_columns = list(df.columns)
        X = df.values.astype(np.float32)

        X_train, X_val, X_test, y_train, y_val, y_test = self.split(X, y)
        X_train, X_val, X_test = self.scale_features(X_train, X_val, X_test)

        logger.info("train: %d  val: %d  test: %d", len(X_train), len(X_val), len(X_test))
        return X_train, X_val, X_test, y_train, y_val, y_test

    def get_class_names(self):
        return list(self.label_encoder.classes_)
