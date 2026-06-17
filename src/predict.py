"""
Inference script — run a trained model on new network traffic data.

Usage:
    python src/predict.py --model models/random_forest.joblib --data data/raw/test.csv --type rf
    python src/predict.py --model models/ann_model --data data/raw/test.csv --type ann
"""
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

from data.preprocessor import DataPreprocessor
from models.random_forest_model import RandomForestModel
from models.kmeans_model import KMeansAnomalyDetector
from models.ann_model import ANNModel
from models.cnn_lstm_model import CNNLSTMModel
from utils.logger import setup_logger

logger = setup_logger()
CONFIG = "configs/config.yaml"


def parse_args():
    p = argparse.ArgumentParser(description="Run inference with a trained DDoS detection model")
    p.add_argument("--model", required=True, help="Path to saved model")
    p.add_argument("--data", required=True, help="Path to input CSV file")
    p.add_argument("--type", required=True, choices=["rf", "kmeans", "ann", "cnn_lstm"], help="Model type")
    p.add_argument("--output", default="reports/predictions.csv", help="Output CSV path")
    return p.parse_args()


def load_and_preprocess(data_path: str):
    preprocessor = DataPreprocessor(CONFIG)
    df = preprocessor.load_dataset(data_path)
    df = preprocessor.clean(df)
    target_col = preprocessor.config["data"]["target_column"]
    if target_col in df.columns:
        y_true = preprocessor.label_encoder.fit_transform(df[target_col].str.strip())
        df = df.drop(columns=[target_col])
    else:
        y_true = None
    X = df.values.astype("float32")
    return X, y_true, preprocessor


def main():
    args = parse_args()
    X, y_true, preprocessor = load_and_preprocess(args.data)

    if args.type == "rf":
        model = RandomForestModel(CONFIG)
        model.load(args.model)
        preds = model.predict(X)
    elif args.type == "kmeans":
        model = KMeansAnomalyDetector(CONFIG)
        model.load(args.model)
        preds = model.predict(X)
    elif args.type == "ann":
        model = ANNModel(CONFIG)
        model.load(args.model)
        preds = model.predict(X)
    elif args.type == "cnn_lstm":
        model = CNNLSTMModel(CONFIG)
        model.load(args.model)
        preds = model.predict(X, y_true if y_true is not None else np.zeros(len(X)))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result_df = pd.DataFrame({"prediction": preds})
    if y_true is not None:
        result_df["true_label"] = y_true[: len(preds)]
        correct = (result_df["prediction"] == result_df["true_label"]).mean()
        logger.info("Accuracy on provided data: %.4f", correct)
    result_df.to_csv(out_path, index=False)
    logger.info("Predictions saved to %s", out_path)


if __name__ == "__main__":
    main()
