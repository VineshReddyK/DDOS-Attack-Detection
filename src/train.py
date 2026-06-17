"""
Main training pipeline — trains all four models and saves results.

Usage:
    python src/train.py --data data/raw/CIC-DDoS2019.csv
    python src/train.py --data data/raw/ --merge          # merge all CSVs in folder
"""
import argparse
import json
from pathlib import Path

from data.preprocessor import DataPreprocessor
from models.random_forest_model import RandomForestModel
from models.kmeans_model import KMeansAnomalyDetector
from models.ann_model import ANNModel
from models.cnn_lstm_model import CNNLSTMModel
from utils.metrics import compute_metrics, save_metrics, compare_models
from utils.visualizer import (
    plot_confusion_matrix, plot_training_history,
    plot_feature_importance, plot_model_comparison, plot_anomaly_scores,
)
from utils.logger import setup_logger

import numpy as np

logger = setup_logger(log_file="reports/training.log")
CONFIG = "configs/config.yaml"
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")
FIGURES_DIR = REPORTS_DIR / "figures"


def parse_args():
    p = argparse.ArgumentParser(description="Train DDoS detection models")
    p.add_argument("--data", required=True, help="Path to CSV file or directory of CSVs")
    p.add_argument("--merge", action="store_true", help="Merge all CSVs in --data directory")
    p.add_argument("--skip-rf", action="store_true")
    p.add_argument("--skip-kmeans", action="store_true")
    p.add_argument("--skip-ann", action="store_true")
    p.add_argument("--skip-cnn-lstm", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    preprocessor = DataPreprocessor(CONFIG)

    if args.merge:
        df = preprocessor.load_and_merge(args.data)
        df = preprocessor.clean(df)
        df, y = preprocessor.encode_labels(df)
        feature_cols = list(df.columns)
        X = df.values.astype("float32")
        X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split(X, y)
        X_train, X_val, X_test = preprocessor.scale_features(X_train, X_val, X_test)
    else:
        X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.prepare(args.data)
        feature_cols = preprocessor.feature_columns

    class_names = preprocessor.get_class_names()
    num_classes = len(class_names)
    n_features = X_train.shape[1]
    all_results = {}

    # Random Forest
    if not args.skip_rf:
        logger.info("=== Random Forest ===")
        rf = RandomForestModel(CONFIG)
        rf.train(X_train, y_train)
        rf_results = rf.evaluate(X_test, y_test, class_names)
        rf_metrics = compute_metrics(y_test, rf.predict(X_test), class_names=class_names)
        save_metrics(rf_metrics, str(REPORTS_DIR / "rf_metrics.json"))
        rf.save(str(MODELS_DIR / "random_forest.joblib"))
        if feature_cols:
            importances = rf.feature_importances(feature_cols)
            plot_feature_importance(importances, save_path=str(FIGURES_DIR / "rf_feature_importance.png"))
        plot_confusion_matrix(
            np.array(rf_results["confusion_matrix"]), class_names,
            "Random Forest Confusion Matrix",
            save_path=str(FIGURES_DIR / "rf_confusion_matrix.png"),
        )
        all_results["Random Forest"] = rf_metrics

    # KMeans
    if not args.skip_kmeans:
        logger.info("=== KMeans Anomaly Detection ===")
        km = KMeansAnomalyDetector(CONFIG)
        km.train(X_train)
        km_eval = km.evaluate(X_test)
        scores = km.anomaly_scores(X_test)
        plot_anomaly_scores(scores, km.threshold, save_path=str(FIGURES_DIR / "kmeans_anomaly_scores.png"))
        save_metrics(km_eval, str(REPORTS_DIR / "kmeans_metrics.json"))
        km.save(str(MODELS_DIR / "kmeans.joblib"))

    # ANN
    if not args.skip_ann:
        logger.info("=== Artificial Neural Network ===")
        ann = ANNModel(CONFIG)
        ann.build(n_features, num_classes)
        ann.train(X_train, y_train, X_val, y_val)
        ann_results = ann.evaluate(X_test, y_test, class_names)
        ann_metrics = compute_metrics(y_test, ann.predict(X_test), class_names=class_names)
        save_metrics(ann_metrics, str(REPORTS_DIR / "ann_metrics.json"))
        ann.save(str(MODELS_DIR / "ann_model"))
        plot_training_history(ann.history, "ANN", save_path=str(FIGURES_DIR / "ann_training_history.png"))
        plot_confusion_matrix(
            np.array(ann_results["confusion_matrix"]), class_names,
            "ANN Confusion Matrix",
            save_path=str(FIGURES_DIR / "ann_confusion_matrix.png"),
        )
        all_results["ANN"] = ann_metrics

    # CNN-LSTM
    if not args.skip_cnn_lstm:
        logger.info("=== CNN-LSTM ===")
        cnn_lstm = CNNLSTMModel(CONFIG)
        cnn_lstm.build(n_features, num_classes)
        cnn_lstm.train(X_train, y_train, X_val, y_val)
        cl_results = cnn_lstm.evaluate(X_test, y_test, class_names)
        cl_metrics = compute_metrics(
            y_test[cnn_lstm.seq_len:],
            cnn_lstm.predict(X_test, y_test),
            class_names=class_names,
        )
        save_metrics(cl_metrics, str(REPORTS_DIR / "cnn_lstm_metrics.json"))
        cnn_lstm.save(str(MODELS_DIR / "cnn_lstm_model"))
        plot_training_history(cnn_lstm.history, "CNN-LSTM", save_path=str(FIGURES_DIR / "cnn_lstm_training_history.png"))
        plot_confusion_matrix(
            np.array(cl_results["confusion_matrix"]), class_names,
            "CNN-LSTM Confusion Matrix",
            save_path=str(FIGURES_DIR / "cnn_lstm_confusion_matrix.png"),
        )
        all_results["CNN-LSTM"] = cl_metrics

    if len(all_results) > 1:
        compare_models(all_results)
        plot_model_comparison(all_results, save_path=str(FIGURES_DIR / "model_comparison.png"))
        save_metrics(all_results, str(REPORTS_DIR / "all_models_comparison.json"))

    logger.info("Training pipeline complete. Results saved to reports/")


if __name__ == "__main__":
    main()
