import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def plot_confusion_matrix(cm: np.ndarray, class_names: list, title: str, save_path: str = None) -> None:
    fig, ax = plt.subplots(figsize=(max(8, len(class_names)), max(6, len(class_names) - 1)))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names, ax=ax,
    )
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved confusion matrix to %s", save_path)
    plt.show()


def plot_training_history(history, title: str = "Training History", save_path: str = None) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(history.history["loss"], label="Train Loss")
    axes[0].plot(history.history["val_loss"], label="Val Loss")
    axes[0].set_title(f"{title} - Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(history.history["accuracy"], label="Train Acc")
    axes[1].plot(history.history["val_accuracy"], label="Val Acc")
    axes[1].set_title(f"{title} - Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved training history to %s", save_path)
    plt.show()


def plot_feature_importance(importances: dict, top_n: int = 20, save_path: str = None) -> None:
    items = list(importances.items())[:top_n]
    features, scores = zip(*items)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=list(scores), y=list(features), palette="viridis", ax=ax)
    ax.set_title(f"Top {top_n} Feature Importances (Random Forest)", fontweight="bold")
    ax.set_xlabel("Importance Score")
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved feature importance plot to %s", save_path)
    plt.show()


def plot_model_comparison(results: dict, save_path: str = None) -> None:
    models = list(results.keys())
    metrics_to_plot = ["accuracy", "precision", "recall", "f1_score"]
    values = {m: [results[model].get(m, 0) for model in models] for m in metrics_to_plot}

    x = np.arange(len(models))
    width = 0.2
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, (metric, vals) in enumerate(values.items()):
        ax.bar(x + i * width, vals, width, label=metric.replace("_", " ").title())

    ax.set_title("Model Comparison", fontsize=14, fontweight="bold")
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(models, rotation=15)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.legend()
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved model comparison to %s", save_path)
    plt.show()


def plot_anomaly_scores(scores: np.ndarray, threshold: float, save_path: str = None) -> None:
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(scores, alpha=0.6, label="Anomaly Score")
    ax.axhline(threshold, color="red", linestyle="--", label=f"Threshold ({threshold:.4f})")
    ax.fill_between(range(len(scores)), scores, threshold, where=(scores > threshold), alpha=0.3, color="red", label="Anomalies")
    ax.set_title("KMeans Anomaly Detection", fontweight="bold")
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Distance to Centroid")
    ax.legend()
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
