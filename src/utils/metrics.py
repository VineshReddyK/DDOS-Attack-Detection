import numpy as np
import json
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
)
import logging

logger = logging.getLogger(__name__)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray = None, class_names: list = None) -> dict:
    avg = "binary" if len(np.unique(y_true)) == 2 else "weighted"
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average=avg, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average=avg, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, average=avg, zero_division=0)),
    }
    if y_proba is not None:
        try:
            if avg == "binary":
                metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba[:, 1]))
            else:
                metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted"))
        except Exception:
            pass
    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred).tolist()
    if class_names:
        metrics["classification_report"] = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    return metrics


def save_metrics(metrics: dict, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Metrics saved to %s", path)


def compare_models(results: dict) -> None:
    print("\n" + "=" * 60)
    print(f"{'Model':<20} {'Accuracy':>10} {'F1 Score':>10} {'ROC AUC':>10}")
    print("-" * 60)
    for name, m in results.items():
        acc = m.get("accuracy", 0)
        f1 = m.get("f1_score", 0)
        auc = m.get("roc_auc", float("nan"))
        print(f"{name:<20} {acc:>10.4f} {f1:>10.4f} {auc:>10.4f}")
    print("=" * 60)
