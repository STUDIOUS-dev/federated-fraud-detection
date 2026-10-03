from __future__ import annotations
import numpy as np
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, confusion_matrix


def evaluate_probs(y_true: np.ndarray, probs: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    pred = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true.astype(int), pred, labels=[0, 1]).ravel()
    if y_true.sum() == 0:
        pr_auc = float('nan')
    else:
        pr_auc = float(average_precision_score(y_true, probs))
    return {
        "pr_auc": pr_auc,
        "f1": float(f1_score(y_true, pred, zero_division=np.nan)),
        "precision": float(precision_score(y_true, pred, zero_division=np.nan)),
        "recall": float(recall_score(y_true, pred, zero_division=np.nan)),
        "false_positive_rate": float(fp / max(fp + tn, 1)),
    }
