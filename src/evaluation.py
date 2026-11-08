from __future__ import annotations
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score, f1_score, precision_score, recall_score, confusion_matrix


def evaluate_probs(y_true: np.ndarray, probs: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    pred = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true.astype(int), pred, labels=[0, 1]).ravel()
    if y_true.sum() == 0:
        pr_auc = float('nan')
    else:
        pr_auc = float(average_precision_score(y_true, probs))
    n_pos, n_neg = int(y_true.sum()), int(len(y_true) - y_true.sum())
    if n_pos == 0 or n_neg == 0:
        roc_auc = recall_at_fpr1 = float('nan')
    else:
        roc_auc = float(roc_auc_score(y_true, probs))
        fpr, tpr, _ = roc_curve(y_true, probs)
        # Threshold-free operating point: recall when at most 1% of legitimate transactions are flagged.
        recall_at_fpr1 = float(tpr[fpr <= 0.01].max())
    return {
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "recall_at_fpr1": recall_at_fpr1,
        "f1": float(f1_score(y_true, pred, zero_division=np.nan)),
        "precision": float(precision_score(y_true, pred, zero_division=np.nan)),
        "recall": float(recall_score(y_true, pred, zero_division=np.nan)),
        "false_positive_rate": float(fp / max(fp + tn, 1)),
    }
