from typing import Any, Dict, Union
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report
)
from sklearn.preprocessing import label_binarize

def _safe_auc(y_true, y_score) -> Union[float, None]:
    """
    Compute AUC safely for binary or multiclass.
    Returns None if not computable (e.g., only hard labels provided).
    """
    try:
        y_true = np.asarray(y_true)
        y_score = np.asarray(y_score)

        # Binary case
        if y_score.ndim == 1 or (y_score.ndim == 2 and y_score.shape[1] == 1):
            # Need a continuous score (probability or decision function)
            if np.unique(y_score).size <= 2:
                return None  # looks like hard labels; AUC not meaningful
            return float(roc_auc_score(y_true, y_score))

        # Multiclass case (n_classes > 2) expects shape (n_samples, n_classes)
        if y_score.ndim == 2 and y_score.shape[1] > 1:
            classes = np.unique(y_true)
            # Binarize y_true to one-hot to match y_score columns
            y_true_bin = label_binarize(y_true, classes=classes)
            if y_true_bin.shape[1] == 1:  # degenerate after binarization
                return None
            return float(roc_auc_score(y_true_bin, y_score, average="macro", multi_class="ovr"))

        return None
    except Exception:
        return None

def evaluate_scores(
    y_true: Union[np.ndarray, list],
    y_score: Union[np.ndarray, list],
    threshold: float = 0.5
) -> Dict[str, Any]:
    """
    - Works for binary and multiclass.
    - y_score can be probabilities/scores (binary: shape (n,), multiclass: shape (n, C))
      or hard labels (we'll use them directly for metrics that allow it).
    - Returns AUC when possible; otherwise None.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    # Derive predicted labels
    if y_score.ndim == 1:
        # Binary: threshold scores -> labels
        if np.unique(y_score).size <= 2:
            y_pred = y_score.astype(int)  # already hard labels
        else:
            y_pred = (y_score >= threshold).astype(int)
    elif y_score.ndim == 2:
        # Multiclass: take argmax over class scores/probabilities
        y_pred = np.argmax(y_score, axis=1)
    else:
        raise ValueError("y_score must be 1D (binary) or 2D (multiclass)")

    # Choose averaging strategy
    average = "binary" if np.unique(y_true).size == 2 else "macro"

    # Metrics
    metrics: Dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average=average, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average=average, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average=average, zero_division=0)),
        "auc": _safe_auc(y_true, y_score),
        # classification_report needs string output; set output_dict=True if you prefer a dict
        "classification_report": classification_report(y_true, y_pred)
    }
    return metrics
