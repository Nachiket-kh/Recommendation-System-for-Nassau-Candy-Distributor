"""Evaluation helpers for multiclass shipping models."""
from __future__ import annotations
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

def classification_metrics(y_true, y_pred, probabilities=None) -> dict:
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    result = {"accuracy": float(accuracy_score(y_true, y_pred)), "precision_weighted": float(precision), "recall_weighted": float(recall), "f1_weighted": float(f1), "confusion_matrix": confusion_matrix(y_true, y_pred).tolist()}
    if probabilities is not None:
        result["top_1_accuracy"] = result["accuracy"]
        result["top_3_accuracy"] = float(np.mean([target in classes for target, classes in zip(y_true, np.array(probabilities))])) if False else None
    return result
