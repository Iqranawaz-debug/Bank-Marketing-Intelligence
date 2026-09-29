"""Metrics and curves computed from real predictions."""
import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_recall_curve, precision_score,
                             recall_score, roc_auc_score, roc_curve)


def metrics(y, probability, threshold=.5):
    prediction = np.asarray(probability) >= threshold
    return {"Accuracy": float(accuracy_score(y, prediction)),
            "Precision": float(precision_score(y, prediction, zero_division=0)),
            "Recall": float(recall_score(y, prediction, zero_division=0)),
            "F1": float(f1_score(y, prediction, zero_division=0)),
            "ROC-AUC": float(roc_auc_score(y, probability)),
            "Average precision": float(average_precision_score(y, probability))}


def curves(y, probability, threshold):
    fpr, tpr, _ = roc_curve(y, probability)
    precision, recall, _ = precision_recall_curve(y, probability)
    return {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "precision": precision.tolist(),
            "recall": recall.tolist(),
            "confusion_matrix": confusion_matrix(y, np.asarray(probability) >= threshold).tolist()}
