import numpy as np
from sklearn.metrics import (roc_auc_score, average_precision_score, precision_score,
                             recall_score, f1_score, roc_curve)

def metrics_at(y, proba, thr=0.5):
    pred = (proba >= thr).astype(int)
    auc = roc_auc_score(y, proba)
    fpr, tpr, _ = roc_curve(y, proba)
    return {
        "AUC": auc, "Gini": 2 * auc - 1, "KS": float(np.max(tpr - fpr)),
        "PR-AUC": average_precision_score(y, proba),
        "Precision": precision_score(y, pred, zero_division=0),
        "Recall": recall_score(y, pred),
        "F1": f1_score(y, pred),
    }