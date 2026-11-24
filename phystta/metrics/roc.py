from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def auroc_per_class(labels: np.ndarray, probs: np.ndarray) -> Dict[int, float]:
    try:
        from sklearn.metrics import roc_auc_score
    except Exception as e:
        raise RuntimeError("scikit-learn is required for AUROC") from e
    results = {}
    for i in range(probs.shape[1]):
        y = labels[:, i]
        p = probs[:, i]
        valid = y >= 0
        if valid.sum() < 2:
            continue
        yv = y[valid]
        pv = p[valid]
        if np.unique(yv).size < 2:
            results[i] = float("nan")
            continue
        try:
            results[i] = float(roc_auc_score(yv, pv))
        except Exception:
            results[i] = float("nan")
    return results


def auprc_per_class(labels: np.ndarray, probs: np.ndarray) -> Dict[int, float]:
    try:
        from sklearn.metrics import average_precision_score
    except Exception as e:
        raise RuntimeError("scikit-learn is required for AUPRC") from e
    results = {}
    for i in range(probs.shape[1]):
        y = labels[:, i]
        p = probs[:, i]
        valid = y >= 0
        if valid.sum() < 2:
            continue
        try:
            results[i] = float(average_precision_score(y[valid], p[valid]))
        except Exception:
            results[i] = float("nan")
    return results


def macro_average(metrics: Dict[int, float]) -> float:
    vals = [v for v in metrics.values() if not np.isnan(v)]
    if not vals:
        return float("nan")
    return float(np.mean(vals))
