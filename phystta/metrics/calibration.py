from __future__ import annotations

from typing import Dict

import numpy as np


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    probs = np.asarray(probs).reshape(-1)
    labels = np.asarray(labels).reshape(-1)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    confidences = np.maximum(probs, 1 - probs)
    preds = (probs >= 0.5).astype(float)
    n = len(probs)
    ece = 0.0
    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (confidences >= bins[i]) & (confidences <= bins[i + 1])
        else:
            mask = (confidences >= bins[i]) & (confidences < bins[i + 1])
        count = mask.sum()
        if count == 0:
            continue
        acc = (preds[mask] == labels[mask]).mean()
        conf = confidences[mask].mean()
        ece += (count / n) * abs(acc - conf)
    return float(ece)


def maximum_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    probs = np.asarray(probs).reshape(-1)
    labels = np.asarray(labels).reshape(-1)
    mce = 0.0
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    confidences = np.maximum(probs, 1 - probs)
    preds = (probs >= 0.5).astype(float)
    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (confidences >= bins[i]) & (confidences <= bins[i + 1])
        else:
            mask = (confidences >= bins[i]) & (confidences < bins[i + 1])
        if mask.sum() == 0:
            continue
        acc = (preds[mask] == labels[mask]).mean()
        conf = confidences[mask].mean()
        mce = max(mce, abs(acc - conf))
    return float(mce)


def brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean((probs - labels) ** 2))


def reliability_curve(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> Dict[str, np.ndarray]:
    probs = np.asarray(probs).reshape(-1)
    labels = np.asarray(labels).reshape(-1)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    confidences = np.maximum(probs, 1 - probs)
    preds = (probs >= 0.5).astype(float)
    accs = []
    confs = []
    counts = []
    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (confidences >= bins[i]) & (confidences <= bins[i + 1])
        else:
            mask = (confidences >= bins[i]) & (confidences < bins[i + 1])
        if mask.sum() == 0:
            continue
        accs.append((preds[mask] == labels[mask]).mean())
        confs.append(confidences[mask].mean())
        counts.append(mask.sum())
    return {
        "accuracy": np.array(accs),
        "confidence": np.array(confs),
        "count": np.array(counts),
        "bins": bins,
    }


def tail_ece(probs: np.ndarray, labels: np.ndarray, threshold: float = 0.9, n_bins: int = 10) -> float:
    probs = np.asarray(probs).reshape(-1)
    labels = np.asarray(labels).reshape(-1)
    conf = np.maximum(probs, 1 - probs)
    mask = conf >= threshold
    if mask.sum() == 0:
        return float("nan")
    return expected_calibration_error(probs[mask], labels[mask], n_bins=n_bins)
