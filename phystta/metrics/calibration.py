from __future__ import annotations

from typing import Dict

import numpy as np


def _observations(probs, labels):
    probs = np.asarray(probs, dtype=float).reshape(-1)
    labels = np.asarray(labels).reshape(-1)
    if probs.size != labels.size:
        raise ValueError("Every probability must have one binary label")
    if not np.isfinite(probs).all() or ((probs < 0) | (probs > 1)).any():
        raise ValueError("Expected finite probabilities in [0, 1]")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("Expected binary labels")
    return probs, labels


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    curve = reliability_curve(probs, labels, n_bins)
    if not curve["count"].size:
        return float("nan")
    return float(np.average(np.abs(curve["accuracy"] - curve["confidence"]), weights=curve["count"]))


def maximum_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    curve = reliability_curve(probs, labels, n_bins)
    if not curve["count"].size:
        return float("nan")
    return float(np.max(np.abs(curve["accuracy"] - curve["confidence"])))


def brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    probs, labels = _observations(probs, labels)
    return float(np.mean((probs - labels) ** 2)) if probs.size else float("nan")


def reliability_curve(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> Dict[str, np.ndarray]:
    probs, labels = _observations(probs, labels)
    if n_bins < 1:
        raise ValueError("n_bins must be positive")
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
    probs, labels = _observations(probs, labels)
    conf = np.maximum(probs, 1 - probs)
    mask = conf >= threshold
    if mask.sum() == 0:
        return float("nan")
    return expected_calibration_error(probs[mask], labels[mask], n_bins=n_bins)
