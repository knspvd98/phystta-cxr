from __future__ import annotations

import numpy as np


def selective_risk(errors: np.ndarray, confidences: np.ndarray, coverage: float) -> float:
    if coverage <= 0:
        return float("nan")
    e = np.asarray(errors, dtype=float).reshape(-1)
    c = np.asarray(confidences, dtype=float).reshape(-1)
    valid = np.isfinite(e) & np.isfinite(c)
    if valid.sum() == 0:
        return float("nan")
    e = e[valid]
    c = c[valid]
    q = float(np.clip(1.0 - coverage, 0.0, 1.0))
    threshold = float(np.quantile(c, q))
    mask = c >= threshold
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(e[mask]))


def coverage_at_risk(errors: np.ndarray, confidences: np.ndarray, target_risk: float) -> float:
    e = np.asarray(errors, dtype=float).reshape(-1)
    c = np.asarray(confidences, dtype=float).reshape(-1)
    valid = np.isfinite(e) & np.isfinite(c)
    if valid.sum() == 0:
        return float("nan")
    e = e[valid]
    c = c[valid]
    thresholds = np.unique(np.quantile(c, np.linspace(0.0, 1.0, 101)))
    best_cov = 0.0
    for t in thresholds:
        mask = c >= t
        if mask.sum() == 0:
            continue
        risk = float(np.mean(e[mask]))
        if risk <= target_risk:
            best_cov = max(best_cov, float(np.mean(mask)))
    return float(best_cov)


def selective_risk_multilabel(
    probs: np.ndarray,
    labels: np.ndarray,
    coverage: float,
    threshold: float | np.ndarray = 0.5,
    cost_fn: float = 1.0,
    cost_fp: float = 1.0,
) -> float:
    """
    Sample-level selective risk for multilabel predictions with ignore-mask support.
    labels < 0 are ignored.
    """
    p = np.asarray(probs, dtype=float)
    y = np.asarray(labels, dtype=float)
    if p.ndim != 2 or y.ndim != 2:
        raise ValueError("probs and labels must be [N,C]")
    if p.shape != y.shape:
        raise ValueError("probs and labels shape mismatch")
    th = np.asarray(threshold, dtype=float)
    if th.ndim == 0:
        th = np.full((p.shape[1],), float(th), dtype=float)
    if th.shape[0] != p.shape[1]:
        raise ValueError("threshold length mismatch")

    valid = y >= 0
    preds = (p >= th[None, :]).astype(int)
    fn = ((preds == 0) & (y == 1) & valid).astype(float)
    fp = ((preds == 1) & (y == 0) & valid).astype(float)
    denom = np.maximum(valid.sum(axis=1), 1)
    sample_error = (cost_fn * fn.sum(axis=1) + cost_fp * fp.sum(axis=1)) / denom
    conf = np.maximum(p, 1.0 - p)
    sample_conf = np.mean(np.where(valid, conf, np.nan), axis=1)
    sample_conf = np.where(np.isfinite(sample_conf), sample_conf, 0.5)
    return selective_risk(sample_error, sample_conf, coverage=coverage)
