from __future__ import annotations

from typing import Dict

import numpy as np


def risk_coverage_curve(errors: np.ndarray, confidences: np.ndarray, points: int = 50) -> Dict[str, np.ndarray]:
    e = np.asarray(errors, dtype=float).reshape(-1)
    c = np.asarray(confidences, dtype=float).reshape(-1)
    valid = np.isfinite(e) & np.isfinite(c)
    if valid.sum() == 0:
        nan = np.full((points,), np.nan, dtype=float)
        z = np.zeros((points,), dtype=float)
        return {"thresholds": np.linspace(0.0, 1.0, points), "risk": nan, "coverage": z}
    e = e[valid]
    c = c[valid]
    thresholds = np.quantile(c, np.linspace(0.0, 1.0, points))
    risks = []
    coverages = []
    for t in thresholds:
        mask = c >= t
        if mask.sum() == 0:
            risks.append(np.nan)
            coverages.append(0.0)
        else:
            risks.append(float(np.mean(e[mask])))
            coverages.append(float(np.mean(mask)))
    return {"thresholds": thresholds, "risk": np.array(risks), "coverage": np.array(coverages)}


def aurc(errors: np.ndarray, confidences: np.ndarray, points: int = 50) -> float:
    curve = risk_coverage_curve(errors, confidences, points=points)
    risk = curve["risk"]
    coverage = curve["coverage"]
    if np.all(np.isnan(risk)):
        return float("nan")
    risk = np.nan_to_num(risk, nan=float(np.nanmax(risk)))
    order = np.argsort(coverage)
    cov_s = coverage[order]
    risk_s = risk[order]
    return float(np.trapezoid(risk_s, cov_s))


def risk_coverage_inputs_multilabel(
    probs: np.ndarray,
    labels: np.ndarray,
    threshold: float | np.ndarray = 0.5,
    cost_fn: float = 1.0,
    cost_fp: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
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
    count = valid.sum(axis=1)
    observed = count > 0
    harm = cost_fn * fn.sum(axis=1) + cost_fp * fp.sum(axis=1)
    confidence = np.where(valid, np.maximum(p, 1.0 - p), 0.0).sum(axis=1)
    # Keep row alignment with patient IDs for cluster bootstrap resampling.
    sample_error = np.divide(harm, count, out=np.full(count.shape, np.nan), where=observed)
    sample_conf = np.divide(confidence, count, out=np.full(count.shape, np.nan), where=observed)
    return sample_error, sample_conf


def risk_coverage_curve_multilabel(
    probs: np.ndarray,
    labels: np.ndarray,
    threshold: float | np.ndarray = 0.5,
    points: int = 50,
    cost_fn: float = 1.0,
    cost_fp: float = 1.0,
) -> Dict[str, np.ndarray]:
    sample_error, sample_conf = risk_coverage_inputs_multilabel(
        probs,
        labels,
        threshold=threshold,
        cost_fn=cost_fn,
        cost_fp=cost_fp,
    )
    return risk_coverage_curve(sample_error, sample_conf, points=points)
