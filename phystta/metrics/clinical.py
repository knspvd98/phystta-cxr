from __future__ import annotations

from typing import Iterable, Optional

import numpy as np

from ..safety.policy_harm import _resolve_class_costs


def clinical_risk(
    probs: np.ndarray,
    labels: np.ndarray,
    cost_fn: float = 10.0,
    cost_fp: float = 2.0,
    class_cost_fn: Optional[Iterable[float]] = None,
    class_cost_fp: Optional[Iterable[float]] = None,
) -> float:
    return clinical_risk_at_threshold(
        probs,
        labels,
        threshold=0.5,
        cost_fn=cost_fn,
        cost_fp=cost_fp,
        class_cost_fn=class_cost_fn,
        class_cost_fp=class_cost_fp,
    )


def clinical_risk_at_threshold(
    probs: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    cost_fn: float = 10.0,
    cost_fp: float = 2.0,
    class_cost_fn: Optional[Iterable[float]] = None,
    class_cost_fp: Optional[Iterable[float]] = None,
) -> float:
    p = np.asarray(probs, dtype=float)
    y = np.asarray(labels, dtype=float)
    if p.shape != y.shape:
        raise ValueError("probs and labels shape mismatch")
    preds = (p >= threshold).astype(int)
    valid = y >= 0
    if p.ndim == 1:
        p = p.reshape(-1, 1)
        y = y.reshape(-1, 1)
        preds = preds.reshape(-1, 1)
        valid = valid.reshape(-1, 1)
    cost_fn_vec = _resolve_class_costs(p.shape[1], cost_fn, class_cost_fn, "class_cost_fn").reshape(1, -1)
    cost_fp_vec = _resolve_class_costs(p.shape[1], cost_fp, class_cost_fp, "class_cost_fp").reshape(1, -1)
    fn = ((y == 1) & (preds == 0) & valid).astype(float)
    fp = ((y == 0) & (preds == 1) & valid).astype(float)
    total = valid.sum()
    if total == 0:
        return float("nan")
    cost = cost_fn_vec * fn + cost_fp_vec * fp
    return float(cost.sum() / total)


def cost_matrix(cost_fn: float = 10.0, cost_fp: float = 2.0) -> np.ndarray:
    return np.array([[0.0, cost_fp], [cost_fn, 0.0]])


def expected_cost_from_probs(
    probs: np.ndarray,
    labels: np.ndarray,
    cost_fn: float = 10.0,
    cost_fp: float = 2.0,
    class_cost_fn: Optional[Iterable[float]] = None,
    class_cost_fp: Optional[Iterable[float]] = None,
) -> float:
    p = np.asarray(probs, dtype=float)
    y = np.asarray(labels, dtype=float)
    if p.shape != y.shape:
        raise ValueError("probs and labels shape mismatch")
    valid = y >= 0
    preds = (p >= 0.5).astype(int)
    if p.ndim == 1:
        p = p.reshape(-1, 1)
        y = y.reshape(-1, 1)
        valid = valid.reshape(-1, 1)
        preds = preds.reshape(-1, 1)
    if valid.sum() == 0:
        return float("nan")
    cost_fn_vec = _resolve_class_costs(p.shape[1], cost_fn, class_cost_fn, "class_cost_fn").reshape(1, -1)
    cost_fp_vec = _resolve_class_costs(p.shape[1], cost_fp, class_cost_fp, "class_cost_fp").reshape(1, -1)
    fn = ((y == 1) & (preds == 0) & valid).astype(float)
    fp = ((y == 0) & (preds == 1) & valid).astype(float)
    cost = cost_fn_vec * fn + cost_fp_vec * fp
    return float(cost[valid].mean())


def risk_reduction(baseline: float, adapted: float) -> float:
    if baseline <= 0:
        return 0.0
    return float((baseline - adapted) / baseline)


def cost_sensitivity_analysis(probs: np.ndarray, labels: np.ndarray, cost_ratios=None, threshold: float = 0.5):
    if cost_ratios is None:
        cost_ratios = [1, 2, 5, 10, 20, 50, 100]
    results = []
    for ratio in cost_ratios:
        risk = clinical_risk_at_threshold(probs, labels, threshold=threshold, cost_fn=float(ratio), cost_fp=1.0)
        results.append({"cost_ratio": float(ratio), "clinical_risk": float(risk)})
    return results


def optimal_thresholds(labels: np.ndarray, probs: np.ndarray, cost_fn: float = 10.0, cost_fp: float = 1.0) -> np.ndarray:
    thresholds = []
    for i in range(probs.shape[1]):
        y = labels[:, i]
        p = probs[:, i]
        valid = y >= 0
        if valid.sum() == 0:
            thresholds.append(0.5)
            continue
        yv = y[valid]
        pv = p[valid]
        grid = np.linspace(0.0, 1.0, 101)
        best_th = 0.5
        best_cost = float("inf")
        for th in grid:
            preds = (pv >= th).astype(int)
            fn = ((preds == 0) & (yv == 1)).astype(float)
            fp = ((preds == 1) & (yv == 0)).astype(float)
            cost = cost_fn * fn.mean() + cost_fp * fp.mean()
            if cost < best_cost:
                best_cost = cost
                best_th = float(th)
        thresholds.append(best_th)
    return np.array(thresholds, dtype=float)
