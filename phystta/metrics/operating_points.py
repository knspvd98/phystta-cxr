from __future__ import annotations

from typing import Iterable, List, Optional

import numpy as np


def _prepare_binary(y_true: np.ndarray, y_score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true, dtype=float).reshape(-1)
    p = np.asarray(y_score, dtype=float).reshape(-1)
    valid = (y >= 0) & np.isfinite(p)
    y = y[valid]
    p = p[valid]
    if y.size == 0:
        return y.astype(int), p
    return y.astype(int), p


def _confusion_counts(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict:
    y, p = _prepare_binary(y_true, y_score)
    if y.size == 0:
        return {"tp": 0, "tn": 0, "fp": 0, "fn": 0, "n": 0}
    pred = (p >= float(threshold)).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn, "n": int(y.size)}


def operating_point_stats(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
    prevalence: Optional[float] = None,
) -> dict:
    counts = _confusion_counts(y_true, y_score, threshold)
    n = counts["n"]
    if n == 0:
        return {
            "n_valid": 0.0,
            "threshold": float(threshold),
            "sensitivity": float("nan"),
            "specificity": float("nan"),
            "ppv": float("nan"),
            "npv": float("nan"),
            "prevalence": float("nan"),
        }
    tp = counts["tp"]
    tn = counts["tn"]
    fp = counts["fp"]
    fn = counts["fn"]
    pos = tp + fn
    neg = tn + fp
    sens = float(tp / pos) if pos > 0 else float("nan")
    spec = float(tn / neg) if neg > 0 else float("nan")
    prev_obs = float(pos / n)
    prev = prev_obs if prevalence is None else float(np.clip(prevalence, 1e-8, 1.0 - 1e-8))
    if np.isfinite(sens) and np.isfinite(spec):
        den_ppv = sens * prev + (1.0 - spec) * (1.0 - prev)
        den_npv = spec * (1.0 - prev) + (1.0 - sens) * prev
        ppv = float((sens * prev) / den_ppv) if den_ppv > 0 else float("nan")
        npv = float((spec * (1.0 - prev)) / den_npv) if den_npv > 0 else float("nan")
    else:
        ppv = float("nan")
        npv = float("nan")
    return {
        "n_valid": float(n),
        "threshold": float(threshold),
        "sensitivity": sens,
        "specificity": spec,
        "ppv": ppv,
        "npv": npv,
        "prevalence": prev_obs,
    }


def _candidate_thresholds(y_true: np.ndarray, y_score: np.ndarray) -> np.ndarray:
    _, p = _prepare_binary(y_true, y_score)
    if p.size == 0:
        return np.array([0.5], dtype=float)
    uniq = np.unique(p)
    eps = 1e-12
    # include edges: all-positive and all-negative predictions
    cands = np.concatenate(([float(np.max(uniq) + eps)], uniq[::-1], [float(np.min(uniq) - eps)]))
    return cands.astype(float)


def threshold_at_sensitivity(y_true: np.ndarray, y_score: np.ndarray, target_sensitivity: float) -> float:
    target = float(np.clip(target_sensitivity, 0.0, 1.0))
    y, _ = _prepare_binary(y_true, y_score)
    if y.size == 0 or int((y == 1).sum()) == 0:
        return float("nan")
    cands = _candidate_thresholds(y_true, y_score)
    best = cands[-1]
    for th in cands:
        stats = operating_point_stats(y_true, y_score, float(th))
        if np.isfinite(stats["sensitivity"]) and stats["sensitivity"] >= target:
            best = float(th)
            break
    return float(best)


def threshold_at_specificity(y_true: np.ndarray, y_score: np.ndarray, target_specificity: float) -> float:
    target = float(np.clip(target_specificity, 0.0, 1.0))
    y, _ = _prepare_binary(y_true, y_score)
    if y.size == 0 or int((y == 0).sum()) == 0:
        return float("nan")
    cands = _candidate_thresholds(y_true, y_score)
    best = cands[0]
    for th in cands[::-1]:
        stats = operating_point_stats(y_true, y_score, float(th))
        if np.isfinite(stats["specificity"]) and stats["specificity"] >= target:
            best = float(th)
            break
    return float(best)


def report_at_operating_points(
    labels: np.ndarray,
    probs: np.ndarray,
    label_names: Optional[Iterable[str]] = None,
    target_sensitivities: Optional[Iterable[float]] = None,
    target_specificities: Optional[Iterable[float]] = None,
    prevalence: Optional[float] = None,
) -> List[dict]:
    y = np.asarray(labels, dtype=float)
    p = np.asarray(probs, dtype=float)
    if y.shape != p.shape or y.ndim != 2:
        raise ValueError("report_at_operating_points expects labels/probs with shape [N,C]")
    c = y.shape[1]
    names = list(label_names) if label_names is not None else []
    if len(names) < c:
        names = names + [f"class_{i}" for i in range(len(names), c)]
    sens_targets = list(target_sensitivities or [])
    spec_targets = list(target_specificities or [])
    rows: List[dict] = []
    for i in range(c):
        yi = y[:, i]
        pi = p[:, i]
        for t in sens_targets:
            th = threshold_at_sensitivity(yi, pi, float(t))
            stats = operating_point_stats(yi, pi, threshold=th, prevalence=prevalence)
            rows.append(
                {
                    "label": names[i],
                    "target_kind": "sensitivity",
                    "target_value": float(t),
                    **stats,
                }
            )
        for t in spec_targets:
            th = threshold_at_specificity(yi, pi, float(t))
            stats = operating_point_stats(yi, pi, threshold=th, prevalence=prevalence)
            rows.append(
                {
                    "label": names[i],
                    "target_kind": "specificity",
                    "target_value": float(t),
                    **stats,
                }
            )
    return rows
