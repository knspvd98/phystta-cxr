from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from .roc import auroc_per_class, auprc_per_class
from .calibration import expected_calibration_error


def _group_masks(groups: np.ndarray) -> Dict[str, np.ndarray]:
    masks = {}
    for g in np.unique(groups):
        masks[str(g)] = groups == g
    return masks


def _valid_group(mask: np.ndarray, labels: np.ndarray, min_group_size: int) -> bool:
    if mask.sum() < min_group_size:
        return False
    if labels.ndim == 1:
        y = labels[mask]
        return len(np.unique(y)) > 1
    return True


def worst_group_auc(labels: np.ndarray, probs: np.ndarray, groups: np.ndarray, min_group_size: int = 50) -> Dict[str, float]:
    masks = _group_masks(groups)
    worst_auroc = {}
    worst_auprc = {}
    gap_auroc = {}
    gap_auprc = {}
    for i in range(probs.shape[1]):
        group_scores = {}
        group_scores_pr = {}
        for g, mask in masks.items():
            y = labels[:, i]
            p = probs[:, i]
            valid = (y >= 0) & mask
            if valid.sum() < min_group_size:
                continue
            if len(np.unique(y[valid])) < 2:
                continue
            try:
                group_scores[g] = auroc_per_class(y[valid].reshape(-1, 1), p[valid].reshape(-1, 1))[0]
                group_scores_pr[g] = auprc_per_class(y[valid].reshape(-1, 1), p[valid].reshape(-1, 1))[0]
            except Exception:
                continue
        if group_scores:
            vals = list(group_scores.values())
            worst_auroc[i] = float(np.min(vals))
            gap_auroc[i] = float(np.max(vals) - np.min(vals))
        if group_scores_pr:
            vals = list(group_scores_pr.values())
            worst_auprc[i] = float(np.min(vals))
            gap_auprc[i] = float(np.max(vals) - np.min(vals))
    def _macro(d: Dict[int, float]) -> float:
        if not d:
            return float("nan")
        return float(np.mean(list(d.values())))
    return {
        "worst_group_auroc_macro": _macro(worst_auroc),
        "worst_group_auprc_macro": _macro(worst_auprc),
        "auroc_gap_macro": _macro(gap_auroc),
        "auprc_gap_macro": _macro(gap_auprc),
    }


def underdiagnosis_gap(labels: np.ndarray, probs: np.ndarray, groups: np.ndarray, threshold: float | np.ndarray = 0.5, min_group_size: int = 50) -> Dict[str, float]:
    masks = _group_masks(groups)
    gaps = {}
    for i in range(probs.shape[1]):
        th_i = float(threshold[i]) if isinstance(threshold, np.ndarray) else float(threshold)
        fnr_by_group = []
        for g, mask in masks.items():
            y = labels[:, i]
            p = probs[:, i]
            valid = (y >= 0) & mask
            if valid.sum() < min_group_size:
                continue
            if len(np.unique(y[valid])) < 2:
                continue
            preds = (p[valid] >= th_i).astype(int)
            fn = ((preds == 0) & (y[valid] == 1)).sum()
            tp = ((preds == 1) & (y[valid] == 1)).sum()
            fnr = fn / (fn + tp + 1e-8)
            fnr_by_group.append(fnr)
        if fnr_by_group:
            gaps[i] = float(np.max(fnr_by_group) - np.min(fnr_by_group))
    if not gaps:
        return {"fnr_gap_macro": float("nan")}
    return {"fnr_gap_macro": float(np.mean(list(gaps.values())))}


def ece_gap(labels: np.ndarray, probs: np.ndarray, groups: np.ndarray, n_bins: int = 15, min_group_size: int = 50) -> Dict[str, float]:
    masks = _group_masks(groups)
    eces = []
    for g, mask in masks.items():
        y = labels[mask]
        p = probs[mask]
        valid = y >= 0
        if valid.sum() < min_group_size:
            continue
        eces.append(expected_calibration_error(p[valid], y[valid], n_bins=n_bins))
    if len(eces) < 2:
        return {"ece_gap": float("nan")}
    return {"ece_gap": float(np.max(eces) - np.min(eces))}


def bin_age(ages: np.ndarray, bins: List[int]) -> np.ndarray:
    ages = np.asarray(ages)
    labels = []
    for a in ages:
        if np.isnan(a):
            labels.append("unknown")
            continue
        idx = np.digitize(a, bins, right=False)
        labels.append(f"bin_{idx}")
    return np.array(labels)
