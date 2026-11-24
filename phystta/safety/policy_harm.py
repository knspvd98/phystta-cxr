from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import numpy as np


@dataclass
class HarmConfig:
    cost_fn: float = 10.0
    cost_fp: float = 1.0
    abstain_cost: float = 1.0
    thresholds: Optional[np.ndarray] = None


def normalize_action(action: str) -> str:
    a = str(action)
    if a == "apply_eraser":
        return "eraser"
    return a


def _resolve_class_costs(
    num_classes: int,
    scalar_value: float,
    per_class_value: Optional[Iterable[float]],
    name: str,
) -> np.ndarray:
    if per_class_value is None:
        return np.full((num_classes,), float(scalar_value), dtype=float)
    arr = np.asarray(list(per_class_value), dtype=float).reshape(-1)
    if arr.size == 1:
        arr = np.full((num_classes,), float(arr[0]), dtype=float)
    if arr.shape[0] != num_classes:
        raise ValueError(f"{name} length mismatch: expected {num_classes}, got {arr.shape[0]}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values")
    return arr


def sample_harm_multilabel(
    probs: np.ndarray,
    labels: np.ndarray,
    thresholds: Optional[np.ndarray] = None,
    cost_fn: float = 10.0,
    cost_fp: float = 1.0,
    critical_idx: Optional[Iterable[int]] = None,
    class_cost_fn: Optional[Iterable[float]] = None,
    class_cost_fp: Optional[Iterable[float]] = None,
) -> np.ndarray:
    if probs.ndim != 2 or labels.ndim != 2:
        raise ValueError("probs and labels must be [N,C]")
    c = probs.shape[1]
    th = np.full(c, 0.5, dtype=float) if thresholds is None else np.asarray(thresholds, dtype=float).reshape(-1)
    if th.size == 1:
        th = np.full(c, float(th[0]), dtype=float)
    if th.shape[0] != c:
        raise ValueError("threshold length mismatch")
    cost_fn_vec = _resolve_class_costs(c, cost_fn, class_cost_fn, "class_cost_fn")
    cost_fp_vec = _resolve_class_costs(c, cost_fp, class_cost_fp, "class_cost_fp")
    if critical_idx is None:
        critical = list(range(c))
    else:
        critical = list(critical_idx)
    n = probs.shape[0]
    harms = np.zeros((n,), dtype=float)
    denom = np.zeros((n,), dtype=float)
    for j in critical:
        y = labels[:, j]
        p = probs[:, j]
        valid = y >= 0
        if valid.sum() == 0:
            continue
        pred = (p >= th[j]).astype(int)
        yy = y.astype(int)
        fn = ((pred == 0) & (yy == 1) & valid).astype(float)
        fp = ((pred == 1) & (yy == 0) & valid).astype(float)
        harms += cost_fn_vec[j] * fn + cost_fp_vec[j] * fp
        denom += valid.astype(float)
    denom = np.maximum(denom, 1.0)
    return harms / denom


def expected_harm_multilabel(
    probs: np.ndarray,
    labels: np.ndarray,
    thresholds: Optional[np.ndarray] = None,
    cost_fn: float = 10.0,
    cost_fp: float = 1.0,
    critical_idx: Optional[Iterable[int]] = None,
    class_cost_fn: Optional[Iterable[float]] = None,
    class_cost_fp: Optional[Iterable[float]] = None,
) -> float:
    sample = sample_harm_multilabel(
        probs,
        labels,
        thresholds=thresholds,
        cost_fn=cost_fn,
        cost_fp=cost_fp,
        critical_idx=critical_idx,
        class_cost_fn=class_cost_fn,
        class_cost_fp=class_cost_fp,
    )
    return float(np.mean(sample)) if sample.size else float("nan")


def policy_regret(
    action: str,
    harm_no_eraser: float,
    harm_eraser: float,
    abstain_cost: float = 1.0,
    harm_recalibrate: Optional[float] = None,
) -> Dict[str, float]:
    a = normalize_action(action)
    if a == "eraser":
        chosen = harm_eraser
    elif a == "recalibrate":
        chosen = harm_recalibrate if (harm_recalibrate is not None and np.isfinite(harm_recalibrate)) else harm_no_eraser
    elif a == "no_eraser":
        chosen = harm_no_eraser
    else:
        chosen = float(abstain_cost)
    oracle_candidates = [harm_no_eraser, harm_eraser, abstain_cost]
    if harm_recalibrate is not None:
        oracle_candidates.append(harm_recalibrate)
    oracle = np.nanmin(oracle_candidates)
    regret = float(chosen - oracle) if np.isfinite(chosen) and np.isfinite(oracle) else float("nan")
    return {"policy_harm": float(chosen), "oracle_harm": float(oracle), "policy_regret": regret}


def regret_by_shift(
    decisions: List[Dict[str, object]],
    harms_no: List[float],
    harms_er: List[float],
    abstain_cost: float = 1.0,
    harms_recalibrate: Optional[List[float]] = None,
) -> Dict[str, float]:
    vals = []
    for idx, (d, h0, h1) in enumerate(zip(decisions, harms_no, harms_er)):
        action = normalize_action(str(d.get("decision", "abstain")))
        hrec = harms_recalibrate[idx] if harms_recalibrate is not None and idx < len(harms_recalibrate) else None
        vals.append(policy_regret(action, h0, h1, abstain_cost=abstain_cost, harm_recalibrate=hrec)["policy_regret"])
    arr = np.asarray(vals, dtype=float)
    return {
        "regret_mean": float(np.nanmean(arr)) if arr.size else float("nan"),
        "regret_p95": float(np.nanquantile(arr, 0.95)) if arr.size else float("nan"),
        "n": int(arr.size),
    }


def regret_decomposition(
    decisions: List[Dict[str, object]],
    harms_no: List[float],
    harms_er: List[float],
    abstain_cost: float = 1.0,
    harms_recalibrate: Optional[List[float]] = None,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Returns regret summary grouped by:
    - shift label (`row['shift']`)
    - policy action (`row['decision']`)
    """
    rows = []
    for idx, (d, h0, h1) in enumerate(zip(decisions, harms_no, harms_er)):
        action = normalize_action(str(d.get("decision", "abstain")))
        shift = str(d.get("shift", "unknown"))
        hrec = harms_recalibrate[idx] if harms_recalibrate is not None and idx < len(harms_recalibrate) else None
        reg = policy_regret(action, h0, h1, abstain_cost=abstain_cost, harm_recalibrate=hrec)
        rows.append((shift, action, reg["policy_regret"]))
    if not rows:
        return {"by_shift": {}, "by_action": {}}

    def _summ(values: List[float]) -> Dict[str, float]:
        arr = np.asarray(values, dtype=float)
        return {
            "regret_mean": float(np.nanmean(arr)),
            "regret_p95": float(np.nanquantile(arr, 0.95)),
            "n": int(arr.size),
        }

    by_shift: Dict[str, List[float]] = {}
    by_action: Dict[str, List[float]] = {}
    for shift, action, r in rows:
        by_shift.setdefault(shift, []).append(r)
        by_action.setdefault(action, []).append(r)
    return {
        "by_shift": {k: _summ(v) for k, v in by_shift.items()},
        "by_action": {k: _summ(v) for k, v in by_action.items()},
    }
