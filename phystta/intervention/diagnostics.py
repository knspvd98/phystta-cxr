from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, ShuffleSplit

from ..utils import cluster_bootstrap_statistic, sample_cluster_bootstrap_indices
from .adversary_eval import ProbeSuiteConfig, build_presence_task, evaluate_probe_suite
from .eraser import TaskPreservingEraser, principal_angle_overlap


def _safe_groups(groups: Optional[np.ndarray], n: int) -> np.ndarray:
    if groups is None:
        return np.asarray([f"sample_{i}" for i in range(n)], dtype=str)
    g = np.asarray(groups).reshape(-1).astype(str)
    if g.shape[0] != n:
        raise ValueError(f"group length mismatch: expected {n}, got {g.shape[0]}")
    return g


def _split_indices(X: np.ndarray, y: np.ndarray, groups: np.ndarray, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    uniq = np.unique(groups)
    if uniq.size > 1:
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
        tr, te = next(splitter.split(X, y, groups=groups))
        return tr, te
    splitter = ShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
    tr, te = next(splitter.split(X, y))
    return tr, te


def _binary_metrics(y_true: np.ndarray, probs: np.ndarray) -> Dict[str, float]:
    y = np.asarray(y_true, dtype=int).reshape(-1)
    p = np.asarray(probs, dtype=float).reshape(-1)
    pred = (p >= 0.5).astype(int)
    out = {
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)) if y.size else float("nan"),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)) if y.size else float("nan"),
    }
    if y.size and len(np.unique(y)) > 1:
        out["auroc"] = float(roc_auc_score(y, p))
    else:
        out["auroc"] = float("nan")
    return out


def _fit_binary_shortcut_probe(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    seed: int = 2024,
    bootstrap_samples: int = 200,
) -> Dict[str, float]:
    y = np.asarray(y, dtype=int).reshape(-1)
    if len(np.unique(y)) < 2:
        return {
            "balanced_accuracy": float("nan"),
            "macro_f1": float("nan"),
            "auroc": float("nan"),
            "auroc_ci_low": float("nan"),
            "auroc_ci_high": float("nan"),
            "n_test": 0.0,
            "n_test_groups": 0.0,
        }
    tr, te = _split_indices(X, y, groups, seed=seed)
    if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
        splitter = ShuffleSplit(n_splits=1, test_size=0.25, random_state=seed + 11)
        tr, te = next(splitter.split(X, y))
    if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
        return {
            "balanced_accuracy": float("nan"),
            "macro_f1": float("nan"),
            "auroc": float("nan"),
            "balanced_accuracy_ci_low": float("nan"),
            "balanced_accuracy_ci_high": float("nan"),
            "macro_f1_ci_low": float("nan"),
            "macro_f1_ci_high": float("nan"),
            "auroc_ci_low": float("nan"),
            "auroc_ci_high": float("nan"),
            "n_test": 0.0,
            "n_test_groups": 0.0,
        }
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(X[tr], y[tr])
    probs = clf.predict_proba(X[te])[:, 1]
    point = _binary_metrics(y[te], probs)
    test_groups = groups[te]
    if len(np.unique(test_groups)) > 1:
        boot = cluster_bootstrap_statistic(
            test_groups,
            lambda idx: np.asarray(
                [
                    _binary_metrics(y[te][idx], probs[idx])["balanced_accuracy"],
                    _binary_metrics(y[te][idx], probs[idx])["macro_f1"],
                    _binary_metrics(y[te][idx], probs[idx])["auroc"],
                ],
                dtype=float,
            ),
            n_boot=bootstrap_samples,
            seed=seed,
        )
        ci_low = np.asarray(boot["ci_low"], dtype=float)
        ci_high = np.asarray(boot["ci_high"], dtype=float)
    else:
        ci_low = np.asarray([float("nan")] * 3, dtype=float)
        ci_high = np.asarray([float("nan")] * 3, dtype=float)
    return {
        "balanced_accuracy": float(point["balanced_accuracy"]),
        "macro_f1": float(point["macro_f1"]),
        "auroc": float(point["auroc"]),
        "balanced_accuracy_ci_low": float(ci_low[0]),
        "balanced_accuracy_ci_high": float(ci_high[0]),
        "macro_f1_ci_low": float(ci_low[1]),
        "macro_f1_ci_high": float(ci_high[1]),
        "auroc_ci_low": float(ci_low[2]),
        "auroc_ci_high": float(ci_high[2]),
        "n_test": float(len(te)),
        "n_test_groups": float(len(np.unique(test_groups))),
    }


def extract_eraser_basis(eraser) -> Optional[np.ndarray]:
    obj = eraser
    if isinstance(obj, TaskPreservingEraser):
        obj = obj.eraser
    basis = getattr(obj, "V", None)
    if basis is None:
        return None
    if hasattr(basis, "detach"):
        basis = basis.detach().cpu().numpy()
    arr = np.asarray(basis, dtype=float)
    if arr.ndim != 2 or arr.size == 0:
        return None
    return arr


def logit_recovery_diagnostic(
    z_clean: np.ndarray,
    z_shifted: np.ndarray,
    eraser,
    head,
    *,
    critical_idx: Optional[Sequence[int]] = None,
    groups: Optional[np.ndarray] = None,
    bootstrap_samples: int = 200,
    seed: int = 2024,
) -> Dict[str, object]:
    import torch

    z_clean_np = np.asarray(z_clean, dtype=np.float32)
    z_shifted_np = np.asarray(z_shifted, dtype=np.float32)
    if z_clean_np.shape != z_shifted_np.shape:
        raise ValueError("z_clean and z_shifted must have matching shapes")
    if z_clean_np.ndim != 2:
        raise ValueError("z_clean and z_shifted must be [N,D]")

    device = next(head.parameters()).device if hasattr(head, "parameters") else torch.device("cpu")
    z_clean_t = torch.as_tensor(z_clean_np, dtype=torch.float32, device=device)
    z_shifted_t = torch.as_tensor(z_shifted_np, dtype=torch.float32, device=device)
    eraser = eraser.to(device) if hasattr(eraser, "to") else eraser
    head = head.to(device) if hasattr(head, "to") else head
    if hasattr(eraser, "eval"):
        eraser.eval()
    if hasattr(head, "eval"):
        head.eval()
    with torch.no_grad():
        logits_clean = head(z_clean_t).detach().cpu().numpy()
        logits_shift = head(z_shifted_t).detach().cpu().numpy()
        logits_erased = head(eraser(z_shifted_t)).detach().cpu().numpy()
    if critical_idx is not None:
        crit = list(critical_idx)
        logits_clean = logits_clean[:, crit]
        logits_shift = logits_shift[:, crit]
        logits_erased = logits_erased[:, crit]
    abs_shift_vs_clean = np.abs(logits_shift - logits_clean)
    abs_erased_vs_shift = np.abs(logits_erased - logits_shift)
    abs_erased_vs_clean = np.abs(logits_erased - logits_clean)
    recovery_indicator = np.mean(abs_erased_vs_clean < abs_shift_vs_clean, axis=1)
    summary: Dict[str, object] = {
        "mean_abs_shift_vs_clean": float(abs_shift_vs_clean.mean()),
        "mean_abs_erased_vs_shift": float(abs_erased_vs_shift.mean()),
        "mean_abs_erased_vs_clean": float(abs_erased_vs_clean.mean()),
        "recovery_fraction": float(recovery_indicator.mean()),
    }
    if groups is not None:
        safe_groups = _safe_groups(np.asarray(groups), len(recovery_indicator))
        if len(np.unique(safe_groups)) > 1:
            boot = cluster_bootstrap_statistic(
                safe_groups,
                lambda idx: float(np.mean(recovery_indicator[idx])),
                n_boot=bootstrap_samples,
                seed=seed,
            )
            summary["recovery_fraction_ci_low"] = float(boot["ci_low"])
            summary["recovery_fraction_ci_high"] = float(boot["ci_high"])
    return summary


def probe_spuriousness_diagnostic(
    z_clean: np.ndarray,
    z_shifted: np.ndarray,
    labels: np.ndarray,
    *,
    groups: Optional[np.ndarray] = None,
    metadata_df: Optional[pd.DataFrame] = None,
    strata_keys: Optional[Sequence[str]] = None,
    min_samples: int = 24,
    seed: int = 2024,
    bootstrap_samples: int = 200,
) -> tuple[Dict[str, object], pd.DataFrame]:
    if z_clean.shape != z_shifted.shape:
        raise ValueError("z_clean and z_shifted must have matching shapes")
    n = z_clean.shape[0]
    g = _safe_groups(groups, n)
    cfg = ProbeSuiteConfig(seed=seed)
    X, y, g_task = build_presence_task(z_clean, z_shifted, groups=g)
    overall = evaluate_probe_suite(X, y, groups=g_task, target="presence", cfg=cfg)
    primary_metric = "linear_auc" if np.isfinite(overall.get("linear_auc", np.nan)) else "mlp_auc"
    rows: List[Dict[str, object]] = [
        {
            "diagnostic": "probe_spuriousness",
            "scope": "overall",
            "stratum_key": "overall",
            "stratum_value": "all",
            **overall,
        }
    ]

    label_burden = np.sum(np.where(labels >= 0, labels, 0.0), axis=1)
    burden_threshold = float(np.median(label_burden)) if label_burden.size else 0.0
    severity_high = (label_burden > burden_threshold).astype(int)
    severity_metrics = _fit_binary_shortcut_probe(
        z_clean,
        severity_high,
        g,
        seed=seed,
        bootstrap_samples=bootstrap_samples,
    )

    max_gap = 0.0
    n_strata = 0
    if metadata_df is not None and not metadata_df.empty:
        strata = list(strata_keys or ["view_family", "procedure_context", "orientation_context", "label_burden_stratum"])
        meta = metadata_df.copy().reset_index(drop=True)
        if "label_burden_stratum" not in meta.columns:
            meta["label_burden_stratum"] = np.where(severity_high > 0, "high_burden", "low_burden")
        for key in strata:
            if key not in meta.columns:
                continue
            values = meta[key].fillna("unknown").astype(str)
            for value, idx in values.groupby(values).groups.items():
                idx_arr = np.asarray(list(idx), dtype=int)
                if idx_arr.size < int(min_samples):
                    continue
                X_sub, y_sub, g_sub = build_presence_task(z_clean[idx_arr], z_shifted[idx_arr], groups=g[idx_arr])
                if len(np.unique(y_sub)) < 2:
                    continue
                res = evaluate_probe_suite(X_sub, y_sub, groups=g_sub, target="presence", cfg=cfg)
                cur_primary = float(res.get(primary_metric, np.nan))
                if np.isfinite(cur_primary) and np.isfinite(overall.get(primary_metric, np.nan)):
                    max_gap = max(max_gap, abs(cur_primary - float(overall[primary_metric])))
                n_strata += 1
                rows.append(
                    {
                        "diagnostic": "probe_spuriousness",
                        "scope": "within_stratum",
                        "stratum_key": str(key),
                        "stratum_value": str(value),
                        "n_base_samples": int(idx_arr.size),
                        **res,
                    }
                )

    summary = {
        "primary_metric": primary_metric,
        "overall_primary_metric": float(overall.get(primary_metric, np.nan)),
        "severity_shortcut_auroc": float(severity_metrics.get("auroc", float("nan"))),
        "severity_shortcut_auroc_ci_low": float(severity_metrics.get("auroc_ci_low", float("nan"))),
        "severity_shortcut_auroc_ci_high": float(severity_metrics.get("auroc_ci_high", float("nan"))),
        "severity_shortcut_balanced_accuracy": float(severity_metrics.get("balanced_accuracy", float("nan"))),
        "max_strata_gap": float(max_gap),
        "n_strata_evaluated": int(n_strata),
    }
    rows.append(
        {
            "diagnostic": "probe_spuriousness",
            "scope": "severity_shortcut",
            "stratum_key": "label_burden",
            "stratum_value": "high_vs_low",
            **severity_metrics,
        }
    )
    return summary, pd.DataFrame(rows)


def eraser_subspace_stability_diagnostic(
    z_before: np.ndarray,
    z_after: np.ndarray,
    *,
    groups: Optional[np.ndarray] = None,
    base_basis: Optional[np.ndarray] = None,
    n_resamples: int = 12,
    k: Optional[int] = None,
    seed: int = 2024,
) -> tuple[Dict[str, object], pd.DataFrame]:
    if z_before.shape != z_after.shape:
        raise ValueError("z_before and z_after must have matching shapes")
    delta = np.asarray(z_before, dtype=float) - np.asarray(z_after, dtype=float)
    if delta.ndim != 2 or delta.shape[0] < 4:
        return (
            {
                "n_resamples": int(n_resamples),
                "k": int(k or 0),
                "mean_pairwise_overlap": float("nan"),
                "min_pairwise_overlap": float("nan"),
                "mean_reference_overlap": float("nan"),
            },
            pd.DataFrame(),
        )
    n, d = delta.shape
    kk = int(max(1, min(k or min(8, d), d)))
    g = _safe_groups(groups, n)
    uniq = np.unique(g)
    rng = np.random.default_rng(seed)

    def _fit_basis(arr: np.ndarray) -> np.ndarray:
        _, _, vt = np.linalg.svd(arr, full_matrices=False)
        return vt[:kk].T

    bases: List[np.ndarray] = []
    for _ in range(max(int(n_resamples), 2)):
        if len(uniq) > 1:
            idx = sample_cluster_bootstrap_indices(g, rng)
        else:
            idx = rng.choice(n, size=n, replace=True)
        idx = np.asarray(idx, dtype=int)
        if idx.size < kk:
            continue
        bases.append(_fit_basis(delta[idx]))
    if len(bases) < 2:
        return (
            {
                "n_resamples": int(len(bases)),
                "k": int(kk),
                "mean_pairwise_overlap": float("nan"),
                "min_pairwise_overlap": float("nan"),
                "mean_reference_overlap": float("nan"),
            },
            pd.DataFrame(),
        )
    rows: List[Dict[str, object]] = []
    overlaps: List[float] = []
    ref_overlaps: List[float] = []
    ref = None if base_basis is None else np.asarray(base_basis, dtype=float)[:, :kk]
    for i in range(len(bases)):
        if ref is not None:
            ref_overlap = principal_angle_overlap(ref, bases[i])
            ref_overlaps.append(float(ref_overlap))
            rows.append(
                {
                    "diagnostic": "eraser_stability",
                    "comparison": "reference_vs_resample",
                    "resample_i": int(i),
                    "resample_j": int(-1),
                    "subspace_overlap": float(ref_overlap),
                }
            )
        for j in range(i + 1, len(bases)):
            overlap = principal_angle_overlap(bases[i], bases[j])
            overlaps.append(float(overlap))
            rows.append(
                {
                    "diagnostic": "eraser_stability",
                    "comparison": "resample_vs_resample",
                    "resample_i": int(i),
                    "resample_j": int(j),
                    "subspace_overlap": float(overlap),
                }
            )
    summary = {
        "n_resamples": int(len(bases)),
        "k": int(kk),
        "mean_pairwise_overlap": float(np.mean(overlaps)) if overlaps else float("nan"),
        "min_pairwise_overlap": float(np.min(overlaps)) if overlaps else float("nan"),
        "mean_reference_overlap": float(np.mean(ref_overlaps)) if ref_overlaps else float("nan"),
    }
    return summary, pd.DataFrame(rows)
