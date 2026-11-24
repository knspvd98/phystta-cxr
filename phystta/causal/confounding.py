from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, ShuffleSplit

from ..safety.shift_id import ShiftIdentifierV2
from ..utils import cluster_bootstrap_statistic
from .pair_quality import annotate_pair_quality, summarize_pair_quality


CONFOUNDING_CONTEXT_KEYS = [
    "view",
    "view_family",
    "procedure_context",
    "orientation_context",
    "resolution_bucket",
]


def available_confounding_context_keys(df: pd.DataFrame) -> List[str]:
    keys: List[str] = []
    for key in CONFOUNDING_CONTEXT_KEYS:
        if key not in df.columns:
            continue
        vals = df[key].fillna("unknown").astype(str)
        uniq = sorted({v for v in vals.tolist() if v.strip()})
        if len(uniq) > 1:
            keys.append(key)
    return keys


def _safe_groups(df: pd.DataFrame) -> np.ndarray:
    for key in ["global_patient_id", "patient_id", "study_id", "image_path"]:
        if key in df.columns:
            return df[key].fillna("unknown").astype(str).to_numpy()
    return np.asarray([f"row_{i}" for i in range(len(df))], dtype=str)


def _split_indices(X: np.ndarray, y: np.ndarray, groups: np.ndarray, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    if len(np.unique(groups)) > 1:
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
        tr, te = next(splitter.split(X, y, groups=groups))
        return tr, te
    splitter = ShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
    tr, te = next(splitter.split(X, y))
    return tr, te


def _safe_multiclass_auc(y_true: np.ndarray, probs: np.ndarray, classes: np.ndarray) -> float:
    y_arr = np.asarray(y_true)
    if len(np.unique(y_arr)) < 2:
        return float("nan")
    try:
        return float(roc_auc_score(y_arr, probs, multi_class="ovr", average="macro", labels=list(classes)))
    except Exception:
        return float("nan")


def _one_hot_context_features(df: pd.DataFrame) -> pd.DataFrame:
    work = pd.DataFrame(index=df.index)
    for key in CONFOUNDING_CONTEXT_KEYS:
        if key in df.columns:
            work[key] = df[key].fillna("unknown").astype(str)
    for key in ["rows", "columns"]:
        if key in df.columns:
            numeric = pd.to_numeric(df[key], errors="coerce")
            median = float(numeric.median()) if numeric.notna().any() else 0.0
            work[key] = numeric.fillna(median)
    if work.empty:
        return pd.DataFrame(index=df.index)
    cat_cols = [c for c in work.columns if work[c].dtype == object]
    num_cols = [c for c in work.columns if c not in cat_cols]
    out = []
    if cat_cols:
        out.append(pd.get_dummies(work[cat_cols], prefix=cat_cols, dtype=float))
    if num_cols:
        num_df = work[num_cols].astype(float).copy()
        for col in num_cols:
            std = float(num_df[col].std(ddof=0))
            if np.isfinite(std) and std > 1e-8:
                num_df[col] = (num_df[col] - float(num_df[col].mean())) / std
            else:
                num_df[col] = 0.0
        out.append(num_df)
    if not out:
        return pd.DataFrame(index=df.index)
    return pd.concat(out, axis=1)


def _multiclass_predictability(
    X: np.ndarray,
    y_raw: Sequence[object],
    groups: np.ndarray,
    *,
    seed: int = 2024,
    bootstrap_samples: int = 200,
) -> Dict[str, float]:
    y_series = pd.Series(list(y_raw)).fillna("unknown").astype(str)
    valid = y_series.str.len() > 0
    X = np.asarray(X, dtype=float)[valid.to_numpy()]
    y = y_series[valid].to_numpy()
    groups = np.asarray(groups)[valid.to_numpy()]
    if X.shape[0] < 8 or len(np.unique(y)) < 2:
        return {
            "balanced_accuracy": float("nan"),
            "macro_f1": float("nan"),
            "auroc_macro": float("nan"),
            "balanced_accuracy_ci_low": float("nan"),
            "balanced_accuracy_ci_high": float("nan"),
            "macro_f1_ci_low": float("nan"),
            "macro_f1_ci_high": float("nan"),
            "auroc_macro_ci_low": float("nan"),
            "auroc_macro_ci_high": float("nan"),
            "n_test": 0.0,
            "n_classes": float(len(np.unique(y))),
        }
    tr, te = _split_indices(X, y, groups, seed=seed)
    if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
        splitter = ShuffleSplit(n_splits=1, test_size=0.25, random_state=seed + 7)
        tr, te = next(splitter.split(X, y))
    if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
        return {
            "balanced_accuracy": float("nan"),
            "macro_f1": float("nan"),
            "auroc_macro": float("nan"),
            "balanced_accuracy_ci_low": float("nan"),
            "balanced_accuracy_ci_high": float("nan"),
            "macro_f1_ci_low": float("nan"),
            "macro_f1_ci_high": float("nan"),
            "auroc_macro_ci_low": float("nan"),
            "auroc_macro_ci_high": float("nan"),
            "n_test": 0.0,
            "n_classes": float(len(np.unique(y))),
        }
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(X[tr], y[tr])
    pred = clf.predict(X[te])
    probs = clf.predict_proba(X[te])
    point_bal = float(balanced_accuracy_score(y[te], pred))
    point_f1 = float(f1_score(y[te], pred, average="macro", zero_division=0))
    point_auc = _safe_multiclass_auc(y[te], probs, clf.classes_)
    test_groups = np.asarray(groups)[te]
    if len(np.unique(test_groups)) > 1:
        boot = cluster_bootstrap_statistic(
            test_groups,
            lambda idx: np.asarray(
                [
                    balanced_accuracy_score(np.asarray(y[te])[idx], np.asarray(pred)[idx]),
                    f1_score(np.asarray(y[te])[idx], np.asarray(pred)[idx], average="macro", zero_division=0),
                    _safe_multiclass_auc(np.asarray(y[te])[idx], np.asarray(probs)[idx], clf.classes_),
                ],
                dtype=float,
            ),
            n_boot=bootstrap_samples,
            seed=seed,
        )
        lo = np.asarray(boot["ci_low"], dtype=float)
        hi = np.asarray(boot["ci_high"], dtype=float)
    else:
        lo = np.asarray([float("nan")] * 3, dtype=float)
        hi = np.asarray([float("nan")] * 3, dtype=float)
    return {
        "balanced_accuracy": point_bal,
        "macro_f1": point_f1,
        "auroc_macro": point_auc,
        "balanced_accuracy_ci_low": float(lo[0]),
        "balanced_accuracy_ci_high": float(hi[0]),
        "macro_f1_ci_low": float(lo[1]),
        "macro_f1_ci_high": float(hi[1]),
        "auroc_macro_ci_low": float(lo[2]),
        "auroc_macro_ci_high": float(hi[2]),
        "n_test": float(len(te)),
        "n_classes": float(len(np.unique(y))),
    }


def _binary_predictability(
    X: np.ndarray,
    y_raw: Sequence[object],
    groups: np.ndarray,
    *,
    seed: int = 2024,
    bootstrap_samples: int = 200,
) -> Dict[str, float]:
    y_series = pd.Series(list(y_raw))
    valid = pd.to_numeric(y_series, errors="coerce").notna()
    X = np.asarray(X, dtype=float)[valid.to_numpy()]
    y = pd.to_numeric(y_series[valid], errors="coerce").to_numpy(dtype=float)
    groups = np.asarray(groups)[valid.to_numpy()]
    y = (y > 0).astype(int)
    if X.shape[0] < 8 or len(np.unique(y)) < 2:
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
        }
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(X[tr], y[tr])
    probs = clf.predict_proba(X[te])[:, 1]
    pred = (probs >= 0.5).astype(int)
    point_bal = float(balanced_accuracy_score(y[te], pred))
    point_f1 = float(f1_score(y[te], pred, average="macro", zero_division=0))
    point_auc = float(roc_auc_score(y[te], probs)) if len(np.unique(y[te])) > 1 else float("nan")
    test_groups = np.asarray(groups)[te]
    if len(np.unique(test_groups)) > 1:
        boot = cluster_bootstrap_statistic(
            test_groups,
            lambda idx: np.asarray(
                [
                    balanced_accuracy_score(y[te][idx], pred[idx]),
                    f1_score(y[te][idx], pred[idx], average="macro", zero_division=0),
                    roc_auc_score(y[te][idx], probs[idx]) if len(np.unique(y[te][idx])) > 1 else np.nan,
                ],
                dtype=float,
            ),
            n_boot=bootstrap_samples,
            seed=seed,
        )
        lo = np.asarray(boot["ci_low"], dtype=float)
        hi = np.asarray(boot["ci_high"], dtype=float)
    else:
        lo = np.asarray([float("nan")] * 3, dtype=float)
        hi = np.asarray([float("nan")] * 3, dtype=float)
    return {
        "balanced_accuracy": point_bal,
        "macro_f1": point_f1,
        "auroc": point_auc,
        "balanced_accuracy_ci_low": float(lo[0]),
        "balanced_accuracy_ci_high": float(hi[0]),
        "macro_f1_ci_low": float(lo[1]),
        "macro_f1_ci_high": float(hi[1]),
        "auroc_ci_low": float(lo[2]),
        "auroc_ci_high": float(hi[2]),
        "n_test": float(len(te)),
    }


def _choose_shift_id_group_key(df: pd.DataFrame, preferred: Optional[str] = None) -> Optional[str]:
    if preferred and preferred in df.columns and df[preferred].nunique(dropna=True) > 1:
        return str(preferred)
    for key in ["view_family", "procedure_context", "orientation_context", "resolution_bucket", "view"]:
        if key in df.columns and df[key].nunique(dropna=True) > 1:
            return key
    return None


def build_predictability_audit(
    *,
    features: np.ndarray,
    probs: np.ndarray,
    labels: np.ndarray,
    metadata_df: pd.DataFrame,
    label_cols: Sequence[str],
    bootstrap_samples: int = 200,
    seed: int = 2024,
) -> pd.DataFrame:
    groups = _safe_groups(metadata_df)
    rows: List[Dict[str, object]] = []
    context_keys = available_confounding_context_keys(metadata_df)
    label_feature_matrix = np.asarray(np.where(labels >= 0, labels, 0.0), dtype=float)
    context_feature_df = _one_hot_context_features(metadata_df)
    context_feature_matrix = context_feature_df.to_numpy(dtype=float) if not context_feature_df.empty else np.empty((len(metadata_df), 0))

    for key in context_keys:
        target = metadata_df[key].fillna("unknown").astype(str).to_numpy()
        rows.append(
            {
                "audit_family": "context_from_embeddings",
                "target": str(key),
                **_multiclass_predictability(
                    features,
                    target,
                    groups,
                    seed=seed,
                    bootstrap_samples=bootstrap_samples,
                ),
            }
        )
        rows.append(
            {
                "audit_family": "context_from_labels",
                "target": str(key),
                **_multiclass_predictability(
                    label_feature_matrix,
                    target,
                    groups,
                    seed=seed + 13,
                    bootstrap_samples=bootstrap_samples,
                ),
            }
        )

    if context_feature_matrix.size > 0:
        for col_idx, label_name in enumerate(label_cols):
            rows.append(
                {
                    "audit_family": "labels_from_context",
                    "target": str(label_name),
                    **_binary_predictability(
                        context_feature_matrix,
                        labels[:, col_idx],
                        groups,
                        seed=seed + 31 + col_idx,
                        bootstrap_samples=bootstrap_samples,
                    ),
                }
            )
        label_burden = np.sum(np.where(labels >= 0, labels, 0.0), axis=1)
        threshold = float(np.median(label_burden)) if label_burden.size else 0.0
        rows.append(
            {
                "audit_family": "severity_from_context",
                "target": "label_burden_high",
                **_binary_predictability(
                    context_feature_matrix,
                    (label_burden > threshold).astype(float),
                    groups,
                    seed=seed + 97,
                    bootstrap_samples=bootstrap_samples,
                ),
            }
        )
        predicted_burden = np.mean(probs, axis=1) if probs.ndim == 2 and probs.size else np.zeros(len(metadata_df), dtype=float)
        p_threshold = float(np.median(predicted_burden)) if predicted_burden.size else 0.0
        rows.append(
            {
                "audit_family": "severity_from_context",
                "target": "predicted_burden_high",
                **_binary_predictability(
                    context_feature_matrix,
                    (predicted_burden > p_threshold).astype(float),
                    groups,
                    seed=seed + 113,
                    bootstrap_samples=bootstrap_samples,
                ),
            }
        )
    return pd.DataFrame(rows)


def build_shift_id_proxy_audit(
    *,
    features: np.ndarray,
    probs: np.ndarray,
    labels: np.ndarray,
    metadata_df: pd.DataFrame,
    preferred_group_key: Optional[str] = None,
    score_family: Optional[Sequence[str]] = None,
    seed: int = 2024,
) -> tuple[pd.DataFrame, Dict[str, object]]:
    group_key = _choose_shift_id_group_key(metadata_df, preferred=preferred_group_key)
    if group_key is None:
        return pd.DataFrame(), {"active_group_key": None, "reason": "no_valid_context_group_key"}
    groups = metadata_df[group_key].fillna("unknown").astype(str).to_numpy()
    calibration_errors: List[str] = []
    shift_id = None
    import time, warnings
    from sklearn.exceptions import UndefinedMetricWarning

    requested_family = list(score_family or ["energy", "mmd", "ks", "cond_mmd"])
    family_try_order = [
        ["energy", "js"],
        ["energy", "js", "mmd"],
        requested_family,
    ]

    seen = set()
    deduped = []
    for fam in family_try_order:
        key = tuple(fam)
        if key not in seen:
            seen.add(key)
            deduped.append(fam)

    for family in deduped:
        try:
            print(f"[shift_id] calibrate start family={family}", flush=True)
            t0 = time.perf_counter()
            candidate = ShiftIdentifierV2(score_family=list(family), calibration="none")
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
                warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")
                candidate.calibrate(
                    features=features,
                    probs=probs,
                    labels=labels,
                    group=groups,
                    seed=seed,
                    n_repeats=1,
                    n_splits=1,
                )
            dt = time.perf_counter() - t0
            print(f"[shift_id] calibrate done family={family} sec={dt:.2f}", flush=True)
            shift_id = candidate
            break
        except Exception as e:
            print(f"[shift_id] calibrate fail family={family} err={e}", flush=True)
            calibration_errors.append(f"{family}:{e}")

    if shift_id is None:
        return pd.DataFrame(), {"active_group_key": group_key, "reason": "shift_id_calibration_failed", "errors": calibration_errors}
    rows: List[Dict[str, object]] = []
    uniq = metadata_df[group_key].fillna("unknown").astype(str).value_counts()
    top_context = list(uniq.index[:2])
    label_burden = np.sum(np.where(labels >= 0, labels, 0.0), axis=1)
    high_severity = label_burden > (float(np.median(label_burden)) if label_burden.size else 0.0)

    def _append_scenario(name: str, ref_mask: np.ndarray, tgt_mask: np.ndarray) -> None:
        if ref_mask.sum() < 8 or tgt_mask.sum() < 8:
            return
        print(f"[shift_id] predict scenario={name} n_ref={int(ref_mask.sum())} n_tgt={int(tgt_mask.sum())}", flush=True)
        pred = shift_id.predict(features[ref_mask], probs[ref_mask], features[tgt_mask], probs[tgt_mask])
        print(f"[shift_id] predict done scenario={name}", flush=True)
        rows.append(
            {
                "scenario_kind": str(name),
                "active_group_key": str(group_key),
                **{k: v for k, v in pred.items()},
                "n_reference": int(ref_mask.sum()),
                "n_target": int(tgt_mask.sum()),
            }
        )

    if len(top_context) >= 2:
        _append_scenario(
            "context_split",
            metadata_df[group_key].fillna("unknown").astype(str).to_numpy() == top_context[0],
            metadata_df[group_key].fillna("unknown").astype(str).to_numpy() == top_context[1],
        )
    if "procedure_context" in metadata_df.columns:
        proc = metadata_df["procedure_context"].fillna("unknown").astype(str).str.lower().to_numpy()
        portable = proc == "portable_like"
        non_portable = (proc != "portable_like") & (proc != "unknown")
        _append_scenario("care_setting_proxy", portable, non_portable)
    _append_scenario("severity_proxy", ~high_severity, high_severity)
    return pd.DataFrame(rows), {"active_group_key": group_key, "classes": list(shift_id.classes_ or [])}


def build_pair_pathology_delta_audit(
    *,
    index_df: pd.DataFrame,
    pairs_df: pd.DataFrame,
    label_cols: Sequence[str],
    bootstrap_samples: int = 200,
    seed: int = 2024,
) -> tuple[pd.DataFrame, Dict[str, object]]:
    if index_df is None or pairs_df is None or pairs_df.empty or "image_path" not in index_df.columns:
        return pd.DataFrame(), {"n_pairs_total": 0, "n_pairs_technical_dominant": 0}
    technical_pairs = pairs_df[pairs_df.get("pair_quality_bucket", "ambiguous").astype(str) == "technical_dominant"].copy()
    if technical_pairs.empty:
        return pd.DataFrame(), {"n_pairs_total": int(len(pairs_df)), "n_pairs_technical_dominant": 0}
    idx = index_df.set_index("image_path")
    rows: List[Dict[str, object]] = []
    pair_groups = technical_pairs.get("global_patient_id", technical_pairs.get("patient_id", pd.Series(range(len(technical_pairs))))).astype(str).to_numpy()
    for label_name in label_cols:
        if label_name not in idx.columns:
            continue
        vals_a = pd.to_numeric(idx.reindex(technical_pairs["image_path_a"])[label_name], errors="coerce").to_numpy(dtype=float)
        vals_b = pd.to_numeric(idx.reindex(technical_pairs["image_path_b"])[label_name], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(vals_a) & np.isfinite(vals_b) & (vals_a >= 0) & (vals_b >= 0)
        if valid.sum() < 4:
            continue
        abs_delta = np.abs(vals_b[valid] - vals_a[valid])
        signed_delta = vals_b[valid] - vals_a[valid]
        groups = pair_groups[valid]
        if len(np.unique(groups)) > 1:
            boot_abs = cluster_bootstrap_statistic(
                groups,
                lambda idx_sel: float(np.mean(abs_delta[idx_sel])),
                n_boot=bootstrap_samples,
                seed=seed,
            )
            boot_signed = cluster_bootstrap_statistic(
                groups,
                lambda idx_sel: float(np.mean(signed_delta[idx_sel])),
                n_boot=bootstrap_samples,
                seed=seed + 17,
            )
        else:
            boot_abs = {"ci_low": float("nan"), "ci_high": float("nan")}
            boot_signed = {"ci_low": float("nan"), "ci_high": float("nan")}
        rows.append(
            {
                "label": str(label_name),
                "mean_abs_delta": float(np.mean(abs_delta)),
                "mean_abs_delta_ci_low": float(boot_abs.get("ci_low", float("nan"))),
                "mean_abs_delta_ci_high": float(boot_abs.get("ci_high", float("nan"))),
                "mean_signed_delta": float(np.mean(signed_delta)),
                "mean_signed_delta_ci_low": float(boot_signed.get("ci_low", float("nan"))),
                "mean_signed_delta_ci_high": float(boot_signed.get("ci_high", float("nan"))),
                "prevalence_a": float(np.mean(vals_a[valid] > 0)),
                "prevalence_b": float(np.mean(vals_b[valid] > 0)),
                "n_pairs_valid": int(valid.sum()),
            }
        )
    summary = summarize_pair_quality(technical_pairs)
    summary["n_pairs_total"] = int(len(pairs_df))
    summary["n_pairs_technical_dominant"] = int(len(technical_pairs))
    return pd.DataFrame(rows), summary


def run_confounding_audit(
    *,
    features: np.ndarray,
    probs: np.ndarray,
    labels: np.ndarray,
    metadata_df: pd.DataFrame,
    label_cols: Sequence[str],
    index_df: Optional[pd.DataFrame] = None,
    pairs_df: Optional[pd.DataFrame] = None,
    pair_quality_cfg=None,
    shift_id_group_key: Optional[str] = None,
    shift_id_score_family: Optional[Sequence[str]] = None,
    bootstrap_samples: int = 200,
    seed: int = 2024,
) -> Dict[str, object]:
    print("[confounding] start build_predictability_audit", flush=True)
    predictability_df = build_predictability_audit(
        features=features,
        probs=probs,
        labels=labels,
        metadata_df=metadata_df,
        label_cols=label_cols,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )
    print("[confounding] done build_predictability_audit", flush=True)
    print("[confounding] start build_shift_id_proxy_audit", flush=True)
    shift_id_df, shift_id_summary = build_shift_id_proxy_audit(
        features=features,
        probs=probs,
        labels=labels,
        metadata_df=metadata_df,
        preferred_group_key=shift_id_group_key,
        score_family=shift_id_score_family,
        seed=seed,
    )
    pair_delta_df = pd.DataFrame()
    pair_summary: Dict[str, object] = {"n_pairs_total": 0, "n_pairs_technical_dominant": 0}
    if index_df is not None and pairs_df is not None and not pairs_df.empty:
        pair_work = pairs_df
        if "pair_quality_bucket" not in pair_work.columns:
            pair_work = annotate_pair_quality(pair_work, index_df=index_df, label_cols=list(label_cols), cfg=pair_quality_cfg)
        print("[confounding] done build_shift_id_proxy_audit", flush=True)
        print("[confounding] start build_pair_pathology_delta_audit", flush=True)
        pair_delta_df, pair_summary = build_pair_pathology_delta_audit(
            index_df=index_df,
            pairs_df=pair_work,
            label_cols=label_cols,
            bootstrap_samples=bootstrap_samples,
            seed=seed,
        )
    print("[confounding] done build_pair_pathology_delta_audit", flush=True)
    summary = {
        "n_samples": int(len(metadata_df)),
        "context_keys": available_confounding_context_keys(metadata_df),
        "predictability_tests": int(len(predictability_df)),
        "shift_id_proxy_tests": int(len(shift_id_df)),
        "active_shift_id_group_key": shift_id_summary.get("active_group_key"),
        "n_technical_pairs_audited": int(pair_summary.get("n_pairs_technical_dominant", 0)),
    }
    return {
        "predictability_df": predictability_df,
        "shift_id_proxy_df": shift_id_df,
        "technical_pair_delta_df": pair_delta_df,
        "summary": summary,
        "shift_id_summary": shift_id_summary,
        "pair_quality_summary": pair_summary,
    }
