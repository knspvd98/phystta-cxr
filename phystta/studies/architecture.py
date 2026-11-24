from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


def infer_paradigm(encoder_source: str) -> str:
    s = (encoder_source or "").lower()
    if s in {"baseline_cnn", "cnn", "resnet", "densenet", "efficientnet"}:
        return "CNN"
    if s in {"timm_vit", "vit", "deit", "swin"}:
        return "ViT"
    if s in {"hf_clip_or_vit", "clip", "openclip", "medclip"}:
        return "CLIP/Foundation"
    if "ssl" in s or "self_supervised" in s:
        return "SSL-Med"
    if "vlm" in s or "llava" in s or "biovil" in s:
        return "VLM"
    if "foundation" in s:
        return "Foundation-Med"
    return "Other"


def infer_pretraining(encoder_source: str, model_id: Optional[str] = None) -> str:
    s = (encoder_source or "").lower()
    mid = (model_id or "").lower()
    if s == "baseline_cnn":
        return "ImageNet/Medical"
    if "clip" in s or "clip" in mid:
        return "Foundation"
    if "timm_vit" in s or "vit" in s:
        return "ImageNet"
    if "ssl" in s:
        return "SSL-Medical"
    if "vlm" in s:
        return "VLM-pretrain"
    return "Unknown"


def build_interaction_table(
    metrics_df: pd.DataFrame,
    metric_cols: Optional[Iterable[str]] = None,
    baseline_method: str = "no_eraser",
    compare_method: str = "eraser",
) -> pd.DataFrame:
    if metrics_df.empty:
        return pd.DataFrame()
    req = {"profile", "shift", "intensity", "method"}
    missing = req - set(metrics_df.columns)
    if missing:
        raise ValueError(f"metrics_df missing required columns: {sorted(missing)}")
    if metric_cols is None:
        default_cols = [
            "auroc_macro",
            "auprc_macro",
            "ece",
            "risk_at_coverage",
            "expected_harm",
            "sex_worst_auroc",
            "race_worst_auroc",
            "age_worst_auroc",
        ]
        metric_cols = [c for c in default_cols if c in metrics_df.columns]
    metric_cols = list(metric_cols)
    if not metric_cols:
        return pd.DataFrame()

    rows: List[Dict[str, object]] = []
    group_cols = ["profile", "paradigm", "pretraining", "shift", "intensity"]
    for keys, sub in metrics_df.groupby(group_cols, dropna=False):
        b = sub[sub["method"] == baseline_method]
        e = sub[sub["method"] == compare_method]
        if b.empty or e.empty:
            continue
        rb = b.iloc[0]
        re = e.iloc[0]
        row: Dict[str, object] = {k: v for k, v in zip(group_cols, keys)}
        row["baseline_method"] = baseline_method
        row["compare_method"] = compare_method
        for m in metric_cols:
            vb = rb.get(m, np.nan)
            ve = re.get(m, np.nan)
            if pd.notna(vb) and pd.notna(ve):
                row[f"delta_{m}"] = float(ve - vb)
            else:
                row[f"delta_{m}"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_architecture_matrix(metrics_df: pd.DataFrame) -> pd.DataFrame:
    if metrics_df.empty:
        return pd.DataFrame()
    cols = [c for c in ["auroc_macro", "auprc_macro", "ece", "risk_at_coverage", "expected_harm"] if c in metrics_df.columns]
    if not cols:
        return pd.DataFrame()
    grouped = (
        metrics_df.groupby(["paradigm", "profile", "pretraining", "method"], dropna=False)[cols]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    # flatten multiindex columns
    grouped.columns = [
        c[0] if isinstance(c, tuple) and c[1] == "" else f"{c[0]}_{c[1]}" if isinstance(c, tuple) else c
        for c in grouped.columns
    ]
    return grouped


def _lower_is_better(metric_name: str) -> bool:
    m = metric_name.lower()
    tokens = ("ece", "risk", "harm", "fnr", "error", "loss", "gap", "brier")
    return any(t in m for t in tokens)


def build_robustness_table(
    metrics_df: pd.DataFrame,
    metric_cols: Optional[Iterable[str]] = None,
    clean_intensity: float = 0.0,
) -> pd.DataFrame:
    """
    Build architecture x shift robustness degradation table.

    Degradation is positive when robustness worsens under shift:
    - higher-is-better metrics: clean - worst_shift
    - lower-is-better metrics: worst_shift - clean
    """
    if metrics_df.empty:
        return pd.DataFrame()
    req = {"profile", "paradigm", "pretraining", "method", "shift", "intensity"}
    missing = req - set(metrics_df.columns)
    if missing:
        raise ValueError(f"metrics_df missing required columns: {sorted(missing)}")
    if metric_cols is None:
        default_cols = [
            "auroc_macro",
            "auprc_macro",
            "ece",
            "risk_at_coverage",
            "expected_harm",
            "sex_worst_auroc",
            "race_worst_auroc",
            "age_worst_auroc",
        ]
        metric_cols = [c for c in default_cols if c in metrics_df.columns]
    metric_cols = list(metric_cols)
    if not metric_cols:
        return pd.DataFrame()

    df = metrics_df.copy()
    df["intensity_num"] = pd.to_numeric(df["intensity"], errors="coerce")
    rows: List[Dict[str, object]] = []
    group_cols = ["profile", "paradigm", "pretraining", "method", "shift"]
    for keys, sub in df.groupby(group_cols, dropna=False):
        clean = sub[np.isclose(sub["intensity_num"], clean_intensity, equal_nan=False)]
        shifted = sub[sub["intensity_num"] > clean_intensity]
        if clean.empty or shifted.empty:
            continue
        clean_row = clean.iloc[0]
        for metric in metric_cols:
            cval = clean_row.get(metric, np.nan)
            sval_series = shifted[["intensity_num", metric]].dropna()
            if pd.isna(cval) or sval_series.empty:
                continue
            if _lower_is_better(metric):
                worst = sval_series.sort_values(metric, ascending=False).iloc[0]
                degradation = float(worst[metric] - cval)
            else:
                worst = sval_series.sort_values(metric, ascending=True).iloc[0]
                degradation = float(cval - worst[metric])
            relative_degradation = np.nan
            if np.isfinite(float(cval)) and float(cval) != 0.0:
                relative_degradation = float(degradation / float(cval))
            rows.append(
                {
                    "profile": keys[0],
                    "paradigm": keys[1],
                    "pretraining": keys[2],
                    "method": keys[3],
                    "shift": keys[4],
                    "metric": metric,
                    "clean_value": float(cval),
                    "worst_shift_value": float(worst[metric]),
                    "worst_shift_intensity": float(worst["intensity_num"]),
                    "degradation": degradation,
                    "relative_degradation": relative_degradation,
                }
            )
    return pd.DataFrame(rows)
