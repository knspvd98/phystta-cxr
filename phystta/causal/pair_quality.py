from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


@dataclass
class PairQualityConfig:
    max_delta_hours: float = 24.0
    label_change_weight: float = 0.45
    view_change_weight: float = 0.20
    context_change_weight: float = 0.20
    sentinel_change_weight: float = 0.15
    procedure_change_weight: float = 0.15
    orientation_change_weight: float = 0.15
    resolution_jump_weight: float = 0.10
    support_device_change_weight: float = 0.10
    technical_dominant_max_risk: float = 0.25
    same_view_context_shift_max_risk: float = 0.45
    clinical_change_high_risk: float = 0.65
    resolution_ratio_tolerance: float = 0.20


def _safe_str(x) -> str:
    if x is None:
        return "unknown"
    if isinstance(x, (float, np.floating)) and np.isnan(x):
        return "unknown"
    s = str(x).strip()
    return s if s else "unknown"


def _same_or_unknown(a, b) -> bool:
    aa = _safe_str(a).lower()
    bb = _safe_str(b).lower()
    if aa == "unknown" or bb == "unknown":
        return True
    return aa == bb


def _label_change_score(
    row: pd.Series,
    index_df: Optional[pd.DataFrame],
    label_cols: Optional[Iterable[str]],
) -> float:
    if index_df is None or not label_cols:
        return 0.0
    if "image_path" not in index_df.columns:
        return 0.0
    idx = index_df.set_index("image_path")
    pa = row.get("image_path_a")
    pb = row.get("image_path_b")
    if pa not in idx.index or pb not in idx.index:
        return 0.0
    try:
        la = np.asarray(idx.loc[pa, list(label_cols)], dtype=float)
        lb = np.asarray(idx.loc[pb, list(label_cols)], dtype=float)
    except Exception:
        return 0.0
    valid = (la >= 0) & (lb >= 0)
    if valid.sum() == 0:
        return 0.0
    return float(np.mean(la[valid] != lb[valid]))


def _context_change_score(row: pd.Series) -> float:
    keys = ["icu_flag", "ed_flag", "care_unit", "admission_type"]
    votes = []
    for k in keys:
        a = _safe_str(row.get(f"{k}_a"))
        b = _safe_str(row.get(f"{k}_b"))
        if a == "unknown" and b == "unknown":
            continue
        votes.append(float(a != b))
    return float(np.mean(votes)) if votes else 0.0


def _view_change_score(row: pd.Series) -> float:
    a = _safe_str(row.get("view_a"))
    b = _safe_str(row.get("view_b"))
    if a == "unknown" and b == "unknown":
        return 0.0
    return float(a != b)


def _view_family_change_score(row: pd.Series) -> float:
    a = _safe_str(row.get("view_family_a")).lower()
    b = _safe_str(row.get("view_family_b")).lower()
    if "unknown" in {a, b}:
        return 0.0
    return float(a != b)


def _procedure_change_score(row: pd.Series) -> float:
    return float(not _same_or_unknown(row.get("procedure_context_a"), row.get("procedure_context_b")))


def _orientation_change_score(row: pd.Series) -> float:
    return float(not _same_or_unknown(row.get("orientation_context_a"), row.get("orientation_context_b")))


def _resolution_jump_score(row: pd.Series, cfg: PairQualityConfig) -> float:
    try:
        rows_a = float(row.get("rows_a"))
        cols_a = float(row.get("columns_a"))
        rows_b = float(row.get("rows_b"))
        cols_b = float(row.get("columns_b"))
    except Exception:
        bucket_a = _safe_str(row.get("resolution_bucket_a")).lower()
        bucket_b = _safe_str(row.get("resolution_bucket_b")).lower()
        if "unknown" in {bucket_a, bucket_b}:
            return 0.0
        return float(bucket_a != bucket_b)
    if not all(np.isfinite(v) and v > 0 for v in [rows_a, cols_a, rows_b, cols_b]):
        bucket_a = _safe_str(row.get("resolution_bucket_a")).lower()
        bucket_b = _safe_str(row.get("resolution_bucket_b")).lower()
        if "unknown" in {bucket_a, bucket_b}:
            return 0.0
        return float(bucket_a != bucket_b)
    area_a = rows_a * cols_a
    area_b = rows_b * cols_b
    ratio = max(area_a, area_b) / max(min(area_a, area_b), 1.0)
    if ratio <= 1.0 + float(cfg.resolution_ratio_tolerance):
        return 0.0
    return float(min(1.0, (ratio - 1.0) / max(cfg.resolution_ratio_tolerance, 1e-6)))


def _support_device_change_score(row: pd.Series) -> float:
    pair_mode = _safe_str(row.get("pair_mode"))
    if pair_mode != "strict_device_change":
        return 0.0
    man_a = _safe_str(row.get("manufacturer_a"))
    man_b = _safe_str(row.get("manufacturer_b"))
    sta_a = _safe_str(row.get("station_name_a"))
    sta_b = _safe_str(row.get("station_name_b"))
    if "unknown" in {man_a, man_b, sta_a, sta_b}:
        return 1.0
    return float((man_a == man_b) and (sta_a == sta_b))


def _sentinel_change_score(
    row: pd.Series,
    sentinel_scores: Optional[Dict[str, float]],
) -> float:
    if sentinel_scores is None:
        return 0.0
    pa = row.get("image_path_a")
    pb = row.get("image_path_b")
    if pa not in sentinel_scores or pb not in sentinel_scores:
        return 0.0
    da = float(sentinel_scores[pa])
    db = float(sentinel_scores[pb])
    return float(min(1.0, abs(db - da)))


def annotate_pair_quality(
    pairs_df: pd.DataFrame,
    index_df: Optional[pd.DataFrame] = None,
    label_cols: Optional[List[str]] = None,
    sentinel_scores: Optional[Dict[str, float]] = None,
    cfg: Optional[PairQualityConfig] = None,
) -> pd.DataFrame:
    """
    Scores real within-patient pairs for acquisition-dominant assumption quality.
    Output columns:
      - pair_quality: same_view_low_risk | same_view_context_shift | cross_view_ambiguous | clinical_change_suspected | ambiguous
      - pair_quality_bucket: technical_dominant | ambiguous
      - acquisition_context_risk: [0,1]
      - quality_reason: semicolon-separated reason codes
    """
    if cfg is None:
        cfg = PairQualityConfig()
    if pairs_df is None or len(pairs_df) == 0:
        return pairs_df

    out = pairs_df.copy()
    risks = []
    acquisition_risks = []
    labels = []
    buckets = []
    reasons = []
    for _, row in out.iterrows():
        d_hours = float(row.get("delta_hours", 1e9))
        label_score = _label_change_score(row, index_df, label_cols)
        view_score = _view_change_score(row)
        view_family_score = _view_family_change_score(row)
        ctx_score = _context_change_score(row)
        proc_score = _procedure_change_score(row)
        orient_score = _orientation_change_score(row)
        resolution_score = _resolution_jump_score(row, cfg)
        device_support_score = _support_device_change_score(row)
        sent_score = _sentinel_change_score(row, sentinel_scores)

        acquisition_risk = (
            cfg.view_change_weight * max(view_score, view_family_score)
            + cfg.context_change_weight * ctx_score
            + cfg.procedure_change_weight * proc_score
            + cfg.orientation_change_weight * orient_score
            + cfg.resolution_jump_weight * resolution_score
            + cfg.support_device_change_weight * device_support_score
        )
        risk = (
            acquisition_risk
            + cfg.label_change_weight * label_score
            + cfg.sentinel_change_weight * sent_score
        )
        risk = float(min(1.0, risk))
        reason = []
        if d_hours > cfg.max_delta_hours:
            risk = 1.0
            reason.append("outside_time_window")
        if label_score > 0.0:
            reason.append("label_change")
        if view_family_score > 0.0:
            reason.append("view_family_change")
        elif view_score > 0.0:
            reason.append("view_change")
        if ctx_score > 0.0:
            reason.append("context_change")
        if proc_score > 0.0:
            reason.append("procedure_change")
        if orient_score > 0.0:
            reason.append("orientation_change")
        if resolution_score > 0.0:
            reason.append("resolution_jump")
        if device_support_score > 0.0:
            reason.append("unsupported_device_change")
        if sent_score > 0.0:
            reason.append("sentinel_change")

        same_view_family = view_family_score == 0.0
        exact_view_stable = same_view_family and view_score == 0.0
        acquisition_stable = (
            exact_view_stable
            and ctx_score == 0.0
            and proc_score == 0.0
            and orient_score == 0.0
            and resolution_score == 0.0
            and device_support_score == 0.0
        )
        if risk >= cfg.clinical_change_high_risk or label_score > 0.0 or sent_score > 0.0:
            label = "clinical_change_suspected"
            bucket = "ambiguous"
        elif not same_view_family:
            label = "cross_view_ambiguous"
            bucket = "ambiguous"
        elif acquisition_stable and risk <= cfg.technical_dominant_max_risk:
            label = "same_view_low_risk"
            bucket = "technical_dominant"
        elif same_view_family and (
            view_score > 0.0
            or ctx_score > 0.0
            or proc_score > 0.0
            or orient_score > 0.0
            or resolution_score > 0.0
            or device_support_score > 0.0
            or risk <= cfg.same_view_context_shift_max_risk
        ):
            label = "same_view_context_shift"
            bucket = "ambiguous"
        else:
            label = "ambiguous"
            bucket = "ambiguous"

        if not reason:
            reason.append("stable_pair" if bucket == "technical_dominant" else "high_risk")
        risks.append(float(risk))
        acquisition_risks.append(float(acquisition_risk))
        labels.append(label)
        buckets.append(bucket)
        reasons.append(";".join(reason))

    out["clinical_change_risk"] = np.asarray(risks, dtype=float)
    out["acquisition_context_risk"] = np.asarray(acquisition_risks, dtype=float)
    out["pair_quality"] = labels
    out["pair_quality_bucket"] = buckets
    out["quality_reason"] = reasons
    return out


def split_pairs_by_quality(pairs_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    if pairs_df is None or len(pairs_df) == 0:
        return {"technical_dominant": pd.DataFrame(), "ambiguous": pd.DataFrame()}
    bucket_col = "pair_quality_bucket" if "pair_quality_bucket" in pairs_df.columns else "pair_quality"
    return {
        "technical_dominant": pairs_df[pairs_df[bucket_col] == "technical_dominant"].copy(),
        "ambiguous": pairs_df[pairs_df[bucket_col] != "technical_dominant"].copy(),
    }


def summarize_pair_quality(pairs_df: pd.DataFrame) -> Dict[str, object]:
    if pairs_df is None or len(pairs_df) == 0:
        return {
            "n_pairs_total": 0,
            "bucket_counts": {},
            "pair_quality_counts": {},
            "quality_reason_counts": {},
            "clinical_change_risk_quantiles": {},
            "acquisition_context_risk_quantiles": {},
        }

    def _quantiles(series: pd.Series) -> Dict[str, float]:
        vals = pd.to_numeric(series, errors="coerce").dropna()
        if vals.empty:
            return {"q50": float("nan"), "q90": float("nan")}
        arr = vals.to_numpy(dtype=float)
        return {
            "q50": float(np.quantile(arr, 0.5)),
            "q90": float(np.quantile(arr, 0.9)),
        }

    reason_counts: Dict[str, int] = {}
    for value in pairs_df.get("quality_reason", pd.Series(["unknown"] * len(pairs_df))).fillna("unknown").astype(str):
        tokens = [token.strip() for token in value.split(";") if token.strip()]
        if not tokens:
            tokens = ["unknown"]
        for token in tokens:
            reason_counts[token] = reason_counts.get(token, 0) + 1

    bucket_col = "pair_quality_bucket" if "pair_quality_bucket" in pairs_df.columns else "pair_quality"
    accepted = pairs_df[pairs_df[bucket_col] == "technical_dominant"]
    rejected = pairs_df[pairs_df[bucket_col] != "technical_dominant"]
    return {
        "n_pairs_total": int(len(pairs_df)),
        "bucket_counts": {
            str(k): int(v)
            for k, v in pairs_df[bucket_col].fillna("unknown").value_counts().to_dict().items()
        },
        "pair_quality_counts": {
            str(k): int(v)
            for k, v in pairs_df.get("pair_quality", pd.Series(["unknown"] * len(pairs_df))).fillna("unknown").value_counts().to_dict().items()
        },
        "quality_reason_counts": reason_counts,
        "accepted_pairs": int(len(accepted)),
        "rejected_pairs": int(len(rejected)),
        "accepted_view_families": sorted(
            {
                str(v)
                for col in ["view_family_a", "view_family_b"]
                if col in accepted.columns
                for v in accepted[col].fillna("unknown").astype(str).tolist()
                if str(v).strip() and str(v).strip().lower() != "unknown"
            }
        ),
        "accepted_procedure_contexts": sorted(
            {
                str(v)
                for col in ["procedure_context_a", "procedure_context_b"]
                if col in accepted.columns
                for v in accepted[col].fillna("unknown").astype(str).tolist()
                if str(v).strip() and str(v).strip().lower() != "unknown"
            }
        ),
        "accepted_orientation_contexts": sorted(
            {
                str(v)
                for col in ["orientation_context_a", "orientation_context_b"]
                if col in accepted.columns
                for v in accepted[col].fillna("unknown").astype(str).tolist()
                if str(v).strip() and str(v).strip().lower() != "unknown"
            }
        ),
        "clinical_change_risk_quantiles": {
            "all": _quantiles(pairs_df.get("clinical_change_risk", pd.Series([], dtype=float))),
            "technical_dominant": _quantiles(accepted.get("clinical_change_risk", pd.Series([], dtype=float))),
            "ambiguous": _quantiles(rejected.get("clinical_change_risk", pd.Series([], dtype=float))),
        },
        "acquisition_context_risk_quantiles": {
            "all": _quantiles(pairs_df.get("acquisition_context_risk", pd.Series([], dtype=float))),
            "technical_dominant": _quantiles(accepted.get("acquisition_context_risk", pd.Series([], dtype=float))),
            "ambiguous": _quantiles(rejected.get("acquisition_context_risk", pd.Series([], dtype=float))),
        },
    }


def ensure_pairs_group_key(
    pairs_df: pd.DataFrame,
    group_key: str,
    default_source: Optional[str] = None,
) -> pd.DataFrame:
    """
    Ensure paired index has the same grouping key semantics as split dataframe.
    This avoids filtering pairs with incompatible key spaces (patient_id vs global_patient_id).
    """
    if pairs_df is None or len(pairs_df) == 0:
        return pairs_df
    out = pairs_df.copy()
    if group_key in out.columns:
        return out
    if group_key == "patient_id":
        if "patient_id" not in out.columns:
            raise RuntimeError("pairs_df missing patient_id for group_key=patient_id")
        return out
    if group_key == "global_patient_id":
        if "patient_id" not in out.columns:
            raise RuntimeError("pairs_df missing patient_id for group_key=global_patient_id")
        if "source_a" in out.columns:
            source = out["source_a"].astype(str)
        elif "source" in out.columns:
            source = out["source"].astype(str)
        elif default_source is not None:
            source = pd.Series([str(default_source)] * len(out), index=out.index)
        else:
            raise RuntimeError(
                "pairs_df missing source/source_a to build global_patient_id; "
                "provide default_source from the selected dataset source."
            )
        out["global_patient_id"] = source.astype(str) + ":" + out["patient_id"].astype(str)
        return out
    raise RuntimeError(
        f"Unsupported group_key='{group_key}' for paired index filtering. "
        f"Available columns: {sorted(out.columns)}"
    )
