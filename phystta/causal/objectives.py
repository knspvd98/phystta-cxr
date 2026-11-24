from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd


@dataclass
class StructuralVariables:
    image: str = "X"
    representation: str = "Z"
    labels: str = "Y"
    nuisance: str = "S"


def target_statement() -> str:
    return "Target conditional invariance: Z ⫫ S | Y (deployment-conditional, not unconditional independence)."


def build_within_patient_pairs(
    df: pd.DataFrame,
    patient_col: str = "patient_id",
    time_col: str = "study_datetime",
    source_col: str = "source",
    manufacturer_col: str = "manufacturer",
    station_col: str = "station_name",
    view_col: str = "view",
    view_family_col: str = "view_family",
    procedure_context_col: str = "procedure_context",
    orientation_context_col: str = "orientation_context",
    max_hours: float = 24.0,
    pair_mode: str = "temporal_same_view",
) -> pd.DataFrame:
    required = [patient_col, time_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing columns for pair building: {missing}")
    if pair_mode == "strict_device_change":
        for col in [manufacturer_col, station_col]:
            if col not in df.columns:
                raise RuntimeError(
                    f"pair_mode='strict_device_change' requires column '{col}'. Available columns: {sorted(df.columns)}"
                )
    work = df.copy()
    work[time_col] = pd.to_datetime(work[time_col], errors="coerce")
    work = work.dropna(subset=[time_col])

    def _norm(value) -> str:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return "unknown"
        token = str(value).strip()
        return token if token else "unknown"

    def _same_or_unknown(a: object, b: object) -> bool:
        aa = _norm(a).lower()
        bb = _norm(b).lower()
        if aa == "unknown" or bb == "unknown":
            return True
        return aa == bb

    def _view_family_jump(a: object, b: object) -> bool:
        aa = _norm(a).lower()
        bb = _norm(b).lower()
        if "unknown" in {aa, bb}:
            return False
        return aa != bb

    def _device_changed(a: dict, b: dict) -> bool:
        man_a = _norm(a.get(manufacturer_col))
        man_b = _norm(b.get(manufacturer_col))
        sta_a = _norm(a.get(station_col))
        sta_b = _norm(b.get(station_col))
        return (man_a != man_b) or (sta_a != sta_b)

    passthrough_cols = [
        "view",
        "view_family",
        "sex",
        "age",
        "race",
        "ethnicity",
        "manufacturer",
        "station_name",
        "study_id",
        "image_path",
        "procedure_context",
        "orientation_context",
        "resolution_bucket",
        "rows",
        "columns",
        "icu_flag",
        "ed_flag",
        "care_unit",
        "admission_type",
        "source",
    ]
    rows = []
    max_dt = pd.Timedelta(hours=max_hours)
    for pid, g in work.groupby(patient_col):
        if len(g) < 2:
            continue
        g = g.sort_values(time_col)
        vals = g.to_dict("records")
        for i in range(len(vals)):
            for j in range(i + 1, len(vals)):
                ti = vals[i][time_col]
                tj = vals[j][time_col]
                if pd.isna(ti) or pd.isna(tj):
                    continue
                dt = tj - ti
                if dt > max_dt:
                    break
                row_i = vals[i]
                row_j = vals[j]
                view_family_i = row_i.get(view_family_col)
                view_family_j = row_j.get(view_family_col)
                procedure_i = row_i.get(procedure_context_col)
                procedure_j = row_j.get(procedure_context_col)
                orientation_i = row_i.get(orientation_context_col)
                orientation_j = row_j.get(orientation_context_col)
                if pair_mode == "strict_device_change":
                    if not _device_changed(row_i, row_j):
                        continue
                elif pair_mode == "temporal_acquisition_context":
                    if _view_family_jump(view_family_i, view_family_j):
                        continue
                    if not _same_or_unknown(procedure_i, procedure_j):
                        continue
                    if not _same_or_unknown(orientation_i, orientation_j):
                        continue
                elif pair_mode == "temporal_same_view":
                    if _view_family_jump(view_family_i, view_family_j):
                        continue
                    if _norm(view_family_i).lower() == "unknown":
                        continue
                    if not _same_or_unknown(procedure_i, procedure_j):
                        continue
                    if not _same_or_unknown(orientation_i, orientation_j):
                        continue
                else:
                    raise RuntimeError(f"Unknown pair_mode='{pair_mode}'")

                pair = {
                    "pair_id": f"{pid}:{row_i.get('study_id', i)}->{row_j.get('study_id', j)}:{pair_mode}",
                    "patient_id": str(pid),
                    "source": _norm(row_i.get(source_col)),
                    "source_a": _norm(row_i.get(source_col)),
                    "source_b": _norm(row_j.get(source_col)),
                    "global_patient_id": f"{_norm(row_i.get(source_col))}:{pid}",
                    "image_path_a": row_i.get("image_path"),
                    "image_path_b": row_j.get("image_path"),
                    "delta_hours": float(dt.total_seconds() / 3600.0),
                    "study_datetime_a": ti,
                    "study_datetime_b": tj,
                    "pair_mode": pair_mode,
                }
                for col in passthrough_cols:
                    pair[f"{col}_a"] = row_i.get(col)
                    pair[f"{col}_b"] = row_j.get(col)
                rows.append(pair)
    return pd.DataFrame(rows)


def paired_effect_ci(
    values: np.ndarray,
    groups: Optional[np.ndarray],
    seed: int = 0,
    n_boot: int = 500,
) -> Dict[str, float]:
    v = np.asarray(values, dtype=float).reshape(-1)
    if v.size == 0:
        return {"effect_mean": float("nan"), "effect_ci_low": float("nan"), "effect_ci_high": float("nan"), "n": 0}
    mean = float(np.mean(v))
    if groups is None:
        groups = np.arange(len(v)).astype(str)
    else:
        groups = np.asarray(groups).astype(str)
    uniq = np.unique(groups)
    if len(uniq) < 2:
        return {"effect_mean": mean, "effect_ci_low": float("nan"), "effect_ci_high": float("nan"), "n": int(len(v))}
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        g = rng.choice(uniq, size=len(uniq), replace=True)
        m = np.isin(groups, g)
        if m.sum() == 0:
            continue
        boots.append(float(np.mean(v[m])))
    if not boots:
        return {"effect_mean": mean, "effect_ci_low": float("nan"), "effect_ci_high": float("nan"), "n": int(len(v))}
    low, high = np.percentile(np.array(boots), [2.5, 97.5])
    return {"effect_mean": mean, "effect_ci_low": float(low), "effect_ci_high": float(high), "n": int(len(v))}


def dose_response_slope(
    intensity: np.ndarray,
    effect: np.ndarray,
    groups: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    x = np.asarray(intensity, dtype=float).reshape(-1)
    y = np.asarray(effect, dtype=float).reshape(-1)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    if groups is not None:
        g = np.asarray(groups).astype(str)[m]
    else:
        g = None
    if len(x) < 3 or np.std(x) < 1e-8:
        return {"dose_response_slope": float("nan"), "dose_response_intercept": float("nan"), "n": int(len(x))}
    if g is not None and len(np.unique(g)) > 1:
        # group-aggregated slope to reduce pseudo-replication
        gx = []
        gy = []
        for gg in np.unique(g):
            mg = g == gg
            gx.append(float(np.mean(x[mg])))
            gy.append(float(np.mean(y[mg])))
        x_fit = np.array(gx)
        y_fit = np.array(gy)
    else:
        x_fit = x
        y_fit = y
    b1, b0 = np.polyfit(x_fit, y_fit, 1)
    return {"dose_response_slope": float(b1), "dose_response_intercept": float(b0), "n": int(len(x_fit))}
