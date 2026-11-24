from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..config import DatasetCapabilitySnapshot, DatasetSourceConfig


VIEW_FAMILY_KEYS = {
    "AP": "frontal",
    "PA": "frontal",
    "LL": "lateral",
    "LAT": "lateral",
    "LATERAL": "lateral",
}

PROCEDURE_PORTABLE_TOKENS = ("portable", "port", "bedside", "ward", "icu")
ORIENTATION_SUPINE_TOKENS = ("supine", "decub", "erect", "upright", "semi-erect")
CONTEXT_COLUMN_ALIASES = (
    "icu_flag",
    "ed_flag",
    "care_unit",
    "admission_type",
)


def _clean_string(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, (float, np.floating)) and np.isnan(value):
        return ""
    return str(value).strip()


def _normalize_token(value: object) -> str:
    token = _clean_string(value).upper()
    return token.replace("-", "").replace("_", "").replace(" ", "")


def normalize_view_value(value: object) -> str:
    raw = _clean_string(value).upper()
    if not raw:
        return "unknown"
    token = _normalize_token(raw)
    if token in {"AP", "PA", "LAT", "LL", "LATERAL"}:
        return "LAT" if token == "LATERAL" else token
    if token in {"FRONTAL", "FRONT"}:
        return "FRONTAL"
    if token in {"PORTABLEAP", "PORTAP"}:
        return "AP"
    return raw


def derive_view_family(view: object, explicit_family: object = None) -> str:
    family = _clean_string(explicit_family).lower()
    if family in {"frontal", "lateral"}:
        return family
    view_norm = normalize_view_value(view)
    if view_norm in {"AP", "PA", "FRONTAL"}:
        return "frontal"
    if view_norm in {"LAT", "LL"}:
        return "lateral"
    return "unknown"


def derive_is_frontal(view_family: object) -> bool:
    return str(view_family).lower() == "frontal"


def derive_is_lateral(view_family: object) -> bool:
    return str(view_family).lower() == "lateral"


def derive_procedure_context(
    performed_procedure_step_description: object,
    procedure_code_meaning: object,
    view_value: object = None,
) -> str:
    joined = " ".join(
        [
            _clean_string(performed_procedure_step_description).lower(),
            _clean_string(procedure_code_meaning).lower(),
            _clean_string(view_value).lower(),
        ]
    ).strip()
    if not joined:
        return "unknown"
    if any(token in joined for token in PROCEDURE_PORTABLE_TOKENS):
        return "portable_like"
    if "lateral" in joined:
        return "lateral_series"
    if "pa" in joined or "posteroanterior" in joined:
        return "upright_pa"
    if "ap" in joined or "anteroposterior" in joined:
        return "ap_like"
    return joined


def derive_orientation_context(
    patient_orientation_code_meaning: object,
    view_code_meaning: object = None,
) -> str:
    joined = " ".join(
        [
            _clean_string(patient_orientation_code_meaning).lower(),
            _clean_string(view_code_meaning).lower(),
        ]
    ).strip()
    if not joined:
        return "unknown"
    if any(token in joined for token in ORIENTATION_SUPINE_TOKENS):
        return joined
    if "left" in joined or "right" in joined:
        return joined
    return joined


def derive_resolution_bucket(rows: object, columns: object) -> str:
    try:
        r = float(rows)
        c = float(columns)
    except Exception:
        return "unknown"
    if not np.isfinite(r) or not np.isfinite(c) or r <= 0 or c <= 0:
        return "unknown"
    area = r * c
    if area < 1_000_000:
        return "small"
    if area < 4_000_000:
        return "standard"
    return "high"


def derive_study_datetime(df: pd.DataFrame) -> pd.Series:
    if "study_datetime" in df.columns:
        return pd.to_datetime(df["study_datetime"], errors="coerce")
    if {"study_date", "study_time"}.issubset(df.columns):
        study_date = df["study_date"].astype(str).str.replace(r"\.0$", "", regex=True)
        study_time = df["study_time"].astype(str).str.split(".").str[0].str.zfill(6)
        return pd.to_datetime(study_date + study_time, format="%Y%m%d%H%M%S", errors="coerce")
    if "study_date" in df.columns:
        return pd.to_datetime(df["study_date"], errors="coerce")
    return pd.Series(pd.NaT, index=df.index)


def enrich_index_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "view" in out.columns:
        out["view"] = out["view"].map(normalize_view_value)
    if "view_family" in out.columns:
        out["view_family"] = out["view_family"].fillna("unknown").astype(str).str.lower()
    else:
        out["view_family"] = [
            derive_view_family(view=row.get("view"), explicit_family=row.get("view_family"))
            for _, row in out.iterrows()
        ]
    out["is_frontal"] = out["view_family"].map(derive_is_frontal)
    out["is_lateral"] = out["view_family"].map(derive_is_lateral)

    if "procedure_context" not in out.columns:
        out["procedure_context"] = [
            derive_procedure_context(
                row.get("performed_procedure_step_description"),
                row.get("procedure_code_meaning"),
                row.get("view"),
            )
            for _, row in out.iterrows()
        ]
    if "orientation_context" not in out.columns:
        out["orientation_context"] = [
            derive_orientation_context(
                row.get("patient_orientation_code_meaning"),
                row.get("view_code_meaning"),
            )
            for _, row in out.iterrows()
        ]
    if "resolution_bucket" not in out.columns:
        out["resolution_bucket"] = [
            derive_resolution_bucket(row.get("rows"), row.get("columns")) for _, row in out.iterrows()
        ]
    if "study_datetime" not in out.columns:
        out["study_datetime"] = derive_study_datetime(out)
    for key in CONTEXT_COLUMN_ALIASES:
        if key not in out.columns:
            out[key] = ""
    return out


def source_columns_snapshot(df: pd.DataFrame) -> List[str]:
    cols = []
    for col in df.columns:
        if column_supported(df, str(col)):
            cols.append(str(col))
    return sorted(cols)


def declared_source_columns_snapshot(src_cfg: DatasetSourceConfig) -> List[str]:
    cols: List[str] = []
    for key, alias in [
        ("patient_id_col", "patient_id"),
        ("study_id_col", "study_id"),
        ("dicom_id_col", "dicom_id"),
        ("view_col", "view"),
        ("view_family_col", "view_family"),
        ("study_datetime_col", "study_datetime"),
        ("study_date_col", "study_date"),
        ("study_time_col", "study_time"),
        ("sex_col", "sex"),
        ("age_col", "age"),
        ("race_col", "race"),
        ("manufacturer_col", "manufacturer"),
        ("station_name_col", "station_name"),
        ("performed_procedure_step_description_col", "performed_procedure_step_description"),
        ("procedure_code_meaning_col", "procedure_code_meaning"),
        ("view_code_meaning_col", "view_code_meaning"),
        ("patient_orientation_code_meaning_col", "patient_orientation_code_meaning"),
        ("rows_col", "rows"),
        ("columns_col", "columns"),
    ]:
        if getattr(src_cfg, key, None):
            cols.append(alias)
    if getattr(src_cfg, "boxes_csv", None) is not None or bool(getattr(src_cfg, "supports_boxes", False)):
        cols.extend(["box_xmin", "box_ymin", "box_xmax", "box_ymax"])
    if "view" in cols and "view_family" not in cols:
        cols.extend(["view_family", "is_frontal", "is_lateral"])
    if any(c in cols for c in ["performed_procedure_step_description", "procedure_code_meaning"]) and "procedure_context" not in cols:
        cols.append("procedure_context")
    if any(c in cols for c in ["patient_orientation_code_meaning", "view_code_meaning"]) and "orientation_context" not in cols:
        cols.append("orientation_context")
    if any(c in cols for c in ["rows", "columns"]) and "resolution_bucket" not in cols:
        cols.append("resolution_bucket")
    return sorted(set(cols))


def capability_snapshot_from_source_config(src_cfg: DatasetSourceConfig) -> DatasetCapabilitySnapshot:
    cols = set(declared_source_columns_snapshot(src_cfg))
    has_any = lambda keys: any(key in cols for key in keys)
    return DatasetCapabilitySnapshot(
        source=str(src_cfg.domain_label or src_cfg.name),
        supports_temporal_pairs=bool(
            src_cfg.supports_temporal_pairs
            or "study_datetime" in cols
            or {"study_date", "study_time"} <= cols
        ),
        supports_view_metadata=bool(src_cfg.supports_view_metadata or has_any({"view", "view_family", "is_frontal", "is_lateral"})),
        supports_demographics=bool(src_cfg.supports_demographics or has_any({"age", "sex", "race", "ethnicity"})),
        supports_orientation_metadata=bool(
            src_cfg.supports_orientation_metadata
            or has_any({"orientation_context", "patient_orientation_code_meaning", "view_code_meaning"})
        ),
        supports_resolution_metadata=bool(src_cfg.supports_resolution_metadata or has_any({"rows", "columns", "resolution_bucket"})),
        supports_boxes=bool(src_cfg.supports_boxes or has_any({"box_xmin", "box_ymin", "box_xmax", "box_ymax"})),
        supports_device_metadata=bool(src_cfg.supports_device_metadata or has_any({"manufacturer", "station_name"})),
        supports_procedure_metadata=bool(
            src_cfg.supports_procedure_metadata
            or has_any({"procedure_context", "performed_procedure_step_description", "procedure_code_meaning"})
        ),
        available_columns=sorted(cols),
    )


def pair_mode_required_capabilities(pair_mode: Optional[str], require_resolution: bool = False) -> List[str]:
    if not pair_mode:
        return []
    mapping = {
        "strict_device_change": ["supports_temporal_pairs", "supports_device_metadata"],
        "temporal_acquisition_context": [
            "supports_temporal_pairs",
            "supports_view_metadata",
            "supports_procedure_metadata",
        ],
        "temporal_same_view": [
            "supports_temporal_pairs",
            "supports_view_metadata",
            "supports_orientation_metadata",
            "supports_procedure_metadata",
        ],
    }
    required = list(mapping.get(str(pair_mode), []))
    if require_resolution and pair_mode != "strict_device_change" and "supports_resolution_metadata" not in required:
        required.append("supports_resolution_metadata")
    return required


def pair_mode_support_status(
    snapshot: DatasetCapabilitySnapshot,
    pair_mode: Optional[str],
    require_resolution: bool = False,
) -> Dict[str, object]:
    required = pair_mode_required_capabilities(pair_mode, require_resolution=require_resolution)
    missing = [cap for cap in required if not bool(getattr(snapshot, cap, False))]
    return {
        "pair_mode": pair_mode,
        "required_capabilities": required,
        "missing_capabilities": missing,
        "supports_requested_pair_mode": len(missing) == 0,
    }


def capability_snapshot_from_frame(
    source: str,
    df: pd.DataFrame,
    src_cfg: Optional[DatasetSourceConfig] = None,
) -> DatasetCapabilitySnapshot:
    cols = set(source_columns_snapshot(df))
    has_any = lambda keys: any(key in cols for key in keys)
    snapshot = DatasetCapabilitySnapshot(
        source=str(source),
        supports_temporal_pairs=("study_datetime" in cols) or ({"study_date", "study_time"} <= cols),
        supports_view_metadata=has_any({"view", "view_family", "is_frontal", "is_lateral"}),
        supports_demographics=has_any({"age", "sex", "race", "ethnicity"}),
        supports_orientation_metadata=has_any({"orientation_context", "patient_orientation_code_meaning", "view_code_meaning"}),
        supports_resolution_metadata=has_any({"rows", "columns", "resolution_bucket"}),
        supports_boxes=has_any({"box_xmin", "box_ymin", "box_xmax", "box_ymax", "box_count"}),
        supports_device_metadata=has_any({"manufacturer", "station_name"}),
        supports_procedure_metadata=has_any(
            {"procedure_context", "performed_procedure_step_description", "procedure_code_meaning"}
        ),
        available_columns=sorted(cols),
    )
    if src_cfg is not None and src_cfg.boxes_csv is not None:
        snapshot.supports_boxes = True
    return snapshot


def validate_declared_capabilities(
    source: str,
    src_cfg: DatasetSourceConfig,
    snapshot: DatasetCapabilitySnapshot,
) -> None:
    for key in [
        "supports_temporal_pairs",
        "supports_view_metadata",
        "supports_demographics",
        "supports_orientation_metadata",
        "supports_resolution_metadata",
        "supports_boxes",
        "supports_device_metadata",
        "supports_procedure_metadata",
    ]:
        declared = getattr(src_cfg, key)
        if declared is None:
            continue
        actual = bool(getattr(snapshot, key))
        if bool(declared) != actual:
            raise RuntimeError(
                f"{source}: declared capability {key}={declared} but materialized index reports {actual}. "
                f"Available columns: {snapshot.available_columns}"
            )


def column_supported(df: pd.DataFrame, key: str) -> bool:
    if key not in df.columns:
        return False
    series = df[key]
    if pd.api.types.is_datetime64_any_dtype(series):
        return bool(series.notna().any())
    if pd.api.types.is_bool_dtype(series):
        return bool(series.any()) or bool((~series.isna()).any())
    if pd.api.types.is_numeric_dtype(series):
        return bool(series.notna().any())
    cleaned = series.fillna("").astype(str).str.strip().str.lower()
    valid = cleaned[~cleaned.isin({"", "nan", "none", "unknown", "nat"})]
    return bool(len(valid) > 0)


def validate_required_columns(
    *,
    df: pd.DataFrame,
    source: str,
    keys: Sequence[str],
    context: str,
) -> None:
    missing = [str(key) for key in keys if not column_supported(df, str(key))]
    if missing:
        raise RuntimeError(
            f"{source}: {context} requested unsupported keys {missing}. "
            f"Materialized columns: {source_columns_snapshot(df)}"
        )
