from __future__ import annotations

from typing import Optional, Sequence

import pandas as pd


def resolve_group_key(df: pd.DataFrame, preferred: Optional[str] = None) -> Optional[str]:
    if preferred and preferred in df.columns:
        if df[preferred].notna().any():
            return preferred
    for key in ["global_patient_id", "patient_id", "study_id"]:
        if key in df.columns and df[key].notna().any():
            return key
    return None


def require_group_key(df: pd.DataFrame, preferred: Optional[str] = None) -> str:
    key = resolve_group_key(df, preferred=preferred)
    if key is None:
        raise RuntimeError("Group key missing: patient_id/study_id required. NOT CLINICALLY VALID.")
    return key


def ensure_no_overlap(df_a: pd.DataFrame, df_b: pd.DataFrame, group_key: str) -> None:
    if group_key not in df_a.columns or group_key not in df_b.columns:
        raise RuntimeError(f"Group key {group_key} missing for overlap check")
    a = set(df_a[group_key].astype(str))
    b = set(df_b[group_key].astype(str))
    if a & b:
        raise RuntimeError(f"Group leakage detected between splits for key {group_key}")


def sample_ids_from_batch(batch: dict) -> Optional[Sequence[str]]:
    """
    Return stable per-sample identifiers for deterministic shift seeding.
    Preference order avoids absolute paths when possible.
    """
    for key in ("global_image_id", "dicom_id", "study_id", "image_path"):
        if key in batch:
            vals = batch[key]
            try:
                return [str(v) for v in vals]
            except Exception:
                return None
    return None
