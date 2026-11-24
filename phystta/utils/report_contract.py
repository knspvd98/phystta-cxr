from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np

try:
    import pandas as pd
except Exception:  # pragma: no cover - pandas is available in tests, but keep helper import-safe
    pd = None


def _unwrap_scalar_like(value: Any, field_name: str) -> Any:
    if pd is not None and isinstance(value, pd.Series):
        if len(value) != 1:
            raise RuntimeError(f"Field '{field_name}' expected a scalar-like Series, got {len(value)} elements.")
        value = value.iloc[0]
    elif isinstance(value, np.ndarray):
        if value.size != 1:
            raise RuntimeError(f"Field '{field_name}' expected a scalar-like ndarray, got {value.size} elements.")
        value = value.reshape(-1)[0]
    return value


def to_scalar_json_value(value: Any, field_name: str) -> Any:
    value = _unwrap_scalar_like(value, field_name)
    if isinstance(value, np.generic):
        value = value.item()

    if value is None or isinstance(value, (bool, str)):
        return value

    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return int(value)

    if isinstance(value, (float, np.floating)):
        out = float(value)
        if not math.isfinite(out):
            raise RuntimeError(f"Field '{field_name}' contains non-finite scalar value.")
        return out

    if pd is not None and isinstance(value, pd.Timestamp):
        return value.isoformat()

    raise RuntimeError(f"Field '{field_name}' is not a JSON-safe scalar: {type(value).__name__}.")


def require_finite_scalar(value: Any, field_name: str) -> Any:
    out = to_scalar_json_value(value, field_name)
    if isinstance(out, (int, float)) and not isinstance(out, bool) and not math.isfinite(float(out)):
        raise RuntimeError(f"Field '{field_name}' contains non-finite scalar value.")
    return out


def vector_finiteness_status(values: Any, field_name: str) -> dict[str, Any]:
    if values is None:
        return {
            "field": field_name,
            "present": False,
            "length": 0,
            "n_nonfinite": 0,
            "all_finite": False,
        }

    arr = np.asarray(values)
    if arr.ndim == 0:
        arr = arr.reshape(1)

    length = int(arr.size)
    if length == 0:
        return {
            "field": field_name,
            "present": True,
            "length": 0,
            "n_nonfinite": 0,
            "all_finite": True,
        }

    try:
        flat = np.asarray(arr, dtype=float).reshape(-1)
        n_nonfinite = int((~np.isfinite(flat)).sum())
    except Exception:
        n_nonfinite = length

    return {
        "field": field_name,
        "present": True,
        "length": length,
        "n_nonfinite": n_nonfinite,
        "all_finite": bool(n_nonfinite == 0),
    }


def contract_status(errors: Any, incomplete_fields: Any) -> str:
    if errors and len(list(errors)) > 0:
        return "error"
    if incomplete_fields and len(list(incomplete_fields)) > 0:
        return "incomplete"
    return "ok"


def serialize_operating_point_row(
    row_like: Mapping[str, Any] | Any,
    required_fields: tuple[str, ...] | list[str] | None = None,
) -> dict[str, Any]:
    required = tuple(
        required_fields
        if required_fields is not None
        else (
            "label",
            "target_kind",
            "target_value",
            "threshold",
            "sensitivity",
            "specificity",
            "coverage",
            "n_support",
        )
    )
    if isinstance(row_like, Mapping):
        row = dict(row_like)
    elif hasattr(row_like, "to_dict"):
        row = dict(row_like.to_dict())
    else:
        raise RuntimeError("Operating-point row must be a mapping or pandas row-like object.")

    for field in required:
        if field not in row:
            raise RuntimeError(f"Operating-point row is missing required field '{field}'.")

    out: dict[str, Any] = {}
    string_fields = {"label", "target_kind"}
    for key, value in row.items():
        if key in string_fields:
            out[key] = str(to_scalar_json_value(value, key))
        else:
            out[key] = require_finite_scalar(value, key)
    return out
