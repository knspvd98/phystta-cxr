#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path

import pandas as pd

# Allow direct script execution without manual PYTHONPATH exports.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phystta.data.templates import OPEN_CXR_CANONICAL_LABELS


CANONICAL_LABELS = list(OPEN_CXR_CANONICAL_LABELS)

NIH_CANONICAL_FINDINGS = {
    "No Finding": {"no finding"},
    "Atelectasis": {"atelectasis"},
    "Cardiomegaly": {"cardiomegaly"},
    "Consolidation": {"consolidation"},
    "Edema": {"edema"},
    "Pneumothorax": {"pneumothorax"},
    "Pleural Effusion": {"effusion"},
    "Pneumonia": {"pneumonia"},
}

PADCHEST_CANONICAL_PATTERNS = {
    "No Finding": {"normal", "no finding", "no findings"},
    "Atelectasis": {"atelectasis"},
    "Cardiomegaly": {"cardiomegaly", "cardiac enlargement"},
    "Consolidation": {"consolidation"},
    "Edema": {"edema", "pulmonary edema"},
    "Pneumothorax": {"pneumothorax"},
    "Pleural Effusion": {"pleural effusion", "effusion"},
    "Pneumonia": {"pneumonia"},
}


def _find_first(root: Path, candidates: list[str]) -> Path:
    for candidate in candidates:
        path = root / candidate
        if path.exists():
            return path
    raise FileNotFoundError(f"No metadata CSV found under {root} for candidates: {candidates}")


def _first_present(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _normalize_text(value) -> str:
    text = str(value).strip().lower()
    text = text.replace("_", " ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _normalize_sex(value) -> str:
    text = _normalize_text(value)
    if text in {"m", "male"}:
        return "M"
    if text in {"f", "female"}:
        return "F"
    return "Unknown"


def _normalize_view(value) -> str:
    text = str(value).strip()
    return text.upper() if text else "UNKNOWN"


def _normalize_age(value) -> float:
    if pd.isna(value):
        return float("nan")
    text = str(value).strip()
    if not text:
        return float("nan")
    match = re.search(r"(\d+(?:\.\d+)?)", text)
    if match is None:
        return float("nan")
    age = float(match.group(1))
    suffix = text.upper()[-1]
    if suffix == "M":
        return age / 12.0
    if suffix == "D":
        return age / 365.25
    return age


def _normalize_label_value(value) -> float:
    if value in {1, 1.0, "1", "1.0"}:
        return 1.0
    if value in {0, 0.0, "0", "0.0"}:
        return 0.0
    if value in {-1, -1.0, "-1", "-1.0"}:
        return -1.0
    if pd.isna(value):
        return -1.0
    try:
        numeric = float(value)
    except Exception:
        return -1.0
    if numeric == 1.0:
        return 1.0
    if numeric == 0.0:
        return 0.0
    if numeric == -1.0:
        return -1.0
    return -1.0


def _extract_regex(value: str, pattern: str, fallback_prefix: str) -> str:
    match = re.search(pattern, value)
    if match:
        return match.group(1)
    stem = Path(str(value)).stem
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("_")
    if normalized:
        return f"{fallback_prefix}_{normalized}"
    return f"{fallback_prefix}_{abs(hash(str(value))) % (10 ** 10)}"


def _empty_labels(n_rows: int) -> pd.DataFrame:
    return pd.DataFrame({label: [-1.0] * n_rows for label in CANONICAL_LABELS})


def _finalize_no_finding(df: pd.DataFrame) -> pd.DataFrame:
    pathology_cols = [col for col in CANONICAL_LABELS if col != "No Finding"]
    if not pathology_cols:
        return df
    pathology_positive = (df[pathology_cols] == 1.0).any(axis=1)
    df.loc[pathology_positive, "No Finding"] = 0.0
    return df


def _parse_pipe_tokens(value) -> set[str]:
    if pd.isna(value):
        return set()
    text = str(value).strip()
    if not text:
        return set()
    return {_normalize_text(token) for token in text.split("|") if str(token).strip()}


def _parse_padchest_tokens(value) -> set[str]:
    if pd.isna(value):
        return set()
    text = str(value).strip()
    if not text:
        return set()
    parsed: list[str] = []
    try:
        obj = ast.literal_eval(text)
        if isinstance(obj, (list, tuple, set)):
            parsed = [str(item) for item in obj]
        else:
            parsed = [str(obj)]
    except Exception:
        parsed = re.split(r"[|,;]", text.strip("[]"))
    return {_normalize_text(token) for token in parsed if str(token).strip()}


def _save_normalized(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)


def prepare_chexpert_metadata(root: str | Path, out: str | Path, metadata_csv: str | Path | None = None) -> pd.DataFrame:
    root_path = Path(root).expanduser()
    csv_path = Path(metadata_csv).expanduser() if metadata_csv else _find_first(
        root_path,
        ["train.csv", "CheXpert-v1.0/train.csv", "CheXpert-v1.0-small/train.csv"],
    )
    meta = pd.read_csv(csv_path)
    if "Path" not in meta.columns:
        raise KeyError(f"CheXpert metadata must contain 'Path': {csv_path}")

    path_series = meta["Path"].astype(str)
    patient_ids = (
        meta["PatientID"].astype(str)
        if "PatientID" in meta.columns
        else path_series.map(lambda p: _extract_regex(p, r"patient(\d+)", "patient"))
    )
    if "Study" in meta.columns:
        study_ids = meta["Study"].astype(str)
    else:
        study_nums = path_series.map(lambda p: _extract_regex(p, r"study(\d+)", "study"))
        study_ids = patient_ids.astype(str) + "_study_" + study_nums.astype(str)

    base = pd.DataFrame(
        {
            "path": path_series,
            "patient_id": patient_ids.astype(str),
            "study_id": study_ids.astype(str),
            "view": meta.get("AP/PA", pd.Series(["UNKNOWN"] * len(meta))).map(_normalize_view),
            "age": meta.get("Age", pd.Series([float("nan")] * len(meta))).map(_normalize_age),
            "sex": meta.get("Sex", pd.Series(["Unknown"] * len(meta))).map(_normalize_sex),
            "report_text": meta.get("Report", pd.Series([""] * len(meta))).fillna("").astype(str),
        }
    )
    labels = _empty_labels(len(meta))
    for label in CANONICAL_LABELS:
        if label in meta.columns:
            labels[label] = meta[label].map(_normalize_label_value)
    out_df = pd.concat([base, labels], axis=1)
    out_df = _finalize_no_finding(out_df)
    _save_normalized(out_df, Path(out).expanduser())
    return out_df


def prepare_nih_metadata(root: str | Path, out: str | Path, metadata_csv: str | Path | None = None) -> pd.DataFrame:
    root_path = Path(root).expanduser()
    csv_path = Path(metadata_csv).expanduser() if metadata_csv else _find_first(
        root_path,
        ["Data_Entry_2017.csv", "data_entry_2017.csv"],
    )
    meta = pd.read_csv(csv_path)
    image_col = _first_present(meta, ["Image Index", "image_index", "path"])
    patient_col = _first_present(meta, ["Patient ID", "patient_id"])
    findings_col = _first_present(meta, ["Finding Labels", "finding_labels"])
    if image_col is None or patient_col is None or findings_col is None:
        raise KeyError(f"NIH metadata missing required columns in {csv_path}")

    paths = meta[image_col].astype(str)
    patient_ids = meta[patient_col].astype(str)
    study_ids = paths.map(lambda p: f"nih_{Path(str(p)).stem}")
    findings = meta[findings_col].fillna("")

    base = pd.DataFrame(
        {
            "path": paths,
            "patient_id": patient_ids,
            "study_id": study_ids,
            "view": meta.get("View Position", pd.Series(["UNKNOWN"] * len(meta))).map(_normalize_view),
            "age": meta.get("Patient Age", pd.Series([float("nan")] * len(meta))).map(_normalize_age),
            "sex": meta.get("Patient Gender", pd.Series(["Unknown"] * len(meta))).map(_normalize_sex),
            "report_text": meta.get("Report", pd.Series([""] * len(meta))).fillna("").astype(str),
        }
    )
    labels = _empty_labels(len(meta))
    available_mask = findings.astype(str).str.strip() != ""
    labels.loc[available_mask, CANONICAL_LABELS] = 0.0
    token_sets = findings.map(_parse_pipe_tokens)
    for idx, tokens in token_sets.items():
        if not available_mask.iloc[idx]:
            continue
        for canonical, source_terms in NIH_CANONICAL_FINDINGS.items():
            if tokens & source_terms:
                labels.at[idx, canonical] = 1.0
    out_df = pd.concat([base, labels], axis=1)
    out_df = _finalize_no_finding(out_df)
    _save_normalized(out_df, Path(out).expanduser())
    return out_df


def prepare_padchest_metadata(root: str | Path, out: str | Path, metadata_csv: str | Path | None = None) -> pd.DataFrame:
    root_path = Path(root).expanduser()
    csv_path = Path(metadata_csv).expanduser() if metadata_csv else _find_first(
        root_path,
        [
            "PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv",
            "padchest.csv",
            "padchest_metadata.csv",
        ],
    )
    meta = pd.read_csv(csv_path, low_memory=False)
    image_col = _first_present(meta, ["ImageID", "image_id", "path"])
    patient_col = _first_present(meta, ["PatientID", "patient_id"])
    label_col = _first_present(meta, ["Labels", "labels", "Finding Labels", "finding_labels"])
    if image_col is None or patient_col is None:
        raise KeyError(f"PadChest metadata missing required columns in {csv_path}")

    paths = meta[image_col].astype(str)
    patient_ids = meta[patient_col].astype(str)
    if "StudyID" in meta.columns:
        study_ids = meta["StudyID"].astype(str)
    elif "study_id" in meta.columns:
        study_ids = meta["study_id"].astype(str)
    else:
        study_ids = patient_ids + "_padchest_" + paths.map(lambda p: Path(str(p)).stem)

    base = pd.DataFrame(
        {
            "path": paths,
            "patient_id": patient_ids,
            "study_id": study_ids,
            "view": meta.get("Projection", meta.get("ViewPosition_DICOM", pd.Series(["UNKNOWN"] * len(meta)))).map(
                _normalize_view
            ),
            "age": meta.get("PatientAge", meta.get("age", pd.Series([float("nan")] * len(meta)))).map(_normalize_age),
            "sex": meta.get("PatientSex_DICOM", meta.get("sex", pd.Series(["Unknown"] * len(meta)))).map(
                _normalize_sex
            ),
            "report_text": meta.get("Report", meta.get("report_text", pd.Series([""] * len(meta)))).fillna("").astype(str),
        }
    )
    labels = _empty_labels(len(meta))
    if label_col is not None:
        raw_labels = meta[label_col].fillna("")
        available_mask = raw_labels.astype(str).str.strip() != ""
        labels.loc[available_mask, CANONICAL_LABELS] = 0.0
        token_sets = raw_labels.map(_parse_padchest_tokens)
        for idx, tokens in token_sets.items():
            if not available_mask.iloc[idx]:
                continue
            for canonical, patterns in PADCHEST_CANONICAL_PATTERNS.items():
                if any(token == pattern or pattern in token for token in tokens for pattern in patterns):
                    labels.at[idx, canonical] = 1.0
    out_df = pd.concat([base, labels], axis=1)
    out_df = _finalize_no_finding(out_df)
    _save_normalized(out_df, Path(out).expanduser())
    return out_df


def _build_summary(dataset: str, df: pd.DataFrame, out_path: Path) -> dict:
    mapping_contract = {
        "chexpert": {
            "label_space": "open_cxr_canonical_8",
            "mapping_mode": "exact",
            "note": "Utility normalization into the 8-label Open-CXR schema. This is not the 14-label external head-space harmonizer.",
        },
        "nih": {
            "label_space": "open_cxr_canonical_8",
            "mapping_mode": "explicit_exact_or_proxy",
            "note": (
                "NIH normalization uses explicit source-term rules into the 8-label Open-CXR schema. "
                "For the 14-label external branch use phystta.data.harmonization instead of assuming ontology equivalence."
            ),
        },
        "padchest": {
            "label_space": "open_cxr_canonical_8",
            "mapping_mode": "pattern_proxy",
            "note": "PadChest normalization is a lightweight pattern-matching utility and is not the canonical external harmonization layer.",
        },
    }.get(dataset, {})
    return {
        "dataset": dataset,
        "out_csv": str(out_path),
        "rows": int(len(df)),
        "columns": list(df.columns),
        "mapping_contract": mapping_contract,
        "positive_counts": {
            label: int((pd.to_numeric(df[label], errors="coerce") == 1.0).sum())
            for label in CANONICAL_LABELS
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize open chest X-ray metadata into the PhysTTA canonical schema.")
    subparsers = parser.add_subparsers(dest="dataset", required=True)

    for dataset_name in ("chexpert", "nih", "padchest"):
        sub = subparsers.add_parser(dataset_name)
        sub.add_argument("--root", required=True)
        sub.add_argument("--out", required=True)
        sub.add_argument("--metadata_csv", default=None)

    args = parser.parse_args()
    out_path = Path(args.out).expanduser()

    if args.dataset == "chexpert":
        df = prepare_chexpert_metadata(args.root, out_path, metadata_csv=args.metadata_csv)
    elif args.dataset == "nih":
        df = prepare_nih_metadata(args.root, out_path, metadata_csv=args.metadata_csv)
    elif args.dataset == "padchest":
        df = prepare_padchest_metadata(args.root, out_path, metadata_csv=args.metadata_csv)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    print(json.dumps(_build_summary(args.dataset, df, out_path), indent=2))


if __name__ == "__main__":
    main()
