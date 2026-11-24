#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

# Allow direct script execution without manual PYTHONPATH exports.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phystta.data.harmonization import canonical_external_label_space, build_label_projection_manifest
from phystta.data.templates import VINDR_LABELS


def _normalize_label_token(token: str) -> str | None:
    text = re.sub(r"\s+", " ", str(token).strip()).lower()
    if not text:
        return None
    aliases = {
        "ild": "ILD",
        "pleural effusion": "Pleural effusion",
        "pleural thickening": "Pleural thickening",
        "nodule/mass": "Nodule/Mass",
    }
    for label in VINDR_LABELS:
        if text == label.lower():
            return label
    return aliases.get(text)


def _parse_labels(value) -> set[str]:
    if pd.isna(value):
        return set()
    tokens = re.split(r"[|;,]", str(value))
    out: set[str] = set()
    for token in tokens:
        normalized = _normalize_label_token(token)
        if normalized is not None:
            out.add(normalized)
    return out


def _resolve_train_image_path(root: Path, image_id: str) -> tuple[Path, str]:
    train_dir = root / "Train"
    direct = train_dir / image_id
    if direct.exists():
        return direct, str(Path("Train") / direct.name)
    for ext in (".dicom", ".dcm"):
        candidate = train_dir / f"{image_id}{ext}"
        if candidate.exists():
            return candidate, str(Path("Train") / candidate.name)
    matches = sorted(train_dir.glob(f"{image_id}.*"))
    if matches:
        return matches[0], str(Path("Train") / matches[0].name)
    raise FileNotFoundError(f"Train image not found for image_id='{image_id}' under {train_dir}")


def _normalize_age(value) -> str:
    text = str(value).strip()
    if not text:
        return ""
    match = re.search(r"(\d+)", text)
    if match is None:
        return ""
    return match.group(1)


def _read_dicom_metadata(path: Path, image_id: str) -> dict[str, str]:
    try:
        import pydicom
    except Exception:
        return {
            "patient_id": image_id,
            "age": "",
            "sex": "",
            "view_position": "",
        }
    ds = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
    patient_id = str(getattr(ds, "PatientID", "") or image_id)
    return {
        "patient_id": patient_id,
        "age": _normalize_age(getattr(ds, "PatientAge", "")),
        "sex": str(getattr(ds, "PatientSex", "") or ""),
        "view_position": str(getattr(ds, "ViewPosition", "") or ""),
    }


def _load_filter_set(image_list: str | Path | None) -> set[str] | None:
    if image_list is None:
        return None
    items = {
        line.strip()
        for line in Path(image_list).expanduser().read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    return items or None


def prepare_vindr_physionet_train_csv(
    root: str | Path,
    out_csv: str | Path | None = None,
    manifest_out: str | Path | None = None,
    existing_only: bool = False,
    image_list: str | Path | None = None,
    max_images: int | None = None,
) -> Path:
    root_path = Path(root).expanduser()
    labels_csv = root_path / "Annotations" / "image_labels_train.csv"
    if not labels_csv.exists():
        raise FileNotFoundError(f"Official PhysioNet VinDr labels CSV not found: {labels_csv}")
    if not (root_path / "Train").exists():
        raise FileNotFoundError(f"Official PhysioNet VinDr Train/ directory not found: {root_path / 'Train'}")

    df = pd.read_csv(labels_csv)
    if "image_id" not in df.columns or "labels" not in df.columns:
        raise KeyError(f"{labels_csv} must contain at least 'image_id' and 'labels' columns")

    keep_tokens = _load_filter_set(image_list)
    rows: list[dict[str, object]] = []
    for image_id, group in df.groupby("image_id", sort=True):
        image_id_str = str(image_id)
        if keep_tokens is not None:
            allowed = {
                image_id_str,
                f"Train/{image_id_str}",
                f"Train/{image_id_str}.dcm",
                f"Train/{image_id_str}.dicom",
            }
            if allowed.isdisjoint(keep_tokens):
                continue
        try:
            image_path_abs, image_path_rel = _resolve_train_image_path(root_path, image_id_str)
        except FileNotFoundError:
            if existing_only:
                continue
            raise
        meta = _read_dicom_metadata(image_path_abs, str(image_id))
        labels = {label: 0.0 for label in VINDR_LABELS}
        for raw_labels in group["labels"].tolist():
            for label in _parse_labels(raw_labels):
                labels[label] = 1.0
        rows.append(
            {
                "image_id": image_path_rel,
                "patient_id": meta["patient_id"],
                "age": meta["age"],
                "sex": meta["sex"],
                "view_position": meta["view_position"],
                **labels,
            }
        )
        if max_images is not None and len(rows) >= max_images:
            break

    out_path = Path(out_csv).expanduser() if out_csv is not None else root_path / "train.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    manifest_path = Path(manifest_out).expanduser() if manifest_out is not None else out_path.with_suffix(".label_harmonization_manifest.json")
    projection_manifest = build_label_projection_manifest(
        "vindr",
        target_labels=canonical_external_label_space(),
    )
    payload = {
        "dataset": "vindr",
        "prepared_csv": str(out_path),
        "native_ontology_id": "vindr_cxr_native_14",
        "native_label_order": list(VINDR_LABELS),
        "projection_target_ontology_id": "mimic_chexpert_14_external",
        "projection_note": (
            "This projection manifest exists for auditability only. VinDr-CXR is not ontology-identical to the "
            "MIMIC/CheXpert 14-label head space and should be treated as an external sanity branch."
        ),
        "projection": projection_manifest,
    }
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert official PhysioNet VinDr-CXR layout into the root-level train.csv expected by PhysTTA."
    )
    parser.add_argument("--root", required=True, help="VinDr-CXR root with Train/, Test/, and Annotations/ subdirectories.")
    parser.add_argument("--out", default=None, help="Output CSV path. Defaults to <root>/train.csv")
    parser.add_argument(
        "--manifest_out",
        default=None,
        help="Optional sidecar JSON path for the explicit VinDr-to-head-space label projection manifest.",
    )
    parser.add_argument(
        "--existing_only",
        action="store_true",
        help="Skip annotation rows whose DICOM file is not present under Train/ (useful for controlled smoke subsets).",
    )
    parser.add_argument("--image_list", default=None, help="Optional text file with one image_id or Train/<file> path per line.")
    parser.add_argument("--max_images", type=int, default=None, help="Optional stable cap after filtering.")
    args = parser.parse_args()

    out_path = prepare_vindr_physionet_train_csv(
        args.root,
        out_csv=args.out,
        manifest_out=args.manifest_out,
        existing_only=args.existing_only,
        image_list=args.image_list,
        max_images=args.max_images,
    )
    print(f"Saved normalized VinDr train CSV: {out_path}")


if __name__ == "__main__":
    main()
