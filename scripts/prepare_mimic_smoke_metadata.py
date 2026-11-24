#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
from pathlib import Path

import pandas as pd


JOIN_CANDIDATES = ["subject_id", "study_id", "dicom_id"]
METADATA_HINTS = ["path", "ViewPosition", "view", "PatientSex", "sex", "anchor_age", "age"]


def _read_csv(path: Path) -> pd.DataFrame:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return pd.read_csv(f)
    return pd.read_csv(path)


def _first_existing(root: Path, names: list[str]) -> Path | None:
    for name in names:
        path = root / name
        if path.exists():
            return path
    return None


def _resolve_inputs(root: Path, labels_csv: Path | None, metadata_csv: Path | None) -> tuple[Path, Path]:
    labels = labels_csv or _first_existing(
        root,
        [
            "mimic-cxr-2.0.0-chexpert.csv",
            "mimic-cxr-2.0.0-chexpert.csv.gz",
            "mimic-cxr-2.0.0/mimic-cxr-2.0.0-chexpert.csv",
            "mimic-cxr-2.0.0/mimic-cxr-2.0.0-chexpert.csv.gz",
        ],
    )
    metadata = metadata_csv or _first_existing(
        root,
        [
            "mimic-cxr-2.0.0-metadata.csv",
            "mimic-cxr-2.0.0-metadata.csv.gz",
            "mimic-cxr-2.0.0/mimic-cxr-2.0.0-metadata.csv",
            "mimic-cxr-2.0.0/mimic-cxr-2.0.0-metadata.csv.gz",
        ],
    )
    if labels is None:
        raise FileNotFoundError("Could not find mimic-cxr-2.0.0-chexpert.csv(.gz) under the MIMIC root.")
    if metadata is None:
        raise FileNotFoundError("Could not find mimic-cxr-2.0.0-metadata.csv(.gz) under the MIMIC root.")
    return labels, metadata


def prepare_mimic_smoke_metadata(
    root: str | Path,
    labels_csv: str | Path | None = None,
    metadata_csv: str | Path | None = None,
    out_csv: str | Path | None = None,
    image_list: str | Path | None = None,
    max_images: int | None = None,
) -> Path:
    root_path = Path(root).expanduser()
    labels_path, metadata_path = _resolve_inputs(
        root_path,
        Path(labels_csv).expanduser() if labels_csv is not None else None,
        Path(metadata_csv).expanduser() if metadata_csv is not None else None,
    )

    labels_df = _read_csv(labels_path)
    metadata_df = _read_csv(metadata_path)
    join_cols = [col for col in JOIN_CANDIDATES if col in labels_df.columns and col in metadata_df.columns]
    if not {"subject_id", "study_id"}.issubset(set(join_cols)):
        raise KeyError(
            "MIMIC smoke metadata requires at least subject_id and study_id in both labels and metadata tables."
        )

    keep_cols = list(dict.fromkeys(join_cols + [c for c in METADATA_HINTS if c in metadata_df.columns]))
    merged = labels_df.merge(metadata_df[keep_cols], on=join_cols, how="left", suffixes=("", "_meta"))
    if "path" not in merged.columns:
        raise KeyError("Merged MIMIC smoke metadata is missing a path column after merge.")

    if image_list is not None:
        keep_paths = {
            line.strip()
            for line in Path(image_list).expanduser().read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        merged = merged[merged["path"].astype(str).isin(keep_paths)]
    else:
        merged = merged[merged["path"].astype(str).map(lambda p: (root_path / str(p)).exists())]

    merged = merged.drop_duplicates(subset=["path"]).sort_values("path", kind="stable").reset_index(drop=True)
    if max_images is not None:
        merged = merged.head(max_images).reset_index(drop=True)
    if len(merged) == 0:
        raise RuntimeError("No MIMIC rows remain after filtering to real files.")

    out_path = Path(out_csv).expanduser() if out_csv is not None else root_path / "mimic-cxr-2.0.0-chexpert.smoke.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create an image-level MIMIC smoke metadata CSV aligned to real downloaded JPG files."
    )
    parser.add_argument("--root", required=True, help="MIMIC-CXR-JPG root.")
    parser.add_argument("--labels_csv", default=None, help="Optional override for mimic-cxr-2.0.0-chexpert.csv(.gz).")
    parser.add_argument("--metadata_csv", default=None, help="Optional override for mimic-cxr-2.0.0-metadata.csv(.gz).")
    parser.add_argument("--out", default=None, help="Output CSV path. Defaults to <root>/mimic-cxr-2.0.0-chexpert.smoke.csv")
    parser.add_argument("--image_list", default=None, help="Optional text file with one relative image path per line to keep.")
    parser.add_argument("--max_images", type=int, default=None, help="Optional stable cap after filtering.")
    args = parser.parse_args()

    out_path = prepare_mimic_smoke_metadata(
        root=args.root,
        labels_csv=args.labels_csv,
        metadata_csv=args.metadata_csv,
        out_csv=args.out,
        image_list=args.image_list,
        max_images=args.max_images,
    )
    print(f"Saved MIMIC smoke metadata CSV: {out_path}")


if __name__ == "__main__":
    main()
