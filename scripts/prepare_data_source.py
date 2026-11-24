#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Dict

from phystta.config.loader import load_config, save_config
from phystta.data import DataEngine
from phystta.data.templates import expected_files_message, resolve_dataset_source


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _copytree(src: Path, dst: Path) -> None:
    if not src.exists():
        raise RuntimeError(f"--copy_from does not exist: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        raise RuntimeError(f"Destination root already exists: {dst}. Refusing to overwrite.")
    shutil.copytree(src, dst)


def _update_cfg_source(cfg, source_name: str, root: Path, metadata_csv: Path | None, image_root: Path | None) -> bool:
    updated = False
    for src in cfg.data.sources:
        key = src.domain_label or src.name
        if src.name == source_name or key == source_name:
            src.root = root
            if metadata_csv is not None:
                src.metadata_csv = metadata_csv
            if image_root is not None:
                src.image_root = image_root
            updated = True
            break
    return updated


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Prepare/validate local dataset source for RunPod and write deterministic data manifest/index."
    )
    ap.add_argument("--config", required=True)
    ap.add_argument("--source_name", required=True, help="Source key from config (name or domain_label).")
    ap.add_argument("--dataset_root", required=True)
    ap.add_argument("--metadata_csv", default=None)
    ap.add_argument("--image_root", default=None)
    ap.add_argument("--copy_from", default=None, help="Optional local path to copy dataset into --dataset_root.")
    ap.add_argument("--index_out", default=None, help="Optional output CSV for deterministic prebuilt index.")
    ap.add_argument("--manifest_out", default=None, help="Path to write data manifest JSON.")
    ap.add_argument("--set_config", action="store_true", help="Patch config source paths and prebuilt index fields.")
    ap.add_argument("--force_index", action="store_true")
    args = ap.parse_args()

    cfg_path = Path(args.config).expanduser()
    cfg = load_config(str(cfg_path))
    dataset_root = Path(args.dataset_root).expanduser()
    metadata_csv = Path(args.metadata_csv).expanduser() if args.metadata_csv else None
    image_root = Path(args.image_root).expanduser() if args.image_root else None

    if args.copy_from:
        _copytree(Path(args.copy_from).expanduser(), dataset_root)

    if not dataset_root.exists():
        raise RuntimeError(f"dataset_root does not exist: {dataset_root}")

    # Validate template-level expected files if metadata_csv is not explicit.
    template_status: Dict[str, Any] = {"ok": True}
    if metadata_csv is None:
        try:
            resolved = resolve_dataset_source(args.source_name, str(dataset_root))
            metadata_csv = resolved.metadata_csv
            if image_root is None:
                image_root = resolved.image_root
        except Exception as e:
            template_status = {
                "ok": False,
                "error": f"{type(e).__name__}: {e}",
                "expected_files": expected_files_message(args.source_name),
            }
            raise

    if args.set_config:
        if not _update_cfg_source(cfg, args.source_name, dataset_root, metadata_csv, image_root):
            raise RuntimeError(f"Source '{args.source_name}' not found in config {cfg_path}")
        save_config(str(cfg_path), cfg)
        cfg = load_config(str(cfg_path))

    # Build deterministic index (optional but recommended).
    index_csv = Path(args.index_out).expanduser() if args.index_out else None
    index_sha = None
    n_rows = 0
    n_patients = 0
    label_cols = []
    if index_csv is not None:
        cfg2 = load_config(str(cfg_path))
        cfg2.data.prebuilt_index_csv = None
        cfg2.data.prebuilt_index_sha256 = None
        engine = DataEngine(cfg2.data)
        # When source paths are updated in this command, avoid stale cached indexes.
        force_build = bool(
            args.force_index
            or args.set_config
            or metadata_csv is not None
            or image_root is not None
        )
        df = engine.build_index(force=force_build)
        df.to_csv(index_csv, index=False)
        index_sha = _sha256(index_csv)
        n_rows = int(len(df))
        n_patients = int(df.get("global_patient_id", df["patient_id"]).astype(str).nunique())
        label_cols = list(engine.label_cols)
        if args.set_config:
            cfg3 = load_config(str(cfg_path))
            cfg3.data.prebuilt_index_csv = index_csv
            cfg3.data.prebuilt_index_sha256 = index_sha
            save_config(str(cfg_path), cfg3)

    manifest = {
        "config": str(cfg_path),
        "source_name": args.source_name,
        "dataset_root": str(dataset_root),
        "metadata_csv": str(metadata_csv) if metadata_csv is not None else None,
        "image_root": str(image_root) if image_root is not None else None,
        "template_validation": template_status,
        "index_csv": str(index_csv) if index_csv is not None else None,
        "index_sha256": index_sha,
        "n_rows": n_rows,
        "n_patients": n_patients,
        "label_cols": label_cols,
    }
    manifest_out = (
        Path(args.manifest_out).expanduser()
        if args.manifest_out
        else (index_csv.with_suffix(".data_manifest.json") if index_csv is not None else dataset_root / "data_manifest.json")
    )
    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    manifest_out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Saved data manifest: {manifest_out}")
    if index_csv is not None:
        print(f"Saved index: {index_csv}")
        print(f"index_sha256={index_sha}")


if __name__ == "__main__":
    main()
