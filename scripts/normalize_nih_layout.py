#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# Allow direct script execution without manual PYTHONPATH exports.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


def _find_fragment_dirs(root: Path) -> list[Path]:
    candidates = sorted(p for p in root.glob("images_*/images") if p.is_dir())
    if not candidates and (root / "images").is_dir():
        candidates = [root / "images"]
    return candidates


def normalize_nih_layout(
    root: str | Path,
    unified_images_dir: str | Path | None = None,
    metadata_csv: str | Path | None = None,
    smoke_out: str | Path | None = None,
    max_images: int | None = None,
) -> tuple[Path, Path | None]:
    root_path = Path(root).expanduser()
    images_dir = Path(unified_images_dir).expanduser() if unified_images_dir is not None else root_path / "images"
    metadata_path = Path(metadata_csv).expanduser() if metadata_csv is not None else (
        root_path / "Data_Entry_2017.csv" if (root_path / "Data_Entry_2017.csv").exists() else root_path / "data_entry_2017.csv"
    )
    if not metadata_path.exists():
        raise FileNotFoundError(f"NIH metadata CSV not found: {metadata_path}")

    fragment_dirs = _find_fragment_dirs(root_path)
    if not fragment_dirs:
        raise FileNotFoundError(f"No NIH image directories found under {root_path} (expected images_*/images or images/).")

    images_dir.mkdir(parents=True, exist_ok=True)
    linked = 0
    skipped = 0
    for src_dir in fragment_dirs:
        for src in sorted(src_dir.iterdir()):
            if not src.is_file() or src.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            dst = images_dir / src.name
            if dst.exists():
                if dst.resolve() != src.resolve():
                    raise RuntimeError(f"NIH image name collision: {dst.name} maps to multiple files.")
                skipped += 1
                continue
            dst.symlink_to(src.resolve())
            linked += 1

    smoke_path = None
    if smoke_out is not None:
        df = pd.read_csv(metadata_path)
        if "Image Index" not in df.columns:
            raise KeyError(f"{metadata_path} is missing 'Image Index'")
        keep = df["Image Index"].astype(str).map(lambda name: (images_dir / name).exists())
        out = df.loc[keep].copy()
        out = out.sort_values("Image Index", kind="stable").reset_index(drop=True)
        if max_images is not None:
            out = out.head(max_images).reset_index(drop=True)
        smoke_path = Path(smoke_out).expanduser()
        smoke_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(smoke_path, index=False)

    print(f"NIH unified images dir: {images_dir}")
    print(f"linked={linked} skipped_existing={skipped}")
    if smoke_path is not None:
        print(f"Saved NIH smoke metadata CSV: {smoke_path}")
    return images_dir, smoke_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Normalize NIH ChestX-ray14 layout into one unified images/ directory via symlinks, optionally emitting a smoke CSV."
    )
    parser.add_argument("--root", required=True, help="NIH dataset root.")
    parser.add_argument("--images_dir", default=None, help="Unified images directory. Defaults to <root>/images")
    parser.add_argument("--metadata_csv", default=None, help="Metadata CSV. Defaults to Data_Entry_2017.csv under root.")
    parser.add_argument("--smoke_out", default=None, help="Optional filtered smoke metadata CSV path.")
    parser.add_argument("--max_images", type=int, default=None, help="Optional stable cap for smoke metadata rows.")
    args = parser.parse_args()

    normalize_nih_layout(
        root=args.root,
        unified_images_dir=args.images_dir,
        metadata_csv=args.metadata_csv,
        smoke_out=args.smoke_out,
        max_images=args.max_images,
    )


if __name__ == "__main__":
    main()
