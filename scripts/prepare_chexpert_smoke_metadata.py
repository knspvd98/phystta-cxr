#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path, PurePosixPath

import pandas as pd

# Allow direct script execution without manual PYTHONPATH exports.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


CHEXPERT_PREFIXES = ("CheXpert-v1.0", "CheXpert-v1.0-small")


def _find_chexpert_csv(root: Path) -> Path:
    candidates = [
        root / "train.csv",
        root / "CheXpert-v1.0" / "train.csv",
        root / "CheXpert-v1.0-small" / "train.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"CheXpert train.csv not found under {root}")


def _normalized_candidates(raw_path: str) -> list[str]:
    text = str(PurePosixPath(str(raw_path).replace("\\", "/"))).lstrip("./")
    out = [text]
    trimmed = [text]
    for prefix in CHEXPERT_PREFIXES:
        needle = prefix + "/"
        if text.startswith(needle):
            trimmed.append(text[len(needle):])
    for item in trimmed:
        if item not in out:
            out.append(item)
        for prefix in CHEXPERT_PREFIXES:
            candidate = f"{prefix}/{item}"
            if candidate not in out:
                out.append(candidate)
    return out


def _resolve_relpath(root: Path, raw_path: str) -> str | None:
    for candidate in _normalized_candidates(raw_path):
        if (root / candidate).exists():
            return candidate
    return None


def prepare_chexpert_smoke_metadata(
    root: str | Path,
    csv_path: str | Path | None = None,
    out_csv: str | Path | None = None,
    smoke_out: str | Path | None = None,
    max_images: int | None = None,
    strict_resolved: bool = False,
) -> tuple[Path | None, Path | None]:
    root_path = Path(root).expanduser()
    src_csv = Path(csv_path).expanduser() if csv_path is not None else _find_chexpert_csv(root_path)
    df = pd.read_csv(src_csv)
    if "Path" not in df.columns:
        raise KeyError(f"{src_csv} is missing 'Path'")

    resolved = df["Path"].astype(str).map(lambda p: _resolve_relpath(root_path, p))
    missing = int(resolved.isna().sum())
    if strict_resolved and missing > 0:
        raise RuntimeError(f"{src_csv}: {missing} rows do not resolve under {root_path}")

    resolved_out = None
    if out_csv is not None:
        if missing > 0:
            raise RuntimeError(
                f"{src_csv}: cannot write fully resolved CSV because {missing} rows do not resolve under {root_path}"
            )
        out = df.copy()
        out["Path"] = resolved.astype(str)
        resolved_out = Path(out_csv).expanduser()
        resolved_out.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(resolved_out, index=False)

    smoke_csv = None
    if smoke_out is not None:
        smoke = df.loc[resolved.notna()].copy()
        smoke["Path"] = resolved.loc[resolved.notna()].astype(str)
        smoke = smoke.sort_values("Path", kind="stable").reset_index(drop=True)
        if max_images is not None:
            smoke = smoke.head(max_images).reset_index(drop=True)
        smoke_csv = Path(smoke_out).expanduser()
        smoke_csv.parent.mkdir(parents=True, exist_ok=True)
        smoke.to_csv(smoke_csv, index=False)

    print(f"CheXpert source CSV: {src_csv}")
    print(f"resolved_rows={int(resolved.notna().sum())} missing_rows={missing}")
    if resolved_out is not None:
        print(f"Saved resolved CheXpert CSV: {resolved_out}")
    if smoke_csv is not None:
        print(f"Saved CheXpert smoke metadata CSV: {smoke_csv}")
    return resolved_out, smoke_csv


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Normalize CheXpert CSV paths against a chosen dataset root and optionally emit a filtered smoke CSV."
    )
    parser.add_argument("--root", required=True, help="CheXpert dataset root.")
    parser.add_argument("--csv", default=None, help="Input train CSV. Defaults to the first standard CheXpert train.csv found.")
    parser.add_argument("--out", default=None, help="Optional fully resolved CSV output path.")
    parser.add_argument("--smoke_out", default=None, help="Optional filtered smoke metadata CSV output path.")
    parser.add_argument("--max_images", type=int, default=None, help="Optional stable cap for the smoke CSV.")
    parser.add_argument("--strict_resolved", action="store_true", help="Fail if any CSV row does not resolve under the chosen root.")
    args = parser.parse_args()

    prepare_chexpert_smoke_metadata(
        root=args.root,
        csv_path=args.csv,
        out_csv=args.out,
        smoke_out=args.smoke_out,
        max_images=args.max_images,
        strict_resolved=args.strict_resolved,
    )


if __name__ == "__main__":
    main()
