#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


def main() -> None:
    ap = argparse.ArgumentParser(description="Create a tiny deterministic CXR-like dataset for smoke tests.")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_patients", type=int, default=24)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--source_name", default="mini")
    ap.add_argument("--write_config_patch", action="store_true", help="Write a config patch JSON with data.source fields.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir).expanduser()
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    rows = []
    for i in range(args.n_patients):
        pid = f"p{i:04d}"
        label = float(i % 2 == 0)
        manufacturer = "GE" if (i % 3 == 0) else ("Philips" if (i % 3 == 1) else "Siemens")
        for t in (0, 1):
            base = 0.60 if label == 1.0 else 0.40
            arr = rng.normal(loc=base + (0.01 * t), scale=0.03, size=(64, 64)).astype(np.float32)
            arr = np.clip(arr, 0.0, 1.0)
            img_path = img_dir / f"{pid}_t{t}.png"
            Image.fromarray((arr * 255.0).astype(np.uint8), mode="L").save(img_path)
            rows.append(
                {
                    "path": img_path.name,
                    "patient_id": pid,
                    "study_id": f"s{i:05d}_{t}",
                    "view": "PA" if t == 0 else "AP",
                    "manufacturer": manufacturer,
                    "station_name": "A" if t == 0 else "B",
                    "study_date": "20240101",
                    "study_time": "080000" if t == 0 else "083000",
                    "No Finding": label,
                }
            )
    meta_csv = out_dir / "metadata.csv"
    pd.DataFrame(rows).to_csv(meta_csv, index=False)

    report = {
        "out_dir": str(out_dir),
        "images_dir": str(img_dir),
        "metadata_csv": str(meta_csv),
        "n_rows": int(len(rows)),
        "n_patients": int(args.n_patients),
        "source_name": args.source_name,
    }
    (out_dir / "minicxr_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    if args.write_config_patch:
        patch = {
            "data": {
                "sources": [
                    {
                        "name": args.source_name,
                        "root": str(out_dir),
                        "metadata_csv": str(meta_csv),
                        "image_root": str(img_dir),
                        "path_col": "path",
                        "patient_id_col": "patient_id",
                        "study_id_col": "study_id",
                        "view_col": "view",
                        "manufacturer_col": "manufacturer",
                        "station_name_col": "station_name",
                        "study_date_col": "study_date",
                        "study_time_col": "study_time",
                        "label_cols": ["No Finding"],
                    }
                ]
            }
        }
        (out_dir / "minicxr_config_patch.json").write_text(json.dumps(patch, indent=2), encoding="utf-8")

    print(f"Saved mini dataset: {out_dir}")
    print(f"metadata_csv={meta_csv}")


if __name__ == "__main__":
    main()
