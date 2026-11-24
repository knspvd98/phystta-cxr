#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from phystta.config.loader import load_config, save_config
from phystta.data import DataEngine


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description="Build deterministic prebuilt index for paper-tier runs")
    ap.add_argument("--config", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--out_manifest", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--set_config", action="store_true", help="Write data.prebuilt_index_csv/sha256 into config")
    args = ap.parse_args()

    cfg_path = Path(args.config).expanduser()
    cfg = load_config(str(cfg_path))

    out_csv = Path(args.out_csv).expanduser()
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # Avoid recursion through prebuilt mode while generating new index.
    cfg.data.prebuilt_index_csv = None
    cfg.data.prebuilt_index_sha256 = None

    engine = DataEngine(cfg.data)
    index = engine.build_index(force=args.force)
    index.to_csv(out_csv, index=False)
    digest = _sha256(out_csv)

    manifest = {
        "index_csv": str(out_csv),
        "sha256": digest,
        "n_rows": int(len(index)),
        "columns": sorted(index.columns.tolist()),
        "split_group_key": cfg.data.split_group_key,
        "label_cols": list(engine.label_cols),
        "sources": sorted(set(index["source"].astype(str).tolist())) if "source" in index.columns else [],
    }
    out_manifest = Path(args.out_manifest).expanduser() if args.out_manifest else out_csv.with_suffix(".manifest.json")
    out_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if args.set_config:
        cfg2 = load_config(str(cfg_path))
        cfg2.data.prebuilt_index_csv = out_csv
        cfg2.data.prebuilt_index_sha256 = digest
        save_config(str(cfg_path), cfg2)

    print(f"Saved prebuilt index: {out_csv}")
    print(f"Saved manifest: {out_manifest}")
    print(f"sha256={digest}")


if __name__ == "__main__":
    main()
