#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from phystta.config.loader import load_config
from phystta.data import DataEngine, collect_dicom_leakage_features, audit_site_leakage
from phystta.utils import set_global_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit site/vendor leakage in preprocessed CXR tensors")
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default=None)
    parser.add_argument("--group_col", default="manufacturer")
    parser.add_argument("--fallback_col", default="station_name")
    parser.add_argument("--max_batches", type=int, default=100)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else cfg.runtime.seed
    set_global_seed(seed, deterministic=cfg.runtime.deterministic, benchmark=cfg.runtime.benchmark)

    split = args.split if args.split is not None else cfg.data.calib_split
    engine = DataEngine(cfg.data)
    engine.build_index()
    engine.create_splits()
    loaders = engine.build_dataloaders(batch_size=cfg.train.batch_size, num_workers=cfg.train.num_workers, splits=[split])
    loader = loaders[split]
    X, y, groups = collect_dicom_leakage_features(
        loader,
        group_col=args.group_col,
        fallback_col=args.fallback_col,
        max_batches=args.max_batches,
    )
    report = audit_site_leakage(X, y, groups=groups, seed=seed)
    report["split"] = split
    report["group_col"] = args.group_col
    report["fallback_col"] = args.fallback_col
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(f"Saved audit to {out_path}")


if __name__ == "__main__":
    main()
