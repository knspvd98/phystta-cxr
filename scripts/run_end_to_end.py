#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List

# Allow direct script execution without manual PYTHONPATH exports.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phystta.config.loader import load_config, save_config


def _run(cmd: List[str], dry_run: bool) -> None:
    print("RUN:", " ".join(cmd))
    if dry_run:
        return
    env = os.environ.copy()
    root = str(Path(__file__).resolve().parents[1])
    env["PYTHONPATH"] = root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    subprocess.run(cmd, check=True, env=env, cwd=root)


def _first_shift(cfg, override: str | None) -> str:
    if override:
        return override
    for s in list(cfg.intervention.shift_types or []) + list(cfg.erasure_eval.shift_types or []):
        if str(s) != "mixed":
            return str(s)
    return "heel"


def _patch_head_paths(cfg, head_path: Path, head_meta: Path, encoder_profile: str | None) -> None:
    if encoder_profile:
        found = False
        for p in cfg.model.encoder_profiles or []:
            if p.name == encoder_profile:
                p.encoder.head_path = head_path
                p.encoder.head_meta_path = head_meta
                found = True
                break
        if not found:
            raise RuntimeError(f"encoder_profile '{encoder_profile}' not found in config.")
    else:
        cfg.model.encoder.head_path = head_path
        cfg.model.encoder.head_meta_path = head_meta


def main() -> None:
    ap = argparse.ArgumentParser(
        description="End-to-end runner: validate_env -> build_index -> train_head -> collect_probes -> "
        "fit_eraser -> calibrate_realism -> paper_suite"
    )
    ap.add_argument("--config", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--run_id", default=None)
    ap.add_argument("--encoder_profile", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--shift_type", default=None)
    ap.add_argument("--head_epochs", type=int, default=3)
    ap.add_argument("--probe_batch_size", type=int, default=16)
    ap.add_argument("--max_images", type=int, default=200)
    ap.add_argument("--max_pairs", type=int, default=300)
    ap.add_argument("--bootstrap_samples", type=int, default=500)
    ap.add_argument("--skip_realism_calibration", action="store_true")
    ap.add_argument("--build_index_if_missing", action="store_true")
    ap.add_argument("--bootstrap_cache", action="store_true")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    cfg_src = Path(args.config).expanduser()
    cfg = load_config(str(cfg_src))
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_root = Path(args.out_dir).expanduser() / run_id
    run_root.mkdir(parents=True, exist_ok=True)

    heads_dir = run_root / "heads"
    probes_dir = run_root / "probes"
    eraser_dir = run_root / "eraser"
    realism_dir = run_root / "realism"
    suite_dir = run_root / "paper_suite"
    env_dir = run_root / "env"
    for d in [heads_dir, probes_dir, eraser_dir, realism_dir, suite_dir, env_dir]:
        d.mkdir(parents=True, exist_ok=True)

    effective_cfg = run_root / "config.effective.yaml"
    head_path = heads_dir / "head.pt"
    head_meta = heads_dir / "head_meta.json"
    probes_path = probes_dir / "probes.pt"
    eraser_path = eraser_dir / "eraser.pt"
    eraser_meta = eraser_dir / "eraser_meta.json"
    realism_path = realism_dir / "realism_calibrator.json"

    cfg_out = load_config(str(cfg_src))
    _patch_head_paths(cfg_out, head_path, head_meta, args.encoder_profile)
    cfg_out.intervention.realism_path = realism_path
    if cfg_out.data.prebuilt_index_csv is None:
        cfg_out.data.prebuilt_index_csv = run_root / "data_index.csv"
        cfg_out.data.prebuilt_index_sha256 = None
    save_config(str(effective_cfg), cfg_out)

    shift_type = _first_shift(cfg_out, args.shift_type)

    cmds: List[List[str]] = []
    cmds.append(
        [
            sys.executable,
            "scripts/validate_env.py",
            "--config",
            str(effective_cfg),
            "--out",
            str(env_dir / "env.json"),
        ]
    )
    if args.build_index_if_missing:
        cmds.append(
            [
                sys.executable,
                "scripts/build_data_index.py",
                "--config",
                str(effective_cfg),
                "--out_csv",
                str(cfg_out.data.prebuilt_index_csv),
                "--set_config",
            ]
        )
    cmds.append(
        [
            sys.executable,
            "scripts/train_linear_head.py",
            "--config",
            str(effective_cfg),
            "--epochs",
            str(args.head_epochs),
            "--out_head",
            str(head_path),
            "--out_meta",
            str(head_meta),
            "--batch_size",
            str(cfg_out.train.batch_size),
        ]
    )
    cmds.append(
        [
            sys.executable,
            "scripts/collect_probes.py",
            "--config",
            str(effective_cfg),
            "--shift_types",
            shift_type,
            "--batch_size",
            str(args.probe_batch_size),
            "--max_images",
            str(args.max_images),
            "--out",
            str(probes_path),
        ]
    )
    cmds.append(
        [
            sys.executable,
            "scripts/fit_eraser.py",
            "--config",
            str(effective_cfg),
            "--probes",
            str(probes_path),
            "--shift_type",
            shift_type,
            "--out",
            str(eraser_path),
            "--meta",
            str(eraser_meta),
            "--self_check",
        ]
    )
    if not args.skip_realism_calibration:
        cmds.append(
            [
                sys.executable,
                "scripts/calibrate_physics_realism.py",
                "--config",
                str(effective_cfg),
                "--out",
                str(realism_path),
                "--max_pairs",
                str(args.max_pairs),
                "--max_images",
                str(args.max_images),
                "--write_config",
            ]
        )
    paper_cmd = [
        sys.executable,
        "scripts/run_paper_suite.py",
        "--config",
        str(effective_cfg),
        "--eraser",
        str(eraser_path),
        "--out_dir",
        str(suite_dir),
        "--max_pairs",
        str(args.max_pairs),
        "--max_images",
        str(args.max_images),
        "--bootstrap_samples",
        str(args.bootstrap_samples),
    ]
    if args.bootstrap_cache:
        paper_cmd.append("--bootstrap_cache")
    if args.build_index_if_missing:
        paper_cmd.append("--build_index_if_missing")
    if args.skip_realism_calibration:
        paper_cmd.append("--skip_realism_calibration")
    else:
        # realism already produced in this runner
        paper_cmd.append("--skip_realism_calibration")
    cmds.append(paper_cmd)

    if args.encoder_profile:
        for cmd in cmds:
            cmd.extend(["--encoder_profile", args.encoder_profile])
    if args.seed is not None:
        for cmd in cmds:
            cmd.extend(["--seed", str(args.seed)])
    if args.device:
        for cmd in cmds:
            cmd.extend(["--device", args.device])

    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source_config": str(cfg_src),
        "effective_config": str(effective_cfg),
        "run_root": str(run_root),
        "encoder_profile": args.encoder_profile,
        "shift_type": shift_type,
        "commands": cmds,
        "artifacts": {
            "head": str(head_path),
            "head_meta": str(head_meta),
            "probes": str(probes_path),
            "eraser": str(eraser_path),
            "eraser_meta": str(eraser_meta),
            "realism": str(realism_path),
            "paper_suite_dir": str(suite_dir),
            "env": str(env_dir / "env.json"),
        },
    }
    (run_root / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if args.dry_run:
        print(f"Saved dry-run manifest: {run_root / 'run_manifest.json'}")
        for cmd in cmds:
            _run(cmd, dry_run=True)
        return

    for cmd in cmds:
        _run(cmd, dry_run=False)

    print(f"Run complete. Manifest: {run_root / 'run_manifest.json'}")


if __name__ == "__main__":
    main()
