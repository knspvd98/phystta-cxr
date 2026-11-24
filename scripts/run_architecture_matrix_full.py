#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

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


def _select_profiles(cfg, raw: str | None) -> List[str]:
    if raw:
        return [p.strip() for p in raw.split(",") if p.strip()]
    if cfg.architecture_study.encoder_profiles:
        return list(cfg.architecture_study.encoder_profiles)
    if cfg.model.encoder_profiles:
        return [p.name for p in cfg.model.encoder_profiles]
    return ["__default__"]


def _first_shift(cfg, override: str | None) -> str:
    if override:
        return str(override)
    for s in list(cfg.intervention.shift_types or []) + list(cfg.erasure_eval.shift_types or []):
        if str(s) != "mixed":
            return str(s)
    return "heel"


def _patch_profile_config(cfg, profile: str, head_path: Path, head_meta: Path, realism_path: Path) -> None:
    if profile == "__default__":
        cfg.model.encoder.head_path = head_path
        cfg.model.encoder.head_meta_path = head_meta
    else:
        found = False
        for p in cfg.model.encoder_profiles or []:
            if p.name == profile:
                p.encoder.head_path = head_path
                p.encoder.head_meta_path = head_meta
                found = True
                break
        if not found:
            raise RuntimeError(f"encoder_profile '{profile}' not found in model.encoder_profiles")
    cfg.intervention.realism_path = realism_path


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Full architecture sweep runner: train_linear_head -> collect_probes -> fit_eraser -> "
            "run_erasure_eval -> run_safety_policy_eval for each profile, then aggregate."
        )
    )
    ap.add_argument("--config", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--profiles", default=None, help="comma-separated encoder profiles; default=architecture_study.encoder_profiles")
    ap.add_argument("--shift_type", default=None, help="single shift for eraser fit/probes")
    ap.add_argument("--modes", default=None, help="modes for run_erasure_eval")
    ap.add_argument("--intensity_grid", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--head_epochs", type=int, default=3)
    ap.add_argument("--probe_batch_size", type=int, default=16)
    ap.add_argument("--max_images", type=int, default=200)
    ap.add_argument("--max_pairs", type=int, default=300)
    ap.add_argument("--bootstrap_samples", type=int, default=500)
    ap.add_argument("--target_sensitivities", default="0.90,0.95")
    ap.add_argument("--target_specificities", default="0.90,0.95")
    ap.add_argument("--skip_realism_calibration", action="store_true")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    profiles = _select_profiles(cfg, args.profiles)
    fit_shift = _first_shift(cfg, args.shift_type)
    modes = args.modes or ",".join(cfg.architecture_study.modes or (cfg.erasure_eval.modes or [cfg.erasure_eval.mode]))
    root = Path(args.config).expanduser()

    by_profile_cmds: Dict[str, List[List[str]]] = {}
    for profile in profiles:
        profile_key = str(profile)
        profile_dir = out_dir / profile_key
        profile_dir.mkdir(parents=True, exist_ok=True)
        profile_cfg_path = profile_dir / "config.profile.yaml"
        head_path = profile_dir / "heads" / "head.pt"
        head_meta = profile_dir / "heads" / "head_meta.json"
        probes_path = profile_dir / "probes" / "probes.pt"
        eraser_path = profile_dir / "eraser" / "eraser.pt"
        eraser_meta = profile_dir / "eraser" / "eraser_meta.json"
        realism_path = profile_dir / "realism" / "realism_calibrator.json"
        eval_dir = profile_dir / "eval"
        safety_dir = profile_dir / "safety"
        shift_id_path = safety_dir / "shift_id.pkl"
        policy_json_path = safety_dir / "policy.json"

        head_path.parent.mkdir(parents=True, exist_ok=True)
        probes_path.parent.mkdir(parents=True, exist_ok=True)
        eraser_path.parent.mkdir(parents=True, exist_ok=True)
        realism_path.parent.mkdir(parents=True, exist_ok=True)
        eval_dir.mkdir(parents=True, exist_ok=True)
        safety_dir.mkdir(parents=True, exist_ok=True)

        cfg_profile = load_config(str(root))
        _patch_profile_config(cfg_profile, profile_key, head_path, head_meta, realism_path)
        save_config(str(profile_cfg_path), cfg_profile)

        cmds: List[List[str]] = []
        cmd_head = [
            sys.executable,
            "scripts/train_linear_head.py",
            "--config",
            str(profile_cfg_path),
            "--epochs",
            str(args.head_epochs),
            "--out_head",
            str(head_path),
            "--out_meta",
            str(head_meta),
        ]
        if profile_key != "__default__":
            cmd_head += ["--encoder_profile", profile_key]
        if args.seed is not None:
            cmd_head += ["--seed", str(args.seed)]
        if args.device:
            cmd_head += ["--device", args.device]
        cmds.append(cmd_head)

        if not args.skip_realism_calibration:
            cmd_real = [
                sys.executable,
                "scripts/calibrate_physics_realism.py",
                "--config",
                str(profile_cfg_path),
                "--out",
                str(realism_path),
                "--max_pairs",
                str(args.max_pairs),
                "--max_images",
                str(args.max_images),
                "--write_config",
            ]
            if profile_key != "__default__":
                cmd_real += ["--encoder_profile", profile_key]
            if args.seed is not None:
                cmd_real += ["--seed", str(args.seed)]
            if args.device:
                cmd_real += ["--device", args.device]
            cmds.append(cmd_real)

        cmd_probes = [
            sys.executable,
            "scripts/collect_probes.py",
            "--config",
            str(profile_cfg_path),
            "--shift_types",
            fit_shift,
            "--batch_size",
            str(args.probe_batch_size),
            "--max_images",
            str(args.max_images),
            "--out",
            str(probes_path),
        ]
        if profile_key != "__default__":
            cmd_probes += ["--encoder_profile", profile_key]
        if args.seed is not None:
            cmd_probes += ["--seed", str(args.seed)]
        if args.device:
            cmd_probes += ["--device", args.device]
        cmds.append(cmd_probes)

        cmd_eraser = [
            sys.executable,
            "scripts/fit_eraser.py",
            "--config",
            str(profile_cfg_path),
            "--probes",
            str(probes_path),
            "--shift_type",
            fit_shift,
            "--out",
            str(eraser_path),
            "--meta",
            str(eraser_meta),
            "--self_check",
        ]
        if args.seed is not None:
            cmd_eraser += ["--seed", str(args.seed)]
        cmds.append(cmd_eraser)

        cmd_eval = [
            sys.executable,
            "scripts/run_erasure_eval.py",
            "--config",
            str(profile_cfg_path),
            "--eraser",
            str(eraser_path),
            "--out_dir",
            str(eval_dir),
            "--modes",
            modes,
        ]
        if profile_key != "__default__":
            cmd_eval += ["--encoder_profile", profile_key]
        if args.shift_type:
            cmd_eval += ["--shift_type", args.shift_type]
        if args.intensity_grid:
            cmd_eval += ["--intensity_grid", args.intensity_grid]
        if args.seed is not None:
            cmd_eval += ["--seed", str(args.seed)]
        if args.device:
            cmd_eval += ["--device", args.device]
        cmds.append(cmd_eval)

        cmd_safety = [
            sys.executable,
            "scripts/run_safety_policy_eval.py",
            "--config",
            str(profile_cfg_path),
            "--eraser",
            str(eraser_path),
            "--out_dir",
            str(safety_dir),
            "--save_shift_id",
            str(shift_id_path),
            "--save_policy",
            str(policy_json_path),
            "--bootstrap_samples",
            str(args.bootstrap_samples),
            "--target_sensitivities",
            args.target_sensitivities,
            "--target_specificities",
            args.target_specificities,
        ]
        if profile_key != "__default__":
            cmd_safety += ["--encoder_profile", profile_key]
        if args.shift_type:
            cmd_safety += ["--shift_type", args.shift_type]
        if args.intensity_grid:
            cmd_safety += ["--intensity_grid", args.intensity_grid]
        if args.seed is not None:
            cmd_safety += ["--seed", str(args.seed)]
        if args.device:
            cmd_safety += ["--device", args.device]
        cmds.append(cmd_safety)

        by_profile_cmds[profile_key] = cmds
        for cmd in cmds:
            _run(cmd, dry_run=args.dry_run)

    aggregate_cmd = [
        sys.executable,
        "scripts/run_architecture_matrix.py",
        "--config",
        args.config,
        "--out_dir",
        str(out_dir),
        "--aggregate_only",
    ]
    if args.shift_type:
        aggregate_cmd += ["--shift_type", args.shift_type]
    if args.intensity_grid:
        aggregate_cmd += ["--intensity_grid", args.intensity_grid]
    if args.modes:
        aggregate_cmd += ["--modes", args.modes]
    _run(aggregate_cmd, dry_run=args.dry_run)

    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config": str(root),
        "profiles": profiles,
        "fit_shift_type": fit_shift,
        "out_dir": str(out_dir),
        "commands": by_profile_cmds,
        "aggregate_command": aggregate_cmd,
    }
    (out_dir / "architecture_matrix_full_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved manifest: {out_dir / 'architecture_matrix_full_manifest.json'}")


if __name__ == "__main__":
    main()
