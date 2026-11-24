#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List

# Allow direct script execution without manual PYTHONPATH exports.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phystta.config.loader import load_config
from phystta.data import DataEngine
from phystta.intervention import eraser_fingerprint_mismatches, load_eraser
from phystta.models import (
    EncoderFactory,
    build_encoder_fingerprint,
    load_head_meta,
    validate_head_meta,
)
from phystta.shifts.realism import calibrator_fingerprint_mismatches, load_calibrator
from phystta.utils import preprocess_signature


def _run(cmd: List[str], dry_run: bool) -> None:
    print("RUN:", " ".join(cmd))
    if dry_run:
        return
    env = os.environ.copy()
    root = str(Path(__file__).resolve().parents[1])
    env["PYTHONPATH"] = root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    subprocess.run(cmd, check=True, env=env, cwd=root)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_if_exists(path: Path) -> str | None:
    if not path.exists():
        return None
    return _sha256_file(path)


def _aggregate_hash_map(items: dict[str, str | None]) -> str:
    """Build one deterministic digest over per-file hashes for manifest pinning."""
    h = hashlib.sha256()
    for key in sorted(items):
        val = items[key] if items[key] is not None else "<missing>"
        h.update(f"{key}:{val}\n".encode("utf-8"))
    return h.hexdigest()


def _first_anchor_shift(cfg, shift_type_arg: str | None) -> str:
    if shift_type_arg:
        return shift_type_arg
    preferred = list(cfg.erasure_eval.shift_types or [])
    for s in preferred + list(cfg.intervention.shift_types or []):
        if str(s) != "mixed":
            return str(s)
    return "heel"


def _expected_label_cols(cfg) -> List[str]:
    if cfg.data.label_cols:
        return list(cfg.data.label_cols)
    out = []
    for src in cfg.data.sources:
        for col in src.label_cols or []:
            if col not in out:
                out.append(col)
    return out


def _pair_metadata_status(cfg, *, pair_mode: str | None, require_resolution: bool) -> list[dict]:
    return (
        DataEngine(cfg.data)
        .attach_runtime_contract(cfg)
        .pair_metadata_status(pair_mode=pair_mode, require_resolution=require_resolution)
    )


def _format_pair_metadata_gaps(statuses: list[dict]) -> str:
    parts = []
    for status in statuses:
        missing = status.get("missing_capabilities") or status.get("missing_columns") or []
        if missing:
            parts.append(f"{status.get('source')}[{','.join(str(x) for x in missing)}]")
    return ", ".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run full paper-tier suite: realism -> erasure -> safety -> anchor -> report")
    ap.add_argument("--config", required=True)
    ap.add_argument("--eraser", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--encoder_profile", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--shift_type", default=None)
    ap.add_argument("--bootstrap_samples", type=int, default=500)
    ap.add_argument("--target_sensitivities", default="0.90,0.95")
    ap.add_argument("--target_specificities", default="0.90,0.95")
    ap.add_argument("--max_pairs", type=int, default=300)
    ap.add_argument("--max_images", type=int, default=300)
    ap.add_argument("--calibrate_source", default=None)
    ap.add_argument("--external_eval_sources", default=None, help="comma-separated external source names for frozen external eval")
    ap.add_argument("--bootstrap_cache", action="store_true", help="Run scripts/cache_models.py before paper suite.")
    ap.add_argument("--cache_profiles", default="study", choices=["none", "study", "all"])
    ap.add_argument("--build_index_if_missing", action="store_true", help="Auto-run scripts/build_data_index.py if prebuilt index is missing.")
    ap.add_argument("--skip_realism_calibration", action="store_true")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    preflight_errors: List[str] = []
    preflight_notes: List[str] = []
    open_data_fallback = bool(getattr(cfg.experiment, "open_data_fallback", False))
    if not open_data_fallback:
        if cfg.experiment.tier != "paper":
            raise RuntimeError("run_paper_suite requires experiment.tier=paper")
        if not cfg.intervention.use_realism:
            raise RuntimeError("run_paper_suite requires intervention.use_realism=true")
        if cfg.intervention.realism_path is None:
            raise RuntimeError("run_paper_suite requires intervention.realism_path")
        if cfg.runtime.allow_download:
            raise RuntimeError("run_paper_suite requires runtime.allow_download=false in paper tier.")
    else:
        preflight_notes.append(
            "Open-data fallback run enabled: this is not the full credentialed paper-tier suite."
        )
        preflight_notes.append("Synthetic intervention grid only: real-pair realism is disabled in open-data fallback mode.")
        preflight_notes.append("Real-anchor evaluation is skipped in open-data fallback mode.")

    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    erasure_dir = out_dir / "erasure"
    safety_dir = out_dir / "safety"
    anchor_dir = out_dir / "anchor"
    report_dir = out_dir / "paper_report"
    dissertation_dir = out_dir / "dissertation_artifacts"
    vlm_dir = out_dir / "vlm" if cfg.vlm is not None and cfg.vlm.enabled else None
    external_dir = out_dir / "external" if args.external_eval_sources else None
    realism_out = (
        Path(cfg.intervention.realism_path).expanduser()
        if cfg.intervention.realism_path is not None
        else out_dir / "realism" / "realism_calibrator.json"
    )
    realism_out.parent.mkdir(parents=True, exist_ok=True)
    effective_use_realism = bool(cfg.intervention.use_realism) and not open_data_fallback
    scenario_mode = str(getattr(cfg.erasure_eval, "scenario_mode", "atomic_calibrated_eval"))
    realism_pair_mode = str(getattr(cfg.intervention, "pair_mode", "")) if effective_use_realism else None
    pair_metadata_status = _pair_metadata_status(
        cfg,
        pair_mode=realism_pair_mode,
        require_resolution=bool(effective_use_realism and realism_pair_mode and realism_pair_mode != "strict_device_change"),
    )
    sources_without_real_pairs = [s for s in pair_metadata_status if not bool(s.get("supports_real_pairs"))]
    sources_without_pair_metadata = [
        s for s in pair_metadata_status
        if realism_pair_mode and not bool(s.get("supports_requested_pair_mode"))
    ]
    enable_real_anchor = bool(effective_use_realism and scenario_mode == "atomic_calibrated_eval")
    if scenario_mode == "mixed_stress_eval" and effective_use_realism:
        preflight_errors.append("mixed_stress_eval must keep intervention.use_realism=false.")
    if open_data_fallback:
        if sources_without_real_pairs:
            gap_msg = _format_pair_metadata_gaps(sources_without_real_pairs)
            preflight_notes.append(f"Sources without real-pair metadata: {gap_msg}")
        else:
            preflight_notes.append("Pair-metadata-dependent stages remain disabled because experiment.open_data_fallback=true.")
    elif sources_without_pair_metadata:
        gap_msg = _format_pair_metadata_gaps(sources_without_pair_metadata)
        if effective_use_realism:
            preflight_errors.append(
                f"Real-pair realism with pair_mode='{realism_pair_mode}' requires capability-compatible metadata for all sources: "
                + gap_msg
            )
    elif effective_use_realism:
        preflight_notes.append(
            f"Real-pair realism preflight passed for pair_mode='{realism_pair_mode}' under capability-aware acquisition-context checks."
        )
    if scenario_mode == "mixed_stress_eval" and not open_data_fallback:
        preflight_notes.append("Real-anchor evaluation is disabled for mixed_stress_eval because this contour is not calibrated on real pairs.")
    eraser_path = Path(args.eraser).expanduser()
    enc_cfg = EncoderFactory.resolve_encoder_config(cfg, profile=args.encoder_profile)
    expected_fp = build_encoder_fingerprint(enc_cfg)
    expected_fp["preprocess_signature"] = preprocess_signature(cfg)
    prebuilt_csv = Path(cfg.data.prebuilt_index_csv).expanduser() if cfg.data.prebuilt_index_csv is not None else None
    if prebuilt_csv is None:
        if open_data_fallback:
            preflight_notes.append(
                "data.prebuilt_index_csv is not set; open-data fallback will rely on source metadata directly."
            )
        else:
            preflight_errors.append("data.prebuilt_index_csv is not set (paper-tier requires deterministic index).")
    else:
        if not prebuilt_csv.exists():
            if args.build_index_if_missing:
                preflight_notes.append(f"prebuilt index missing; will build at {prebuilt_csv}")
            else:
                preflight_errors.append(
                    f"Prebuilt index not found: {prebuilt_csv}. Use --build_index_if_missing or run scripts/build_data_index.py."
                )
        elif cfg.data.prebuilt_index_sha256:
            got = _sha256_file(prebuilt_csv)
            if got != cfg.data.prebuilt_index_sha256:
                preflight_errors.append(
                    f"Prebuilt index SHA mismatch: expected={cfg.data.prebuilt_index_sha256} got={got} ({prebuilt_csv})"
                )
    if not eraser_path.exists():
        preflight_errors.append(f"Eraser artifact not found: {eraser_path}")
    else:
        preflight_notes.append(f"eraser={eraser_path}")
        try:
            eraser = load_eraser(str(eraser_path))
            mismatches = eraser_fingerprint_mismatches(eraser, expected_fp)
            if mismatches:
                preflight_errors.append(
                    "Eraser fingerprint mismatch against active encoder/preprocess: " + "; ".join(mismatches)
                )
        except Exception as e:
            preflight_errors.append(f"Failed to load eraser artifact {eraser_path}: {e}")

    if enc_cfg.head_path is None:
        preflight_errors.append(
            "Selected encoder profile has no head_path configured. "
            "Paper suite requires a trained compatible head + metadata."
        )
    else:
        hp = Path(enc_cfg.head_path).expanduser()
        if not hp.exists():
            preflight_errors.append(f"Head artifact not found: {hp}")
        else:
            expected_labels = _expected_label_cols(cfg)
            meta = load_head_meta(
                hp,
                explicit_meta=Path(enc_cfg.head_meta_path).expanduser() if enc_cfg.head_meta_path is not None else None,
            )
            if meta is None:
                preflight_errors.append(
                    f"Head metadata not found for {hp}. Provide sidecar (*.meta.json / *_meta.json / head_meta.json) "
                    "or set encoder.head_meta_path."
                )
            else:
                errs = validate_head_meta(
                    meta,
                    enc_cfg=enc_cfg,
                    expected_num_classes=cfg.model.num_classes,
                    expected_label_cols=expected_labels if expected_labels else None,
                )
                if errs:
                    preflight_errors.append("Head metadata validation failed: " + "; ".join(errs))

    if effective_use_realism and cfg.intervention.realism_path is None:
        preflight_errors.append("intervention.use_realism=true but intervention.realism_path is not set.")

    if effective_use_realism and args.skip_realism_calibration:
        if not realism_out.exists():
            preflight_errors.append(
                f"--skip_realism_calibration was set but realism calibrator file is missing: {realism_out}"
            )
        else:
            try:
                calib = load_calibrator(str(realism_out))
                mismatches = calibrator_fingerprint_mismatches(calib, expected_fp)
                if mismatches:
                    preflight_errors.append(
                        "Realism calibrator fingerprint mismatch against active encoder/preprocess: "
                        + "; ".join(mismatches)
                    )
            except Exception as e:
                preflight_errors.append(f"Failed to load realism calibrator {realism_out}: {e}")
    if cfg.vlm is not None and cfg.vlm.enabled:
        if cfg.vlm.model_path is not None and not Path(cfg.vlm.model_path).expanduser().exists():
            preflight_errors.append(f"VLM model_path does not exist: {cfg.vlm.model_path}")

    cmds: List[List[str]] = []
    if prebuilt_csv is not None and (args.build_index_if_missing and not prebuilt_csv.exists()):
        cmd_index = [
            sys.executable,
            "scripts/build_data_index.py",
            "--config",
            args.config,
            "--out_csv",
            str(prebuilt_csv),
            "--set_config",
        ]
        cmds.append(cmd_index)

    if args.bootstrap_cache:
        cmd_cache = [
            sys.executable,
            "scripts/cache_models.py",
            "--config",
            args.config,
            "--profiles",
            args.cache_profiles,
            "--out_report",
            str(out_dir / "bootstrap" / "model_cache_report.json"),
        ]
        cmds.append(cmd_cache)
    if effective_use_realism and not args.skip_realism_calibration:
        cmd_realism = [
            sys.executable,
            "scripts/calibrate_physics_realism.py",
            "--config",
            args.config,
            "--out",
            str(realism_out),
            "--max_pairs",
            str(args.max_pairs),
            "--max_images",
            str(args.max_images),
        ]
        if args.calibrate_source:
            cmd_realism += ["--source", args.calibrate_source]
        if args.encoder_profile:
            cmd_realism += ["--encoder_profile", args.encoder_profile]
        if args.seed is not None:
            cmd_realism += ["--seed", str(args.seed)]
        if args.device:
            cmd_realism += ["--device", args.device]
        cmds.append(cmd_realism)

    modes = ",".join(cfg.erasure_eval.modes or [cfg.erasure_eval.mode])
    cmd_erasure = [
        sys.executable,
        "scripts/run_erasure_eval.py",
        "--config",
        args.config,
        "--eraser",
        args.eraser,
        "--out_dir",
        str(erasure_dir),
        "--modes",
        modes,
    ]
    if args.shift_type:
        cmd_erasure += ["--shift_type", args.shift_type]
    if args.encoder_profile:
        cmd_erasure += ["--encoder_profile", args.encoder_profile]
    if args.seed is not None:
        cmd_erasure += ["--seed", str(args.seed)]
    if args.device:
        cmd_erasure += ["--device", args.device]
    cmds.append(cmd_erasure)

    shift_id_path = safety_dir / "shift_id.pkl"
    policy_json_path = safety_dir / "policy.json"
    cmd_safety = [
        sys.executable,
        "scripts/run_safety_policy_eval.py",
        "--config",
        args.config,
        "--eraser",
        args.eraser,
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
    if args.shift_type:
        cmd_safety += ["--shift_type", args.shift_type]
    if args.encoder_profile:
        cmd_safety += ["--encoder_profile", args.encoder_profile]
    if args.seed is not None:
        cmd_safety += ["--seed", str(args.seed)]
    if args.device:
        cmd_safety += ["--device", args.device]
    cmds.append(cmd_safety)

    if cfg.vlm is not None and cfg.vlm.enabled:
        cmd_vlm = [
            sys.executable,
            "scripts/run_vlm_erasure_eval.py",
            "--config",
            args.config,
            "--eraser",
            args.eraser,
            "--out_dir",
            str(vlm_dir),
            "--split",
            cfg.data.eval_split,
            "--max_images",
            str(args.max_images),
        ]
        if args.shift_type:
            cmd_vlm += ["--shift_type", args.shift_type]
        if args.seed is not None:
            cmd_vlm += ["--seed", str(args.seed)]
        if args.device:
            cmd_vlm += ["--device", args.device]
        cmds.append(cmd_vlm)

    if enable_real_anchor:
        anchor_shift = _first_anchor_shift(cfg, args.shift_type)
        cmd_anchor = [
            sys.executable,
            "scripts/run_real_anchor_eval.py",
            "--config",
            args.config,
            "--shift_type",
            anchor_shift,
            "--out_dir",
            str(anchor_dir),
            "--k",
            str(cfg.intervention.k),
            "--max_pairs",
            str(args.max_pairs),
            "--max_images",
            str(args.max_images),
            "--bootstrap_samples",
            str(args.bootstrap_samples),
        ]
        if args.encoder_profile:
            cmd_anchor += ["--encoder_profile", args.encoder_profile]
        if args.seed is not None:
            cmd_anchor += ["--seed", str(args.seed)]
        if args.device:
            cmd_anchor += ["--device", args.device]
        cmds.append(cmd_anchor)

    cmd_report = [
        sys.executable,
        "scripts/build_paper_report.py",
        "--config",
        args.config,
        "--out_dir",
        str(report_dir),
        "--erasure_dir",
        str(erasure_dir),
        "--safety_dir",
        str(safety_dir),
        "--real_anchor_dir",
        str(anchor_dir),
    ]
    if vlm_dir is not None:
        cmd_report += ["--vlm_dir", str(vlm_dir)]
    cmds.append(cmd_report)

    if args.external_eval_sources:
        if cfg.runtime.allow_download:
            raise RuntimeError("external paper eval requires runtime.allow_download=false")
        cmd_external = [
            sys.executable,
            "scripts/run_external_policy_eval.py",
            "--config",
            args.config,
            "--eraser",
            args.eraser,
            "--shift_id_model",
            str(shift_id_path),
            "--policy_json",
            str(policy_json_path),
            "--eval_sources",
            args.external_eval_sources,
            "--out_dir",
            str(external_dir),
            "--bootstrap_samples",
            str(args.bootstrap_samples),
            "--target_sensitivities",
            args.target_sensitivities,
            "--target_specificities",
            args.target_specificities,
        ]
        if args.shift_type:
            cmd_external += ["--shift_type", args.shift_type]
        if args.encoder_profile:
            cmd_external += ["--encoder_profile", args.encoder_profile]
        if args.seed is not None:
            cmd_external += ["--seed", str(args.seed)]
        if args.device:
            cmd_external += ["--device", args.device]
        cmds.append(cmd_external)

    cmd_dissertation = [
        sys.executable,
        "scripts/build_dissertation_artifacts.py",
        "--erasure_dir",
        str(erasure_dir),
        "--safety_dir",
        str(safety_dir),
        "--real_anchor_dir",
        str(anchor_dir),
        "--out_dir",
        str(dissertation_dir),
    ]
    if vlm_dir is not None:
        cmd_dissertation += ["--vlm_dir", str(vlm_dir)]
    if external_dir is not None:
        cmd_dissertation += ["--external_dir", str(external_dir)]
    cmds.append(cmd_dissertation)

    manifest = {
        "config": args.config,
        "eraser": args.eraser,
        "out_dir": str(out_dir),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_env": {
            "python": sys.version,
            "platform": platform.platform(),
            "cwd": os.getcwd(),
        },
        "config_sha256": _sha256_file(Path(args.config).expanduser()),
        "run_mode": {
            "open_data_fallback": open_data_fallback,
            "full_paper_tier": not open_data_fallback,
            "real_pair_realism_enabled": effective_use_realism,
            "real_anchor_enabled": enable_real_anchor,
            "scenario_mode": scenario_mode,
            "pair_mode": realism_pair_mode,
            "pair_metadata_status": pair_metadata_status,
        },
        "preflight": {
            "errors": preflight_errors,
            "notes": preflight_notes,
            "prebuilt_index_csv": str(prebuilt_csv) if prebuilt_csv is not None else None,
            "prebuilt_index_sha256": cfg.data.prebuilt_index_sha256,
        },
        "commands": cmds,
        "artifact_dirs": {
            "realism": str(realism_out),
            "erasure": str(erasure_dir),
            "safety": str(safety_dir),
            "anchor": str(anchor_dir),
            "report": str(report_dir),
            "vlm": str(vlm_dir) if vlm_dir is not None else None,
            "external": str(external_dir) if external_dir is not None else None,
            "dissertation": str(dissertation_dir),
        },
    }
    key_code_files = [
        "phystta/config/schema.py",
        "phystta/config/loader.py",
        "phystta/data/engine.py",
        "phystta/models/encoder_factory.py",
        "phystta/safety/policy.py",
        "phystta/safety/decision.py",
        "phystta/shifts/realism.py",
        "scripts/run_paper_suite.py",
        "scripts/run_erasure_eval.py",
        "scripts/run_safety_policy_eval.py",
        "scripts/run_real_anchor_eval.py",
    ]
    per_file_hashes = {
        rel: _sha256_if_exists(Path(rel)) for rel in key_code_files
    }
    manifest["code_sha256_files"] = per_file_hashes
    manifest["code_sha256"] = _aggregate_hash_map(per_file_hashes)
    if args.dry_run:
        (out_dir / "paper_suite_plan.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        for c in cmds:
            _run(c, dry_run=True)
        print(f"Saved dry-run plan: {out_dir / 'paper_suite_plan.json'}")
        return

    if preflight_errors:
        (out_dir / "paper_suite_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        raise RuntimeError("Paper suite preflight failed: " + " | ".join(preflight_errors))

    for c in cmds:
        _run(c, dry_run=False)

    required = [
        erasure_dir / "metrics.csv",
        erasure_dir / "fairness.csv",
        erasure_dir / "causal_invariance.csv",
        safety_dir / "safety.csv",
        safety_dir / "policy_decisions.csv",
        safety_dir / "policy_bootstrap.csv",
        safety_dir / "policy_operating_points.csv",
        safety_dir / "policy_operating_points_targeted.csv",
        report_dir / "paper_report.json",
        dissertation_dir / "dissertation_artifacts_report.json",
    ]
    if enable_real_anchor:
        required.extend(
            [
                anchor_dir / "principal_angles.json",
                anchor_dir / "falsification_report.json",
            ]
        )
    if vlm_dir is not None:
        required.extend(
            [
                vlm_dir / "vlm_metrics.csv",
                vlm_dir / "vlm_summary.json",
            ]
        )
    if external_dir is not None:
        required.append(external_dir / "external_policy_metrics.csv")
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise RuntimeError(f"Paper suite finished with missing required artifacts: {missing}")

    manifest["required_artifacts"] = [str(p) for p in required]
    (out_dir / "paper_suite_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved manifest: {out_dir / 'paper_suite_manifest.json'}")


if __name__ == "__main__":
    main()
