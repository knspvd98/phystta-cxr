#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phystta.config.loader import load_config, save_config
from scripts.runpod_preflight import run_preflight


STAGE_ORDER = [
    "collect_probes",
    "fit_eraser",
    "erasure_eval",
    "safety_eval",
]
RUNNER_REPORT_NAME = "mixed_stress_runner_report.json"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _read_json(path: Path) -> Dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _run_subprocess(cmd: List[str], log_path: Path) -> None:
    env = os.environ.copy()
    root = str(_repo_root())
    env["PYTHONPATH"] = root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"\n[{_now_iso()}] RUN {' '.join(cmd)}\n")
        f.flush()
        proc = subprocess.run(
            cmd,
            cwd=root,
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if proc.returncode != 0:
        raise RuntimeError(f"stage command failed with exit_code={proc.returncode}")


def _stage_state_dir(out_root: Path) -> Path:
    return out_root / ".stage_state"


def _stage_state_path(out_root: Path, stage: str) -> Path:
    return _stage_state_dir(out_root) / f"{stage}.json"


def _stage_log_path(out_root: Path, stage: str) -> Path:
    return out_root / "launcher_logs" / f"{stage}.log"


def _verify_paths(paths: List[str]) -> List[str]:
    missing: List[str] = []
    for raw in paths:
        path = Path(raw).expanduser()
        if not path.exists():
            missing.append(str(path))
    return missing


def _mixed_stress_identity_status(cfg) -> Dict[str, Any]:
    errors: List[str] = []
    scenario_mode = str(getattr(cfg.erasure_eval, "scenario_mode", "atomic_calibrated_eval"))
    erasure_shift_types = [str(st) for st in getattr(cfg.erasure_eval, "shift_types", [])]
    intervention_shift_types = [str(st) for st in getattr(cfg.intervention, "shift_types", [])]
    family = str(getattr(cfg.model, "family", ""))
    encoder_source = str(getattr(cfg.model.encoder, "encoder_source", ""))
    num_classes = int(getattr(cfg.model, "num_classes", -1))

    if scenario_mode != "mixed_stress_eval":
        errors.append(f"mixed stress runner requires erasure_eval.scenario_mode='mixed_stress_eval', got '{scenario_mode}'.")
    if set(erasure_shift_types) != {"mixed"}:
        errors.append(f"mixed stress runner requires erasure_eval.shift_types=['mixed'], got {erasure_shift_types}.")
    if set(intervention_shift_types) != {"mixed"}:
        errors.append(f"mixed stress runner requires intervention.shift_types=['mixed'], got {intervention_shift_types}.")
    if bool(getattr(cfg.intervention, "use_realism", False)):
        errors.append("mixed stress runner requires intervention.use_realism=false.")
    if family != "cnn":
        errors.append(f"mixed stress runner requires model.family='cnn', got '{family}'.")
    if encoder_source != "baseline_cnn":
        errors.append(f"mixed stress runner requires encoder_source='baseline_cnn', got '{encoder_source}'.")
    if num_classes != 6:
        errors.append(f"mixed stress runner requires model.num_classes=6, got {num_classes}.")

    return {
        "checked": True,
        "eligible": len(errors) == 0,
        "scenario_mode": scenario_mode,
        "erasure_shift_types": erasure_shift_types,
        "intervention_shift_types": intervention_shift_types,
        "use_realism": bool(getattr(cfg.intervention, "use_realism", False)),
        "model_family": family,
        "encoder_source": encoder_source,
        "num_classes": num_classes,
        "reasons": errors,
    }


def _append_optional_args(cmd: List[str], *, encoder_profile: str | None, seed: int | None, device: str | None) -> None:
    if encoder_profile:
        cmd.extend(["--encoder_profile", str(encoder_profile)])
    if seed is not None:
        cmd.extend(["--seed", str(seed)])
    if device:
        cmd.extend(["--device", str(device)])


def _runner_report_path(out_root: Path) -> Path:
    return out_root / RUNNER_REPORT_NAME


def _build_stage_status_surface(
    stages: List[Dict[str, Any]],
    *,
    default_status: str,
    stage_updates: Dict[str, Dict[str, Any]] | None = None,
) -> List[Dict[str, Any]]:
    updates = stage_updates or {}
    surface: List[Dict[str, Any]] = []
    for idx, row in enumerate(stages, start=1):
        extra = dict(updates.get(str(row["stage"]), {}))
        entry = {
            "stage_index": idx,
            "stage": row["stage"],
            "status": extra.pop("status", default_status),
            "command": list(row["command"]),
            "command_display": " ".join(row["command"]),
            "log_path": row["log_path"],
            "state_path": row["state_path"],
            "expected_inputs": list(row["expected_inputs"]),
            "expected_outputs": list(row["expected_outputs"]),
        }
        entry.update(extra)
        surface.append(entry)
    return surface


def _build_restart_surface(stages: List[Dict[str, Any]]) -> Dict[str, Any]:
    stage_observations: List[Dict[str, Any]] = []
    existing_stage_state_detected = False
    existing_successful_stage_detected = False
    existing_outputs_detected = False
    for idx, row in enumerate(stages, start=1):
        state_path = Path(row["state_path"]).expanduser()
        log_path = Path(row["log_path"]).expanduser()
        state_payload = _read_json(state_path)
        outputs_present = [str(Path(raw).expanduser()) for raw in row["expected_outputs"] if Path(raw).expanduser().exists()]
        outputs_missing = [str(Path(raw).expanduser()) for raw in row["expected_outputs"] if not Path(raw).expanduser().exists()]
        state_status = state_payload.get("status") if isinstance(state_payload, dict) else None
        existing_stage_state_detected = existing_stage_state_detected or state_payload is not None
        existing_successful_stage_detected = existing_successful_stage_detected or state_status == "success"
        existing_outputs_detected = existing_outputs_detected or bool(outputs_present)
        stage_observations.append(
            {
                "stage_index": idx,
                "stage": row["stage"],
                "state_path": str(state_path),
                "existing_state_detected": state_payload is not None,
                "existing_state_status": state_status,
                "existing_log_detected": log_path.exists(),
                "existing_outputs_present": outputs_present,
                "existing_outputs_missing": outputs_missing,
                "rerun_required": True,
            }
        )
    return {
        "auto_resume_supported": False,
        "resume_mode": "manual_rebuild_only",
        "existing_stage_state_detected": existing_stage_state_detected,
        "existing_successful_stage_detected": existing_successful_stage_detected,
        "existing_outputs_detected": existing_outputs_detected,
        "stage_observations": stage_observations,
        "notes": [
            "Mixed stress runner records stage-state and useful outputs, but does not implement automatic resume.",
            "Existing stage-state or outputs are forensic/restart clues only; reruns still require explicit operator judgment.",
        ],
    }


def _append_runner_event(
    events: List[Dict[str, Any]],
    *,
    event: str,
    stage: str | None = None,
    status: str | None = None,
    detail: str | None = None,
) -> None:
    row: Dict[str, Any] = {"timestamp_utc": _now_iso(), "event": event}
    if stage is not None:
        row["stage"] = stage
    if status is not None:
        row["status"] = status
    if detail is not None:
        row["detail"] = detail
    events.append(row)


def _build_runner_report(
    payload: Dict[str, Any],
    *,
    run_mode: str,
    runner_status: str,
    stage_status_surface: List[Dict[str, Any]],
    runner_events: List[Dict[str, Any]] | None = None,
    blocking_reasons: List[str] | None = None,
    final_error: str | None = None,
) -> Dict[str, Any]:
    out_root = Path(payload["out_dir"])
    preflight = payload.get("preflight", {})
    preflight_errors = list(preflight.get("errors", [])) if isinstance(preflight, dict) else []
    preflight_contract = preflight.get("mixed_stress_contract", {}) if isinstance(preflight, dict) else {}
    preflight_eligible = bool(isinstance(preflight_contract, dict) and preflight_contract.get("eligible", False)) and not preflight_errors
    notes = [
        "This artifact is a runner/reporting surface, not a scientific summary contract.",
        "Dry-run surfaces do not imply runtime proof or scientific proof.",
        "Mixed stress remains a legacy CNN / 6-class / mixed_stress_eval contour and is not equivalent to the hardened bounded MIMIC or external lines.",
    ]
    return {
        "kind": "mixed_stress_runner_report",
        "timestamp_utc": _now_iso(),
        "run_mode": run_mode,
        "runner_status": runner_status,
        "real_run_attempted": run_mode == "real_run",
        "source_config": payload["source_config"],
        "effective_config": payload["effective_config"],
        "out_dir": payload["out_dir"],
        "canonical_contract": payload["canonical_contract"],
        "scenario_mode": payload["scenario_mode"],
        "selected_stages": list(payload["selected_stages"]),
        "preflight_eligible": preflight_eligible,
        "preflight_artifacts": dict(payload["preflight_artifacts"]),
        "runtime_diagnostics": dict(payload.get("runtime_diagnostics", {})),
        "reporting_artifacts": {
            "plan_json": str(out_root / "mixed_stress_plan.json"),
            "runner_report_json": str(_runner_report_path(out_root)),
            "real_run_manifest_json": str(out_root / "mixed_stress_manifest.json"),
        },
        "artifacts": dict(payload["artifacts"]),
        "stage_status_surface": stage_status_surface,
        "restart_surface": _build_restart_surface(payload["stages"]),
        "runner_events": list(runner_events or []),
        "blocking_reasons": list(blocking_reasons or []),
        "preflight_errors": preflight_errors,
        "final_error": final_error,
        "runtime_proof_implied": False,
        "scientific_proof_implied": False,
        "scientific_summary_equivalent": False,
        "notes": notes,
    }


def _write_runner_report(
    payload: Dict[str, Any],
    *,
    run_mode: str,
    runner_status: str,
    stage_status_surface: List[Dict[str, Any]],
    runner_events: List[Dict[str, Any]] | None = None,
    blocking_reasons: List[str] | None = None,
    final_error: str | None = None,
) -> Dict[str, Any]:
    report = _build_runner_report(
        payload,
        run_mode=run_mode,
        runner_status=runner_status,
        stage_status_surface=stage_status_surface,
        runner_events=runner_events,
        blocking_reasons=blocking_reasons,
        final_error=final_error,
    )
    _write_json(_runner_report_path(Path(payload["out_dir"])), report)
    return report


def build_mixed_stress_plan(
    *,
    config: str | Path,
    out_dir: str | Path,
    encoder_profile: str | None = None,
    seed: int | None = None,
    device: str | None = None,
    probe_batch_size: int | None = None,
    max_images: int | None = None,
    bootstrap_samples: int | None = None,
    eval_batch_size: int | None = None,
    eval_num_workers: int | None = None,
    checkpoint_every_eval_batches: int | None = None,
) -> Dict[str, Any]:
    config_path = Path(config).expanduser()
    out_root = Path(out_dir).expanduser()
    out_root.mkdir(parents=True, exist_ok=True)

    cfg = load_config(str(config_path))
    cfg.output.root = out_root
    if seed is not None:
        cfg.runtime.seed = int(seed)
    if device is not None:
        cfg.runtime.device = str(device)

    effective_cfg_path = out_root / "mixed_stress.effective.yaml"
    save_config(str(effective_cfg_path), cfg)

    identity_contract = _mixed_stress_identity_status(cfg)
    preflight_path = out_root / "runpod_preflight.json"
    rehearsal_path = out_root / "mixed_stress_rehearsal.json"

    preflight_payload: Dict[str, Any] | None = None
    try:
        preflight_payload = run_preflight(
            effective_cfg_path,
            out_json=preflight_path,
            encoder_profile=encoder_profile,
        )
    except SystemExit:
        preflight_payload = _read_json(preflight_path)
        if preflight_payload is None:
            preflight_payload = {
                "config": str(effective_cfg_path),
                "errors": ["mixed-stress preflight exited without writing a JSON payload."],
                "warnings": [],
            }
            _write_json(preflight_path, preflight_payload)

    if preflight_payload is None:
        preflight_payload = {"config": str(effective_cfg_path), "errors": [], "warnings": []}

    preflight_errors = list(preflight_payload.get("errors", []))
    if identity_contract["reasons"]:
        preflight_errors.extend(identity_contract["reasons"])
    preflight_payload["errors"] = preflight_errors
    preflight_payload["mixed_stress_identity"] = identity_contract
    _write_json(preflight_path, preflight_payload)

    probes_path = out_root / "probes" / "probes.pt"
    probes_manifest_path = out_root / "probes" / "artifact_manifest.json"
    eraser_path = out_root / "eraser" / "eraser.pt"
    eraser_meta_path = out_root / "eraser" / "eraser_meta.json"
    erasure_dir = out_root / "erasure"
    safety_dir = out_root / "safety"
    shift_id_path = safety_dir / "shift_id.pkl"
    policy_path = safety_dir / "policy.json"
    safety_thresholds_path = safety_dir / "policy_thresholds.json"

    preflight_cmd = [
        sys.executable,
        "scripts/runpod_preflight.py",
        "--config",
        str(effective_cfg_path),
        "--out_json",
        str(preflight_path),
    ]
    _append_optional_args(preflight_cmd, encoder_profile=encoder_profile, seed=None, device=None)

    collect_cmd = [
        sys.executable,
        "scripts/collect_probes.py",
        "--config",
        str(effective_cfg_path),
        "--shift_types",
        "mixed",
        "--out",
        str(probes_path),
    ]
    if probe_batch_size is not None:
        collect_cmd.extend(["--batch_size", str(int(probe_batch_size))])
    if max_images is not None:
        collect_cmd.extend(["--max_images", str(int(max_images))])
    _append_optional_args(collect_cmd, encoder_profile=encoder_profile, seed=seed, device=device)

    fit_cmd = [
        sys.executable,
        "scripts/fit_eraser.py",
        "--config",
        str(effective_cfg_path),
        "--probes",
        str(probes_path),
        "--shift_type",
        "mixed",
        "--out",
        str(eraser_path),
        "--meta",
        str(eraser_meta_path),
        "--self_check",
    ]
    _append_optional_args(fit_cmd, encoder_profile=encoder_profile, seed=seed, device=device)

    erasure_cmd = [
        sys.executable,
        "scripts/run_erasure_eval.py",
        "--config",
        str(effective_cfg_path),
        "--eraser",
        str(eraser_path),
        "--out_dir",
        str(erasure_dir),
        "--shift_type",
        "mixed",
        "--scenario_mode",
        "mixed_stress_eval",
    ]
    if bootstrap_samples is not None:
        erasure_cmd.extend(["--bootstrap_samples", str(int(bootstrap_samples))])
    if eval_batch_size is not None:
        erasure_cmd.extend(["--eval_batch_size", str(int(eval_batch_size))])
    if eval_num_workers is not None:
        erasure_cmd.extend(["--eval_num_workers", str(int(eval_num_workers))])
    if checkpoint_every_eval_batches is not None:
        erasure_cmd.extend(["--checkpoint_every_eval_batches", str(int(checkpoint_every_eval_batches))])
    _append_optional_args(erasure_cmd, encoder_profile=encoder_profile, seed=seed, device=device)

    safety_cmd = [
        sys.executable,
        "scripts/run_safety_policy_eval.py",
        "--config",
        str(effective_cfg_path),
        "--eraser",
        str(eraser_path),
        "--out_dir",
        str(safety_dir),
        "--shift_type",
        "mixed",
        "--save_shift_id",
        str(shift_id_path),
        "--save_policy",
        str(policy_path),
    ]
    if bootstrap_samples is not None:
        safety_cmd.extend(["--bootstrap_samples", str(int(bootstrap_samples))])
    if eval_batch_size is not None:
        safety_cmd.extend(["--eval_batch_size", str(int(eval_batch_size))])
    if eval_num_workers is not None:
        safety_cmd.extend(["--eval_num_workers", str(int(eval_num_workers))])
    _append_optional_args(safety_cmd, encoder_profile=encoder_profile, seed=seed, device=device)

    stages = [
        {
            "stage": "collect_probes",
            "command": collect_cmd,
            "log_path": str(_stage_log_path(out_root, "collect_probes")),
            "state_path": str(_stage_state_path(out_root, "collect_probes")),
            "expected_inputs": [str(effective_cfg_path)],
            "expected_outputs": [str(probes_path), str(probes_manifest_path)],
        },
        {
            "stage": "fit_eraser",
            "command": fit_cmd,
            "log_path": str(_stage_log_path(out_root, "fit_eraser")),
            "state_path": str(_stage_state_path(out_root, "fit_eraser")),
            "expected_inputs": [str(effective_cfg_path), str(probes_path)],
            "expected_outputs": [str(eraser_path), str(eraser_meta_path)],
        },
        {
            "stage": "erasure_eval",
            "command": erasure_cmd,
            "log_path": str(_stage_log_path(out_root, "erasure_eval")),
            "state_path": str(_stage_state_path(out_root, "erasure_eval")),
            "expected_inputs": [str(effective_cfg_path), str(eraser_path)],
            "expected_outputs": [str(erasure_dir / "metrics.csv"), str(erasure_dir / "artifact_manifest.json")],
        },
        {
            "stage": "safety_eval",
            "command": safety_cmd,
            "log_path": str(_stage_log_path(out_root, "safety_eval")),
            "state_path": str(_stage_state_path(out_root, "safety_eval")),
            "expected_inputs": [str(effective_cfg_path), str(eraser_path)],
            "expected_outputs": [
                str(safety_dir / "safety.csv"),
                str(policy_path),
                str(shift_id_path),
                str(safety_thresholds_path),
                str(safety_dir / "artifact_manifest.json"),
            ],
        },
    ]

    payload = {
        "timestamp_utc": _now_iso(),
        "source_config": str(config_path),
        "effective_config": str(effective_cfg_path),
        "out_dir": str(out_root),
        "canonical_contract": "docs/canonical_mixed_stress_runpod_packet.md",
        "scenario_mode": str(getattr(cfg.erasure_eval, "scenario_mode", "")),
        "identity_contract": identity_contract,
        "preflight": preflight_payload,
        "preflight_command": preflight_cmd,
        "preflight_artifacts": {
            "runpod_preflight": str(preflight_path),
            "mixed_stress_rehearsal": str(rehearsal_path),
        },
        "logs_dir": str(out_root / "launcher_logs"),
        "stage_state_dir": str(out_root / ".stage_state"),
        "selected_stages": list(STAGE_ORDER),
        "stages": stages,
        "commands": [collect_cmd, fit_cmd, erasure_cmd, safety_cmd],
        "artifacts": {
            "probes": str(probes_path),
            "probes_manifest": str(probes_manifest_path),
            "eraser": str(eraser_path),
            "eraser_meta": str(eraser_meta_path),
            "erasure_metrics": str(erasure_dir / "metrics.csv"),
            "safety_csv": str(safety_dir / "safety.csv"),
            "shift_id": str(shift_id_path),
            "policy_json": str(policy_path),
            "policy_thresholds": str(safety_thresholds_path),
        },
        "reporting_artifacts": {
            "plan_json": str(out_root / "mixed_stress_plan.json"),
            "runner_report_json": str(_runner_report_path(out_root)),
            "real_run_manifest_json": str(out_root / "mixed_stress_manifest.json"),
        },
        "runtime_diagnostics": {
            "python_executable": sys.executable,
            "repo_root": str(_repo_root()),
            "device": str(getattr(cfg.runtime, "device", "")),
            "seed": int(getattr(cfg.runtime, "seed", 0)),
            "encoder_profile": encoder_profile,
            "probe_batch_size": probe_batch_size,
            "max_images": max_images,
            "bootstrap_samples": bootstrap_samples,
            "eval_batch_size": eval_batch_size,
            "eval_num_workers": eval_num_workers,
            "checkpoint_every_eval_batches": checkpoint_every_eval_batches,
            "freeze_artifacts": bool(getattr(cfg.runtime, "freeze_artifacts", False)),
            "allow_download": bool(getattr(cfg.runtime, "allow_download", False)),
            "logs_dir": str(out_root / "launcher_logs"),
            "stage_state_dir": str(out_root / ".stage_state"),
        },
    }
    _write_json(out_root / "mixed_stress_plan.json", payload)
    dry_run_status = "dry_run_blocked" if preflight_errors else "dry_run_ready"
    dry_run_stage_status = _build_stage_status_surface(
        stages,
        default_status="blocked_preflight" if preflight_errors else "planned",
    )
    dry_run_events: List[Dict[str, Any]] = []
    _append_runner_event(
        dry_run_events,
        event="dry_run_plan_materialized",
        status=dry_run_status,
        detail=f"plan={out_root / 'mixed_stress_plan.json'}",
    )
    _write_runner_report(
        payload,
        run_mode="dry_run",
        runner_status=dry_run_status,
        stage_status_surface=dry_run_stage_status,
        runner_events=dry_run_events,
        blocking_reasons=preflight_errors,
    )
    return payload


def run_mixed_stress_suite(**kwargs) -> Dict[str, Any]:
    payload = build_mixed_stress_plan(**kwargs)
    stages = payload["stages"]
    preflight_errors = list(payload.get("preflight", {}).get("errors", []))
    runner_events: List[Dict[str, Any]] = []
    _append_runner_event(runner_events, event="real_run_started", status="running")
    if preflight_errors:
        _append_runner_event(
            runner_events,
            event="real_run_blocked_preflight",
            status="blocked_preflight",
            detail=" | ".join(str(err) for err in preflight_errors),
        )
        _write_runner_report(
            payload,
            run_mode="real_run",
            runner_status="blocked_preflight",
            stage_status_surface=_build_stage_status_surface(stages, default_status="blocked_preflight"),
            runner_events=runner_events,
            blocking_reasons=preflight_errors,
            final_error="mixed-stress preflight failed before execution",
        )
        raise SystemExit(
            "mixed-stress preflight failed before execution: "
            + " | ".join(str(err) for err in preflight_errors)
        )

    out_root = Path(payload["out_dir"])
    stage_updates: Dict[str, Dict[str, Any]] = {}
    _write_runner_report(
        payload,
        run_mode="real_run",
        runner_status="running",
        stage_status_surface=_build_stage_status_surface(stages, default_status="pending", stage_updates=stage_updates),
        runner_events=runner_events,
    )
    for row in stages:
        state_path = Path(row["state_path"])
        stage_name = str(row["stage"])
        started_at = _now_iso()
        stage_updates[stage_name] = {"status": "running", "started_at_utc": started_at}
        _append_runner_event(runner_events, event="stage_started", stage=stage_name, status="running")
        _write_json(
            state_path,
            {
                "stage": row["stage"],
                "status": "running",
                "command": row["command"],
                "expected_inputs": row["expected_inputs"],
                "expected_outputs": row["expected_outputs"],
                "started_at_utc": started_at,
            },
        )
        _write_runner_report(
            payload,
            run_mode="real_run",
            runner_status="running",
            stage_status_surface=_build_stage_status_surface(stages, default_status="pending", stage_updates=stage_updates),
            runner_events=runner_events,
        )
        missing_inputs = _verify_paths(row["expected_inputs"])
        if missing_inputs:
            error_message = f"missing stage inputs: {missing_inputs}"
            stage_updates[stage_name] = {
                "status": "error",
                "started_at_utc": started_at,
                "error": error_message,
            }
            _append_runner_event(runner_events, event="stage_failed", stage=stage_name, status="error", detail=error_message)
            _write_json(
                state_path,
                {
                    "stage": row["stage"],
                    "status": "error",
                    "error": error_message,
                    "command": row["command"],
                    "expected_inputs": row["expected_inputs"],
                    "expected_outputs": row["expected_outputs"],
                },
            )
            _write_runner_report(
                payload,
                run_mode="real_run",
                runner_status="failed",
                stage_status_surface=_build_stage_status_surface(stages, default_status="pending", stage_updates=stage_updates),
                runner_events=runner_events,
                blocking_reasons=[error_message],
                final_error=error_message,
            )
            raise RuntimeError(f"Mixed stress stage '{row['stage']}' requires missing inputs: {missing_inputs}")
        try:
            _run_subprocess(row["command"], Path(row["log_path"]))
        except Exception as e:
            error_message = f"{type(e).__name__}: {e}"
            stage_updates[stage_name] = {
                "status": "error",
                "started_at_utc": started_at,
                "error": error_message,
            }
            _append_runner_event(runner_events, event="stage_failed", stage=stage_name, status="error", detail=error_message)
            _write_json(
                state_path,
                {
                    "stage": row["stage"],
                    "status": "error",
                    "error": error_message,
                    "command": row["command"],
                    "expected_inputs": row["expected_inputs"],
                    "expected_outputs": row["expected_outputs"],
                },
            )
            _write_runner_report(
                payload,
                run_mode="real_run",
                runner_status="failed",
                stage_status_surface=_build_stage_status_surface(stages, default_status="pending", stage_updates=stage_updates),
                runner_events=runner_events,
                blocking_reasons=[error_message],
                final_error=error_message,
            )
            raise
        missing_outputs = _verify_paths(row["expected_outputs"])
        if missing_outputs:
            error_message = f"missing stage outputs: {missing_outputs}"
            stage_updates[stage_name] = {
                "status": "error",
                "started_at_utc": started_at,
                "error": error_message,
            }
            _append_runner_event(runner_events, event="stage_failed", stage=stage_name, status="error", detail=error_message)
            _write_json(
                state_path,
                {
                    "stage": row["stage"],
                    "status": "error",
                    "error": error_message,
                    "command": row["command"],
                    "expected_inputs": row["expected_inputs"],
                    "expected_outputs": row["expected_outputs"],
                },
            )
            _write_runner_report(
                payload,
                run_mode="real_run",
                runner_status="failed",
                stage_status_surface=_build_stage_status_surface(stages, default_status="pending", stage_updates=stage_updates),
                runner_events=runner_events,
                blocking_reasons=[error_message],
                final_error=error_message,
            )
            raise RuntimeError(f"Mixed stress stage '{row['stage']}' did not materialize outputs: {missing_outputs}")
        finished_at = _now_iso()
        stage_updates[stage_name] = {
            "status": "success",
            "started_at_utc": started_at,
            "finished_at_utc": finished_at,
        }
        _append_runner_event(runner_events, event="stage_completed", stage=stage_name, status="success")
        _write_json(
            state_path,
            {
                "stage": row["stage"],
                "status": "success",
                "command": row["command"],
                "expected_inputs": row["expected_inputs"],
                "expected_outputs": row["expected_outputs"],
                "finished_at_utc": finished_at,
            },
        )
        _write_runner_report(
            payload,
            run_mode="real_run",
            runner_status="running",
            stage_status_surface=_build_stage_status_surface(stages, default_status="pending", stage_updates=stage_updates),
            runner_events=runner_events,
        )

    manifest = dict(payload)
    manifest["kind"] = "mixed_stress_manifest"
    manifest["run_mode"] = "real_run"
    manifest["status"] = "success"
    manifest["runner_status"] = "completed"
    manifest["finished_at_utc"] = _now_iso()
    manifest["stage_status_surface"] = _build_stage_status_surface(stages, default_status="pending", stage_updates=stage_updates)
    _append_runner_event(runner_events, event="real_run_completed", status="completed")
    manifest["runner_events"] = list(runner_events)
    manifest["runtime_proof_implied"] = False
    manifest["scientific_proof_implied"] = False
    manifest["scientific_summary_equivalent"] = False
    _write_json(out_root / "mixed_stress_manifest.json", manifest)
    _write_runner_report(
        payload,
        run_mode="real_run",
        runner_status="completed",
        stage_status_surface=manifest["stage_status_surface"],
        runner_events=runner_events,
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Mixed-stress contour runner")
    parser.add_argument("--config", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--encoder_profile", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--probe_batch_size", type=int, default=None)
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--bootstrap_samples", type=int, default=None)
    parser.add_argument("--eval_batch_size", type=int, default=None)
    parser.add_argument("--eval_num_workers", type=int, default=None)
    parser.add_argument("--checkpoint_every_eval_batches", type=int, default=None)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    kwargs = {
        "config": args.config,
        "out_dir": args.out_dir,
        "encoder_profile": args.encoder_profile,
        "seed": args.seed,
        "device": args.device,
        "probe_batch_size": args.probe_batch_size,
        "max_images": args.max_images,
        "bootstrap_samples": args.bootstrap_samples,
        "eval_batch_size": args.eval_batch_size,
        "eval_num_workers": args.eval_num_workers,
        "checkpoint_every_eval_batches": args.checkpoint_every_eval_batches,
    }
    if args.dry_run:
        payload = build_mixed_stress_plan(**kwargs)
        print(f"Saved dry-run plan: {Path(payload['out_dir']) / 'mixed_stress_plan.json'}")
        return

    manifest = run_mixed_stress_suite(**kwargs)
    print(f"Mixed-stress run complete. Manifest: {Path(manifest['out_dir']) / 'mixed_stress_manifest.json'}")


if __name__ == "__main__":
    main()
