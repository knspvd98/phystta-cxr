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
from scripts.run_public_mainline import STAGE_ORDER as PUBLIC_MAINLINE_STAGE_ORDER
from scripts.runpod_preflight import run_preflight


RUNNER_REPORT_NAME = "broader_shift_family_runner_report.json"


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
        raise RuntimeError(f"delegated public-mainline command failed with exit_code={proc.returncode}")


def _runner_state_dir(out_root: Path) -> Path:
    return out_root / ".branch_runner_state"


def _runner_state_path(out_root: Path, stage: str) -> Path:
    return _runner_state_dir(out_root) / f"{stage}.json"


def _runner_log_path(out_root: Path, stage: str) -> Path:
    return out_root / "branch_runner_logs" / f"{stage}.log"


def _delegated_stage_state_path(out_root: Path, stage: str) -> Path:
    return out_root / ".stage_state" / f"{stage}.json"


def _delegated_stage_log_path(out_root: Path, stage: str) -> Path:
    return out_root / "launcher_logs" / f"{stage}.log"


def _verify_paths(paths: List[str]) -> List[str]:
    missing: List[str] = []
    for raw in paths:
        if not Path(raw).expanduser().exists():
            missing.append(str(Path(raw).expanduser()))
    return missing


def _classify_paths(paths: List[str]) -> tuple[List[str], List[str]]:
    found: List[str] = []
    missing: List[str] = []
    for raw in paths:
        path = str(Path(raw).expanduser())
        if Path(path).exists():
            found.append(path)
        else:
            missing.append(path)
    return found, missing


def _broader_shift_family_identity_status(cfg) -> Dict[str, Any]:
    reasons: List[str] = []
    scenario_mode = str(getattr(cfg.erasure_eval, "scenario_mode", ""))
    intervention_shift_types = [str(x) for x in getattr(cfg.intervention, "shift_types", [])]
    erasure_shift_types = [str(x) for x in getattr(cfg.erasure_eval, "shift_types", [])]
    family = str(getattr(cfg.model, "family", ""))
    encoder_source = str(getattr(cfg.model.encoder, "encoder_source", ""))
    num_classes = int(getattr(cfg.model, "num_classes", -1))
    use_realism = bool(getattr(cfg.intervention, "use_realism", False))
    expected_shift_types = ["heel", "scatter", "exposure_poisson", "motion_blur"]

    if scenario_mode != "atomic_calibrated_eval":
        reasons.append(
            "broader shift-family runner requires erasure_eval.scenario_mode='atomic_calibrated_eval'."
        )
    if intervention_shift_types != expected_shift_types:
        reasons.append(
            "broader shift-family runner requires intervention.shift_types="
            f"{expected_shift_types}."
        )
    if erasure_shift_types != expected_shift_types:
        reasons.append(
            "broader shift-family runner requires erasure_eval.shift_types="
            f"{expected_shift_types}."
        )
    if family != "cnn":
        reasons.append("broader shift-family runner requires model.family='cnn'.")
    if encoder_source != "baseline_cnn":
        reasons.append("broader shift-family runner requires encoder_source='baseline_cnn'.")
    if num_classes != 14:
        reasons.append("broader shift-family runner requires the legacy 14-class head lineage.")
    if not use_realism:
        reasons.append("broader shift-family runner requires intervention.use_realism=true.")

    return {
        "checked": True,
        "eligible": len(reasons) == 0,
        "scenario_mode": scenario_mode,
        "family": family,
        "encoder_source": encoder_source,
        "expected_num_classes": num_classes,
        "intervention_shift_types": intervention_shift_types,
        "erasure_eval_shift_types": erasure_shift_types,
        "use_realism": use_realism,
        "reasons": reasons,
    }


def _append_optional_args(
    cmd: List[str],
    *,
    encoder_profile: str | None,
    seed: int | None,
    device: str | None,
    bootstrap_samples: int | None,
    max_pairs: int | None,
    max_images: int | None,
    probe_batch_size: int | None,
    resume: bool,
    dry_run: bool,
) -> None:
    if encoder_profile:
        cmd.extend(["--encoder_profile", str(encoder_profile)])
    if seed is not None:
        cmd.extend(["--seed", str(seed)])
    if device:
        cmd.extend(["--device", str(device)])
    if bootstrap_samples is not None:
        cmd.extend(["--bootstrap_samples", str(int(bootstrap_samples))])
    if max_pairs is not None:
        cmd.extend(["--max_pairs", str(int(max_pairs))])
    if max_images is not None:
        cmd.extend(["--max_images", str(int(max_images))])
    if probe_batch_size is not None:
        cmd.extend(["--probe_batch_size", str(int(probe_batch_size))])
    if resume:
        cmd.append("--resume")
    if dry_run:
        cmd.append("--dry_run")


def _build_runner_stage_status_surface(
    payload: Dict[str, Any],
    *,
    status: str,
    extra: Dict[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    stage = payload["runner_stage"]
    state_path = Path(stage["state_path"]).expanduser()
    log_path = Path(stage["log_path"]).expanduser()
    state_payload = _read_json(state_path)
    existing_outputs_found, existing_outputs_missing = _classify_paths(list(stage["expected_outputs"]))
    entry = {
        "stage_index": 1,
        "stage": stage["stage"],
        "status": status,
        "command": list(stage["command"]),
        "command_display": " ".join(stage["command"]),
        "dry_run_command": list(stage["dry_run_command"]),
        "dry_run_command_display": " ".join(stage["dry_run_command"]),
        "log_path": str(log_path),
        "state_path": str(state_path),
        "expected_inputs": list(stage["expected_inputs"]),
        "expected_outputs": list(stage["expected_outputs"]),
        "state_detected": state_payload is not None,
        "state_status": state_payload.get("status") if isinstance(state_payload, dict) else None,
        "log_detected": log_path.exists(),
        "started_at_utc": state_payload.get("started_at_utc") if isinstance(state_payload, dict) else None,
        "finished_at_utc": state_payload.get("finished_at_utc") if isinstance(state_payload, dict) else None,
        "error": state_payload.get("error") if isinstance(state_payload, dict) else None,
        "existing_outputs_found": existing_outputs_found,
        "existing_outputs_missing": existing_outputs_missing,
    }
    if extra:
        entry.update(extra)
    return [entry]


def _build_delegated_stage_status_surface(payload: Dict[str, Any], *, default_status: str) -> List[Dict[str, Any]]:
    out_root = Path(payload["out_dir"])
    surface: List[Dict[str, Any]] = []
    for idx, stage in enumerate(PUBLIC_MAINLINE_STAGE_ORDER, start=1):
        state_path = _delegated_stage_state_path(out_root, stage)
        log_path = _delegated_stage_log_path(out_root, stage)
        state_payload = _read_json(state_path)
        surface.append(
            {
                "stage_index": idx,
                "stage": stage,
                "status": (
                    str(state_payload.get("status"))
                    if isinstance(state_payload, dict) and state_payload.get("status") is not None
                    else default_status
                ),
                "state_path": str(state_path),
                "log_path": str(log_path),
                "state_detected": state_payload is not None,
                "state_status": state_payload.get("status") if isinstance(state_payload, dict) else None,
                "log_detected": log_path.exists(),
                "started_at_utc": state_payload.get("started_at_utc") if isinstance(state_payload, dict) else None,
                "finished_at_utc": state_payload.get("finished_at_utc") if isinstance(state_payload, dict) else None,
                "error": state_payload.get("error") if isinstance(state_payload, dict) else None,
                "expected_outputs": [],
                "existing_outputs_found": [],
                "existing_outputs_missing": [],
                "expected_outputs_known": False,
            }
        )
    return surface


def _build_delegated_artifact_surface(payload: Dict[str, Any]) -> Dict[str, Any]:
    reporting = dict(payload.get("reporting_artifacts", {}))
    delegated_keys = [
        "delegated_effective_config",
        "delegated_preflight_json",
        "delegated_plan_json",
        "delegated_manifest_json",
        "delegated_summary_json",
    ]
    expected_artifacts: List[Dict[str, Any]] = []
    found_artifacts: List[Dict[str, Any]] = []
    missing_artifacts: List[Dict[str, Any]] = []
    for key in delegated_keys:
        raw = reporting.get(key)
        if not raw:
            continue
        path = Path(str(raw)).expanduser()
        row = {
            "name": key,
            "path": str(path),
            "exists": path.exists(),
        }
        expected_artifacts.append(row)
        if row["exists"]:
            found_artifacts.append(row)
        else:
            missing_artifacts.append(row)
    return {
        "reused_public_mainline_reporting": True,
        "branch_owned_scientific_summary": False,
        "expected_artifacts": expected_artifacts,
        "found_artifacts": found_artifacts,
        "missing_artifacts": missing_artifacts,
        "notes": [
            "This broader-family branch reuses delegated public-mainline reporting artifacts.",
            "Presence of delegated artifacts is useful execution evidence, not a branch-owned scientific summary contract.",
        ],
    }


def _build_restart_surface(payload: Dict[str, Any], *, resume_requested: bool) -> Dict[str, Any]:
    out_root = Path(payload["out_dir"])
    observations: List[Dict[str, Any]] = []
    existing_stage_state_detected = False
    successful_stage_detected = False
    branch_owned_artifacts_present: List[str] = []
    branch_owned_artifacts_missing: List[str] = []
    for raw in [
        payload.get("preflight_artifacts", {}).get("runpod_preflight"),
        payload.get("preflight_artifacts", {}).get("broader_shift_family_rehearsal"),
        payload.get("reporting_artifacts", {}).get("plan_json"),
        payload.get("reporting_artifacts", {}).get("branch_manifest_json"),
    ]:
        if not raw:
            continue
        path = str(Path(str(raw)).expanduser())
        if Path(path).exists():
            branch_owned_artifacts_present.append(path)
        else:
            branch_owned_artifacts_missing.append(path)
    for idx, stage in enumerate(PUBLIC_MAINLINE_STAGE_ORDER, start=1):
        state_path = _delegated_stage_state_path(out_root, stage)
        log_path = _delegated_stage_log_path(out_root, stage)
        state_payload = _read_json(state_path)
        state_status = state_payload.get("status") if isinstance(state_payload, dict) else None
        existing_stage_state_detected = existing_stage_state_detected or state_payload is not None
        successful_stage_detected = successful_stage_detected or state_status == "success"
        observations.append(
            {
                "stage_index": idx,
                "stage": stage,
                "state_path": str(state_path),
                "state_detected": state_payload is not None,
                "state_status": state_status,
                "log_path": str(log_path),
                "log_detected": log_path.exists(),
                "existing_outputs_present": [],
                "existing_outputs_missing": [],
                "rerun_required": True,
            }
        )
    delegated_artifact_surface = _build_delegated_artifact_surface(payload)
    return {
        "runner_auto_resume_supported": False,
        "delegated_resume_flag_supported": True,
        "resume_requested": resume_requested,
        "existing_delegated_stage_state_detected": existing_stage_state_detected,
        "existing_successful_stage_detected": successful_stage_detected,
        "branch_owned_artifacts_present": branch_owned_artifacts_present,
        "branch_owned_artifacts_missing": branch_owned_artifacts_missing,
        "delegated_reporting_artifacts_found": [row["path"] for row in delegated_artifact_surface["found_artifacts"]],
        "delegated_reporting_artifacts_missing": [row["path"] for row in delegated_artifact_surface["missing_artifacts"]],
        "rerun_required_for_all_stages": True,
        "stage_observations": observations,
        "notes": [
            "The broader-family runner does not implement automatic resume.",
            "It can delegate an explicit --resume request to scripts/run_public_mainline.py.",
            "Existing delegated stage-state remains an operator-visible restart clue, not standalone runtime proof.",
            "Branch-owned and delegated artifact presence are forensic/restart clues only; explicit operator judgment is still required before rerun.",
        ],
    }


def _append_runner_event(
    events: List[Dict[str, Any]],
    *,
    event: str,
    status: str,
    detail: str | None = None,
) -> None:
    row: Dict[str, Any] = {
        "timestamp_utc": _now_iso(),
        "event": event,
        "status": status,
    }
    if detail is not None:
        row["detail"] = detail
    events.append(row)


def _runner_report_path(out_root: Path) -> Path:
    return out_root / RUNNER_REPORT_NAME


def _build_runner_report(
    payload: Dict[str, Any],
    *,
    run_mode: str,
    runner_status: str,
    runner_stage_status_surface: List[Dict[str, Any]],
    delegated_stage_status_surface: List[Dict[str, Any]],
    runner_events: List[Dict[str, Any]] | None = None,
    blocking_reasons: List[str] | None = None,
    final_error: str | None = None,
) -> Dict[str, Any]:
    out_root = Path(payload["out_dir"])
    broader_contract = payload.get("preflight", {}).get("broader_shift_family_contract", {})
    preflight_errors = list(payload.get("preflight", {}).get("errors", []))
    preflight_eligible = bool(isinstance(broader_contract, dict) and broader_contract.get("eligible", False)) and not preflight_errors
    delegated_artifact_surface = _build_delegated_artifact_surface(payload)
    return {
        "kind": "broader_shift_family_runner_report",
        "timestamp_utc": _now_iso(),
        "run_mode": run_mode,
        "runner_status": runner_status,
        "real_run_attempted": run_mode == "real_run",
        "source_config": payload["source_config"],
        "effective_config": payload["effective_config"],
        "out_dir": payload["out_dir"],
        "canonical_contract": payload["canonical_contract"],
        "contour_identity": dict(payload["identity_contract"]),
        "preflight_eligible": preflight_eligible,
        "preflight_artifacts": dict(payload["preflight_artifacts"]),
        "delegated_execution": dict(payload["delegated_execution"]),
        "reporting_artifacts": dict(payload["reporting_artifacts"]),
        "delegated_artifact_surface": delegated_artifact_surface,
        "runtime_diagnostics": dict(payload["runtime_diagnostics"]),
        "runner_stage_status_surface": runner_stage_status_surface,
        "delegated_stage_status_surface": delegated_stage_status_surface,
        "restart_surface": _build_restart_surface(
            payload,
            resume_requested=bool(payload["runtime_diagnostics"].get("resume")),
        ),
        "runner_events": list(runner_events or []),
        "blocking_reasons": list(blocking_reasons or []),
        "preflight_errors": preflight_errors,
        "final_error": final_error,
        "runtime_proof_implied": False,
        "scientific_proof_implied": False,
        "scientific_summary_equivalent": False,
        "notes": [
            "This artifact is a branch-owned execution/reporting surface, not a scientific summary contract.",
            "The broader-family contour remains legacy CNN + realism-enabled + atomic_calibrated_eval + heel/scatter/exposure_poisson/motion_blur.",
            "Completion of the delegated launcher path does not by itself imply scientific proof or pod-run evidence.",
        ],
    }


def _write_runner_report(
    payload: Dict[str, Any],
    *,
    run_mode: str,
    runner_status: str,
    runner_stage_status_surface: List[Dict[str, Any]],
    delegated_stage_status_surface: List[Dict[str, Any]],
    runner_events: List[Dict[str, Any]] | None = None,
    blocking_reasons: List[str] | None = None,
    final_error: str | None = None,
) -> Dict[str, Any]:
    report = _build_runner_report(
        payload,
        run_mode=run_mode,
        runner_status=runner_status,
        runner_stage_status_surface=runner_stage_status_surface,
        delegated_stage_status_surface=delegated_stage_status_surface,
        runner_events=runner_events,
        blocking_reasons=blocking_reasons,
        final_error=final_error,
    )
    _write_json(_runner_report_path(Path(payload["out_dir"])), report)
    return report


def build_broader_shift_family_plan(
    *,
    config: str | Path,
    out_dir: str | Path,
    encoder_profile: str | None = None,
    seed: int | None = None,
    device: str | None = None,
    bootstrap_samples: int | None = None,
    max_pairs: int | None = None,
    max_images: int | None = None,
    probe_batch_size: int | None = None,
    resume: bool = False,
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

    effective_cfg_path = out_root / "broader_shift_family.effective.yaml"
    preflight_cfg_path = out_root / config_path.name
    save_config(str(effective_cfg_path), cfg)
    save_config(str(preflight_cfg_path), cfg)

    identity_contract = _broader_shift_family_identity_status(cfg)
    preflight_path = out_root / "runpod_preflight.json"
    rehearsal_path = out_root / "broader_shift_family_rehearsal.json"

    preflight_payload: Dict[str, Any] | None = None
    try:
        preflight_payload = run_preflight(
            preflight_cfg_path,
            out_json=preflight_path,
            encoder_profile=encoder_profile,
        )
    except SystemExit:
        preflight_payload = _read_json(preflight_path)
        if preflight_payload is None:
            preflight_payload = {
                "config": str(effective_cfg_path),
                "errors": ["broader-family preflight exited without writing a JSON payload."],
                "warnings": [],
            }
            _write_json(preflight_path, preflight_payload)

    if preflight_payload is None:
        preflight_payload = {"config": str(effective_cfg_path), "errors": [], "warnings": []}

    preflight_errors = list(preflight_payload.get("errors", []))
    if identity_contract["reasons"]:
        preflight_errors.extend(identity_contract["reasons"])
    preflight_payload["errors"] = preflight_errors
    preflight_payload["broader_shift_family_identity"] = identity_contract
    _write_json(preflight_path, preflight_payload)

    real_cmd = [
        sys.executable,
        "scripts/run_public_mainline.py",
        "--config",
        str(effective_cfg_path),
        "--out_dir",
        str(out_root),
    ]
    _append_optional_args(
        real_cmd,
        encoder_profile=encoder_profile,
        seed=seed,
        device=device,
        bootstrap_samples=bootstrap_samples,
        max_pairs=max_pairs,
        max_images=max_images,
        probe_batch_size=probe_batch_size,
        resume=resume,
        dry_run=False,
    )
    dry_run_cmd = list(real_cmd)
    _append_optional_args(
        dry_run_cmd,
        encoder_profile=None,
        seed=None,
        device=None,
        bootstrap_samples=None,
        max_pairs=None,
        max_images=None,
        probe_batch_size=None,
        resume=False,
        dry_run=True,
    )

    runner_stage = {
        "stage": "public_mainline_launcher",
        "command": real_cmd,
        "dry_run_command": dry_run_cmd,
        "log_path": str(_runner_log_path(out_root, "public_mainline_launcher")),
        "state_path": str(_runner_state_path(out_root, "public_mainline_launcher")),
        "expected_inputs": [
            str(effective_cfg_path),
            str(preflight_path),
            str(rehearsal_path),
        ],
        "expected_outputs": [
            str(out_root / "public_mainline_manifest.json"),
            str(out_root / "summary" / "public_mainline_summary.json"),
        ],
    }

    payload = {
        "timestamp_utc": _now_iso(),
        "source_config": str(config_path),
        "effective_config": str(effective_cfg_path),
        "out_dir": str(out_root),
        "canonical_contract": "docs/canonical_broader_shift_family_runpod_packet.md",
        "identity_contract": identity_contract,
        "preflight": preflight_payload,
        "preflight_artifacts": {
            "runpod_preflight": str(preflight_path),
            "broader_shift_family_rehearsal": str(rehearsal_path),
        },
        "runner_stage": runner_stage,
        "delegated_execution": {
            "entrypoint": str(_repo_root() / "scripts" / "run_public_mainline.py"),
            "kind": "public_mainline_launcher",
            "dry_run_command": dry_run_cmd,
            "real_run_command": real_cmd,
        },
        "reporting_artifacts": {
            "plan_json": str(out_root / "broader_shift_family_plan.json"),
            "runner_report_json": str(_runner_report_path(out_root)),
            "branch_manifest_json": str(out_root / "broader_shift_family_manifest.json"),
            "delegated_effective_config": str(out_root / "public_mainline.effective.yaml"),
            "delegated_preflight_json": str(out_root / "public_mainline_preflight.json"),
            "delegated_plan_json": str(out_root / "public_mainline_plan.json"),
            "delegated_manifest_json": str(out_root / "public_mainline_manifest.json"),
            "delegated_summary_json": str(out_root / "summary" / "public_mainline_summary.json"),
        },
        "runtime_diagnostics": {
            "python_executable": sys.executable,
            "repo_root": str(_repo_root()),
            "device": str(getattr(cfg.runtime, "device", "")),
            "seed": int(getattr(cfg.runtime, "seed", 0)),
            "model_family": str(getattr(cfg.model, "family", "")),
            "encoder_source": str(getattr(cfg.model.encoder, "encoder_source", "")),
            "scenario_mode": str(getattr(cfg.erasure_eval, "scenario_mode", "")),
            "intervention_shift_types": [str(x) for x in getattr(cfg.intervention, "shift_types", [])],
            "erasure_eval_shift_types": [str(x) for x in getattr(cfg.erasure_eval, "shift_types", [])],
            "realism_enabled": bool(getattr(cfg.intervention, "use_realism", False)),
            "realism_path": str(getattr(cfg.intervention, "realism_path", "")),
            "encoder_profile": encoder_profile,
            "bootstrap_samples": bootstrap_samples,
            "max_pairs": max_pairs,
            "max_images": max_images,
            "probe_batch_size": probe_batch_size,
            "resume": bool(resume),
            "freeze_artifacts": bool(getattr(cfg.runtime, "freeze_artifacts", False)),
            "allow_download": bool(getattr(cfg.runtime, "allow_download", False)),
            "use_realism": bool(getattr(cfg.intervention, "use_realism", False)),
            "logs_dir": str(out_root / "launcher_logs"),
            "stage_state_dir": str(out_root / ".stage_state"),
            "branch_runner_logs_dir": str(out_root / "branch_runner_logs"),
            "branch_runner_state_dir": str(out_root / ".branch_runner_state"),
            "delegated_outputs_root": str(out_root),
            "branch_runner_entrypoint": str(_repo_root() / "scripts" / "run_broader_shift_family_suite.py"),
            "delegated_public_mainline_reporting_reused": True,
        },
    }

    _write_json(out_root / "broader_shift_family_plan.json", payload)
    dry_run_status = "dry_run_blocked" if preflight_errors else "dry_run_ready"
    dry_run_runner_surface = _build_runner_stage_status_surface(
        payload,
        status="blocked_preflight" if preflight_errors else "planned",
    )
    dry_run_delegated_surface = _build_delegated_stage_status_surface(
        payload,
        default_status="blocked_preflight" if preflight_errors else "planned",
    )
    dry_run_events: List[Dict[str, Any]] = []
    _append_runner_event(
        dry_run_events,
        event="dry_run_plan_materialized",
        status=dry_run_status,
        detail=f"plan={out_root / 'broader_shift_family_plan.json'}",
    )
    _write_runner_report(
        payload,
        run_mode="dry_run",
        runner_status=dry_run_status,
        runner_stage_status_surface=dry_run_runner_surface,
        delegated_stage_status_surface=dry_run_delegated_surface,
        runner_events=dry_run_events,
        blocking_reasons=preflight_errors,
    )
    return payload


def run_broader_shift_family_suite(**kwargs) -> Dict[str, Any]:
    payload = build_broader_shift_family_plan(**kwargs)
    preflight_errors = list(payload.get("preflight", {}).get("errors", []))
    runner_events: List[Dict[str, Any]] = []
    out_root = Path(payload["out_dir"])

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
            runner_stage_status_surface=_build_runner_stage_status_surface(payload, status="blocked_preflight"),
            delegated_stage_status_surface=_build_delegated_stage_status_surface(
                payload,
                default_status="blocked_preflight",
            ),
            runner_events=runner_events,
            blocking_reasons=preflight_errors,
            final_error="broader-family preflight failed before delegated execution",
        )
        raise SystemExit(
            "broader-family preflight failed before delegated execution: "
            + " | ".join(str(err) for err in preflight_errors)
        )

    runner_state_path = Path(payload["runner_stage"]["state_path"])
    runner_log_path = Path(payload["runner_stage"]["log_path"])
    started_at = _now_iso()
    _append_runner_event(runner_events, event="real_run_started", status="running")
    _write_json(
        runner_state_path,
        {
            "stage": payload["runner_stage"]["stage"],
            "status": "running",
            "command": payload["runner_stage"]["command"],
            "expected_inputs": payload["runner_stage"]["expected_inputs"],
            "expected_outputs": payload["runner_stage"]["expected_outputs"],
            "started_at_utc": started_at,
        },
    )
    _write_runner_report(
        payload,
        run_mode="real_run",
        runner_status="running",
        runner_stage_status_surface=_build_runner_stage_status_surface(
            payload,
            status="running",
            extra={"started_at_utc": started_at},
        ),
        delegated_stage_status_surface=_build_delegated_stage_status_surface(payload, default_status="pending"),
        runner_events=runner_events,
    )

    missing_inputs = _verify_paths(payload["runner_stage"]["expected_inputs"])
    if missing_inputs:
        error_message = f"missing branch-owned runner inputs: {missing_inputs}"
        _append_runner_event(runner_events, event="real_run_failed", status="failed", detail=error_message)
        _write_json(
            runner_state_path,
            {
                "stage": payload["runner_stage"]["stage"],
                "status": "error",
                "error": error_message,
                "started_at_utc": started_at,
            },
        )
        _write_runner_report(
            payload,
            run_mode="real_run",
            runner_status="failed",
            runner_stage_status_surface=_build_runner_stage_status_surface(
                payload,
                status="error",
                extra={"started_at_utc": started_at, "error": error_message},
            ),
            delegated_stage_status_surface=_build_delegated_stage_status_surface(payload, default_status="pending"),
            runner_events=runner_events,
            blocking_reasons=[error_message],
            final_error=error_message,
        )
        raise RuntimeError(error_message)

    _append_runner_event(runner_events, event="delegated_run_started", status="running")
    try:
        _run_subprocess(payload["runner_stage"]["command"], runner_log_path)
    except Exception as e:
        error_message = f"{type(e).__name__}: {e}"
        _append_runner_event(runner_events, event="delegated_run_failed", status="error", detail=error_message)
        _append_runner_event(runner_events, event="real_run_failed", status="failed", detail=error_message)
        _write_json(
            runner_state_path,
            {
                "stage": payload["runner_stage"]["stage"],
                "status": "error",
                "error": error_message,
                "started_at_utc": started_at,
            },
        )
        _write_runner_report(
            payload,
            run_mode="real_run",
            runner_status="failed",
            runner_stage_status_surface=_build_runner_stage_status_surface(
                payload,
                status="error",
                extra={"started_at_utc": started_at, "error": error_message},
            ),
            delegated_stage_status_surface=_build_delegated_stage_status_surface(payload, default_status="pending"),
            runner_events=runner_events,
            blocking_reasons=[error_message],
            final_error=error_message,
        )
        raise

    missing_outputs = _verify_paths(payload["runner_stage"]["expected_outputs"])
    if missing_outputs:
        error_message = f"missing delegated public-mainline outputs: {missing_outputs}"
        _append_runner_event(runner_events, event="delegated_run_failed", status="error", detail=error_message)
        _append_runner_event(runner_events, event="real_run_failed", status="failed", detail=error_message)
        _write_json(
            runner_state_path,
            {
                "stage": payload["runner_stage"]["stage"],
                "status": "error",
                "error": error_message,
                "started_at_utc": started_at,
            },
        )
        _write_runner_report(
            payload,
            run_mode="real_run",
            runner_status="failed",
            runner_stage_status_surface=_build_runner_stage_status_surface(
                payload,
                status="error",
                extra={"started_at_utc": started_at, "error": error_message},
            ),
            delegated_stage_status_surface=_build_delegated_stage_status_surface(payload, default_status="pending"),
            runner_events=runner_events,
            blocking_reasons=[error_message],
            final_error=error_message,
        )
        raise RuntimeError(error_message)

    finished_at = _now_iso()
    _append_runner_event(runner_events, event="delegated_run_completed", status="success")
    _append_runner_event(runner_events, event="real_run_completed", status="completed")
    _write_json(
        runner_state_path,
        {
            "stage": payload["runner_stage"]["stage"],
            "status": "success",
            "started_at_utc": started_at,
            "finished_at_utc": finished_at,
        },
    )

    manifest = dict(payload)
    manifest["kind"] = "broader_shift_family_manifest"
    manifest["run_mode"] = "real_run"
    manifest["status"] = "success"
    manifest["runner_status"] = "completed"
    manifest["finished_at_utc"] = finished_at
    manifest["runner_stage_status_surface"] = _build_runner_stage_status_surface(
        payload,
        status="success",
        extra={"started_at_utc": started_at, "finished_at_utc": finished_at},
    )
    manifest["delegated_stage_status_surface"] = _build_delegated_stage_status_surface(payload, default_status="pending")
    manifest["delegated_artifact_surface"] = _build_delegated_artifact_surface(payload)
    manifest["restart_surface"] = _build_restart_surface(
        payload,
        resume_requested=bool(payload["runtime_diagnostics"].get("resume")),
    )
    manifest["runner_events"] = list(runner_events)
    manifest["runtime_proof_implied"] = False
    manifest["scientific_proof_implied"] = False
    manifest["scientific_summary_equivalent"] = False
    manifest["notes"] = [
        "This manifest is a branch-owned execution/reporting artifact, not a scientific summary contract.",
        "Delegated public-mainline artifacts are reused and tracked here machine-readably.",
        "Completed branch-owned runner state does not by itself imply runtime proof or scientific proof.",
    ]
    _write_json(out_root / "broader_shift_family_manifest.json", manifest)
    _write_runner_report(
        payload,
        run_mode="real_run",
        runner_status="completed",
        runner_stage_status_surface=manifest["runner_stage_status_surface"],
        delegated_stage_status_surface=manifest["delegated_stage_status_surface"],
        runner_events=runner_events,
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Broader shift-family contour runner")
    parser.add_argument("--config", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--encoder_profile", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--bootstrap_samples", type=int, default=None)
    parser.add_argument("--max_pairs", type=int, default=None)
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--probe_batch_size", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    kwargs = {
        "config": args.config,
        "out_dir": args.out_dir,
        "encoder_profile": args.encoder_profile,
        "seed": args.seed,
        "device": args.device,
        "bootstrap_samples": args.bootstrap_samples,
        "max_pairs": args.max_pairs,
        "max_images": args.max_images,
        "probe_batch_size": args.probe_batch_size,
        "resume": args.resume,
    }
    if args.dry_run:
        payload = build_broader_shift_family_plan(**kwargs)
        print(f"Saved dry-run plan: {Path(payload['out_dir']) / 'broader_shift_family_plan.json'}")
        return

    manifest = run_broader_shift_family_suite(**kwargs)
    print(
        "Broader shift-family run complete. Manifest: "
        f"{Path(manifest['out_dir']) / 'broader_shift_family_manifest.json'}"
    )


if __name__ == "__main__":
    main()
