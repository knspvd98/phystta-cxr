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
from phystta.intervention import eraser_fingerprint_mismatches, load_eraser
from phystta.models import EncoderFactory, build_encoder_fingerprint
from phystta.models.head_contract import resolve_head_contract
from phystta.utils import preprocess_signature
from scripts.runpod_preflight import run_preflight


RUNNER_REPORT_NAME = "vlm_runner_report.json"


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
        raise RuntimeError(f"vlm stage command failed with exit_code={proc.returncode}")


def _stage_state_dir(out_root: Path) -> Path:
    return out_root / ".vlm_stage_state"


def _stage_state_path(out_root: Path, stage: str) -> Path:
    return _stage_state_dir(out_root) / f"{stage}.json"


def _stage_log_path(out_root: Path, stage: str) -> Path:
    return out_root / "vlm_runner_logs" / f"{stage}.log"


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


def _verify_paths(paths: List[str]) -> List[str]:
    return [str(Path(raw).expanduser()) for raw in paths if not Path(raw).expanduser().exists()]


def _expected_label_cols(cfg) -> List[str]:
    label_cols = list(getattr(cfg.data, "label_cols", None) or [])
    if label_cols:
        return [str(col) for col in label_cols]
    out: List[str] = []
    for src in getattr(cfg.data, "sources", []):
        for col in list(getattr(src, "label_cols", None) or []):
            if col not in out:
                out.append(str(col))
    return out


def _vlm_identity_status(cfg) -> Dict[str, Any]:
    reasons: List[str] = []
    family = str(getattr(cfg.model, "family", ""))
    encoder_source = str(getattr(cfg.model.encoder, "encoder_source", ""))
    num_classes = int(getattr(cfg.model, "num_classes", -1))
    use_realism = bool(getattr(cfg.intervention, "use_realism", False))
    experiment_tier = str(getattr(cfg.experiment, "tier", "dev"))
    vlm_enabled = bool(getattr(cfg.vlm, "enabled", False))
    vlm_model_type = str(getattr(cfg.vlm, "model_type", ""))

    if experiment_tier != "paper":
        reasons.append("VLM runner requires experiment.tier='paper'.")
    if family != "transformer":
        reasons.append("VLM runner requires model.family='transformer'.")
    if encoder_source != "hf_clip_or_vit":
        reasons.append("VLM runner requires encoder_source='hf_clip_or_vit'.")
    if num_classes != 14:
        reasons.append("VLM runner requires the 14-class representation-surgery contour.")
    if not use_realism:
        reasons.append("VLM runner requires intervention.use_realism=true.")
    if not vlm_enabled:
        reasons.append("VLM runner requires vlm.enabled=true.")
    if not vlm_model_type:
        reasons.append("VLM runner requires vlm.model_type.")

    return {
        "checked": True,
        "eligible": len(reasons) == 0,
        "experiment_tier": experiment_tier,
        "model_family": family,
        "encoder_source": encoder_source,
        "num_classes": num_classes,
        "realism_enabled": use_realism,
        "vlm_enabled": vlm_enabled,
        "vlm_model_type": vlm_model_type,
        "intervention_shift_types": [str(x) for x in getattr(cfg.intervention, "shift_types", [])],
        "erasure_eval_shift_types": [str(x) for x in getattr(cfg.erasure_eval, "shift_types", [])],
        "reasons": reasons,
    }


def _build_head_contract(cfg, *, encoder_profile: str | None) -> Dict[str, Any]:
    enc_cfg = EncoderFactory.resolve_encoder_config(cfg, profile=encoder_profile)
    configured_head_path = getattr(enc_cfg, "head_path", None)
    configured_head_meta_path = getattr(enc_cfg, "head_meta_path", None)
    expected_label_cols = _expected_label_cols(cfg)
    reasons: List[str] = []
    try:
        contract = resolve_head_contract(
            cfg,
            label_cols=expected_label_cols if expected_label_cols else None,
            encoder_profile=encoder_profile,
            require_head_path=True,
            strict_meta=True,
            load_artifact=True,
        )
        return {
            "checked": True,
            "eligible": True,
            "status": "ok",
            "head_path": str(contract.head_path) if contract.head_path is not None else None,
            "head_meta_path": (
                str(getattr(contract.enc_cfg, "head_meta_path", None))
                if getattr(contract.enc_cfg, "head_meta_path", None) is not None
                else None
            ),
            "expected_num_classes": int(getattr(cfg.model, "num_classes", 0)),
            "expected_label_cols": expected_label_cols,
            "reasons": reasons,
        }
    except Exception as e:
        reasons.append(
            "VLM runner requires a valid head/head-meta contract before execution: "
            + f"{type(e).__name__}: {e}"
        )
        return {
            "checked": True,
            "eligible": False,
            "status": "error",
            "configured_head_path": str(configured_head_path) if configured_head_path is not None else None,
            "configured_head_meta_path": (
                str(configured_head_meta_path) if configured_head_meta_path is not None else None
            ),
            "expected_num_classes": int(getattr(cfg.model, "num_classes", 0)),
            "expected_label_cols": expected_label_cols,
            "error": f"{type(e).__name__}: {e}",
            "reasons": reasons,
        }


def _build_eraser_contract(cfg, *, eraser_path: Path, encoder_profile: str | None) -> Dict[str, Any]:
    reasons: List[str] = []
    expected_fp = build_encoder_fingerprint(EncoderFactory.resolve_encoder_config(cfg, profile=encoder_profile))
    expected_fp["preprocess_signature"] = preprocess_signature(cfg)
    if not eraser_path.exists():
        reasons.append(f"VLM runner requires an existing eraser artifact: {eraser_path}")
        return {
            "checked": True,
            "eligible": False,
            "path": str(eraser_path),
            "exists": False,
            "fingerprint_status": "missing",
            "reasons": reasons,
        }
    try:
        eraser = load_eraser(str(eraser_path))
        mismatches = eraser_fingerprint_mismatches(eraser, expected_fp)
    except Exception as e:
        mismatches = [f"{type(e).__name__}: {e}"]
    if mismatches:
        reasons.append(
            "VLM runner requires an eraser artifact compatible with the active encoder/preprocess: "
            + "; ".join(str(x) for x in mismatches)
        )
    return {
        "checked": True,
        "eligible": len(reasons) == 0,
        "path": str(eraser_path),
        "exists": True,
        "fingerprint_status": "ok" if not mismatches else "error",
        "fingerprint_mismatches": [str(x) for x in mismatches],
        "expected_encoder_source": expected_fp.get("encoder_source"),
        "expected_preprocess_signature": expected_fp.get("preprocess_signature"),
        "reasons": reasons,
    }


def _build_realism_contract(cfg) -> Dict[str, Any]:
    path = Path(str(getattr(cfg.intervention, "realism_path", ""))).expanduser()
    enabled = bool(getattr(cfg.intervention, "use_realism", False))
    reasons: List[str] = []
    if enabled and not str(getattr(cfg.intervention, "realism_path", "")).strip():
        reasons.append("VLM runner requires intervention.realism_path when realism is enabled.")
    elif enabled and not path.exists():
        reasons.append(f"VLM runner requires an existing realism artifact: {path}")
    return {
        "checked": True,
        "eligible": len(reasons) == 0,
        "enabled": enabled,
        "path": str(path),
        "exists": bool(path.exists()),
        "reasons": reasons,
    }


def _append_optional_args(
    cmd: List[str],
    *,
    seed: int | None,
    device: str | None,
    max_images: int | None,
    split: str | None,
    shift_type: str | None,
) -> None:
    if split:
        cmd.extend(["--split", str(split)])
    if shift_type:
        cmd.extend(["--shift_type", str(shift_type)])
    if max_images is not None:
        cmd.extend(["--max_images", str(int(max_images))])
    if seed is not None:
        cmd.extend(["--seed", str(seed)])
    if device:
        cmd.extend(["--device", str(device)])


def _build_stage_status_surface(
    payload: Dict[str, Any],
    *,
    status: str,
    extra: Dict[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    stage = payload["runner_stage"]
    state_path = Path(stage["state_path"]).expanduser()
    log_path = Path(stage["log_path"]).expanduser()
    state_payload = _read_json(state_path)
    existing_inputs_found, existing_inputs_missing = _classify_paths(list(stage["expected_inputs"]))
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
        "existing_inputs_found": existing_inputs_found,
        "existing_inputs_missing": existing_inputs_missing,
        "expected_outputs": list(stage["expected_outputs"]),
        "expected_outputs_known": True,
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


def _build_branch_owned_artifact_surface(payload: Dict[str, Any]) -> Dict[str, Any]:
    reporting = dict(payload.get("reporting_artifacts", {}))
    preflight_artifacts = dict(payload.get("preflight_artifacts", {}))
    expected_keys = [
        ("runpod_preflight_json", preflight_artifacts.get("runpod_preflight")),
        ("vlm_rehearsal_json", preflight_artifacts.get("vlm_rehearsal")),
        ("vlm_plan_json", reporting.get("plan_json")),
        ("vlm_runner_report_json", reporting.get("runner_report_json")),
        ("vlm_manifest_json", reporting.get("branch_manifest_json")),
        ("vlm_outputs_jsonl", reporting.get("vlm_outputs_jsonl")),
        ("vlm_summary_json", reporting.get("vlm_summary_json")),
        ("vlm_metrics_csv", reporting.get("vlm_metrics_csv")),
    ]
    expected_artifacts: List[Dict[str, Any]] = []
    found_artifacts: List[Dict[str, Any]] = []
    missing_artifacts: List[Dict[str, Any]] = []
    for name, raw in expected_keys:
        if not raw:
            continue
        path = Path(str(raw)).expanduser()
        row = {"name": name, "path": str(path), "exists": path.exists()}
        expected_artifacts.append(row)
        if row["exists"]:
            found_artifacts.append(row)
        else:
            missing_artifacts.append(row)
    return {
        "reused_generic_reporting": False,
        "branch_owned_scientific_summary": False,
        "expected_artifacts": expected_artifacts,
        "found_artifacts": found_artifacts,
        "missing_artifacts": missing_artifacts,
        "notes": [
            "These are branch-owned VLM runner artifacts and stage outputs.",
            "Their presence is useful execution/reporting evidence, not runtime proof or a scientific summary contract.",
        ],
    }


def _build_restart_surface(payload: Dict[str, Any]) -> Dict[str, Any]:
    stage = payload["runner_stage"]
    state_path = Path(stage["state_path"]).expanduser()
    log_path = Path(stage["log_path"]).expanduser()
    state_payload = _read_json(state_path)
    outputs_found, outputs_missing = _classify_paths(list(stage["expected_outputs"]))
    artifact_surface = _build_branch_owned_artifact_surface(payload)
    return {
        "runner_auto_resume_supported": False,
        "resume_requested": False,
        "existing_stage_state_detected": state_payload is not None,
        "existing_successful_stage_detected": (
            bool(isinstance(state_payload, dict) and state_payload.get("status") == "success")
        ),
        "existing_outputs_detected": bool(outputs_found),
        "branch_owned_artifacts_present": [row["path"] for row in artifact_surface["found_artifacts"]],
        "branch_owned_artifacts_missing": [row["path"] for row in artifact_surface["missing_artifacts"]],
        "rerun_required_for_all_stages": True,
        "stage_observations": [
            {
                "stage_index": 1,
                "stage": stage["stage"],
                "state_path": str(state_path),
                "state_detected": state_payload is not None,
                "state_status": state_payload.get("status") if isinstance(state_payload, dict) else None,
                "log_path": str(log_path),
                "log_detected": log_path.exists(),
                "existing_outputs_present": outputs_found,
                "existing_outputs_missing": outputs_missing,
                "rerun_required": True,
            }
        ],
        "notes": [
            "The VLM runner does not implement automatic resume.",
            "Existing stage-state or outputs are forensic/restart clues only and do not prove successful restart semantics.",
            "Existing branch-owned runner artifacts are forensic clues only and do not prove valid restart semantics.",
        ],
    }


def _append_runner_event(
    events: List[Dict[str, Any]],
    *,
    event: str,
    status: str,
    detail: str | None = None,
) -> None:
    row: Dict[str, Any] = {"timestamp_utc": _now_iso(), "event": event, "status": status}
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
    runner_events: List[Dict[str, Any]] | None = None,
    blocking_reasons: List[str] | None = None,
    final_error: str | None = None,
) -> Dict[str, Any]:
    preflight = payload.get("preflight", {})
    preflight_errors = list(preflight.get("errors", [])) if isinstance(preflight, dict) else []
    vlm_contract = preflight.get("vlm_contract", {}) if isinstance(preflight, dict) else {}
    preflight_eligible = bool(isinstance(vlm_contract, dict) and vlm_contract.get("eligible", False)) and not preflight_errors
    return {
        "kind": "vlm_runner_report",
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
        "branch_owned_artifact_surface": _build_branch_owned_artifact_surface(payload),
        "head_contract": dict(payload["head_contract"]),
        "eraser_contract": dict(payload["eraser_contract"]),
        "realism_contract": dict(payload["realism_contract"]),
        "runtime_diagnostics": dict(payload["runtime_diagnostics"]),
        "reporting_artifacts": dict(payload["reporting_artifacts"]),
        "runner_stage_status_surface": runner_stage_status_surface,
        "restart_surface": _build_restart_surface(payload),
        "runner_events": list(runner_events or []),
        "blocking_reasons": list(blocking_reasons or []),
        "preflight_errors": preflight_errors,
        "final_error": final_error,
        "runtime_proof_implied": False,
        "scientific_proof_implied": False,
        "scientific_summary_equivalent": False,
        "notes": [
            "This artifact is a branch-owned VLM execution/reporting surface, not a scientific summary contract.",
            "Completion of the VLM runner does not imply runtime proof on pod or scientific proof.",
            "The contour remains representation-surgery + CLIP-lineage + realism-enabled + VLM-stage specific.",
        ],
    }


def _write_runner_report(
    payload: Dict[str, Any],
    *,
    run_mode: str,
    runner_status: str,
    runner_stage_status_surface: List[Dict[str, Any]],
    runner_events: List[Dict[str, Any]] | None = None,
    blocking_reasons: List[str] | None = None,
    final_error: str | None = None,
) -> Dict[str, Any]:
    report = _build_runner_report(
        payload,
        run_mode=run_mode,
        runner_status=runner_status,
        runner_stage_status_surface=runner_stage_status_surface,
        runner_events=runner_events,
        blocking_reasons=blocking_reasons,
        final_error=final_error,
    )
    _write_json(_runner_report_path(Path(payload["out_dir"])), report)
    return report


def build_vlm_plan(
    *,
    config: str | Path,
    eraser: str | Path,
    out_dir: str | Path,
    encoder_profile: str | None = None,
    seed: int | None = None,
    device: str | None = None,
    max_images: int | None = None,
    split: str | None = None,
    shift_type: str | None = None,
) -> Dict[str, Any]:
    config_path = Path(config).expanduser()
    eraser_path = Path(eraser).expanduser()
    out_root = Path(out_dir).expanduser()
    out_root.mkdir(parents=True, exist_ok=True)

    cfg = load_config(str(config_path))
    if seed is not None:
        cfg.runtime.seed = int(seed)
    if device is not None:
        cfg.runtime.device = str(device)

    effective_cfg_path = out_root / "vlm.effective.yaml"
    save_config(str(effective_cfg_path), cfg)

    preflight_path = out_root / "runpod_preflight.json"
    rehearsal_path = out_root / "vlm_rehearsal.json"
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
                "errors": ["VLM preflight exited without writing a JSON payload."],
                "warnings": [],
            }
            _write_json(preflight_path, preflight_payload)

    if preflight_payload is None:
        preflight_payload = {"config": str(effective_cfg_path), "errors": [], "warnings": []}

    identity_contract = _vlm_identity_status(cfg)
    head_contract = _build_head_contract(cfg, encoder_profile=encoder_profile)
    eraser_contract = _build_eraser_contract(cfg, eraser_path=eraser_path, encoder_profile=encoder_profile)
    realism_contract = _build_realism_contract(cfg)
    blocking_reasons = list(preflight_payload.get("errors", []))
    blocking_reasons.extend(identity_contract["reasons"])
    blocking_reasons.extend(head_contract["reasons"])
    blocking_reasons.extend(eraser_contract["reasons"])
    blocking_reasons.extend(realism_contract["reasons"])

    stage_out_dir = out_root / "vlm_eval"
    real_cmd = [
        sys.executable,
        "scripts/run_vlm_erasure_eval.py",
        "--config",
        str(effective_cfg_path),
        "--eraser",
        str(eraser_path),
        "--out_dir",
        str(stage_out_dir),
    ]
    _append_optional_args(
        real_cmd,
        seed=seed,
        device=device,
        max_images=max_images,
        split=split,
        shift_type=shift_type,
    )
    dry_run_cmd = list(real_cmd)
    dry_run_cmd.append("--max_images")
    dry_run_cmd.append(str(int(max_images) if max_images is not None else 50))

    runner_stage = {
        "stage": "vlm_erasure_eval",
        "command": real_cmd,
        "dry_run_command": dry_run_cmd,
        "log_path": str(_stage_log_path(out_root, "vlm_erasure_eval")),
        "state_path": str(_stage_state_path(out_root, "vlm_erasure_eval")),
        "expected_inputs": [
            path
            for path in [
                str(effective_cfg_path),
                str(preflight_path),
                str(rehearsal_path),
                str(eraser_path),
                str(head_contract.get("head_path") or head_contract.get("configured_head_path") or ""),
                str(head_contract.get("head_meta_path") or head_contract.get("configured_head_meta_path") or ""),
            ]
            if path
        ],
        "expected_outputs": [
            str(stage_out_dir / "vlm_outputs.jsonl"),
            str(stage_out_dir / "vlm_summary.json"),
            str(stage_out_dir / "vlm_metrics.csv"),
        ],
    }

    payload = {
        "timestamp_utc": _now_iso(),
        "source_config": str(config_path),
        "effective_config": str(effective_cfg_path),
        "out_dir": str(out_root),
        "canonical_contract": "docs/canonical_vlm_runpod_packet.md",
        "identity_contract": identity_contract,
        "preflight": preflight_payload,
        "preflight_artifacts": {
            "runpod_preflight": str(preflight_path),
            "vlm_rehearsal": str(rehearsal_path),
        },
        "head_contract": head_contract,
        "eraser_contract": eraser_contract,
        "realism_contract": realism_contract,
        "runner_stage": runner_stage,
        "reporting_artifacts": {
            "plan_json": str(out_root / "vlm_plan.json"),
            "runner_report_json": str(_runner_report_path(out_root)),
            "branch_manifest_json": str(out_root / "vlm_manifest.json"),
            "vlm_outputs_jsonl": str(stage_out_dir / "vlm_outputs.jsonl"),
            "vlm_summary_json": str(stage_out_dir / "vlm_summary.json"),
            "vlm_metrics_csv": str(stage_out_dir / "vlm_metrics.csv"),
        },
        "runtime_diagnostics": {
            "python_executable": sys.executable,
            "repo_root": str(_repo_root()),
            "device": str(getattr(cfg.runtime, "device", "")),
            "seed": int(getattr(cfg.runtime, "seed", 0)),
            "model_family": str(getattr(cfg.model, "family", "")),
            "encoder_source": str(getattr(cfg.model.encoder, "encoder_source", "")),
            "vlm_model_type": str(getattr(cfg.vlm, "model_type", "")),
            "vlm_model_path": str(getattr(cfg.vlm, "model_path", "") or ""),
            "vlm_model_id": getattr(cfg.vlm, "model_id", None),
            "head_path": str(head_contract.get("head_path") or head_contract.get("configured_head_path") or ""),
            "head_meta_path": str(
                head_contract.get("head_meta_path") or head_contract.get("configured_head_meta_path") or ""
            ),
            "head_contract_eligible": bool(head_contract["eligible"]),
            "realism_enabled": bool(getattr(cfg.intervention, "use_realism", False)),
            "realism_path": str(getattr(cfg.intervention, "realism_path", "")),
            "realism_contract_eligible": bool(realism_contract["eligible"]),
            "eraser_path": str(eraser_path),
            "eraser_contract_eligible": bool(eraser_contract["eligible"]),
            "preflight_checked": bool(preflight_payload.get("vlm_contract", {}).get("checked", False)),
            "preflight_eligible": bool(preflight_payload.get("vlm_contract", {}).get("eligible", False))
            and not bool(preflight_payload.get("errors", [])),
            "source_roots": [
                str(row.get("root"))
                for row in preflight_payload.get("sources", [])
                if isinstance(row, dict) and row.get("root") is not None
            ],
            "freeze_artifacts": bool(getattr(cfg.runtime, "freeze_artifacts", False)),
            "allow_download": bool(getattr(cfg.runtime, "allow_download", False)),
            "shift_types": [str(x) for x in getattr(cfg.intervention, "shift_types", [])],
            "eval_shift_types": [str(x) for x in getattr(cfg.erasure_eval, "shift_types", [])],
            "split": split or str(getattr(cfg.data, "eval_split", "")),
            "shift_type": shift_type,
            "max_images": max_images,
            "stage_out_dir": str(stage_out_dir),
            "logs_dir": str(out_root / "vlm_runner_logs"),
            "stage_state_dir": str(out_root / ".vlm_stage_state"),
            "branch_runner_entrypoint": str(_repo_root() / "scripts" / "run_vlm_suite.py"),
        },
        "blocking_reasons": blocking_reasons,
    }

    _write_json(out_root / "vlm_plan.json", payload)
    dry_run_status = "dry_run_blocked" if blocking_reasons else "dry_run_ready"
    dry_run_events: List[Dict[str, Any]] = []
    _append_runner_event(
        dry_run_events,
        event="dry_run_plan_materialized",
        status=dry_run_status,
        detail=f"plan={out_root / 'vlm_plan.json'}",
    )
    _write_runner_report(
        payload,
        run_mode="dry_run",
        runner_status=dry_run_status,
        runner_stage_status_surface=_build_stage_status_surface(
            payload,
            status="blocked_preflight" if blocking_reasons else "planned",
        ),
        runner_events=dry_run_events,
        blocking_reasons=blocking_reasons,
    )
    return payload


def run_vlm_suite(**kwargs) -> Dict[str, Any]:
    payload = build_vlm_plan(**kwargs)
    blocking_reasons = list(payload.get("blocking_reasons", []))
    out_root = Path(payload["out_dir"])
    runner_events: List[Dict[str, Any]] = []

    if blocking_reasons:
        _append_runner_event(runner_events, event="real_run_started", status="running")
        _append_runner_event(
            runner_events,
            event="real_run_blocked_preflight",
            status="blocked_preflight",
            detail=" | ".join(str(x) for x in blocking_reasons),
        )
        _write_runner_report(
            payload,
            run_mode="real_run",
            runner_status="blocked_preflight",
            runner_stage_status_surface=_build_stage_status_surface(payload, status="blocked_preflight"),
            runner_events=runner_events,
            blocking_reasons=blocking_reasons,
            final_error="VLM runner contract failed before execution",
        )
        raise SystemExit("VLM runner contract failed before execution: " + " | ".join(str(x) for x in blocking_reasons))

    state_path = Path(payload["runner_stage"]["state_path"])
    log_path = Path(payload["runner_stage"]["log_path"])
    started_at = _now_iso()
    _append_runner_event(runner_events, event="real_run_started", status="running")
    _append_runner_event(runner_events, event="stage_started", status="running", detail="vlm_erasure_eval")
    _write_json(
        state_path,
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
        runner_stage_status_surface=_build_stage_status_surface(
            payload,
            status="running",
            extra={"started_at_utc": started_at},
        ),
        runner_events=runner_events,
    )

    missing_inputs = _verify_paths(payload["runner_stage"]["expected_inputs"])
    if missing_inputs:
        error_message = f"missing VLM runner inputs: {missing_inputs}"
        _append_runner_event(runner_events, event="stage_failed", status="error", detail=error_message)
        _append_runner_event(runner_events, event="real_run_failed", status="failed", detail=error_message)
        _write_json(
            state_path,
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
            runner_stage_status_surface=_build_stage_status_surface(
                payload,
                status="error",
                extra={"started_at_utc": started_at, "error": error_message},
            ),
            runner_events=runner_events,
            blocking_reasons=[error_message],
            final_error=error_message,
        )
        raise RuntimeError(error_message)

    try:
        _run_subprocess(payload["runner_stage"]["command"], log_path)
    except Exception as e:
        error_message = f"{type(e).__name__}: {e}"
        _append_runner_event(runner_events, event="stage_failed", status="error", detail=error_message)
        _append_runner_event(runner_events, event="real_run_failed", status="failed", detail=error_message)
        _write_json(
            state_path,
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
            runner_stage_status_surface=_build_stage_status_surface(
                payload,
                status="error",
                extra={"started_at_utc": started_at, "error": error_message},
            ),
            runner_events=runner_events,
            final_error=error_message,
        )
        raise

    outputs_found, outputs_missing = _classify_paths(payload["runner_stage"]["expected_outputs"])
    if outputs_missing:
        error_message = f"VLM runner completed with missing expected outputs: {outputs_missing}"
        _append_runner_event(runner_events, event="stage_failed", status="error", detail=error_message)
        _append_runner_event(runner_events, event="real_run_failed", status="failed", detail=error_message)
        _write_json(
            state_path,
            {
                "stage": payload["runner_stage"]["stage"],
                "status": "error",
                "error": error_message,
                "started_at_utc": started_at,
                "finished_at_utc": _now_iso(),
            },
        )
        _write_runner_report(
            payload,
            run_mode="real_run",
            runner_status="failed",
            runner_stage_status_surface=_build_stage_status_surface(
                payload,
                status="error",
                extra={
                    "started_at_utc": started_at,
                    "finished_at_utc": _now_iso(),
                    "error": error_message,
                },
            ),
            runner_events=runner_events,
            final_error=error_message,
        )
        raise RuntimeError(error_message)

    finished_at = _now_iso()
    _append_runner_event(runner_events, event="stage_completed", status="success", detail="vlm_erasure_eval")
    _append_runner_event(runner_events, event="real_run_completed", status="completed")
    _write_json(
        state_path,
        {
            "stage": payload["runner_stage"]["stage"],
            "status": "success",
            "started_at_utc": started_at,
            "finished_at_utc": finished_at,
            "outputs": outputs_found,
        },
    )
    report = _write_runner_report(
        payload,
        run_mode="real_run",
        runner_status="completed",
        runner_stage_status_surface=_build_stage_status_surface(
            payload,
            status="success",
            extra={
                "started_at_utc": started_at,
                "finished_at_utc": finished_at,
            },
        ),
        runner_events=runner_events,
    )
    manifest = dict(report)
    manifest["status"] = "success"
    manifest["runner_status"] = "completed"
    manifest_path = out_root / "vlm_manifest.json"
    _write_json(manifest_path, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Branch-owned VLM runner.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--eraser", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--encoder_profile", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--shift_type", default=None)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    kwargs = {
        "config": args.config,
        "eraser": args.eraser,
        "out_dir": args.out_dir,
        "encoder_profile": args.encoder_profile,
        "seed": args.seed,
        "device": args.device,
        "max_images": args.max_images,
        "split": args.split,
        "shift_type": args.shift_type,
    }
    if args.dry_run:
        payload = build_vlm_plan(**kwargs)
        print(f"Saved dry-run plan: {Path(payload['out_dir']) / 'vlm_plan.json'}")
        return
    manifest = run_vlm_suite(**kwargs)
    print(f"Saved VLM manifest: {Path(manifest['out_dir']) / 'vlm_manifest.json'}")


if __name__ == "__main__":
    main()
