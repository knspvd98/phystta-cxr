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
from typing import Any, Dict, List, Optional

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phystta.config.loader import load_config, save_config
from phystta.data import DataEngine
from phystta.intervention import (
    eraser_fingerprint_mismatches,
    load_eraser_collection,
    resolve_eraser_for_shift,
)
from phystta.models import (
    EncoderFactory,
    build_encoder_fingerprint,
    resolve_head_contract,
)
from phystta.shifts import ShiftManager
from phystta.shifts.realism import (
    calibrator_fingerprint_mismatches,
    load_calibrator,
    scenario_grid_from_calibrator,
)
from phystta.utils import load_torch_artifact, preprocess_signature


STAGE_ORDER = [
    "realism_calibration",
    "collect_probes",
    "fit_eraser",
    "confounding_audit",
    "erasure_eval",
    "safety_eval",
    "summary",
]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_if_exists(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    return _sha256_file(path)


def _aggregate_hash_map(items: dict[str, str | None]) -> str:
    h = hashlib.sha256()
    for key in sorted(items):
        val = items[key] if items[key] is not None else "<missing>"
        h.update(f"{key}:{val}\n".encode("utf-8"))
    return h.hexdigest()


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


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


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


def _source_path_checks(cfg) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for src in cfg.data.sources:
        candidates = {
            "root": src.root,
            "metadata_csv": src.metadata_csv,
            "extra_metadata_csv": src.extra_metadata_csv,
            "image_root": src.image_root,
        }
        for name, value in candidates.items():
            if value is None:
                continue
            path = Path(value).expanduser()
            rows.append(
                {
                    "source": str(src.domain_label or src.name),
                    "field": name,
                    "path": str(path),
                    "exists": bool(path.exists()),
                }
            )
    return rows


def _stage_state_dir(out_root: Path) -> Path:
    return out_root / ".stage_state"


def _stage_state_path(out_root: Path, stage: str) -> Path:
    return _stage_state_dir(out_root) / f"{stage}.json"


def _stage_log_path(out_root: Path, stage: str) -> Path:
    return out_root / "launcher_logs" / f"{stage}.log"


def _load_stage_state(out_root: Path, stage: str) -> Dict[str, Any] | None:
    return _read_json(_stage_state_path(out_root, stage))


def _write_stage_state(out_root: Path, stage: str, payload: Dict[str, Any]) -> None:
    _write_json(_stage_state_path(out_root, stage), payload)


def _output_hashes(paths: List[str]) -> Dict[str, str | None]:
    out: Dict[str, str | None] = {}
    for raw in paths:
        p = Path(raw).expanduser()
        out[str(p)] = _sha256_if_exists(p)
    return out


def _split_shift_string(raw: object) -> set[str]:
    if raw is None:
        return set()
    if isinstance(raw, (list, tuple, set)):
        values = raw
    else:
        values = str(raw).split(",")
    return {str(v).strip() for v in values if str(v).strip()}


def _validate_stage_name(name: str | None, *, allow_none: bool = True) -> str | None:
    if name is None and allow_none:
        return None
    if name not in STAGE_ORDER:
        raise RuntimeError(f"Invalid stage '{name}'. Valid stages: {STAGE_ORDER}")
    return name


def _stage_range(start_from: str | None, stop_after: str | None) -> list[str]:
    start_from = _validate_stage_name(start_from)
    stop_after = _validate_stage_name(stop_after)
    lo = STAGE_ORDER.index(start_from) if start_from is not None else 0
    hi = STAGE_ORDER.index(stop_after) if stop_after is not None else len(STAGE_ORDER) - 1
    if lo > hi:
        raise RuntimeError(f"start_from='{start_from}' must be earlier than or equal to stop_after='{stop_after}'.")
    return STAGE_ORDER[lo : hi + 1]


def _apply_path_overrides(cfg, args, out_root: Path, realism_path: Path) -> None:
    if not cfg.data.sources:
        return
    src = cfg.data.sources[0]
    if args.dataset_root is not None:
        dataset_root = Path(args.dataset_root).expanduser()
        src.root = dataset_root
        src.image_root = Path(args.image_root).expanduser() if args.image_root is not None else dataset_root
        if args.metadata_csv is not None:
            src.metadata_csv = Path(args.metadata_csv).expanduser()
        elif src.metadata_csv is not None:
            src.metadata_csv = dataset_root / Path(src.metadata_csv).name
        if args.extra_metadata_csv is not None:
            src.extra_metadata_csv = Path(args.extra_metadata_csv).expanduser()
        elif src.extra_metadata_csv is not None:
            src.extra_metadata_csv = dataset_root / Path(src.extra_metadata_csv).name
    else:
        if args.image_root is not None:
            src.image_root = Path(args.image_root).expanduser()
        if args.metadata_csv is not None:
            src.metadata_csv = Path(args.metadata_csv).expanduser()
        if args.extra_metadata_csv is not None:
            src.extra_metadata_csv = Path(args.extra_metadata_csv).expanduser()

    if args.index_dir is not None:
        cfg.data.index_dir = Path(args.index_dir).expanduser()
    if args.cache_dir is not None:
        cfg.data.cache_dir = Path(args.cache_dir).expanduser()
    if args.head_path is not None:
        cfg.model.encoder.head_path = Path(args.head_path).expanduser()
    if args.head_meta_path is not None:
        cfg.model.encoder.head_meta_path = Path(args.head_meta_path).expanduser()
    cfg.output.root = out_root
    cfg.intervention.realism_path = realism_path
    if args.seed is not None:
        cfg.runtime.seed = int(args.seed)
    if args.device is not None:
        cfg.runtime.device = str(args.device)


def _build_probe_validation(path: Path, cfg) -> Dict[str, Any]:
    payload = load_torch_artifact(path, map_location="cpu")
    meta = payload.get("meta", {}) if isinstance(payload, dict) else {}
    errs: List[str] = []
    if not isinstance(meta, dict):
        meta = {}
    preprocess_sig = meta.get("preprocess_signature")
    if preprocess_sig is not None and str(preprocess_sig) != str(preprocess_signature(cfg)):
        errs.append(
            f"preprocess_signature mismatch: expected={preprocess_signature(cfg)} got={preprocess_sig}"
        )
    if str(meta.get("group_valid", "True")).lower() == "false":
        errs.append("probe artifact declares group_valid=False")
    if str(meta.get("consistency_ok", "True")).lower() == "false":
        errs.append("probe artifact declares consistency_ok=False")
    expected_shift_families = {str(s) for s in cfg.intervention.shift_types if str(s) != "mixed"}
    actual_shift_families = _split_shift_string(meta.get("shift_family") or meta.get("shift_type"))
    if expected_shift_families and actual_shift_families and not expected_shift_families.issubset(actual_shift_families):
        errs.append(
            f"probe shift families missing canonical members: expected={sorted(expected_shift_families)} got={sorted(actual_shift_families)}"
        )
    split_name = str(meta.get("source_split") or meta.get("split") or "")
    if split_name and split_name != str(cfg.data.eraser_split):
        errs.append(f"probe split mismatch: expected={cfg.data.eraser_split} got={split_name}")
    return {
        "errors": errs,
        "meta": meta,
        "path": str(path),
    }


def _build_eraser_validation(path: Path, expected_fingerprint: Dict[str, object], expected_shift_families: List[str]) -> Dict[str, Any]:
    errs: List[str] = []
    collection, bundle_meta = load_eraser_collection(path)
    for shift_family in expected_shift_families:
        try:
            resolve_eraser_for_shift(collection, shift_family)
        except Exception as e:
            errs.append(str(e))
    per_shift: Dict[str, List[str]] = {}
    for shift_family, eraser_obj in collection.items():
        mismatches = eraser_fingerprint_mismatches(eraser_obj, expected_fingerprint)
        if mismatches:
            label = "default" if shift_family == "__default__" else str(shift_family)
            per_shift[label] = mismatches
            errs.append(
                f"eraser fingerprint mismatch for shift_family='{label}': " + "; ".join(mismatches)
            )
    return {
        "errors": errs,
        "bundle_meta": bundle_meta,
        "per_shift_mismatches": per_shift,
        "path": str(path),
    }


def _build_realism_validation(path: Path, expected_fingerprint: Dict[str, object], expected_shift_families: List[str], pair_mode: str) -> Dict[str, Any]:
    errs: List[str] = []
    calib = load_calibrator(str(path))
    mismatches = calibrator_fingerprint_mismatches(calib, expected_fingerprint)
    if mismatches:
        errs.append("realism calibrator fingerprint mismatch: " + "; ".join(mismatches))
    scenario_map = scenario_grid_from_calibrator(calib, expected_shift_families, graded_only=True)
    missing = [shift for shift in expected_shift_families if shift not in scenario_map]
    if missing:
        errs.append(f"realism calibrator missing graded mappings for shift_families={missing}")
    stored_pair_mode = str((calib.metadata or {}).get("pair_mode", ""))
    if stored_pair_mode and stored_pair_mode != str(pair_mode):
        errs.append(f"realism pair_mode mismatch: expected={pair_mode} got={stored_pair_mode}")
    validity = (calib.metadata or {}).get("realism_validity_summary")
    if isinstance(validity, dict) and validity.get("is_graded") is False:
        errs.append("realism validity summary marks calibrator as non-graded for atomic evaluation")
    return {
        "errors": errs,
        "metadata": calib.metadata,
        "path": str(path),
    }


def _verify_stage_inputs(paths: List[str]) -> List[str]:
    missing: List[str] = []
    for raw in paths:
        if raw is None:
            continue
        p = Path(raw).expanduser()
        if not p.exists():
            missing.append(str(p))
    return missing


def _verify_stage_outputs(paths: List[str]) -> List[str]:
    return _verify_stage_inputs(paths)


def _safe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


def _evaluate_intervention_gate(
    summary_by_bank: List[Dict[str, Any]],
    *,
    min_nonzero_logit_delta: float,
    min_nonzero_prob_delta: float,
    max_zero_logit_damage: float,
    max_zero_prob_damage: float,
) -> Dict[str, Any]:
    """
    Evaluate whether an adjudicated eraser bank is classifier-relevant enough
    to justify expensive downstream erasure/safety stages.

    The gate is intentionally conservative:
    - it requires non-trivial classifier-visible movement under nonzero shifts
    - it rejects banks that already damage clean/zero-shift behavior too much

    It does NOT try to declare scientific success. It only blocks obviously
    classifier-null or obviously destructive banks.
    """
    if not summary_by_bank:
        return {
            "passed": False,
            "reason": "missing_summary_rows",
            "selected_bank": None,
            "selected_row": None,
            "thresholds": {
                "min_nonzero_logit_delta": min_nonzero_logit_delta,
                "min_nonzero_prob_delta": min_nonzero_prob_delta,
                "max_zero_logit_damage": max_zero_logit_damage,
                "max_zero_prob_damage": max_zero_prob_damage,
            },
        }

    ordered = sorted(
        summary_by_bank,
        key=lambda row: (_safe_float(row.get("score")) or float("-inf")),
        reverse=True,
    )
    row = ordered[0]
    bank = row.get("bank")

    nonzero_logit_delta = _safe_float(row.get("mean_nonzero_logit_delta"))
    nonzero_prob_delta = _safe_float(row.get("mean_nonzero_prob_delta"))
    zero_logit_damage = _safe_float(row.get("mean_zero_logit_damage"))
    zero_prob_damage = _safe_float(row.get("mean_zero_prob_damage"))

    failures: List[str] = []

    if nonzero_logit_delta is None or nonzero_logit_delta < min_nonzero_logit_delta:
        failures.append(
            f"mean_nonzero_logit_delta<{min_nonzero_logit_delta:g}"
        )
    if nonzero_prob_delta is None or nonzero_prob_delta < min_nonzero_prob_delta:
        failures.append(
            f"mean_nonzero_prob_delta<{min_nonzero_prob_delta:g}"
        )
    if zero_logit_damage is None or zero_logit_damage > max_zero_logit_damage:
        failures.append(
            f"mean_zero_logit_damage>{max_zero_logit_damage:g}"
        )
    if zero_prob_damage is None or zero_prob_damage > max_zero_prob_damage:
        failures.append(
            f"mean_zero_prob_damage>{max_zero_prob_damage:g}"
        )

    passed = len(failures) == 0
    return {
        "passed": passed,
        "reason": "ok" if passed else " | ".join(failures),
        "selected_bank": bank,
        "selected_row": row,
        "thresholds": {
            "min_nonzero_logit_delta": min_nonzero_logit_delta,
            "min_nonzero_prob_delta": min_nonzero_prob_delta,
            "max_zero_logit_damage": max_zero_logit_damage,
            "max_zero_prob_damage": max_zero_prob_damage,
        },
    }


def _build_intervention_gate_decision_payload(
    *,
    gate_payload: Dict[str, Any],
    decision_json_path: Path,
) -> Dict[str, Any]:
    verdict = gate_payload.get("verdict") or {}
    selected_row = verdict.get("selected_row")
    selected_bank = verdict.get("selected_bank")
    recommended_bank_metadata = gate_payload.get("recommended_bank_metadata")

    return {
        "stage": "intervention_gate",
        "status": gate_payload.get("status"),
        "reason": gate_payload.get("reason"),
        "timestamp_utc": gate_payload.get("timestamp_utc"),
        "decision_json": str(decision_json_path),
        "summary_json": gate_payload.get("summary_json"),
        "log_path": gate_payload.get("log_path"),
        "eraser_target": gate_payload.get("eraser_target"),
        "recommended_bank": selected_bank,
        "recommended_bank_metadata": recommended_bank_metadata,
        "thresholds": verdict.get("thresholds"),
        "verdict": verdict,
        "selected_row": selected_row,
    }


def _write_intervention_gate_decision_payload(
    *,
    gate_payload: Dict[str, Any],
    decision_json_path: Path,
) -> Dict[str, Any]:
    decision_payload = _build_intervention_gate_decision_payload(
        gate_payload=gate_payload,
        decision_json_path=decision_json_path,
    )
    decision_json_path.parent.mkdir(parents=True, exist_ok=True)
    decision_json_path.write_text(
        json.dumps(decision_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return decision_payload


def _normalize_artifact_paths(values: List[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for raw in values:
        if raw is None:
            continue
        path = str(raw)
        if path in seen:
            continue
        seen.add(path)
        ordered.append(path)
    return ordered


def _selected_stage_required_artifacts(stage_results: List[Dict[str, Any]]) -> List[str]:
    selected: List[str] = []
    for row in stage_results:
        for key in ("required_inputs", "expected_outputs"):
            vals = row.get(key) or []
            for item in vals:
                selected.append(str(item))
    return _normalize_artifact_paths(selected)


def _compute_artifact_contract(
    *,
    required_artifacts: List[str],
    stage_results: List[Dict[str, Any]],
) -> Dict[str, List[str]]:
    full_required = _normalize_artifact_paths([str(x) for x in required_artifacts])
    selected_required = _selected_stage_required_artifacts(stage_results)

    missing_selected = [p for p in selected_required if not Path(p).exists()]
    missing_full = [p for p in full_required if not Path(p).exists()]

    return {
        "full_required_artifacts": full_required,
        "selected_required_artifacts": selected_required,
        "missing_selected_artifacts": missing_selected,
        "missing_full_artifacts": missing_full,
    }


def _evaluate_summary_scientific_contract(summary_payload: Dict[str, Any]) -> Dict[str, Any]:
    reporting = summary_payload.get("reporting_contract")
    reporting = reporting if isinstance(reporting, dict) else {}

    report_status = reporting.get("status")
    paper_tier_ready = reporting.get("paper_tier_ready")

    policy_contract_status = (
        summary_payload.get("policy_contract_status")
        or summary_payload.get("contract_status")
        or reporting.get("policy_contract_status")
    )
    safety_contract_status = (
        summary_payload.get("safety_contract_status")
        or reporting.get("safety_contract_status")
    )

    failures: List[str] = []

    if report_status is None:
        failures.append("missing_reporting_contract.status")
    elif str(report_status) != "ok":
        failures.append(f"reporting_contract.status={report_status}")

    if paper_tier_ready is not True:
        failures.append(f"paper_tier_ready={paper_tier_ready}")

    if policy_contract_status is not None and str(policy_contract_status) != "ok":
        failures.append(f"policy_contract_status={policy_contract_status}")

    if safety_contract_status is not None and str(safety_contract_status) != "ok":
        failures.append(f"safety_contract_status={safety_contract_status}")

    return {
        "checked": True,
        "status": "ok" if not failures else "failed",
        "reason": "ok" if not failures else " | ".join(failures),
        "reporting_contract_status": report_status,
        "paper_tier_ready": paper_tier_ready,
        "policy_contract_status": policy_contract_status,
        "safety_contract_status": safety_contract_status,
    }


def _resolve_scientific_contract(
    *,
    contract_scope: str,
    summary_path: Path,
) -> Dict[str, Any]:
    if contract_scope != "full_pipeline":
        return {
            "checked": False,
            "status": "not_applicable",
            "reason": "scientific contract is enforced only for full_pipeline scope",
            "summary_path": str(summary_path),
        }

    if not summary_path.exists():
        return {
            "checked": True,
            "status": "failed",
            "reason": "missing_summary_json",
            "summary_path": str(summary_path),
        }

    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    verdict = _evaluate_summary_scientific_contract(payload)
    verdict["summary_path"] = str(summary_path)
    return verdict


def _run_intervention_gate(
    *,
    out_root: Path,
    repo_root: Path,
    probes_path: Path,
    eraser_target: Path,
    head_path: Path,
    head_meta_path: Optional[Path],
    families: str,
    intensities: str,
    device: str,
    batch_size: int,
    min_nonzero_logit_delta: float,
    min_nonzero_prob_delta: float,
    max_zero_logit_damage: float,
    max_zero_prob_damage: float,
) -> Dict[str, Any]:
    """
    Run bounded adjudication for the currently selected eraser artifact and
    evaluate whether downstream expensive stages should continue.
    """
    adjudication_dir = out_root / "adjudication"
    adjudication_dir.mkdir(parents=True, exist_ok=True)

    log_path = _stage_log_path(out_root, "intervention_gate")
    state_path = _stage_state_path(out_root, "intervention_gate")
    summary_json = adjudication_dir / "delta_summary.json"
    failed_json = adjudication_dir / "intervention_gate_failed.json"
    decision_json = adjudication_dir / "gate_decision_reason.json"

    cmd = [
        sys.executable,
        str(repo_root / "scripts" / "adjudicate_eraser_banks.py"),
        "--probes",
        str(probes_path),
        "--head",
        str(head_path),
        "--bank",
        f"candidate={eraser_target}",
        "--families",
        families,
        "--intensities",
        intensities,
        "--device",
        device,
        "--batch_size",
        str(batch_size),
        "--out_dir",
        str(adjudication_dir),
    ]
    if head_meta_path is not None:
        cmd.extend(["--head_meta", str(head_meta_path)])

    running_payload = {
        "stage": "intervention_gate",
        "status": "running",
        "timestamp_utc": _now_iso(),
        "command": cmd,
        "log_path": str(log_path),
        "summary_json": str(summary_json),
        "eraser_target": str(eraser_target),
    }
    _write_stage_state(out_root, "intervention_gate", running_payload)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as fh:
        proc = subprocess.run(
            cmd,
            cwd=str(repo_root),
            stdout=fh,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )

    if proc.returncode != 0:
        payload = {
            "stage": "intervention_gate",
            "status": "failed",
            "timestamp_utc": _now_iso(),
            "reason": f"adjudication_subprocess_returncode={proc.returncode}",
            "command": cmd,
            "log_path": str(log_path),
            "summary_json": str(summary_json),
            "eraser_target": str(eraser_target),
            "recommended_bank_metadata": None,
            "verdict": {},
        }
        _write_stage_state(out_root, "intervention_gate", payload)
        _write_intervention_gate_decision_payload(
            gate_payload=payload,
            decision_json_path=decision_json,
        )
        failed_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload

    if not summary_json.exists():
        payload = {
            "stage": "intervention_gate",
            "status": "failed",
            "timestamp_utc": _now_iso(),
            "reason": "missing_delta_summary_json",
            "command": cmd,
            "log_path": str(log_path),
            "summary_json": str(summary_json),
            "eraser_target": str(eraser_target),
            "recommended_bank_metadata": None,
            "verdict": {},
        }
        _write_stage_state(out_root, "intervention_gate", payload)
        _write_intervention_gate_decision_payload(
            gate_payload=payload,
            decision_json_path=decision_json,
        )
        failed_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload

    adjudication_payload = json.loads(summary_json.read_text(encoding="utf-8"))
    verdict = _evaluate_intervention_gate(
        adjudication_payload.get("summary_by_bank", []),
        min_nonzero_logit_delta=min_nonzero_logit_delta,
        min_nonzero_prob_delta=min_nonzero_prob_delta,
        max_zero_logit_damage=max_zero_logit_damage,
        max_zero_prob_damage=max_zero_prob_damage,
    )

    payload = {
        "stage": "intervention_gate",
        "status": "success" if verdict["passed"] else "failed",
        "timestamp_utc": _now_iso(),
        "reason": verdict["reason"],
        "command": cmd,
        "log_path": str(log_path),
        "summary_json": str(summary_json),
        "eraser_target": str(eraser_target),
        "recommended_bank_metadata": adjudication_payload.get("recommended_bank_metadata"),
        "verdict": verdict,
    }
    _write_stage_state(out_root, "intervention_gate", payload)
    _write_intervention_gate_decision_payload(
        gate_payload=payload,
        decision_json_path=decision_json,
    )

    if not verdict["passed"]:
        failed_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return payload


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Bounded public-mainline launcher for canonical public-MIMIC acquisition-context experiments."
    )
    ap.add_argument("--config", required=True)
    ap.add_argument("--out_dir", default=None, help="Override output.root for this bounded run.")
    ap.add_argument("--encoder_profile", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--bootstrap_samples", type=int, default=200)
    ap.add_argument("--max_pairs", type=int, default=300)
    ap.add_argument("--max_images", type=int, default=300)
    ap.add_argument("--probe_batch_size", type=int, default=16)
    ap.add_argument("--skip_realism_calibration", action="store_true")
    ap.add_argument("--resume", action="store_true", help="Resume safely from success-marked stages in the same out_dir.")
    ap.add_argument("--start_from", default=None, help=f"Start from one of: {', '.join(STAGE_ORDER)}")
    ap.add_argument("--stop_after", default=None, help=f"Stop after one of: {', '.join(STAGE_ORDER)}")
    ap.add_argument("--dataset_root", default=None, help="Override the first source root and, by default, image_root.")
    ap.add_argument("--image_root", default=None, help="Override the first source image_root.")
    ap.add_argument("--metadata_csv", default=None, help="Override the first source metadata_csv.")
    ap.add_argument("--extra_metadata_csv", default=None, help="Override the first source extra_metadata_csv.")
    ap.add_argument("--index_dir", default=None, help="Override data.index_dir.")
    ap.add_argument("--cache_dir", default=None, help="Override data.cache_dir.")
    ap.add_argument("--head_path", default=None, help="Override model.encoder.head_path.")
    ap.add_argument("--head_meta_path", default=None, help="Override model.encoder.head_meta_path.")
    ap.add_argument("--existing_probes", default=None, help="Reuse an existing probes artifact instead of running collect_probes.")
    ap.add_argument("--existing_eraser", default=None, help="Reuse an existing eraser or eraser bank manifest instead of fit_eraser.")
    ap.add_argument("--existing_realism", default=None, help="Reuse an existing realism calibrator instead of calibrate_physics_realism.")
    ap.add_argument(
        "--adjudicate_after_fit",
        action="store_true",
        help="Run bounded classifier-relevance adjudication before expensive downstream erasure/safety stages.",
    )
    ap.add_argument(
        "--adjudication_probes",
        default=None,
        help="Optional probes artifact path for intervention gate. Defaults to the launcher probes path.",
    )
    ap.add_argument(
        "--adjudication_head",
        default=None,
        help="Required when --adjudicate_after_fit is enabled unless the effective config resolves a head_path.",
    )
    ap.add_argument(
        "--adjudication_head_meta",
        default=None,
        help="Optional head_meta.json path for intervention gate.",
    )
    ap.add_argument(
        "--adjudication_families",
        default="heel,scatter",
        help="Comma-separated families for intervention adjudication.",
    )
    ap.add_argument(
        "--adjudication_intensities",
        default="0.0,0.5,1.0",
        help="Comma-separated intensities for intervention adjudication.",
    )
    ap.add_argument(
        "--adjudication_batch_size",
        type=int,
        default=8192,
        help="Batch size for adjudication script.",
    )
    ap.add_argument(
        "--adjudication_min_nonzero_logit_delta",
        type=float,
        default=1e-5,
        help="Minimum mean nonzero-shift logit delta required to pass intervention gate.",
    )
    ap.add_argument(
        "--adjudication_min_nonzero_prob_delta",
        type=float,
        default=1e-6,
        help="Minimum mean nonzero-shift probability delta required to pass intervention gate.",
    )
    ap.add_argument(
        "--adjudication_max_zero_logit_damage",
        type=float,
        default=5e-3,
        help="Maximum allowed mean zero-shift logit damage.",
    )
    ap.add_argument(
        "--adjudication_max_zero_prob_damage",
        type=float,
        default=5e-4,
        help="Maximum allowed mean zero-shift probability damage.",
    )
    ap.add_argument("--dry_run", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    selected_stages = _stage_range(args.start_from, args.stop_after)
    src_cfg_path = Path(args.config).expanduser()
    cfg = load_config(str(src_cfg_path))

    out_root = Path(args.out_dir).expanduser() if args.out_dir else Path(cfg.output.root).expanduser()
    out_root.mkdir(parents=True, exist_ok=True)

    default_probes_path = out_root / "probes" / "probes.pt"
    default_eraser_dir = out_root / "eraser_bank"
    default_eraser_manifest_path = out_root / "eraser_bank_manifest.json"
    default_eraser_single_path = out_root / "eraser.pt"
    default_eraser_single_meta = out_root / "eraser_meta.json"
    default_realism_dir = out_root / "realism"
    default_realism_path = default_realism_dir / "realism_calibrator.json"
    confounding_dir = out_root / "confounding"
    erasure_dir = out_root / "erasure"
    safety_dir = out_root / "safety"
    summary_dir = out_root / "summary"
    effective_cfg_path = out_root / "public_mainline.effective.yaml"
    preflight_path = out_root / "public_mainline_preflight.json"
    plan_path = out_root / "public_mainline_plan.json"
    manifest_path = out_root / "public_mainline_manifest.json"

    fit_mode = str(getattr(cfg.eraser, "fit_mode", "one_eraser_per_shift_family"))
    probes_path = Path(args.existing_probes).expanduser() if args.existing_probes else default_probes_path
    realism_path = Path(args.existing_realism).expanduser() if args.existing_realism else default_realism_path
    if args.existing_eraser:
        eraser_target = Path(args.existing_eraser).expanduser()
    elif fit_mode == "one_eraser_per_shift_family":
        eraser_target = default_eraser_manifest_path
    else:
        eraser_target = default_eraser_single_path

    cfg_effective = load_config(str(src_cfg_path))
    _apply_path_overrides(cfg_effective, args, out_root, realism_path)
    save_config(str(effective_cfg_path), cfg_effective)

    preflight_errors: List[str] = []
    preflight_notes: List[str] = []
    path_overrides = {
        "out_dir": str(out_root),
        "dataset_root": args.dataset_root,
        "image_root": args.image_root,
        "metadata_csv": args.metadata_csv,
        "extra_metadata_csv": args.extra_metadata_csv,
        "index_dir": args.index_dir,
        "cache_dir": args.cache_dir,
        "head_path": args.head_path,
        "head_meta_path": args.head_meta_path,
        "existing_probes": args.existing_probes,
        "existing_eraser": args.existing_eraser,
        "existing_realism": args.existing_realism,
    }

    source_path_status = _source_path_checks(cfg_effective)
    missing_source_paths = [row for row in source_path_status if not row["exists"]]
    if missing_source_paths:
        preflight_errors.extend(
            [f"Missing source path: {row['source']}:{row['field']} -> {row['path']}" for row in missing_source_paths]
        )

    scenario_mode = str(getattr(cfg_effective.erasure_eval, "scenario_mode", "atomic_calibrated_eval"))
    if scenario_mode != "atomic_calibrated_eval":
        preflight_errors.append(
            f"public-mainline expects erasure_eval.scenario_mode='atomic_calibrated_eval', got '{scenario_mode}'."
        )
    if not bool(cfg_effective.intervention.use_realism):
        preflight_errors.append("public-mainline expects intervention.use_realism=true.")
    if any(str(st) == "mixed" for st in cfg_effective.erasure_eval.shift_types):
        preflight_errors.append("public-mainline does not allow shift_type='mixed' in the canonical bounded launcher.")

    pair_status = _pair_metadata_status(
        cfg_effective,
        pair_mode=str(getattr(cfg_effective.intervention, "pair_mode", "")),
        require_resolution=bool(cfg_effective.intervention.use_realism),
    )
    unsupported_pair_sources = [
        s for s in pair_status if cfg_effective.intervention.use_realism and not bool(s.get("supports_requested_pair_mode"))
    ]
    if unsupported_pair_sources:
        gap_msg = _format_pair_metadata_gaps(unsupported_pair_sources)
        preflight_errors.append(
            f"Capability-aware pair-mode support failed for pair_mode='{cfg_effective.intervention.pair_mode}': {gap_msg}"
        )
    else:
        preflight_notes.append(
            f"Capability-aware pair-mode preflight passed for pair_mode='{cfg_effective.intervention.pair_mode}'."
        )

    split_counts: Dict[str, int] = {}
    source_capabilities: Dict[str, Dict[str, Any]] = {}
    index_materialization = {"status": "skipped"}
    if not missing_source_paths:
        try:
            data_engine = DataEngine(cfg_effective.data).attach_runtime_contract(cfg_effective)
            data_engine.build_index()
            data_engine.create_splits()
            split_counts = {
                str(split): int(len(data_engine.get_split_df(split)))
                for split in [
                    cfg_effective.data.head_split,
                    cfg_effective.data.eraser_split,
                    cfg_effective.data.calib_split,
                    cfg_effective.data.eval_split,
                ]
            }
            source_capabilities = {
                str(name): snap.as_dict() for name, snap in (data_engine.source_capabilities or {}).items()
            }
            index_materialization = {
                "status": "ok",
                "index_dir": str(cfg_effective.data.index_dir),
                "split_counts": split_counts,
            }
        except Exception as e:
            preflight_errors.append(f"Index/split materialization failed: {e}")
            index_materialization = {"status": "error", "message": str(e)}

    enc_cfg = EncoderFactory.resolve_encoder_config(cfg_effective, profile=args.encoder_profile)
    expected_fp = build_encoder_fingerprint(enc_cfg)
    expected_fp["preprocess_signature"] = preprocess_signature(cfg_effective)
    expected_labels = _expected_label_cols(cfg_effective)
    expected_shift_families = [str(s) for s in cfg_effective.erasure_eval.shift_types if str(s) != "mixed"]
    head_status: Dict[str, Any] = {
        "head_path": str(enc_cfg.head_path) if enc_cfg.head_path is not None else None,
        "head_meta_path": str(enc_cfg.head_meta_path) if getattr(enc_cfg, "head_meta_path", None) is not None else None,
    }
    try:
        head_contract = resolve_head_contract(
            cfg_effective,
            label_cols=expected_labels if expected_labels else None,
            encoder_profile=args.encoder_profile,
            require_head_path=True,
            strict_meta=True,
            load_artifact=False,
        )
        head_status["head_path"] = str(head_contract.head_path) if head_contract.head_path is not None else None
        head_status["head_meta_path"] = (
            str(head_contract.enc_cfg.head_meta_path)
            if getattr(head_contract.enc_cfg, "head_meta_path", None) is not None
            else None
        )
        head_status["validation_errors"] = []
        preflight_notes.append("Head metadata validation passed.")
    except Exception as e:
        preflight_errors.append(str(e))

    if args.skip_realism_calibration and args.existing_realism is None:
        preflight_notes.append("skip_realism_calibration requested: launcher will reuse realism target from output root if present.")
    if args.existing_probes is not None:
        preflight_notes.append(f"Reusing probes artifact: {probes_path}")
    if args.existing_eraser is not None:
        preflight_notes.append(f"Reusing eraser artifact: {eraser_target}")
    if args.existing_realism is not None:
        preflight_notes.append(f"Reusing realism calibrator: {realism_path}")

    registered_shifts = sorted(
        [str(name) for name, shift in ShiftManager(seed=cfg_effective.shifts.seed).shifts.items() if shift is not None]
    )
    ready_not_canonical = [
        shift
        for shift in ["collimation_crop"]
        if shift in registered_shifts and shift not in set(cfg_effective.erasure_eval.shift_types)
    ]

    common_tail: list[str] = []
    if args.encoder_profile:
        common_tail += ["--encoder_profile", args.encoder_profile]
    if args.seed is not None:
        common_tail += ["--seed", str(args.seed)]
    if args.device:
        common_tail += ["--device", args.device]

    fit_tail: list[str] = []
    if args.encoder_profile:
        fit_tail += ["--encoder_profile", args.encoder_profile]
    if args.seed is not None:
        fit_tail += ["--seed", str(args.seed)]

    cmd_collect = [
        sys.executable,
        "scripts/collect_probes.py",
        "--config",
        str(effective_cfg_path),
        "--batch_size",
        str(args.probe_batch_size),
        "--max_images",
        str(args.max_images),
        "--out",
        str(probes_path),
        *common_tail,
    ]
    if fit_mode == "one_eraser_per_shift_family":
        cmd_fit = [
            sys.executable,
            "scripts/fit_eraser.py",
            "--config",
            str(effective_cfg_path),
            "--probes",
            str(probes_path),
            "--shift_type",
            "all",
            "--out",
            str(default_eraser_dir),
            "--meta",
            str(default_eraser_manifest_path),
            *fit_tail,
        ]
    else:
        cmd_fit = [
            sys.executable,
            "scripts/fit_eraser.py",
            "--config",
            str(effective_cfg_path),
            "--probes",
            str(probes_path),
            "--shift_type",
            "all",
            "--out",
            str(default_eraser_single_path),
            "--meta",
            str(default_eraser_single_meta),
            *fit_tail,
        ]
    cmd_realism = [
        sys.executable,
        "scripts/calibrate_physics_realism.py",
        "--config",
        str(effective_cfg_path),
        "--out",
        str(realism_path),
        "--max_pairs",
        str(args.max_pairs),
        "--max_images",
        str(args.max_images),
        *common_tail,
    ]
    cmd_confounding = [
        sys.executable,
        "scripts/run_confounding_audit.py",
        "--config",
        str(effective_cfg_path),
        "--out_dir",
        str(confounding_dir),
        "--bootstrap_samples",
        str(args.bootstrap_samples),
        *common_tail,
    ]
    cmd_erasure = [
        sys.executable,
        "scripts/run_erasure_eval.py",
        "--config",
        str(effective_cfg_path),
        "--eraser",
        str(eraser_target),
        "--out_dir",
        str(erasure_dir),
        "--bootstrap_samples",
        str(args.bootstrap_samples),
        *common_tail,
    ]
    cmd_safety = [
        sys.executable,
        "scripts/run_safety_policy_eval.py",
        "--config",
        str(effective_cfg_path),
        "--eraser",
        str(eraser_target),
        "--out_dir",
        str(safety_dir),
        "--save_shift_id",
        str(safety_dir / "shift_id.pkl"),
        "--save_policy",
        str(safety_dir / "policy.json"),
        "--bootstrap_samples",
        str(args.bootstrap_samples),
        *common_tail,
    ]
    cmd_summary = [
        sys.executable,
        "scripts/build_public_mainline_summary.py",
        "--config",
        str(effective_cfg_path),
        "--out_dir",
        str(summary_dir),
        "--realism_dir",
        str(realism_path.parent),
        "--confounding_dir",
        str(confounding_dir),
        "--erasure_dir",
        str(erasure_dir),
        "--safety_dir",
        str(safety_dir),
        "--launcher_root",
        str(out_root),
    ]

    stage_rows: List[Dict[str, Any]] = [
        {
            "stage": "realism_calibration",
            "mode": "reuse" if (args.skip_realism_calibration or args.existing_realism is not None) else "run",
            "command": None if (args.skip_realism_calibration or args.existing_realism is not None) else cmd_realism,
            "required_inputs": [str(realism_path)] if (args.skip_realism_calibration or args.existing_realism is not None) else [str(cfg_effective.data.index_dir)],
            "expected_outputs": [
                str(realism_path),
                str(realism_path.with_suffix(".report.json")),
                str(realism_path.parent / "pair_quality_audit.json"),
            ],
            "validator": "realism",
        },
        {
            "stage": "collect_probes",
            "mode": "reuse" if args.existing_probes is not None else "run",
            "command": None if args.existing_probes is not None else cmd_collect,
            "required_inputs": [str(enc_cfg.head_path) if enc_cfg.head_path is not None else None],
            "expected_outputs": [str(probes_path), str(probes_path.parent / "artifact_manifest.json")],
            "validator": "probes",
        },
        {
            "stage": "fit_eraser",
            "mode": "reuse" if args.existing_eraser is not None else "run",
            "command": None if args.existing_eraser is not None else cmd_fit,
            "required_inputs": [str(probes_path)],
            "expected_outputs": [str(eraser_target)],
            "validator": "eraser",
        },
        {
            "stage": "confounding_audit",
            "mode": "run",
            "command": cmd_confounding,
            "required_inputs": [str(enc_cfg.head_path) if enc_cfg.head_path is not None else None],
            "expected_outputs": [
                str(confounding_dir / "confounding_summary.json"),
                str(confounding_dir / "confounding_predictability.csv"),
            ],
            "validator": None,
        },
        {
            "stage": "erasure_eval",
            "mode": "run",
            "command": cmd_erasure,
            "required_inputs": [str(eraser_target), str(realism_path)],
            "expected_outputs": [
                str(erasure_dir / "metrics.csv"),
                str(erasure_dir / "risk_coverage_curves.csv"),
                str(erasure_dir / "guarded_erasure_diagnostics.csv"),
            ],
            "validator": None,
        },
        {
            "stage": "safety_eval",
            "mode": "run",
            "command": cmd_safety,
            "required_inputs": [str(eraser_target), str(realism_path)],
            "expected_outputs": [
                str(safety_dir / "safety.csv"),
                str(safety_dir / "scenario_trace.csv"),
                str(safety_dir / "policy_bootstrap.csv"),
            ],
            "validator": None,
        },
        {
            "stage": "summary",
            "mode": "run",
            "command": cmd_summary,
            "required_inputs": [
                str(realism_path.with_suffix(".report.json")),
                str(confounding_dir / "confounding_summary.json"),
                str(erasure_dir / "metrics.csv"),
                str(safety_dir / "safety.csv"),
            ],
            "expected_outputs": [str(summary_dir / "public_mainline_summary.json")],
            "validator": None,
        },
    ]

    stage_map = {row["stage"]: row for row in stage_rows}
    selected_rows = [stage_map[name] for name in selected_stages]

    produced_by_selected: set[str] = set()
    for row in selected_rows:
        for raw in row["required_inputs"]:
            if raw is None:
                continue
            path = str(Path(raw).expanduser())
            if Path(path).exists():
                continue
            if path in produced_by_selected:
                continue
            preflight_errors.append(
                f"Selected stage '{row['stage']}' requires missing input artifact: {path}. "
                "Either run earlier stages, reuse an existing artifact path, or change start_from."
            )
        for raw in row["expected_outputs"]:
            produced_by_selected.add(str(Path(raw).expanduser()))

    stage_actions: Dict[str, str] = {}
    for row in selected_rows:
        stage = row["stage"]
        state = _load_stage_state(out_root, stage)
        if row["mode"] == "reuse":
            stage_actions[stage] = "reuse"
        elif args.resume and state and state.get("status") == "success" and not _verify_stage_outputs(row["expected_outputs"]):
            stage_actions[stage] = "resume_skip"
            preflight_notes.append(f"resume: stage '{stage}' has a valid success marker and will be skipped.")
        else:
            stage_actions[stage] = "run"

    probe_validation = None
    need_probe_validation = (
        stage_actions.get("collect_probes") in {"reuse", "resume_skip"}
        or ("collect_probes" not in selected_stages and any(name in selected_stages for name in ["fit_eraser", "confounding_audit", "erasure_eval", "safety_eval", "summary"]))
    )
    if probes_path.exists() and need_probe_validation:
        probe_validation = _build_probe_validation(probes_path, cfg_effective)
        preflight_errors.extend(probe_validation["errors"])

    eraser_validation = None
    need_eraser_validation = (
        stage_actions.get("fit_eraser") in {"reuse", "resume_skip"}
        or ("fit_eraser" not in selected_stages and any(name in selected_stages for name in ["erasure_eval", "safety_eval", "summary"]))
    )
    if Path(eraser_target).exists() and need_eraser_validation:
        eraser_validation = _build_eraser_validation(
            Path(eraser_target),
            expected_fingerprint=expected_fp,
            expected_shift_families=expected_shift_families,
        )
        preflight_errors.extend(eraser_validation["errors"])

    realism_validation = None
    need_realism_validation = (
        stage_actions.get("realism_calibration") in {"reuse", "resume_skip"}
        or ("realism_calibration" not in selected_stages and any(name in selected_stages for name in ["erasure_eval", "safety_eval", "summary"]))
    )
    if Path(realism_path).exists() and need_realism_validation:
        realism_validation = _build_realism_validation(
            Path(realism_path),
            expected_fingerprint=expected_fp,
            expected_shift_families=expected_shift_families,
            pair_mode=str(cfg_effective.intervention.pair_mode),
        )
        preflight_errors.extend(realism_validation["errors"])

    if args.adjudicate_after_fit:
        adjudication_probes = Path(args.adjudication_probes).expanduser() if args.adjudication_probes else probes_path
        adjudication_head = Path(args.adjudication_head).expanduser() if args.adjudication_head else None
        adjudication_head_meta = Path(args.adjudication_head_meta).expanduser() if args.adjudication_head_meta else None

        if adjudication_head is None:
            preflight_errors.append(
                "intervention gate requires --adjudication_head to be provided explicitly."
            )

        if not args.dry_run:
            if not adjudication_probes.exists():
                preflight_errors.append(
                    f"intervention gate probes artifact is missing: {adjudication_probes}"
                )
            if adjudication_head is not None and not adjudication_head.exists():
                preflight_errors.append(
                    f"intervention gate head artifact is missing: {adjudication_head}"
                )
            if adjudication_head_meta is not None and not adjudication_head_meta.exists():
                preflight_errors.append(
                    f"intervention gate head meta artifact is missing: {adjudication_head_meta}"
                )

    logs_dir = out_root / "launcher_logs"
    stage_state_dir = _stage_state_dir(out_root)
    preflight_payload = {
        "config": str(src_cfg_path),
        "effective_config": str(effective_cfg_path),
        "canonical_contract": "docs/canonical_public_mimic_experiment_contract.md",
        "errors": preflight_errors,
        "notes": preflight_notes,
        "selected_stages": selected_stages,
        "resume": bool(args.resume),
        "path_overrides": path_overrides,
        "source_path_status": source_path_status,
        "pair_metadata_status": pair_status,
        "source_capabilities": source_capabilities,
        "index_materialization": index_materialization,
        "head_status": head_status,
        "probe_validation": probe_validation,
        "eraser_validation": eraser_validation,
        "realism_validation": realism_validation,
        "canonical_shift_families": list(cfg_effective.erasure_eval.shift_types),
        "ready_but_not_in_canonical_calibrated_set": ready_not_canonical,
        "probes_target": str(probes_path),
        "eraser_target": str(eraser_target),
        "realism_target": str(realism_path),
        "logs_dir": str(logs_dir),
        "stage_state_dir": str(stage_state_dir),
    }
    _write_json(preflight_path, preflight_payload)

    key_code_files = [
        "phystta/config/schema.py",
        "phystta/config/loader.py",
        "phystta/data/engine.py",
        "phystta/intervention/eraser.py",
        "phystta/causal/confounding.py",
        "scripts/run_public_mainline.py",
        "scripts/collect_probes.py",
        "scripts/fit_eraser.py",
        "scripts/calibrate_physics_realism.py",
        "scripts/run_confounding_audit.py",
        "scripts/run_erasure_eval.py",
        "scripts/run_safety_policy_eval.py",
        "scripts/build_public_mainline_summary.py",
    ]
    per_file_hashes = {rel: _sha256_if_exists(Path(rel)) for rel in key_code_files}
    planned_rows: List[Dict[str, Any]] = []
    commands: List[List[str]] = []
    for row in stage_rows:
        row_copy = dict(row)
        row_copy["log_path"] = str(_stage_log_path(out_root, row["stage"]))
        row_copy["state_path"] = str(_stage_state_path(out_root, row["stage"]))
        row_copy["selected"] = row["stage"] in selected_stages
        row_copy["planned_action"] = stage_actions.get(row["stage"], "not_selected")
        planned_rows.append(row_copy)
        if row_copy["selected"] and row_copy["planned_action"] == "run" and row_copy["command"] is not None:
            commands.append(list(row_copy["command"]))

    plan = {
        "timestamp_utc": _now_iso(),
        "source_config": str(src_cfg_path),
        "effective_config": str(effective_cfg_path),
        "canonical_contract": "docs/canonical_public_mimic_experiment_contract.md",
        "out_dir": str(out_root),
        "runtime_env": {
            "python": sys.version,
            "platform": platform.platform(),
            "cwd": os.getcwd(),
        },
        "selected_stages": selected_stages,
        "resume": bool(args.resume),
        "start_from": args.start_from,
        "stop_after": args.stop_after,
        "preflight": preflight_payload,
        "stages": planned_rows,
        "commands": commands,
        "artifact_paths": {
            "probes": str(probes_path),
            "eraser": str(eraser_target),
            "realism": str(realism_path),
            "confounding": str(confounding_dir),
            "erasure": str(erasure_dir),
            "safety": str(safety_dir),
            "summary": str(summary_dir),
            "adjudication": str(out_root / "adjudication"),
            "intervention_gate_decision": str(out_root / "adjudication" / "gate_decision_reason.json"),
        },
        "intervention_gate": {
            "enabled": bool(args.adjudicate_after_fit),
            "probes": str(Path(args.adjudication_probes).expanduser()) if args.adjudication_probes else str(probes_path),
            "head": str(Path(args.adjudication_head).expanduser()) if args.adjudication_head else None,
            "head_meta": str(Path(args.adjudication_head_meta).expanduser()) if args.adjudication_head_meta else None,
            "decision_json": str(out_root / "adjudication" / "gate_decision_reason.json"),
            "families": args.adjudication_families,
            "intensities": args.adjudication_intensities,
            "batch_size": int(args.adjudication_batch_size),
            "thresholds": {
                "min_nonzero_logit_delta": float(args.adjudication_min_nonzero_logit_delta),
                "min_nonzero_prob_delta": float(args.adjudication_min_nonzero_prob_delta),
                "max_zero_logit_damage": float(args.adjudication_max_zero_logit_damage),
                "max_zero_prob_damage": float(args.adjudication_max_zero_prob_damage),
            },
        },
        "logs_dir": str(logs_dir),
        "stage_state_dir": str(stage_state_dir),
        "code_sha256_files": per_file_hashes,
        "code_sha256": _aggregate_hash_map(per_file_hashes),
    }
    _write_json(plan_path, plan)

    if args.dry_run:
        for row in planned_rows:
            if not row["selected"]:
                continue
            if row["planned_action"] == "run":
                print("RUN:", " ".join(row["command"]))
            elif row["planned_action"] == "reuse":
                print(f"REUSE: {row['stage']} -> {row['expected_outputs'][0]}")
            elif row["planned_action"] == "resume_skip":
                print(f"SKIP(resume): {row['stage']} -> {row['state_path']}")
        print(f"Saved dry-run bounded plan: {plan_path}")
        return

    if preflight_errors:
        _write_json(manifest_path, plan)
        raise RuntimeError(
            "public-mainline preflight failed. "
            f"See {preflight_path} for details. First errors: {' | '.join(preflight_errors[:5])}"
        )

    stage_results: List[Dict[str, Any]] = []
    intervention_gate_done = False
    for row in planned_rows:
        if not row["selected"]:
            continue
        stage = row["stage"]
        log_path = Path(row["log_path"])
        state_path = Path(row["state_path"])
        required_inputs = list(row["required_inputs"])
        expected_outputs = list(row["expected_outputs"])
        validator_name = row.get("validator")
        action = row["planned_action"]

        if (
            args.adjudicate_after_fit
            and not intervention_gate_done
            and stage in {"erasure_eval", "safety_eval", "summary"}
        ):
            gate_payload = _run_intervention_gate(
                out_root=out_root,
                repo_root=Path(__file__).resolve().parents[1],
                probes_path=Path(args.adjudication_probes).expanduser() if args.adjudication_probes else probes_path,
                eraser_target=Path(eraser_target),
                head_path=Path(args.adjudication_head).expanduser(),
                head_meta_path=Path(args.adjudication_head_meta).expanduser() if args.adjudication_head_meta else None,
                families=args.adjudication_families,
                intensities=args.adjudication_intensities,
                device=args.device,
                batch_size=int(args.adjudication_batch_size),
                min_nonzero_logit_delta=float(args.adjudication_min_nonzero_logit_delta),
                min_nonzero_prob_delta=float(args.adjudication_min_nonzero_prob_delta),
                max_zero_logit_damage=float(args.adjudication_max_zero_logit_damage),
                max_zero_prob_damage=float(args.adjudication_max_zero_prob_damage),
            )
            stage_results.append(gate_payload)
            intervention_gate_done = True

            if gate_payload.get("status") != "success":
                _write_json(manifest_path, {**plan, "stage_results": stage_results})
                raise RuntimeError(
                    "intervention gate failed before downstream expensive stages. "
                    f"See {gate_payload.get('log_path')} and {out_root / 'adjudication' / 'intervention_gate_failed.json'}."
                )

        if action == "resume_skip":
            stage_results.append(
                {
                    "stage": stage,
                    "status": "resume_skipped",
                    "state_path": str(state_path),
                    "log_path": str(log_path),
                }
            )
            continue

        if action == "reuse":
            missing_inputs = _verify_stage_inputs(required_inputs if required_inputs else expected_outputs)
            if missing_inputs:
                payload = {
                    "stage": stage,
                    "status": "failed",
                    "mode": "reuse",
                    "timestamp_utc": _now_iso(),
                    "missing_inputs": missing_inputs,
                    "expected_outputs": expected_outputs,
                    "log_path": str(log_path),
                }
                _write_stage_state(out_root, stage, payload)
                _write_json(manifest_path, {**plan, "stage_results": stage_results + [payload]})
                raise RuntimeError(
                    f"Stage '{stage}' was configured for reuse but required artifact(s) are missing: {missing_inputs}. "
                    f"See {state_path}."
                )
            missing_outputs = _verify_stage_outputs(expected_outputs)
            if missing_outputs:
                payload = {
                    "stage": stage,
                    "status": "failed",
                    "mode": "reuse",
                    "timestamp_utc": _now_iso(),
                    "missing_outputs": missing_outputs,
                    "expected_outputs": expected_outputs,
                    "log_path": str(log_path),
                }
                _write_stage_state(out_root, stage, payload)
                _write_json(manifest_path, {**plan, "stage_results": stage_results + [payload]})
                raise RuntimeError(
                    f"Stage '{stage}' reuse target is incomplete. Missing outputs: {missing_outputs}. See {state_path}."
                )
            validator_errors: List[str] = []
            validator_payload: Dict[str, Any] | None = None
            if validator_name == "probes":
                validator_payload = _build_probe_validation(Path(expected_outputs[0]), cfg_effective)
                validator_errors = list(validator_payload["errors"])
            elif validator_name == "eraser":
                validator_payload = _build_eraser_validation(Path(expected_outputs[0]), expected_fp, expected_shift_families)
                validator_errors = list(validator_payload["errors"])
            elif validator_name == "realism":
                validator_payload = _build_realism_validation(
                    Path(expected_outputs[0]),
                    expected_fp,
                    expected_shift_families,
                    str(cfg_effective.intervention.pair_mode),
                )
                validator_errors = list(validator_payload["errors"])
            if validator_errors:
                payload = {
                    "stage": stage,
                    "status": "failed",
                    "mode": "reuse",
                    "timestamp_utc": _now_iso(),
                    "validation_errors": validator_errors,
                    "validation": validator_payload,
                    "log_path": str(log_path),
                }
                _write_stage_state(out_root, stage, payload)
                _write_json(manifest_path, {**plan, "stage_results": stage_results + [payload]})
                raise RuntimeError(
                    f"Stage '{stage}' reuse artifact failed validation: {' | '.join(validator_errors)}. See {state_path}."
                )
            payload = {
                "stage": stage,
                "status": "success",
                "mode": "reuse",
                "timestamp_utc": _now_iso(),
                "expected_outputs": expected_outputs,
                "output_hashes": _output_hashes(expected_outputs),
                "validation": validator_payload,
                "log_path": str(log_path),
            }
            _write_stage_state(out_root, stage, payload)
            stage_results.append(payload)
            continue

        missing_inputs = _verify_stage_inputs(required_inputs)
        if missing_inputs:
            payload = {
                "stage": stage,
                "status": "failed",
                "mode": "run",
                "timestamp_utc": _now_iso(),
                "missing_inputs": missing_inputs,
                "required_inputs": required_inputs,
                "expected_outputs": expected_outputs,
                "log_path": str(log_path),
            }
            _write_stage_state(out_root, stage, payload)
            _write_json(manifest_path, {**plan, "stage_results": stage_results + [payload]})
            raise RuntimeError(
                f"Stage '{stage}' cannot start because required input artifact(s) are missing: {missing_inputs}. "
                f"See {state_path}."
            )

        _write_stage_state(
            out_root,
            stage,
            {
                "stage": stage,
                "status": "running",
                "mode": "run",
                "timestamp_utc": _now_iso(),
                "command": row["command"],
                "required_inputs": required_inputs,
                "expected_outputs": expected_outputs,
                "log_path": str(log_path),
            },
        )
        print(f"[stage:{stage}] running -> {log_path}")
        try:
            _run_subprocess(list(row["command"]), log_path)
        except Exception as e:
            payload = {
                "stage": stage,
                "status": "failed",
                "mode": "run",
                "timestamp_utc": _now_iso(),
                "command": row["command"],
                "error": str(e),
                "log_path": str(log_path),
                "expected_outputs": expected_outputs,
            }
            _write_stage_state(out_root, stage, payload)
            _write_json(manifest_path, {**plan, "stage_results": stage_results + [payload]})
            raise RuntimeError(
                f"Stage '{stage}' failed. See launcher log {log_path} and state marker {state_path}."
            ) from e

        missing_outputs = _verify_stage_outputs(expected_outputs)
        validator_errors: List[str] = []
        validator_payload: Dict[str, Any] | None = None
        if not missing_outputs and validator_name == "probes":
            validator_payload = _build_probe_validation(Path(expected_outputs[0]), cfg_effective)
            validator_errors = list(validator_payload["errors"])
        elif not missing_outputs and validator_name == "eraser":
            validator_payload = _build_eraser_validation(Path(expected_outputs[0]), expected_fp, expected_shift_families)
            validator_errors = list(validator_payload["errors"])
        elif not missing_outputs and validator_name == "realism":
            validator_payload = _build_realism_validation(
                Path(expected_outputs[0]),
                expected_fp,
                expected_shift_families,
                str(cfg_effective.intervention.pair_mode),
            )
            validator_errors = list(validator_payload["errors"])

        if missing_outputs or validator_errors:
            payload = {
                "stage": stage,
                "status": "failed",
                "mode": "run",
                "timestamp_utc": _now_iso(),
                "command": row["command"],
                "missing_outputs": missing_outputs,
                "validation_errors": validator_errors,
                "validation": validator_payload,
                "log_path": str(log_path),
            }
            _write_stage_state(out_root, stage, payload)
            _write_json(manifest_path, {**plan, "stage_results": stage_results + [payload]})
            msg = []
            if missing_outputs:
                msg.append(f"missing outputs={missing_outputs}")
            if validator_errors:
                msg.append(f"validation errors={' | '.join(validator_errors)}")
            raise RuntimeError(
                f"Stage '{stage}' completed but failed output verification ({'; '.join(msg)}). "
                f"See {log_path} and {state_path}."
            )

        payload = {
            "stage": stage,
            "status": "success",
            "mode": "run",
            "timestamp_utc": _now_iso(),
            "command": row["command"],
            "required_inputs": required_inputs,
            "expected_outputs": expected_outputs,
            "output_hashes": _output_hashes(expected_outputs),
            "validation": validator_payload,
            "log_path": str(log_path),
        }
        _write_stage_state(out_root, stage, payload)
        stage_results.append(payload)

    required = [
        Path(probes_path),
        Path(eraser_target),
        confounding_dir / "confounding_summary.json",
        erasure_dir / "metrics.csv",
        safety_dir / "safety.csv",
        summary_dir / "public_mainline_summary.json",
    ]
    if cfg_effective.intervention.use_realism:
        required.extend(
            [
                Path(realism_path),
                Path(realism_path).with_suffix(".report.json"),
                Path(realism_path).parent / "pair_quality_audit.json",
            ]
        )

    artifact_contract = _compute_artifact_contract(
        required_artifacts=[str(path) for path in required],
        stage_results=stage_results,
    )

    payload = {
        **plan,
        "stage_results": stage_results,
        "contract_scope": (
            "full_pipeline"
            if set(selected_stages) == set(STAGE_ORDER)
            else "selected_stages"
        ),
        "required_artifacts": artifact_contract["full_required_artifacts"],
        "selected_required_artifacts": artifact_contract["selected_required_artifacts"],
        "missing_required_artifacts": artifact_contract["missing_selected_artifacts"],
        "missing_full_pipeline_artifacts": artifact_contract["missing_full_artifacts"],
        "completion_timestamp_utc": _now_iso(),
    }

    if (
        artifact_contract["missing_full_artifacts"]
        and not artifact_contract["missing_selected_artifacts"]
        and set(selected_stages) != set(STAGE_ORDER)
    ):
        payload.setdefault("notes", [])
        payload["notes"].append(
            "Selected-stage contract satisfied; full-pipeline artifacts remain absent because the run was stage-limited."
        )

    summary_path = summary_dir / "public_mainline_summary.json"
    scientific_contract = _resolve_scientific_contract(
        contract_scope=payload["contract_scope"],
        summary_path=summary_path,
    )
    payload["scientific_contract"] = scientific_contract

    _write_json(manifest_path, payload)

    if artifact_contract["missing_selected_artifacts"]:
        raise RuntimeError(
            "public-mainline finished with missing artifacts for the selected stage scope: "
            + ", ".join(artifact_contract["missing_selected_artifacts"])
            + f". See {manifest_path}."
        )

    if payload["contract_scope"] == "full_pipeline" and scientific_contract["status"] != "ok":
        raise RuntimeError(
            "public-mainline finished with a scientifically incomplete full-pipeline summary contract: "
            + scientific_contract["reason"]
            + f". See {manifest_path}."
        )

    print(f"Saved bounded public-mainline manifest: {manifest_path}")


if __name__ == "__main__":
    main()
