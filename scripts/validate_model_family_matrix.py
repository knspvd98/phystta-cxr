#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import yaml


SUPPORTED_ENCODER_SOURCES = [
    "baseline_cnn",
    "hf_clip_or_vit",
    "openclip",
    "eva_x",
    "chexfound",
    "llava_rad_encoder",
]
FOUNDATION_BUCKETS = {"biomedical_contrastive", "cxr_foundation"}


def _load_yaml(path: Path) -> Dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise RuntimeError(f"Expected YAML object at {path}, got {type(data).__name__}.")
    return data


def _load_json(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise RuntimeError(f"Expected JSON object at {path}, got {type(data).__name__}.")
    return data


def _as_bool(value: Any) -> bool:
    return bool(value)


def _nonempty_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _available_profiles(base_config: Dict[str, Any]) -> List[str]:
    profiles = base_config.get("model", {}).get("encoder_profiles", [])
    if not isinstance(profiles, list):
        return []
    out: List[str] = []
    for item in profiles:
        if isinstance(item, dict) and isinstance(item.get("name"), str):
            out.append(item["name"])
    return out


def _markdown_report(payload: Dict[str, Any], runplan: Dict[str, Any]) -> str:
    wave_summary = payload["wave_summary"]
    lines = [
        "# Model Family Validation",
        "",
        f"- status: {payload['status']}",
        f"- reference_family: {payload['reference_family']}",
        f"- head_reuse_guard: {payload['head_reuse_guard']}",
        f"- available_profiles: {', '.join(payload['available_profiles']) if payload['available_profiles'] else 'none'}",
        "",
        "## Wave Layout",
        f"- wave1_classifier_families: {', '.join(wave_summary['wave1_classifier_families']) or 'none'}",
        f"- wave2_blocked_foundation_families: {', '.join(wave_summary['wave2_blocked_foundation_families']) or 'none'}",
        f"- wave3_sidecar_families: {', '.join(wave_summary['wave3_sidecar_families']) or 'none'}",
        "",
        "## Hard Guards",
    ]
    for key, value in runplan["hard_guards"].items():
        lines.append(f"- {key}: {str(value).lower()}")
    lines.extend(["", "## Errors"])
    if payload["errors"]:
        lines.extend(f"- {item}" for item in payload["errors"])
    else:
        lines.append("- none")
    lines.extend(["", "## Warnings"])
    if payload["warnings"]:
        lines.extend(f"- {item}" for item in payload["warnings"])
    else:
        lines.append("- none")
    return "\n".join(lines) + "\n"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate declarative model family matrix without loading models.")
    parser.add_argument("--matrix_path", required=True)
    parser.add_argument("--restart_validation_json", default=None)
    parser.add_argument("--restart_bundle_validation_json", default=None)
    parser.add_argument("--future_pod_inputs_json", required=True)
    parser.add_argument(
        "--base_config",
        default="configs/representation_surgery_2026.yaml",
        help="Base config used to discover available encoder profiles.",
    )
    parser.add_argument("--out_dir", required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()

    matrix_path = Path(args.matrix_path).expanduser().resolve()
    restart_validation_arg = args.restart_bundle_validation_json or args.restart_validation_json
    if restart_validation_arg is None:
        raise RuntimeError("One of --restart_validation_json or --restart_bundle_validation_json is required.")
    restart_validation_path = Path(restart_validation_arg).expanduser().resolve()
    future_inputs_path = Path(args.future_pod_inputs_json).expanduser().resolve()
    base_config_path = Path(args.base_config).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    matrix = _load_yaml(matrix_path)
    restart_validation = _load_json(restart_validation_path)
    future_inputs = _load_json(future_inputs_path)
    base_config = _load_yaml(base_config_path)

    shared_contract = matrix.get("shared_contract", {})
    families = matrix.get("families", {})
    available_profiles = _available_profiles(base_config)

    warnings: List[str] = []
    errors: List[str] = []

    if restart_validation.get("status") != "ok":
        errors.append("restart_bundle_validation.json must have status='ok' before validating model family matrix.")
    if future_inputs.get("status") != "ok":
        errors.append("future_pod_inputs.json must have status='ok' before validating model family matrix.")
    if not isinstance(shared_contract, dict):
        errors.append("model_family_matrix.yaml missing top-level shared_contract object.")
        shared_contract = {}
    if not isinstance(families, dict) or not families:
        errors.append("model_family_matrix.yaml missing top-level families mapping.")
        families = {}

    reference_family = shared_contract.get("reference_family")
    if reference_family != "clip_base_patch32_mainline":
        errors.append("shared_contract.reference_family must be 'clip_base_patch32_mainline'.")
    if reference_family not in families:
        errors.append("shared_contract.reference_family is not declared in families.")

    local_sources = future_inputs.get("local_sources", {})
    canonical_pod_restore = future_inputs.get("canonical_pod_restore", {})
    reusable_archived_head_contract = restart_validation.get("reusable_archived_head_contract", {})
    historical_repair_branch_contract = restart_validation.get("historical_repair_branch_contract", {})
    reusable_archived_head = canonical_pod_restore.get("reusable_archived_head", {})
    historical_public_atomic_head_reference = canonical_pod_restore.get("historical_public_atomic_head_reference", {})
    head_meta_path = local_sources.get("head_meta_path_local")
    if not _nonempty_text(head_meta_path):
        errors.append("future_pod_inputs.local_sources.head_meta_path_local is missing.")
        head_meta = {}
    else:
        head_meta_file = Path(head_meta_path).expanduser()
        if not head_meta_file.exists():
            errors.append(f"Local head meta path does not exist: {head_meta_file}")
            head_meta = {}
        else:
            head_meta = _load_json(head_meta_file)
    head_meta_model_id = (
        head_meta.get("fingerprint", {}).get("model_id")
        if isinstance(head_meta.get("fingerprint"), dict)
        else None
    )
    reusable_archived_head_available = reusable_archived_head_contract.get("status") == "ok"
    reusable_archived_head_family = reusable_archived_head.get("family")
    historical_repair_branch_present = bool(historical_repair_branch_contract.get("variants"))
    historical_public_atomic_head_reference_status = str(
        historical_public_atomic_head_reference.get("status")
        or ("meta_only" if historical_public_atomic_head_reference.get("availability") == "meta_only_or_missing" else "available")
    )

    supported_sources = set(SUPPORTED_ENCODER_SOURCES)
    for family_name, family in families.items():
        if not isinstance(family, dict):
            errors.append(f"Family '{family_name}' must be a mapping.")
            continue
        encoder_source = family.get("encoder_source")
        if encoder_source not in supported_sources:
            errors.append(
                f"Family '{family_name}' uses unsupported encoder_source={encoder_source!r}. "
                f"Supported set: {sorted(supported_sources)}"
            )

    reuse_families = [name for name, family in families.items() if isinstance(family, dict) and family.get("head_policy") == "reuse_resolved_head"]
    head_reuse_guard_ok = True
    if reuse_families != ["clip_base_patch32_mainline"]:
        head_reuse_guard_ok = False
        errors.append(
            "Exactly one family may declare head_policy='reuse_resolved_head', and it must be clip_base_patch32_mainline."
        )
    reference_entry = families.get("clip_base_patch32_mainline", {})
    if isinstance(reference_entry, dict):
        if not _as_bool(reference_entry.get("enabled")):
            head_reuse_guard_ok = False
            errors.append("clip_base_patch32_mainline must stay enabled=true.")
        if reference_entry.get("runtime_mode") != "classifier":
            head_reuse_guard_ok = False
            errors.append("clip_base_patch32_mainline must use runtime_mode='classifier'.")
        if reference_entry.get("encoder_source") != "hf_clip_or_vit":
            head_reuse_guard_ok = False
            errors.append("clip_base_patch32_mainline must use encoder_source='hf_clip_or_vit'.")
        if reusable_archived_head_available is not True:
            head_reuse_guard_ok = False
            errors.append("Reusable archived head must be available for clip_base_patch32_mainline.")
        if reusable_archived_head_family != "clip_base_patch32_mainline":
            head_reuse_guard_ok = False
            errors.append("Reusable archived head family must be clip_base_patch32_mainline.")
        if reference_entry.get("model_id") != head_meta_model_id:
            head_reuse_guard_ok = False
            errors.append(
                "clip_base_patch32_mainline model_id must match the archived local head_meta fingerprint.model_id."
            )

    for family_name, family in families.items():
        if not isinstance(family, dict):
            continue
        runtime_mode = family.get("runtime_mode")
        head_policy = family.get("head_policy")
        enabled = _as_bool(family.get("enabled"))
        wave = family.get("wave")
        encoder_source = family.get("encoder_source")
        encoder_profile = family.get("encoder_profile")
        blocked_reason = family.get("blocked_reason")

        if runtime_mode not in {"classifier", "sidecar_only"}:
            errors.append(f"Family '{family_name}' has unsupported runtime_mode={runtime_mode!r}.")

        if encoder_profile is not None and encoder_profile not in available_profiles:
            errors.append(
                f"Family '{family_name}' references missing encoder_profile='{encoder_profile}'. "
                f"Available profiles: {available_profiles}"
            )

        if runtime_mode == "classifier":
            if head_policy == "no_classifier_head":
                errors.append(f"Classifier family '{family_name}' cannot use head_policy='no_classifier_head'.")
            if family.get("requires_classifier_head") is not True:
                errors.append(f"Classifier family '{family_name}' must declare requires_classifier_head=true.")
            if enabled and family_name != "clip_base_patch32_mainline":
                if head_policy != "train_new_head" or family.get("requires_new_head") is not True:
                    errors.append(
                        f"Enabled classifier family '{family_name}' must train a new head unless it is the reference CLIP reuse family."
                    )

        if runtime_mode == "sidecar_only":
            if family.get("requires_classifier_head") is not False:
                errors.append(f"Sidecar family '{family_name}' must declare requires_classifier_head=false.")
            if head_policy != "no_classifier_head":
                errors.append(f"Sidecar family '{family_name}' must declare head_policy='no_classifier_head'.")
            if family.get("eligible_for_intervention_mainline") is not False:
                errors.append(f"Sidecar family '{family_name}' must not be eligible_for_intervention_mainline.")
            if family.get("eligible_for_shift_benchmark") is not False:
                errors.append(f"Sidecar family '{family_name}' must not be eligible_for_shift_benchmark.")

        if encoder_source in {"eva_x", "chexfound", "llava_rad_encoder"}:
            if family.get("requires_local_encoder_path") is not True:
                errors.append(
                    f"Family '{family_name}' with encoder_source='{encoder_source}' must declare requires_local_encoder_path=true."
                )

        if not enabled and wave in {2, 3} and not _nonempty_text(blocked_reason):
            errors.append(f"Disabled wave-{wave} family '{family_name}' must declare a non-empty blocked_reason.")

        if shared_contract.get("runtime_allow_download") is False and encoder_source == "hf_clip_or_vit":
            if not _nonempty_text(family.get("model_id")):
                errors.append(f"hf_clip_or_vit family '{family_name}' must declare model_id when runtime_allow_download=false.")
            if enabled and not _nonempty_text(family.get("revision")):
                errors.append(f"Runnable hf_clip_or_vit family '{family_name}' must declare immutable revision.")

        if encoder_source == "openclip" and not enabled:
            if family.get("requires_prebootstrap_contract") is not True:
                errors.append("Disabled openclip family must declare requires_prebootstrap_contract=true.")
            if not _nonempty_text(blocked_reason):
                errors.append("Disabled openclip family must declare blocked_reason.")

    wave1_classifier_families = [
        family_name
        for family_name, family in families.items()
        if isinstance(family, dict) and _as_bool(family.get("enabled")) and family.get("runtime_mode") == "classifier"
    ]
    wave2_blocked_foundation_families = [
        family_name
        for family_name, family in families.items()
        if isinstance(family, dict)
        and not _as_bool(family.get("enabled"))
        and family.get("runtime_mode") == "classifier"
        and family.get("comparison_bucket") in FOUNDATION_BUCKETS
    ]
    wave3_sidecar_families = [
        family_name
        for family_name, family in families.items()
        if isinstance(family, dict)
        and not _as_bool(family.get("enabled"))
        and family.get("runtime_mode") == "sidecar_only"
    ]

    for family_name in wave1_classifier_families:
        if families[family_name].get("wave") != 1:
            errors.append(f"Enabled classifier family '{family_name}' must live in wave=1.")
    for family_name in wave2_blocked_foundation_families:
        if families[family_name].get("wave") != 2:
            errors.append(f"Blocked foundation family '{family_name}' must live in wave=2.")
    for family_name in wave3_sidecar_families:
        if families[family_name].get("wave") != 3:
            errors.append(f"Sidecar family '{family_name}' must live in wave=3.")

    hard_guards = {
        "no_nonclip_head_reuse": head_reuse_guard_ok,
        "no_sidecar_in_mainline": all(
            not (
                isinstance(family, dict)
                and family.get("runtime_mode") == "sidecar_only"
                and (
                    _as_bool(family.get("eligible_for_intervention_mainline"))
                    or _as_bool(family.get("eligible_for_shift_benchmark"))
                )
            )
            for family in families.values()
        ),
        "no_disabled_family_in_wave1": all(_as_bool(families[name].get("enabled")) for name in wave1_classifier_families),
    }

    payload = {
        "status": "ok" if not errors else "error",
        "warnings": warnings,
        "errors": errors,
        "reference_family": reference_family,
        "supported_encoder_sources": SUPPORTED_ENCODER_SOURCES,
        "available_profiles": available_profiles,
        "validated_families": list(families.keys()),
        "head_reuse_guard": "matched" if head_reuse_guard_ok else "mismatch",
        "wave_summary": {
            "wave1_classifier_families": wave1_classifier_families,
            "wave2_blocked_foundation_families": wave2_blocked_foundation_families,
            "wave3_sidecar_families": wave3_sidecar_families,
        },
        "reusable_archived_head_available": reusable_archived_head_available,
        "reusable_archived_head_family": reusable_archived_head_family,
        "historical_repair_branch_present": historical_repair_branch_present,
        "historical_public_atomic_head_reference_status": historical_public_atomic_head_reference_status,
    }

    family_head_strategy = {}
    for family_name, family in families.items():
        if not isinstance(family, dict):
            continue
        if family_name == "clip_base_patch32_mainline":
            family_head_strategy[family_name] = "reuse_archived_head"
        elif family.get("runtime_mode") == "classifier":
            family_head_strategy[family_name] = "materialize_config_only_no_head_reuse"
        else:
            family_head_strategy[family_name] = "sidecar_only_no_classifier_head"

    runplan = {
        "shared_contract": shared_contract,
        "reference_family": reference_family,
        "wave1_classifier_families": wave1_classifier_families,
        "wave2_blocked_foundation_families": wave2_blocked_foundation_families,
        "wave3_sidecar_families": wave3_sidecar_families,
        "reusable_archived_head_available": reusable_archived_head_available,
        "reusable_archived_head_family": reusable_archived_head_family,
        "historical_repair_branch_present": historical_repair_branch_present,
        "historical_public_atomic_head_reference_status": historical_public_atomic_head_reference_status,
        "family_head_strategy": family_head_strategy,
        "canonical_pod_restore_ref": canonical_pod_restore,
        "hard_guards": hard_guards,
    }

    (out_dir / "model_family_validation.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (out_dir / "future_model_family_runplan.json").write_text(json.dumps(runplan, indent=2), encoding="utf-8")
    (out_dir / "model_family_validation.md").write_text(_markdown_report(payload, runplan), encoding="utf-8")

    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
