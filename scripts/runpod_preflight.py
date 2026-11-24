#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import torch

# Allow direct script execution without manual PYTHONPATH exports.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phystta.config.loader import load_config
from phystta.data import DataEngine
from phystta.models.encoder_factory import EncoderFactory
from phystta.models.head_contract import resolve_head_contract


_BROADER_SHIFT_FAMILY_CONFIGS = {
    "mimic_public_acquisition_atomic.yaml",
    "mimic_public_acquisition_atomic_headsubspace.yaml",
}


def _existing(paths: List[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def _base_requirements(src) -> Tuple[List[Path], List[str]]:
    root = Path(src.root).expanduser()
    name = str(src.name).lower()
    paths: List[Path] = []
    notes: List[str] = []
    if name == "chexpert":
        paths = [
            root / "train.csv",
            root / "CheXpert-v1.0" / "train.csv",
            root / "CheXpert-v1.0-small" / "train.csv",
        ]
    elif name in {"nih", "nih_chestxray14", "chestxray14"}:
        paths = [root / "Data_Entry_2017.csv", root / "data_entry_2017.csv"]
        notes.append(str(root / "images"))
    elif name in {"mimic", "mimic-cxr", "mimic-cxr-jpg"}:
        paths = [
            root / "mimic-cxr-2.0.0-chexpert.csv",
            root / "mimic-cxr-2.0.0-chexpert.csv.gz",
            root / "mimic-cxr-2.0.0" / "mimic-cxr-2.0.0-chexpert.csv",
            root / "mimic-cxr-2.0.0" / "mimic-cxr-2.0.0-chexpert.csv.gz",
            root / "mimic-cxr-2.0.0-metadata.csv",
            root / "mimic-cxr-2.0.0-metadata.csv.gz",
            root / "mimic-cxr-2.0.0" / "mimic-cxr-2.0.0-metadata.csv",
            root / "mimic-cxr-2.0.0" / "mimic-cxr-2.0.0-metadata.csv.gz",
        ]
    elif name in {"vindr", "vindr-cxr"}:
        paths = [
            root / "train.csv",
            root / "Annotations" / "image_labels_train.csv",
        ]
        notes.append(str(root / "Train"))
    return paths, notes


def _expected_smoke_csv(src) -> Path | None:
    root = Path(src.root).expanduser()
    name = str(src.name).lower()
    if name == "chexpert":
        return root / "train.smoke.csv"
    if name in {"nih", "nih_chestxray14", "chestxray14"}:
        return root / "Data_Entry_2017.smoke.csv"
    if name in {"mimic", "mimic-cxr", "mimic-cxr-jpg"}:
        return root / "mimic-cxr-2.0.0-chexpert.smoke.csv"
    if name in {"vindr", "vindr-cxr"}:
        return root / "train.smoke.csv"
    return None


def _path_spec(src) -> tuple[str, Path] | None:
    root = Path(src.root).expanduser()
    name = str(src.name).lower()
    if name == "chexpert":
        return "Path", root
    if name in {"nih", "nih_chestxray14", "chestxray14"}:
        return "Image Index", root / "images"
    if name in {"mimic", "mimic-cxr", "mimic-cxr-jpg"}:
        return "path", root
    if name in {"vindr", "vindr-cxr"}:
        return "image_id", root
    return None


def _check_smoke_rows(src, smoke_csv: Path) -> Dict[str, object]:
    spec = _path_spec(src)
    if spec is None:
        return {"checked": False}
    path_col, base = spec
    df = pd.read_csv(smoke_csv)
    if path_col not in df.columns:
        return {"checked": False, "error": f"missing path column {path_col}"}
    missing = df[path_col].astype(str).map(lambda p: not (base / p).exists()).sum()
    return {
        "checked": True,
        "rows": int(len(df)),
        "missing_files": int(missing),
    }


def _realism_pairing_status(cfg) -> Dict[str, object]:
    enabled = bool(getattr(cfg.intervention, "use_realism", False))
    pair_mode = str(getattr(cfg.intervention, "pair_mode", "")) if enabled else None
    require_resolution = bool(enabled and pair_mode and pair_mode != "strict_device_change")
    statuses = (
        DataEngine(cfg.data)
        .attach_runtime_contract(cfg)
        .pair_metadata_status(pair_mode=pair_mode, require_resolution=require_resolution)
    )
    failing = [status for status in statuses if pair_mode and not bool(status.get("supports_requested_pair_mode"))]
    return {
        "enabled": enabled,
        "pair_mode": pair_mode,
        "require_resolution_metadata": require_resolution,
        "statuses": statuses,
        "failing_sources": failing,
    }


def _expected_label_cols(cfg) -> List[str]:
    label_cols = list(getattr(cfg.data, "label_cols", None) or [])
    if label_cols:
        return label_cols
    out: List[str] = []
    for src in cfg.data.sources:
        for col in list(getattr(src, "label_cols", None) or []):
            if col not in out:
                out.append(str(col))
    return out


def _mixed_stress_contract_status(cfg, *, encoder_profile: str | None) -> Dict[str, object]:
    scenario_mode = str(getattr(cfg.erasure_eval, "scenario_mode", "atomic_calibrated_eval"))
    if scenario_mode != "mixed_stress_eval":
        return {"checked": False}

    enc_cfg = EncoderFactory.resolve_encoder_config(cfg, profile=encoder_profile)
    configured_head_path = getattr(enc_cfg, "head_path", None)
    configured_head_meta_path = getattr(enc_cfg, "head_meta_path", None)
    reasons: List[str] = []

    if bool(getattr(cfg.runtime, "allow_download", False)):
        reasons.append("mixed_stress_eval requires runtime.allow_download=false.")
    if not bool(getattr(cfg.runtime, "freeze_artifacts", False)):
        reasons.append("mixed_stress_eval requires runtime.freeze_artifacts=true.")

    expected_label_cols = _expected_label_cols(cfg)
    head_contract: Dict[str, object]
    try:
        contract = resolve_head_contract(
            cfg,
            label_cols=expected_label_cols if expected_label_cols else None,
            encoder_profile=encoder_profile,
            require_head_path=True,
            strict_meta=True,
            load_artifact=True,
        )
        head_contract = {
            "status": "ok",
            "head_path": str(contract.head_path) if contract.head_path is not None else None,
            "head_meta_path": str(configured_head_meta_path) if configured_head_meta_path is not None else None,
            "expected_num_classes": int(cfg.model.num_classes),
            "expected_label_cols": expected_label_cols,
        }
    except Exception as e:
        head_contract = {
            "status": "error",
            "configured_head_path": str(configured_head_path) if configured_head_path is not None else None,
            "configured_head_meta_path": str(configured_head_meta_path) if configured_head_meta_path is not None else None,
            "expected_num_classes": int(cfg.model.num_classes),
            "expected_label_cols": expected_label_cols,
            "error": f"{type(e).__name__}: {e}",
        }
        reasons.append(
            "mixed_stress_eval requires a valid legacy 6-class head/meta contract before execution: "
            + str(head_contract["error"])
        )

    return {
        "checked": True,
        "eligible": len(reasons) == 0,
        "family": str(getattr(cfg.model, "family", "")),
        "encoder_source": str(getattr(enc_cfg, "encoder_source", "")),
        "expected_num_classes": int(cfg.model.num_classes),
        "runtime": {
            "freeze_artifacts": bool(getattr(cfg.runtime, "freeze_artifacts", False)),
            "allow_download": bool(getattr(cfg.runtime, "allow_download", False)),
        },
        "head_contract": head_contract,
        "reasons": reasons,
    }


def _broader_shift_family_contract_status(
    cfg,
    *,
    config_path: Path,
    encoder_profile: str | None,
) -> Dict[str, object]:
    if config_path.name not in _BROADER_SHIFT_FAMILY_CONFIGS:
        return {"checked": False}

    scenario_mode = str(getattr(cfg.erasure_eval, "scenario_mode", "atomic_calibrated_eval"))
    intervention_shift_types = [str(x) for x in getattr(cfg.intervention, "shift_types", [])]
    erasure_shift_types = [str(x) for x in getattr(cfg.erasure_eval, "shift_types", [])]
    expected_shift_types = ["heel", "scatter", "exposure_poisson", "motion_blur"]
    expected_label_cols = _expected_label_cols(cfg)
    reasons: List[str] = []

    enc_cfg = EncoderFactory.resolve_encoder_config(cfg, profile=encoder_profile)
    configured_head_path = getattr(enc_cfg, "head_path", None)
    configured_head_meta_path = getattr(enc_cfg, "head_meta_path", None)
    realism_path = Path(str(getattr(cfg.intervention, "realism_path", ""))).expanduser()

    if scenario_mode != "atomic_calibrated_eval":
        reasons.append("broader_shift_family requires erasure_eval.scenario_mode=atomic_calibrated_eval.")
    if str(getattr(cfg.model, "family", "")) != "cnn":
        reasons.append("broader_shift_family requires model.family=cnn.")
    if str(getattr(enc_cfg, "encoder_source", "")) != "baseline_cnn":
        reasons.append("broader_shift_family requires encoder_source=baseline_cnn.")
    if int(getattr(cfg.model, "num_classes", 0)) != 14:
        reasons.append("broader_shift_family requires the legacy 14-class public atomic head lineage.")
    if not bool(getattr(cfg.intervention, "use_realism", False)):
        reasons.append("broader_shift_family requires intervention.use_realism=true.")
    if intervention_shift_types != expected_shift_types:
        reasons.append(
            "broader_shift_family requires intervention.shift_types="
            f"{expected_shift_types}."
        )
    if erasure_shift_types != expected_shift_types:
        reasons.append(
            "broader_shift_family requires erasure_eval.shift_types="
            f"{expected_shift_types}."
        )
    if bool(getattr(cfg.runtime, "allow_download", False)):
        reasons.append("broader_shift_family requires runtime.allow_download=false.")
    if not bool(getattr(cfg.runtime, "freeze_artifacts", False)):
        reasons.append("broader_shift_family requires runtime.freeze_artifacts=true.")
    if not str(getattr(cfg.intervention, "realism_path", "")).strip():
        reasons.append("broader_shift_family requires intervention.realism_path.")
    elif not realism_path.exists():
        reasons.append(
            "broader_shift_family requires a calibrated realism artifact before execution: "
            f"{realism_path}"
        )

    head_contract: Dict[str, object]
    try:
        contract = resolve_head_contract(
            cfg,
            label_cols=expected_label_cols if expected_label_cols else None,
            encoder_profile=encoder_profile,
            require_head_path=True,
            strict_meta=True,
            load_artifact=True,
        )
        head_contract = {
            "status": "ok",
            "head_path": str(contract.head_path) if contract.head_path is not None else None,
            "head_meta_path": str(configured_head_meta_path) if configured_head_meta_path is not None else None,
            "expected_num_classes": int(getattr(cfg.model, "num_classes", 0)),
            "expected_label_cols": expected_label_cols,
        }
    except Exception as e:
        head_contract = {
            "status": "error",
            "configured_head_path": str(configured_head_path) if configured_head_path is not None else None,
            "configured_head_meta_path": str(configured_head_meta_path) if configured_head_meta_path is not None else None,
            "expected_num_classes": int(getattr(cfg.model, "num_classes", 0)),
            "expected_label_cols": expected_label_cols,
            "error": f"{type(e).__name__}: {e}",
        }
        reasons.append(
            "broader_shift_family requires a valid legacy 14-class head/meta contract before execution: "
            + str(head_contract["error"])
        )

    return {
        "checked": True,
        "eligible": len(reasons) == 0,
        "config_name": config_path.name,
        "scenario_mode": scenario_mode,
        "family": str(getattr(cfg.model, "family", "")),
        "encoder_source": str(getattr(enc_cfg, "encoder_source", "")),
        "expected_num_classes": int(getattr(cfg.model, "num_classes", 0)),
        "intervention_shift_types": intervention_shift_types,
        "erasure_eval_shift_types": erasure_shift_types,
        "expected_shift_types": expected_shift_types,
        "runtime": {
            "freeze_artifacts": bool(getattr(cfg.runtime, "freeze_artifacts", False)),
            "allow_download": bool(getattr(cfg.runtime, "allow_download", False)),
        },
        "realism": {
            "enabled": bool(getattr(cfg.intervention, "use_realism", False)),
            "realism_path": str(realism_path),
            "realism_path_exists": bool(realism_path.exists()),
        },
        "head_contract": head_contract,
        "reasons": reasons,
    }


def _vlm_contract_status(cfg) -> Dict[str, object]:
    if not hasattr(cfg, "vlm") or cfg.vlm is None or not bool(getattr(cfg.vlm, "enabled", False)):
        return {"checked": False}

    reasons: List[str] = []
    enc_cfg = EncoderFactory.resolve_encoder_config(cfg)
    configured_head_path = getattr(enc_cfg, "head_path", None)
    configured_head_meta_path = getattr(enc_cfg, "head_meta_path", None)
    model_path = Path(cfg.vlm.model_path).expanduser() if getattr(cfg.vlm, "model_path", None) is not None else None
    model_id = getattr(cfg.vlm, "model_id", None)
    revision = getattr(cfg.vlm, "revision", None)
    runtime_allow_download = bool(getattr(cfg.runtime, "allow_download", False))
    runtime_freeze_artifacts = bool(getattr(cfg.runtime, "freeze_artifacts", False))
    offline = bool(getattr(cfg.vlm, "offline", False))
    prompt_template = str(getattr(cfg.vlm, "prompt_template", "") or "")
    json_max_retries = int(getattr(cfg.vlm, "json_max_retries", 0))
    expected_label_cols = _expected_label_cols(cfg)

    if runtime_allow_download:
        reasons.append("vlm contour requires runtime.allow_download=false.")
    if not runtime_freeze_artifacts:
        reasons.append("vlm contour requires runtime.freeze_artifacts=true.")
    if not prompt_template.strip():
        reasons.append("vlm contour requires a non-empty vlm.prompt_template.")
    if json_max_retries < 0:
        reasons.append("vlm contour requires vlm.json_max_retries>=0.")

    source_contract: Dict[str, object]
    if model_path is not None:
        source_contract = {
            "mode": "local_path",
            "status": "ok" if model_path.exists() else "error",
            "model_path": str(model_path),
            "exists": bool(model_path.exists()),
            "local_files_only": True,
        }
        if not model_path.exists():
            reasons.append(
                f"vlm contour requires a local VLM model_path before execution: {model_path}"
            )
    elif model_id:
        source_contract = {
            "mode": "hf_snapshot",
            "status": "pending",
            "model_id": str(model_id),
            "revision": revision,
            "local_snapshot_present": False,
            "local_files_only": True,
        }
        immutable_ok = True
        immutable_checker = getattr(cfg, "_is_immutable_revision", None)
        if callable(immutable_checker):
            immutable_ok = bool(immutable_checker(revision))
        if str(getattr(cfg.experiment, "tier", "dev")) == "paper" and not immutable_ok:
            reasons.append(
                "vlm contour requires immutable vlm.revision when using remote model_id in paper-tier."
            )
        try:
            from huggingface_hub import snapshot_download

            local_dir = snapshot_download(
                repo_id=str(model_id),
                revision=revision,
                local_files_only=True,
            )
            source_contract["status"] = "ok"
            source_contract["local_snapshot_present"] = True
            source_contract["local_dir"] = str(local_dir)
        except Exception as e:
            source_contract["status"] = "error"
            source_contract["error"] = f"{type(e).__name__}: {e}"
            reasons.append(
                "vlm contour requires a local cached HF snapshot or local model_path before execution: "
                + str(source_contract["error"])
            )
    else:
        source_contract = {
            "mode": "unset",
            "status": "error",
            "error": "vlm.enabled=true but both model_path and model_id are unset",
        }
        reasons.append("vlm contour requires vlm.model_path or vlm.model_id.")

    dependency_contract: Dict[str, object]
    try:
        import transformers  # type: ignore

        dependency_contract = {
            "transformers": {
                "status": "ok",
                "version": getattr(transformers, "__version__", None),
            }
        }
    except Exception as e:
        dependency_contract = {
            "transformers": {
                "status": "error",
                "error": f"{type(e).__name__}: {e}",
            }
        }
        reasons.append("vlm contour requires transformers to be installed for execution.")

    head_contract: Dict[str, object]
    try:
        contract = resolve_head_contract(
            cfg,
            label_cols=expected_label_cols if expected_label_cols else None,
            require_head_path=True,
            strict_meta=True,
            load_artifact=True,
        )
        head_contract = {
            "status": "ok",
            "head_path": str(contract.head_path) if contract.head_path is not None else None,
            "head_meta_path": (
                str(getattr(contract.enc_cfg, "head_meta_path", None))
                if getattr(contract.enc_cfg, "head_meta_path", None) is not None
                else None
            ),
            "expected_num_classes": int(getattr(cfg.model, "num_classes", 0)),
            "expected_label_cols": expected_label_cols,
        }
    except Exception as e:
        head_contract = {
            "status": "error",
            "configured_head_path": str(configured_head_path) if configured_head_path is not None else None,
            "configured_head_meta_path": (
                str(configured_head_meta_path) if configured_head_meta_path is not None else None
            ),
            "expected_num_classes": int(getattr(cfg.model, "num_classes", 0)),
            "expected_label_cols": expected_label_cols,
            "error": f"{type(e).__name__}: {e}",
        }
        reasons.append(
            "vlm contour requires a valid head/head-meta contract before execution: "
            + str(head_contract["error"])
        )

    return {
        "checked": True,
        "eligible": len(reasons) == 0,
        "entrypoint": "scripts/run_vlm_erasure_eval.py",
        "model_type": str(getattr(cfg.vlm, "model_type", "")),
        "model_family": str(getattr(cfg.model, "family", "")),
        "encoder_source": str(getattr(enc_cfg, "encoder_source", "")),
        "expected_num_classes": int(getattr(cfg.model, "num_classes", 0)),
        "offline": offline,
        "runtime": {
            "freeze_artifacts": runtime_freeze_artifacts,
            "allow_download": runtime_allow_download,
        },
        "prompt_template_present": bool(prompt_template.strip()),
        "json_max_retries": json_max_retries,
        "source_contract": source_contract,
        "dependency_contract": dependency_contract,
        "head_contract": head_contract,
        "reasons": reasons,
    }


def _vlm_execution_path_audit(
    *,
    config_path: Path,
    out_json_path: Path | None,
) -> Dict[str, object]:
    repo_root = Path(__file__).resolve().parents[1]
    preflight_path = repo_root / "scripts" / "runpod_preflight.py"
    vlm_runner_path = repo_root / "scripts" / "run_vlm_suite.py"
    vlm_eval_path = repo_root / "scripts" / "run_vlm_erasure_eval.py"
    paper_suite_path = repo_root / "scripts" / "run_paper_suite.py"
    end_to_end_path = repo_root / "scripts" / "run_end_to_end.py"

    preflight_command = [
        "python3",
        "scripts/runpod_preflight.py",
        "--config",
        str(config_path),
    ]
    if out_json_path is not None:
        preflight_command.extend(["--out_json", str(out_json_path)])

    runner_out_dir = str(out_json_path.parent) if out_json_path is not None else "<OUT_DIR>"
    eraser_placeholder = "<ERASER_ARTIFACT>"
    vlm_runner_dry_run = [
        "python3",
        "scripts/run_vlm_suite.py",
        "--config",
        str(config_path),
        "--eraser",
        eraser_placeholder,
        "--out_dir",
        runner_out_dir,
        "--dry_run",
    ]
    vlm_runner_run = [
        "python3",
        "scripts/run_vlm_suite.py",
        "--config",
        str(config_path),
        "--eraser",
        eraser_placeholder,
        "--out_dir",
        runner_out_dir,
    ]
    direct_vlm_command = [
        "python3",
        "scripts/run_vlm_erasure_eval.py",
        "--config",
        str(config_path),
        "--eraser",
        eraser_placeholder,
        "--out_dir",
        f"{runner_out_dir}/vlm",
        "--max_images",
        "300",
    ]
    paper_suite_dry_run = [
        "python3",
        "scripts/run_paper_suite.py",
        "--config",
        str(config_path),
        "--eraser",
        eraser_placeholder,
        "--out_dir",
        f"{runner_out_dir}/paper_suite",
        "--dry_run",
    ]
    paper_suite_run = [
        "python3",
        "scripts/run_paper_suite.py",
        "--config",
        str(config_path),
        "--eraser",
        eraser_placeholder,
        "--out_dir",
        f"{runner_out_dir}/paper_suite",
    ]

    return {
        "checked": True,
        "honest_full_execution_path_exists": True,
        "runtime_proven_current_bundle": False,
        "verdict": "branch_owned_runner_available",
        "candidate_paths": [
            {
                "entrypoint": str(preflight_path),
                "status": "preflight_only",
                "eligible": True,
                "reasons": [
                    "This entrypoint validates the VLM runtime/model/dependency contract but does not execute the VLM contour."
                ],
                "command_surface": [
                    {
                        "kind": "runpod_preflight",
                        "argv": preflight_command,
                        "display": " ".join(preflight_command),
                    }
                ],
            },
            {
                "entrypoint": str(vlm_runner_path),
                "status": "branch_owned_runner",
                "eligible": True,
                "reasons": [
                    "scripts/run_vlm_suite.py is the current truthful branch-owned VLM execution/reporting surface.",
                    "It materializes branch-owned dry-run and real-run artifacts (`vlm_plan.json`, `vlm_runner_report.json`, `vlm_manifest.json`) while delegating the stage execution to scripts/run_vlm_erasure_eval.py.",
                    "It does not imply runtime proof, scientific proof, or branch-owned scientific summary semantics.",
                ],
                "command_surface": [
                    {
                        "kind": "run_vlm_suite_dry_run",
                        "argv": vlm_runner_dry_run,
                        "display": " ".join(vlm_runner_dry_run),
                    },
                    {
                        "kind": "run_vlm_suite",
                        "argv": vlm_runner_run,
                        "display": " ".join(vlm_runner_run),
                    },
                ],
                "reporting_surface": {
                    "out_dir": runner_out_dir,
                    "vlm_plan_json": f"{runner_out_dir}/vlm_plan.json",
                    "vlm_runner_report_json": f"{runner_out_dir}/vlm_runner_report.json",
                    "vlm_manifest_json": f"{runner_out_dir}/vlm_manifest.json",
                    "runpod_preflight_json": f"{runner_out_dir}/runpod_preflight.json",
                    "vlm_rehearsal_json": f"{runner_out_dir}/vlm_rehearsal.json",
                    "vlm_outputs_jsonl": f"{runner_out_dir}/vlm_eval/vlm_outputs.jsonl",
                    "vlm_summary_json": f"{runner_out_dir}/vlm_eval/vlm_summary.json",
                    "vlm_metrics_csv": f"{runner_out_dir}/vlm_eval/vlm_metrics.csv",
                },
            },
            {
                "entrypoint": str(vlm_eval_path),
                "status": "stage_execution_entrypoint",
                "eligible": True,
                "reasons": [
                    "scripts/run_vlm_erasure_eval.py is the direct VLM stage entrypoint once a compatible eraser artifact already exists.",
                    "It materializes VLM-owned outputs (`vlm_outputs.jsonl`, `vlm_summary.json`, `vlm_metrics.csv`) but does not provide a branch-owned dry-run or orchestration/reporting surface.",
                ],
                "command_surface": [
                    {
                        "kind": "run_vlm_erasure_eval",
                        "argv": direct_vlm_command,
                        "display": " ".join(direct_vlm_command),
                    }
                ],
                "reporting_surface": {
                    "out_dir": f"{runner_out_dir}/vlm",
                    "vlm_outputs_jsonl": f"{runner_out_dir}/vlm/vlm_outputs.jsonl",
                    "vlm_summary_json": f"{runner_out_dir}/vlm/vlm_summary.json",
                    "vlm_metrics_csv": f"{runner_out_dir}/vlm/vlm_metrics.csv",
                },
            },
            {
                "entrypoint": str(paper_suite_path),
                "status": "generic_paper_suite_orchestration",
                "eligible": True,
                "reasons": [
                    "scripts/run_paper_suite.py remains a truthful generic orchestration path for the VLM contour.",
                    "It already materializes a dry-run plan and paper-tier reporting surfaces while invoking scripts/run_vlm_erasure_eval.py as the VLM stage.",
                    "It still reuses generic paper-suite artifacts and is not the canonical branch-owned VLM runner/reporting surface anymore.",
                ],
                "command_surface": [
                    {
                        "kind": "run_paper_suite_dry_run",
                        "argv": paper_suite_dry_run,
                        "display": " ".join(paper_suite_dry_run),
                    },
                    {
                        "kind": "run_paper_suite",
                        "argv": paper_suite_run,
                        "display": " ".join(paper_suite_run),
                    },
                ],
                "reporting_surface": {
                    "out_dir": f"{runner_out_dir}/paper_suite",
                    "paper_suite_plan_json": f"{runner_out_dir}/paper_suite/paper_suite_plan.json",
                    "paper_report_dir": f"{runner_out_dir}/paper_suite/paper_report",
                    "dissertation_artifacts_dir": f"{runner_out_dir}/paper_suite/dissertation_artifacts",
                    "vlm_outputs_jsonl": f"{runner_out_dir}/paper_suite/vlm/vlm_outputs.jsonl",
                    "vlm_summary_json": f"{runner_out_dir}/paper_suite/vlm/vlm_summary.json",
                    "vlm_metrics_csv": f"{runner_out_dir}/paper_suite/vlm/vlm_metrics.csv",
                },
            },
            {
                "entrypoint": str(end_to_end_path),
                "status": "ineligible",
                "eligible": False,
                "reasons": [
                    "run_end_to_end.py retrains head/eraser/realism artifacts before invoking run_paper_suite.py.",
                    "That path is not the truthful frozen/offline VLM execution surface for the current pre-pod contour.",
                ],
            },
        ],
        "notes": [
            "A truthful VLM execution surface exists today through the branch-owned VLM runner, the direct VLM stage entrypoint, and generic paper-suite orchestration.",
            "The branch-owned VLM runner is now the canonical higher-order execution/reporting surface for this contour.",
            "The direct script remains the stage entrypoint, while run_paper_suite.py remains only a truthful generic secondary orchestration surface.",
            "Neither surface is runtime-proven on current-bundle outputs.",
        ],
    }


def _build_vlm_rehearsal_payload(
    *,
    config_path: Path,
    out_json_path: Path | None,
    payload: Dict[str, object],
) -> Dict[str, object]:
    vlm_contract = payload.get("vlm_contract", {})
    checked = bool(isinstance(vlm_contract, dict) and vlm_contract.get("checked"))
    if not checked:
        return {"checked": False}

    sources = payload.get("sources", [])
    source_roots = [str(row.get("root")) for row in sources if isinstance(row, dict) and row.get("root") is not None]
    contract_reasons = list(vlm_contract.get("reasons", [])) if isinstance(vlm_contract, dict) else []
    errors = list(payload.get("errors", [])) if isinstance(payload.get("errors", []), list) else []
    eligible = bool(vlm_contract.get("eligible", False)) and not errors

    preflight_command = [
        "python3",
        "scripts/runpod_preflight.py",
        "--config",
        str(config_path),
    ]
    if out_json_path is not None:
        preflight_command.extend(["--out_json", str(out_json_path)])

    execution_path_audit = _vlm_execution_path_audit(
        config_path=config_path,
        out_json_path=out_json_path,
    )
    path_exists = bool(execution_path_audit.get("honest_full_execution_path_exists", False))
    primary_execution_command_surface: List[Dict[str, object]] = []
    stage_execution_command_surface: List[Dict[str, object]] = []
    execution_reporting_surface: Dict[str, object] = {}
    stage_reporting_surface: Dict[str, object] = {}
    for row in execution_path_audit.get("candidate_paths", []):
        if not isinstance(row, dict):
            continue
        status = row.get("status")
        if status == "branch_owned_runner" and eligible:
            primary_execution_command_surface = list(row.get("command_surface", []))
            if isinstance(row.get("reporting_surface"), dict):
                execution_reporting_surface = dict(row["reporting_surface"])
        if status == "stage_execution_entrypoint" and eligible:
            stage_execution_command_surface = list(row.get("command_surface", []))
            if isinstance(row.get("reporting_surface"), dict):
                stage_reporting_surface = dict(row["reporting_surface"])

    notes: List[str] = []
    if path_exists and eligible:
        execution_surface_status = "branch_owned_runner_available"
        notes.append(
            "The current truthful higher-order VLM execution path is surfaced through scripts/run_vlm_suite.py."
        )
        notes.append(
            "scripts/run_vlm_erasure_eval.py remains the direct stage entrypoint, while scripts/run_paper_suite.py is now only a truthful generic secondary orchestration surface."
        )
    elif path_exists:
        execution_surface_status = "branch_owned_runner_blocked_by_preflight"
        notes.append(
            "Truthful VLM execution candidates exist, but the current contour is ineligible at preflight and must not be executed."
        )
    else:
        execution_surface_status = str(execution_path_audit.get("verdict", "not_defined"))
        notes.append(
            "The current VLM contour is ineligible at preflight and no truthful execution path has been surfaced."
        )

    return {
        "checked": True,
        "kind": "vlm_rehearsal",
        "config": str(config_path),
        "source_roots": source_roots,
        "vlm_contract": vlm_contract,
        "eligible": eligible,
        "reasons": errors if errors else contract_reasons,
        "runtime_proof_implied": False,
        "scientific_proof_implied": False,
        "preflight_command_surface": [
            {
                "kind": "runpod_preflight",
                "argv": preflight_command,
                "display": " ".join(preflight_command),
            }
        ],
        "execution_path_audit": execution_path_audit,
        "has_execution_surface": bool(primary_execution_command_surface or stage_execution_command_surface),
        "execution_surface_status": execution_surface_status,
        "primary_execution_command_surface": primary_execution_command_surface,
        "stage_execution_command_surface": stage_execution_command_surface,
        "execution_reporting_surface": execution_reporting_surface,
        "stage_reporting_surface": stage_reporting_surface,
        "notes": notes,
    }


def _build_mixed_stress_rehearsal_payload(
    *,
    config_path: Path,
    out_json_path: Path | None,
    payload: Dict[str, object],
) -> Dict[str, object]:
    mixed_stress_contract = payload.get("mixed_stress_contract", {})
    checked = bool(isinstance(mixed_stress_contract, dict) and mixed_stress_contract.get("checked"))
    if not checked:
        return {"checked": False}

    sources = payload.get("sources", [])
    source_roots = [str(row.get("root")) for row in sources if isinstance(row, dict) and row.get("root") is not None]
    contract_reasons = list(mixed_stress_contract.get("reasons", [])) if isinstance(mixed_stress_contract, dict) else []
    errors = list(payload.get("errors", [])) if isinstance(payload.get("errors", []), list) else []
    eligible = bool(mixed_stress_contract.get("eligible", False)) and not errors
    head_contract = mixed_stress_contract.get("head_contract", {}) if isinstance(mixed_stress_contract, dict) else {}

    preflight_command = [
        "python3",
        "scripts/runpod_preflight.py",
        "--config",
        str(config_path),
    ]
    if out_json_path is not None:
        preflight_command.extend(["--out_json", str(out_json_path)])

    execution_path_audit = _mixed_stress_execution_path_audit(
        config_path=config_path,
        out_json_path=out_json_path,
    )
    staged_path_exists = bool(execution_path_audit.get("honest_full_execution_path_exists", False))
    execution_command_surface: List[Dict[str, object]] = []
    execution_reporting_surface: Dict[str, object] = {}
    for row in execution_path_audit.get("candidate_paths", []):
        if not isinstance(row, dict):
            continue
        if row.get("status") != "staged_runner":
            continue
        if eligible:
            execution_command_surface = list(row.get("command_surface", []))
            if isinstance(row.get("reporting_surface"), dict):
                execution_reporting_surface = dict(row["reporting_surface"])
        break

    notes: List[str] = []
    if staged_path_exists and eligible:
        execution_surface_status = "staged_execution_path_available"
        notes.append(
            "A truthful staged mixed-stress execution path now exists via scripts/run_mixed_stress_suite.py."
        )
        notes.append(
            "This staged runner remains legacy CNN / 6-class / mixed_stress_eval and is not yet runtime-proven on current-bundle outputs."
        )
    elif staged_path_exists:
        execution_surface_status = "staged_execution_path_blocked_by_preflight"
        notes.append(
            "A truthful staged mixed-stress execution path exists via scripts/run_mixed_stress_suite.py, but the current contour is ineligible at preflight and must not be executed."
        )
    else:
        execution_surface_status = str(execution_path_audit.get("verdict", "not_defined"))
        notes.append(
            "Mixed stress is ineligible at preflight and no truthful staged execution path has been surfaced."
        )

    return {
        "checked": True,
        "kind": "mixed_stress_rehearsal",
        "config": str(config_path),
        "scenario_mode": payload.get("scenario_mode"),
        "source_roots": source_roots,
        "mixed_stress_contract": mixed_stress_contract,
        "legacy_head_lineage": {
            "head_path": head_contract.get("head_path") or head_contract.get("configured_head_path"),
            "head_meta_path": head_contract.get("head_meta_path") or head_contract.get("configured_head_meta_path"),
            "validation": head_contract,
        },
        "eligible": eligible,
        "reasons": errors if errors else contract_reasons,
        "preflight_command_surface": [
            {
                "kind": "runpod_preflight",
                "argv": preflight_command,
                "display": " ".join(preflight_command),
            }
        ],
        "execution_path_audit": execution_path_audit,
        "has_execution_surface": bool(execution_command_surface),
        "execution_surface_status": execution_surface_status,
        "execution_command_surface": execution_command_surface,
        "execution_reporting_surface": execution_reporting_surface,
        "notes": notes,
    }


def _build_broader_shift_family_rehearsal_payload(
    *,
    config_path: Path,
    out_json_path: Path | None,
    payload: Dict[str, object],
) -> Dict[str, object]:
    broader_shift_family_contract = payload.get("broader_shift_family_contract", {})
    checked = bool(
        isinstance(broader_shift_family_contract, dict)
        and broader_shift_family_contract.get("checked")
    )
    if not checked:
        return {"checked": False}

    sources = payload.get("sources", [])
    source_roots = [str(row.get("root")) for row in sources if isinstance(row, dict) and row.get("root") is not None]
    contract_reasons = (
        list(broader_shift_family_contract.get("reasons", []))
        if isinstance(broader_shift_family_contract, dict)
        else []
    )
    errors = list(payload.get("errors", [])) if isinstance(payload.get("errors", []), list) else []
    eligible = bool(broader_shift_family_contract.get("eligible", False)) and not errors
    head_contract = (
        broader_shift_family_contract.get("head_contract", {})
        if isinstance(broader_shift_family_contract, dict)
        else {}
    )
    realism = (
        broader_shift_family_contract.get("realism", {})
        if isinstance(broader_shift_family_contract, dict)
        else {}
    )

    preflight_command = [
        "python3",
        "scripts/runpod_preflight.py",
        "--config",
        str(config_path),
    ]
    if out_json_path is not None:
        preflight_command.extend(["--out_json", str(out_json_path)])

    execution_path_audit = _broader_shift_family_execution_path_audit(
        config_path=config_path,
        out_json_path=out_json_path,
    )
    path_exists = bool(execution_path_audit.get("honest_full_execution_path_exists", False))
    execution_command_surface: List[Dict[str, object]] = []
    execution_reporting_surface: Dict[str, object] = {}
    for row in execution_path_audit.get("candidate_paths", []):
        if not isinstance(row, dict):
            continue
        if row.get("status") != "public_mainline_launcher":
            continue
        if eligible:
            execution_command_surface = list(row.get("command_surface", []))
            if isinstance(row.get("reporting_surface"), dict):
                execution_reporting_surface = dict(row["reporting_surface"])
        break

    notes: List[str] = []
    if path_exists and eligible:
        execution_surface_status = "public_mainline_execution_path_available"
        notes.append(
            "A truthful broader-family execution path is now surfaced via scripts/run_public_mainline.py."
        )
        notes.append(
            "This path preserves the broader public atomic contour rather than narrowing it into the canonical heel/scatter-only CLIP line."
        )
    elif path_exists:
        execution_surface_status = "public_mainline_execution_path_blocked_by_preflight"
        notes.append(
            "A truthful broader-family execution path exists via scripts/run_public_mainline.py, but the current contour is ineligible at preflight and must not be executed."
        )
    else:
        execution_surface_status = str(execution_path_audit.get("verdict", "not_defined"))
        notes.append(
            "The broader-family contour is ineligible at preflight and no truthful execution path has been surfaced."
        )

    return {
        "checked": True,
        "kind": "broader_shift_family_rehearsal",
        "config": str(config_path),
        "scenario_mode": payload.get("scenario_mode"),
        "source_roots": source_roots,
        "broader_shift_family_contract": broader_shift_family_contract,
        "contour_identity": {
            "config_name": broader_shift_family_contract.get("config_name"),
            "scenario_mode": broader_shift_family_contract.get("scenario_mode"),
            "family": broader_shift_family_contract.get("family"),
            "encoder_source": broader_shift_family_contract.get("encoder_source"),
            "expected_num_classes": broader_shift_family_contract.get("expected_num_classes"),
            "intervention_shift_types": broader_shift_family_contract.get("intervention_shift_types"),
            "erasure_eval_shift_types": broader_shift_family_contract.get("erasure_eval_shift_types"),
        },
        "runtime_contract": broader_shift_family_contract.get("runtime", {}),
        "realism_artifact": realism,
        "legacy_head_lineage": {
            "head_path": head_contract.get("head_path") or head_contract.get("configured_head_path"),
            "head_meta_path": head_contract.get("head_meta_path") or head_contract.get("configured_head_meta_path"),
            "validation": head_contract,
        },
        "eligible": eligible,
        "reasons": errors if errors else contract_reasons,
        "runtime_proof_implied": False,
        "scientific_proof_implied": False,
        "preflight_command_surface": [
            {
                "kind": "runpod_preflight",
                "argv": preflight_command,
                "display": " ".join(preflight_command),
            }
        ],
        "execution_path_audit": execution_path_audit,
        "has_execution_surface": bool(execution_command_surface),
        "execution_surface_status": execution_surface_status,
        "execution_command_surface": execution_command_surface,
        "execution_reporting_surface": execution_reporting_surface,
        "notes": notes,
    }


def _broader_shift_family_execution_path_audit(
    *,
    config_path: Path,
    out_json_path: Path | None,
) -> Dict[str, object]:
    repo_root = Path(__file__).resolve().parents[1]
    preflight_path = repo_root / "scripts" / "runpod_preflight.py"
    public_mainline_path = repo_root / "scripts" / "run_public_mainline.py"
    paper_suite_path = repo_root / "scripts" / "run_paper_suite.py"
    end_to_end_path = repo_root / "scripts" / "run_end_to_end.py"

    preflight_command = [
        "python3",
        "scripts/runpod_preflight.py",
        "--config",
        str(config_path),
    ]
    if out_json_path is not None:
        preflight_command.extend(["--out_json", str(out_json_path)])

    runner_out_dir = str(out_json_path.parent) if out_json_path is not None else "<OUT_DIR>"
    dry_run_command = [
        "python3",
        "scripts/run_public_mainline.py",
        "--config",
        str(config_path),
        "--out_dir",
        runner_out_dir,
        "--dry_run",
    ]
    run_command = [
        "python3",
        "scripts/run_public_mainline.py",
        "--config",
        str(config_path),
        "--out_dir",
        runner_out_dir,
    ]

    reporting_surface = {
        "out_dir": runner_out_dir,
        "effective_config": f"{runner_out_dir}/public_mainline.effective.yaml",
        "preflight_json": f"{runner_out_dir}/public_mainline_preflight.json",
        "plan_json": f"{runner_out_dir}/public_mainline_plan.json",
        "manifest_json": f"{runner_out_dir}/public_mainline_manifest.json",
        "summary_json": f"{runner_out_dir}/summary/public_mainline_summary.json",
        "logs_dir": f"{runner_out_dir}/launcher_logs",
        "stage_state_dir": f"{runner_out_dir}/.stage_state",
    }

    return {
        "checked": True,
        "honest_full_execution_path_exists": True,
        "runtime_proven_current_bundle": False,
        "verdict": "public_mainline_execution_path_available",
        "candidate_paths": [
            {
                "entrypoint": str(preflight_path),
                "status": "preflight_only",
                "eligible": True,
                "reasons": [
                    "This entrypoint validates broader-family identity, realism, frozen/offline runtime, and legacy 14-class head/meta assumptions but does not execute the contour."
                ],
                "command_surface": [
                    {
                        "kind": "runpod_preflight",
                        "argv": preflight_command,
                        "display": " ".join(preflight_command),
                    }
                ],
            },
            {
                "entrypoint": str(public_mainline_path),
                "status": "public_mainline_launcher",
                "eligible": True,
                "reasons": [
                    "scripts/run_public_mainline.py already supports atomic_calibrated_eval with realism enabled and non-mixed shift families.",
                    "The broader-family configs preserve heel/scatter/exposure_poisson/motion_blur and therefore remain distinct from the canonical heel/scatter-only CLIP bounded contour.",
                    "This launcher materializes effective config, preflight, plan, manifest, stage logs/state, and summary surfaces on the current codebase.",
                    "It is still not runtime-proven on current-bundle outputs and must not be confused with runtime or scientific proof.",
                ],
                "command_surface": [
                    {
                        "kind": "run_public_mainline_dry_run",
                        "argv": dry_run_command,
                        "display": " ".join(dry_run_command),
                    },
                    {
                        "kind": "run_public_mainline",
                        "argv": run_command,
                        "display": " ".join(run_command),
                    },
                ],
                "reporting_surface": reporting_surface,
            },
            {
                "entrypoint": str(paper_suite_path),
                "status": "ineligible",
                "eligible": False,
                "reasons": [
                    "run_paper_suite.py requires a pre-existing eraser artifact and paper-tier execution assumptions rather than serving as the truthful entrypoint from the raw broader-family config.",
                    "It is downstream of broader-family orchestration, not the truthful first execution surface for this contour.",
                ],
            },
            {
                "entrypoint": str(end_to_end_path),
                "status": "ineligible",
                "eligible": False,
                "reasons": [
                    "run_end_to_end.py retrains and patches head/realism artifacts as part of a paper-tier end-to-end flow.",
                    "That path is not the truthful frozen/offline execution surface for the current broader-family contour, which expects compatible legacy 14-class head/meta lineage up front.",
                ],
            },
        ],
        "notes": [
            "A truthful broader-family execution path is now surfaced via scripts/run_public_mainline.py.",
            "This path remains distinct from the canonical heel/scatter-only CLIP bounded contour because it preserves the broader shift family set.",
            "The surfaced launcher path is not yet runtime-proven on current-bundle outputs.",
        ],
    }


def _mixed_stress_execution_path_audit(
    *,
    config_path: Path,
    out_json_path: Path | None,
) -> Dict[str, object]:
    repo_root = Path(__file__).resolve().parents[1]
    preflight_path = repo_root / "scripts" / "runpod_preflight.py"
    runner_path = repo_root / "scripts" / "run_mixed_stress_suite.py"
    paper_suite_path = repo_root / "scripts" / "run_paper_suite.py"
    end_to_end_path = repo_root / "scripts" / "run_end_to_end.py"

    preflight_command = [
        "python3",
        "scripts/runpod_preflight.py",
        "--config",
        str(config_path),
    ]
    if out_json_path is not None:
        preflight_command.extend(["--out_json", str(out_json_path)])

    runner_out_dir = str(out_json_path.parent) if out_json_path is not None else "<OUT_DIR>"
    dry_run_command = [
        "python3",
        "scripts/run_mixed_stress_suite.py",
        "--config",
        str(config_path),
        "--out_dir",
        runner_out_dir,
        "--dry_run",
    ]
    run_command = [
        "python3",
        "scripts/run_mixed_stress_suite.py",
        "--config",
        str(config_path),
        "--out_dir",
        runner_out_dir,
    ]

    reporting_surface = {
        "out_dir": runner_out_dir,
        "effective_config": f"{runner_out_dir}/mixed_stress.effective.yaml",
        "plan_json": f"{runner_out_dir}/mixed_stress_plan.json",
        "preflight_json": f"{runner_out_dir}/runpod_preflight.json",
        "rehearsal_json": f"{runner_out_dir}/mixed_stress_rehearsal.json",
        "manifest_json": f"{runner_out_dir}/mixed_stress_manifest.json",
    }

    return {
        "checked": True,
        "honest_full_execution_path_exists": True,
        "runtime_proven_current_bundle": False,
        "verdict": "staged_execution_path_available",
        "candidate_paths": [
            {
                "entrypoint": str(preflight_path),
                "status": "preflight_only",
                "eligible": True,
                "reasons": [
                    "This entrypoint validates mixed-stress runtime and legacy lineage assumptions but does not define full execution orchestration."
                ],
                "command_surface": [
                    {
                        "kind": "runpod_preflight",
                        "argv": preflight_command,
                        "display": " ".join(preflight_command),
                    }
                ],
            },
            {
                "entrypoint": str(runner_path),
                "status": "staged_runner",
                "eligible": True,
                "reasons": [
                    "scripts/run_mixed_stress_suite.py is the truthful staged execution path for the current mixed_stress_eval contour.",
                    "It stays on the legacy CNN / 6-class / realism-disabled mixed-stress surface rather than rebinding into paper-tier or CLIP-lineage orchestration.",
                    "It is still not runtime-proven on current-bundle outputs and should not be presented as equivalent to the hardened bounded MIMIC or external lines.",
                ],
                "command_surface": [
                    {
                        "kind": "run_mixed_stress_suite_dry_run",
                        "argv": dry_run_command,
                        "display": " ".join(dry_run_command),
                    },
                    {
                        "kind": "run_mixed_stress_suite",
                        "argv": run_command,
                        "display": " ".join(run_command),
                    },
                ],
                "reporting_surface": reporting_surface,
            },
            {
                "entrypoint": str(paper_suite_path),
                "status": "ineligible",
                "eligible": False,
                "reasons": [
                    "run_paper_suite.py requires experiment.tier=paper.",
                    "run_paper_suite.py requires intervention.use_realism=true.",
                    "run_paper_suite.py requires intervention.realism_path.",
                    "Those assumptions conflict with the current mixed_stress_eval contour, which is legacy CNN / 6-class and keeps realism disabled.",
                ],
            },
            {
                "entrypoint": str(end_to_end_path),
                "status": "ineligible",
                "eligible": False,
                "reasons": [
                    "run_end_to_end.py terminates in scripts/run_paper_suite.py.",
                    "It therefore inherits the paper-tier realism assumptions that are incompatible with the current mixed_stress_eval contour.",
                ],
            },
        ],
        "notes": [
            "A truthful staged mixed-stress execution path now exists via scripts/run_mixed_stress_suite.py.",
            "This path is runner-based rather than paper-suite-based and remains separate from the canonical CLIP bounded contour.",
            "The staged runner is not yet runtime-proven on current-bundle outputs.",
        ],
    }


def run_preflight(config: str | Path, out_json: str | Path | None = None, encoder_profile: str | None = None) -> dict:
    config_path = Path(config).expanduser()
    cfg = load_config(str(config_path))
    errors: list[str] = []
    warnings: list[str] = []
    realism_pairing = _realism_pairing_status(cfg)

    cuda_info = {
        "available": bool(torch.cuda.is_available()),
        "device_count": int(torch.cuda.device_count()),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    if not cuda_info["available"]:
        errors.append("CUDA is not available.")

    try:
        enc_cfg = EncoderFactory.resolve_encoder_config(cfg, profile=encoder_profile)
        hf_info = {
            "encoder_source": str(getattr(enc_cfg, "encoder_source", "")),
            "model_id": getattr(enc_cfg, "model_id", None),
            "revision": getattr(enc_cfg, "revision", None),
            "status": "not_required",
        }
        if hf_info["encoder_source"] == "hf_clip_or_vit" and hf_info["model_id"]:
            from huggingface_hub import snapshot_download

            try:
                local_dir = snapshot_download(
                    repo_id=hf_info["model_id"],
                    revision=hf_info["revision"],
                    local_files_only=True,
                )
                hf_info["status"] = "present"
                hf_info["local_dir"] = str(local_dir)
            except Exception as e:
                hf_info["status"] = "missing"
                hf_info["error"] = f"{type(e).__name__}: {e}"
                errors.append("Required local HF snapshot is missing for the selected encoder.")
    except Exception as e:
        hf_info = {"status": "error", "error": f"{type(e).__name__}: {e}"}
        errors.append("Failed to resolve encoder config for preflight.")

    scenario_mode = str(getattr(cfg.erasure_eval, "scenario_mode", "atomic_calibrated_eval"))
    if scenario_mode == "mixed_stress_eval" and realism_pairing["enabled"]:
        errors.append("mixed_stress_eval must keep intervention.use_realism=false.")
    mixed_stress_contract = _mixed_stress_contract_status(cfg, encoder_profile=encoder_profile)
    if mixed_stress_contract.get("checked") and not mixed_stress_contract.get("eligible", False):
        errors.extend(str(reason) for reason in mixed_stress_contract.get("reasons", []))
    broader_shift_family_contract = _broader_shift_family_contract_status(
        cfg,
        config_path=config_path,
        encoder_profile=encoder_profile,
    )
    if broader_shift_family_contract.get("checked") and not broader_shift_family_contract.get("eligible", False):
        errors.extend(str(reason) for reason in broader_shift_family_contract.get("reasons", []))
    vlm_contract = _vlm_contract_status(cfg)
    if vlm_contract.get("checked") and not vlm_contract.get("eligible", False):
        errors.extend(str(reason) for reason in vlm_contract.get("reasons", []))
    failing_sources = realism_pairing.get("failing_sources", [])
    if realism_pairing["enabled"] and failing_sources:
        gap = ", ".join(
            f"{status.get('source')}[{','.join(status.get('missing_capabilities', []))}]"
            for status in failing_sources
        )
        errors.append(
            f"Real-pair realism with pair_mode='{realism_pairing['pair_mode']}' is unsupported for: {gap}"
        )

    sources = []
    for src in cfg.data.sources:
        root = Path(src.root).expanduser()
        source_row: dict[str, object] = {
            "name": str(src.name),
            "root": str(root),
            "root_exists": bool(root.exists()),
            "root_is_symlink": bool(root.is_symlink()) if str(root).startswith("/data/") else None,
        }
        if str(root).startswith("/data/"):
            if not root.exists():
                errors.append(f"{src.name}: expected /data symlink root missing: {root}")
            elif not root.is_symlink():
                errors.append(f"{src.name}: expected /data root to be a symlink: {root}")

        required_paths, notes = _base_requirements(src)
        source_row["base_inputs"] = [str(p) for p in required_paths]
        source_row["base_inputs_present"] = [str(p) for p in required_paths if p.exists()]
        source_row["notes"] = notes
        if str(src.name).lower() in {"mimic", "mimic-cxr", "mimic-cxr-jpg"}:
            labels_present = _existing([p for p in required_paths if "chexpert" in p.name])
            meta_present = _existing([p for p in required_paths if "metadata" in p.name])
            if labels_present is None or meta_present is None:
                errors.append(f"{src.name}: required base metadata files are missing.")
        else:
            if not source_row["base_inputs_present"]:
                errors.append(f"{src.name}: required base metadata CSV is missing under {root}.")

        smoke_csv = _expected_smoke_csv(src)
        source_row["expected_smoke_csv"] = str(smoke_csv) if smoke_csv is not None else None
        source_row["smoke_csv_exists"] = bool(smoke_csv and smoke_csv.exists())
        if smoke_csv is not None and smoke_csv.exists():
            smoke_check = _check_smoke_rows(src, smoke_csv)
            source_row["smoke_check"] = smoke_check
            if smoke_check.get("missing_files", 0) > 0:
                errors.append(f"{src.name}: smoke CSV references missing files: {smoke_csv}")
        elif smoke_csv is not None:
            warnings.append(f"{src.name}: smoke CSV not present yet: {smoke_csv}")
        sources.append(source_row)

    payload = {
        "config": str(config_path),
        "encoder_profile": encoder_profile,
        "cuda": cuda_info,
        "hf_snapshot": hf_info,
        "realism_pairing": realism_pairing,
        "scenario_mode": scenario_mode,
        "mixed_stress_contract": mixed_stress_contract,
        "broader_shift_family_contract": broader_shift_family_contract,
        "vlm_contract": vlm_contract,
        "sources": sources,
        "errors": errors,
        "warnings": warnings,
    }
    out_path: Path | None = Path(out_json).expanduser() if out_json is not None else None
    vlm_rehearsal = _build_vlm_rehearsal_payload(
        config_path=config_path,
        out_json_path=out_path,
        payload=payload,
    )
    payload["vlm_rehearsal"] = vlm_rehearsal
    mixed_stress_rehearsal = _build_mixed_stress_rehearsal_payload(
        config_path=config_path,
        out_json_path=out_path,
        payload=payload,
    )
    payload["mixed_stress_rehearsal"] = mixed_stress_rehearsal
    broader_shift_family_rehearsal = _build_broader_shift_family_rehearsal_payload(
        config_path=config_path,
        out_json_path=out_path,
        payload=payload,
    )
    payload["broader_shift_family_rehearsal"] = broader_shift_family_rehearsal

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Saved RunPod preflight report: {out_path}")
        if vlm_rehearsal.get("checked"):
            rehearsal_path = out_path.parent / "vlm_rehearsal.json"
            rehearsal_path.write_text(json.dumps(vlm_rehearsal, indent=2), encoding="utf-8")
            print(f"Saved VLM rehearsal report: {rehearsal_path}")
        if mixed_stress_rehearsal.get("checked"):
            rehearsal_path = out_path.parent / "mixed_stress_rehearsal.json"
            rehearsal_path.write_text(json.dumps(mixed_stress_rehearsal, indent=2), encoding="utf-8")
            print(f"Saved mixed-stress rehearsal report: {rehearsal_path}")
        if broader_shift_family_rehearsal.get("checked"):
            rehearsal_path = out_path.parent / "broader_shift_family_rehearsal.json"
            rehearsal_path.write_text(json.dumps(broader_shift_family_rehearsal, indent=2), encoding="utf-8")
            print(f"Saved broader shift-family rehearsal report: {rehearsal_path}")
    if errors:
        raise SystemExit(1)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run preflight checks for the final RunPod main-path smoke execution."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--out_json", default="outputs/bootstrap/runpod_preflight.json")
    parser.add_argument("--encoder_profile", default=None)
    args = parser.parse_args()

    run_preflight(args.config, out_json=args.out_json, encoder_profile=args.encoder_profile)


if __name__ == "__main__":
    main()
