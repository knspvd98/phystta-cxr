#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from phystta.config.loader import load_config
from phystta.data import DataEngine, canonical_external_label_space, harmonize_external_labels
from phystta.shifts import ShiftManager
from phystta.models import EncoderFactory, build_encoder_fingerprint, load_head_meta, validate_head_meta
from phystta.intervention import (
    eraser_fingerprint_mismatches,
    load_eraser_collection,
    resolve_eraser_for_shift,
    TokenEraserWrapper,
    EntropySteeringAdaptor,
    SteeringEraser,
    TaskPreservingEraser,
)
from phystta.safety import (
    ShiftIdentifierV2,
    SafetyPolicy,
    SafetyPolicyConfig,
    DecisionPolicy,
    LabelShiftRecalibrator,
    sample_harm_multilabel,
    expected_harm_multilabel,
    policy_regret,
    regret_by_shift,
)
from phystta.shifts.realism import calibrator_fingerprint_mismatches, load_calibrator, intensity_grid_from_calibrator
from phystta.metrics import (
    expected_calibration_error,
    auroc_per_class,
    auprc_per_class,
    macro_average,
    selective_risk,
    report_at_operating_points,
    operating_point_stats,
)
from phystta.utils import (
    set_global_seed,
    preprocess_signature,
    cluster_bootstrap_mean,
    sample_ids_from_batch,
    load_torch_artifact,
    write_reprovenance_bundle,
    sha256_file,
)


def eval_metrics(
    probs: np.ndarray,
    labels: np.ndarray,
    coverage: float = 0.8,
    thresholds: np.ndarray | float | None = None,
) -> dict:
    if thresholds is None:
        raise ValueError("eval_metrics requires explicit thresholds; fixed 0.5 defaults are disallowed.")
    valid = labels >= 0
    probs_flat = probs[valid].reshape(-1)
    labels_flat = labels[valid].reshape(-1)
    if probs.ndim == 2:
        c = probs.shape[1]
        th = np.asarray(thresholds, dtype=float).reshape(-1)
        if th.size == 1:
            th = np.full(c, float(th[0]), dtype=float)
        if th.size != c:
            raise ValueError(f"threshold length mismatch in eval_metrics: got {th.size}, expected {c}")
        preds = (probs >= th.reshape(1, -1))[valid].astype(float)
    else:
        t = float(np.asarray(thresholds, dtype=float).reshape(-1)[0])
        preds = (probs_flat >= t).astype(float)
    errors = preds != labels_flat
    conf = np.maximum(probs_flat, 1 - probs_flat)
    metrics = {
        "ece": expected_calibration_error(probs_flat, labels_flat, n_bins=15),
        "risk_at_coverage": selective_risk(errors.astype(float), conf, coverage=coverage),
    }
    try:
        auroc = auroc_per_class(labels, probs)
        auprc = auprc_per_class(labels, probs)
        metrics["auroc_macro"] = macro_average(auroc)
        metrics["auprc_macro"] = macro_average(auprc)
    except Exception:
        metrics["auroc_macro"] = float("nan")
        metrics["auprc_macro"] = float("nan")
    return metrics


def _validate_external_input_paths(
    *,
    config_path: str | Path,
    eraser_path: str | Path,
    shift_id_model_path: str | Path,
    policy_json_path: str | Path,
) -> None:
    missing: list[str] = []
    for flag, raw in [
        ("--config", config_path),
        ("--eraser", eraser_path),
        ("--shift_id_model", shift_id_model_path),
        ("--policy_json", policy_json_path),
    ]:
        path = Path(raw).expanduser()
        if not path.exists():
            missing.append(f"{flag}={path}")
    if missing:
        raise RuntimeError(
            "External follow-up preflight failed: missing required input paths: "
            + ", ".join(missing)
        )


def _load_external_policy_contract(policy_json_path: str | Path) -> dict:
    path = Path(policy_json_path).expanduser()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(
            f"External follow-up preflight failed: policy_json could not be parsed: {path}"
        ) from exc
    if not isinstance(payload, dict):
        raise RuntimeError(
            f"External follow-up preflight failed: policy_json must contain a JSON object: {path}"
        )
    if "class_thresholds" not in payload:
        raise RuntimeError(
            "External follow-up preflight failed: policy_json is missing class_thresholds. "
            "Regenerate policy thresholds from the internal bounded run before external continuation."
        )
    return payload


def _validate_external_followup_contract(
    *,
    cfg,
    enc_cfg,
    config_path: str | Path,
    eraser_path: str | Path,
    shift_id_model_path: str | Path,
    policy_json_path: str | Path,
) -> dict:
    errors: list[str] = []
    config_path = Path(config_path).expanduser()
    eraser_path = Path(eraser_path).expanduser()
    shift_id_model_path = Path(shift_id_model_path).expanduser()
    policy_json_path = Path(policy_json_path).expanduser()

    if not cfg.runtime.freeze_artifacts:
        errors.append("runtime.freeze_artifacts=true is required for external follow-up")
    if bool(getattr(cfg.runtime, "allow_download", True)):
        errors.append("runtime.allow_download=false is required for external follow-up")

    for source in cfg.data.sources:
        for field in ["root", "metadata_csv", "image_root"]:
            value = getattr(source, field, None)
            if value is None:
                continue
            path = Path(value).expanduser()
            if not path.is_absolute():
                errors.append(
                    f"data source '{source.name}' requires absolute {field} in frozen external mode: {path}"
                )
                continue
            if not path.exists():
                errors.append(
                    f"missing external data path for source '{source.name}': {field}={path}"
                )

    head_path = getattr(enc_cfg, "head_path", None)
    if head_path is None:
        errors.append("encoder.head_path is required for external follow-up")
    else:
        hp = Path(head_path).expanduser()
        if not hp.is_absolute():
            errors.append(f"encoder.head_path must be absolute in frozen external mode: {hp}")
        elif not hp.exists():
            errors.append(f"missing frozen head artifact: {hp}")

    head_meta_path = getattr(enc_cfg, "head_meta_path", None)
    if head_meta_path is None:
        errors.append("encoder.head_meta_path is required for external follow-up")
    else:
        hmp = Path(head_meta_path).expanduser()
        if not hmp.is_absolute():
            errors.append(f"encoder.head_meta_path must be absolute in frozen external mode: {hmp}")
        elif not hmp.exists():
            errors.append(f"missing frozen head metadata artifact: {hmp}")

    eraser_collection = None
    eraser_bundle_meta = None
    if eraser_path.exists():
        try:
            eraser_collection, eraser_bundle_meta = load_eraser_collection(eraser_path)
        except Exception as exc:
            errors.append(
                f"failed to load external eraser artifact/bank '{eraser_path}': {exc}"
            )
        else:
            required_shift_families = sorted(
                {
                    str(shift_type)
                    for shift_type in cfg.erasure_eval.shift_types
                    if str(shift_type) not in {"", "none", "mixed"}
                }
            )
            if eraser_bundle_meta.get("kind") == "bank" and required_shift_families:
                available = sorted(str(key) for key in eraser_collection.keys() if str(key) != "__default__")
                missing = [fam for fam in required_shift_families if fam not in available]
                if missing:
                    errors.append(
                        "eraser bank manifest is missing required shift families for external follow-up: "
                        + ", ".join(missing)
                        + f" (available={available})"
                    )

    if errors:
        raise RuntimeError("External follow-up preflight failed: " + " | ".join(errors))

    reference_shift_family = next(
        (
            str(shift_type)
            for shift_type in cfg.erasure_eval.shift_types
            if str(shift_type) not in {"", "none", "mixed"}
        ),
        None,
    )
    if reference_shift_family is None and isinstance(eraser_bundle_meta, dict):
        if eraser_bundle_meta.get("kind") == "bank":
            available = [str(key) for key in eraser_collection.keys() if str(key) != "__default__"]
            if available:
                reference_shift_family = sorted(available)[0]
        elif "__default__" in eraser_collection:
            reference_shift_family = "__default__"

    return {
        "config_path": str(config_path),
        "eraser_path": str(eraser_path),
        "shift_id_model_path": str(shift_id_model_path),
        "policy_json_path": str(policy_json_path),
        "eraser_bundle_kind": eraser_bundle_meta.get("kind") if isinstance(eraser_bundle_meta, dict) else None,
        "reference_shift_family": reference_shift_family,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="External policy eval (no retuning)")
    parser.add_argument("--config", required=True)
    parser.add_argument("--eraser", required=True)
    parser.add_argument("--shift_id_model", required=True)
    parser.add_argument("--policy_json", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--eval_sources", required=True, help="comma-separated source names")
    parser.add_argument("--shift_type", default=None)
    parser.add_argument("--intensity_grid", default=None)
    parser.add_argument("--encoder_profile", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--bootstrap_samples", type=int, default=200)
    parser.add_argument("--target_sensitivities", default="0.90")
    parser.add_argument("--target_specificities", default="0.90")
    args = parser.parse_args()

    _validate_external_input_paths(
        config_path=args.config,
        eraser_path=args.eraser,
        shift_id_model_path=args.shift_id_model,
        policy_json_path=args.policy_json,
    )
    cfg = load_config(args.config)
    if not cfg.runtime.freeze_artifacts:
        raise RuntimeError("runtime.freeze_artifacts must be true for external eval")
    policy_data = _load_external_policy_contract(args.policy_json)
    seed = args.seed if args.seed is not None else cfg.runtime.seed
    device = args.device or cfg.runtime.device
    set_global_seed(seed, deterministic=cfg.runtime.deterministic, benchmark=cfg.runtime.benchmark)
    target_sensitivities = [float(v.strip()) for v in args.target_sensitivities.split(",") if v.strip()] if args.target_sensitivities else []
    target_specificities = [float(v.strip()) for v in args.target_specificities.split(",") if v.strip()] if args.target_specificities else []

    shift_types = [args.shift_type] if args.shift_type else cfg.erasure_eval.shift_types
    enc_cfg = EncoderFactory.resolve_encoder_config(cfg, profile=args.encoder_profile)
    preflight_contract = _validate_external_followup_contract(
        cfg=cfg,
        enc_cfg=enc_cfg,
        config_path=args.config,
        eraser_path=args.eraser,
        shift_id_model_path=args.shift_id_model,
        policy_json_path=args.policy_json,
    )
    expected_calib_fp = build_encoder_fingerprint(enc_cfg)
    expected_calib_fp["preprocess_signature"] = preprocess_signature(cfg)
    strict_meta = bool(cfg.runtime.freeze_artifacts or getattr(cfg.experiment, "tier", "dev") == "paper")
    if enc_cfg.head_path is None:
        raise RuntimeError("External eval requires head_path")
    head_meta = load_head_meta(
        Path(enc_cfg.head_path),
        explicit_meta=Path(enc_cfg.head_meta_path) if getattr(enc_cfg, "head_meta_path", None) is not None else None,
    )
    if head_meta is None and strict_meta:
        raise RuntimeError(
            f"Head metadata not found for head_path={enc_cfg.head_path}. "
            "Provide encoder.head_meta_path or save a sidecar meta json."
        )
    target_label_cols = canonical_external_label_space()
    if isinstance(head_meta, dict) and isinstance(head_meta.get("label_cols"), list) and head_meta.get("label_cols"):
        target_label_cols = list(head_meta["label_cols"])
    if int(cfg.model.num_classes) != int(len(target_label_cols)):
        raise RuntimeError(
            f"External eval label-space mismatch: model.num_classes={cfg.model.num_classes} "
            f"but target label space has {len(target_label_cols)} labels."
        )

    eval_sources = [s.strip() for s in args.eval_sources.split(",") if s.strip()]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data_engine = DataEngine(cfg.data)
    data_engine.build_index()
    harmonized_index, label_manifest = harmonize_external_labels(
        data_engine.index,
        source_map=data_engine.source_map,
        target_labels=target_label_cols,
        eval_sources=eval_sources,
    )
    data_engine.index = harmonized_index
    data_engine.label_cols = list(target_label_cols)
    label_manifest.update(
        {
            "config_path": str(args.config),
            "head_path": str(enc_cfg.head_path),
            "head_meta_path": str(enc_cfg.head_meta_path) if getattr(enc_cfg, "head_meta_path", None) is not None else None,
            "label_order_validated_against_head": bool(head_meta is not None),
        }
    )
    (out_dir / "label_harmonization_manifest.json").write_text(json.dumps(label_manifest, indent=2))
    data_engine.create_splits()
    label_cols = data_engine.label_cols

    eval_split = cfg.data.eval_split
    loaders = data_engine.build_dataloaders(
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        splits=[eval_split],
        sources_map={eval_split: eval_sources},
    )
    eval_loader = loaders[eval_split]
    if cfg.intervention.use_realism:
        if args.intensity_grid:
            raise RuntimeError("intensity_grid overridden by realism calibrator; remove --intensity_grid")
        if cfg.intervention.realism_path is None:
            raise RuntimeError("use_realism=true but realism_path not set")
        calib = load_calibrator(str(cfg.intervention.realism_path))
        mismatches = calibrator_fingerprint_mismatches(calib, expected_calib_fp)
        if mismatches:
            msg = (
                "Realism calibrator fingerprint mismatch against active encoder/preprocess: "
                + "; ".join(mismatches)
            )
            if getattr(cfg.experiment, "tier", "dev") == "paper":
                raise RuntimeError(msg)
            print(f"[warn] {msg}")
        intensity_map = intensity_grid_from_calibrator(calib, shift_types)
    else:
        grid = [float(v) for v in args.intensity_grid.split(",")] if args.intensity_grid else cfg.erasure_eval.intensity_grid
        intensity_map = {st: grid for st in shift_types}

    shift_manager = ShiftManager(seed=cfg.shifts.seed)
    encoder = EncoderFactory.from_config(cfg, profile=args.encoder_profile)
    if hasattr(encoder, "model"):
        encoder.model.to(device)

    head = load_torch_artifact(enc_cfg.head_path, map_location="cpu")
    head.to(device)
    head.eval()
    meta = head_meta
    if meta is None:
        if strict_meta:
            raise RuntimeError(
                f"Head metadata not found for head_path={enc_cfg.head_path}. "
                "Provide encoder.head_meta_path or save a sidecar meta json."
            )
        else:
            print(f"[warn] head metadata not found for {enc_cfg.head_path}; compatibility checks skipped.")
    else:
        errs = validate_head_meta(
            meta,
            enc_cfg=enc_cfg,
            expected_num_classes=cfg.model.num_classes,
            expected_label_cols=target_label_cols,
        )
        if errs and strict_meta:
            raise RuntimeError("Head metadata validation failed: " + "; ".join(errs))
        if errs:
            print("[warn] head metadata validation warnings: " + "; ".join(errs))

    eraser_collection, eraser_bundle_meta = load_eraser_collection(args.eraser)
    for shift_family, eraser_obj in eraser_collection.items():
        eraser_mismatches = eraser_fingerprint_mismatches(eraser_obj, expected_calib_fp)
        if eraser_mismatches:
            label = "default" if shift_family == "__default__" else str(shift_family)
            msg = (
                f"Eraser fingerprint mismatch for shift_family='{label}' against active encoder/preprocess: "
                + "; ".join(eraser_mismatches)
            )
            if getattr(cfg.experiment, "tier", "dev") == "paper":
                raise RuntimeError(msg)
            print(f"[warn] {msg}")
        if hasattr(eraser_obj, "to"):
            eraser_obj = eraser_obj.to(device)
            eraser_collection[shift_family] = eraser_obj
        if hasattr(eraser_obj, "eval"):
            eraser_obj.eval()
    token_wrappers: dict[str, TokenEraserWrapper] = {}
    token_mode_enabled = bool(getattr(cfg, "eraser", None) and cfg.eraser.token_mode != "none")
    eraser_reference_shift_family = preflight_contract.get("reference_shift_family")
    if (
        eraser_bundle_meta.get("kind") == "bank"
        and any(str(shift_type) in {"", "none"} for shift_type in shift_types)
        and eraser_reference_shift_family is not None
    ):
        print(
            "[warn] external follow-up is using reference eraser shift_family="
            f"'{eraser_reference_shift_family}' for non-shift external scenarios."
        )

    def _get_eraser_for_shift(shift_family: str | None):
        key = shift_family
        if str(shift_family or "") in {"", "none"} and eraser_bundle_meta.get("kind") == "bank":
            if eraser_reference_shift_family is None:
                raise RuntimeError(
                    "External follow-up eraser bank manifest does not provide a reference shift family for shift_type='none'."
                )
            key = eraser_reference_shift_family
        return resolve_eraser_for_shift(eraser_collection, str(key))

    if getattr(cfg, "eraser", None) and cfg.eraser.token_mode != "none":
        if eraser_bundle_meta.get("kind") == "bank":
            raise RuntimeError("token_mode is not compatible with eraser bank manifests in external_policy_eval.")
    entropy_adaptor = None
    if getattr(cfg, "eraser", None) and cfg.eraser.online_enabled:
        if eraser_bundle_meta.get("kind") == "bank":
            raise RuntimeError("online_enabled is not compatible with eraser bank manifests in external_policy_eval.")
        if cfg.eraser.token_mode != "none":
            raise RuntimeError("online_enabled not compatible with token_mode in external_policy_eval.")
        if cfg.eraser.online_mode != "entropy":
            raise RuntimeError("external_policy_eval supports online_mode=entropy only.")
        eraser = _get_eraser_for_shift(str(shift_types[0]))
        base = eraser.eraser if isinstance(eraser, TaskPreservingEraser) else eraser
        if not isinstance(base, SteeringEraser):
            raise RuntimeError("online_mode=entropy requires SteeringEraser.")
        entropy_adaptor = EntropySteeringAdaptor(
            base,
            head,
            steps=cfg.eraser.online_entropy_steps,
            lr=cfg.eraser.online_entropy_lr,
            guard_eps=cfg.eraser.online_guard_eps,
        ).to(device)

    def encode_erased(x: torch.Tensor, shift_family: str | None = None) -> torch.Tensor:
        if entropy_adaptor is not None:
            z = encoder.encode(x)
            return entropy_adaptor(z)
        eraser = _get_eraser_for_shift(shift_family)
        if token_mode_enabled:
            key = str(shift_family or eraser_reference_shift_family or "__default__")
            if key not in token_wrappers:
                token_wrappers[key] = TokenEraserWrapper(
                    encoder,
                    eraser,
                    mode=cfg.eraser.token_mode,
                    token_layers=cfg.eraser.token_layers,
                ).to(device)
            return token_wrappers[key](x)
        return eraser(encoder.encode(x))

    shift_id = ShiftIdentifierV2.load(args.shift_id_model)
    policy_cfg = SafetyPolicyConfig()
    policy = SafetyPolicy(policy_cfg)
    decision_policy = None
    if getattr(cfg.safety_policy, "use_decision_policy", False):
        decision_policy = DecisionPolicy(
            uncertainty_abstain_threshold=cfg.safety_policy.uncertainty_abstain_threshold,
            mixed_abstain_threshold=cfg.safety_policy.mixed_abstain_threshold,
            prevalence_action=cfg.safety_policy.prevalence_action,
            apply_on_covariate=cfg.safety_policy.apply_on_covariate,
        )
    policy.thresholds = policy_data.get("thresholds", policy.thresholds)
    if "harm" in policy_data:
        policy.harm = policy_data["harm"]
    if "shift_harm" in policy_data and isinstance(policy_data["shift_harm"], dict):
        policy.shift_harm = policy_data["shift_harm"]
    clinical_thresholds = policy_data.get("class_thresholds")
    if clinical_thresholds is not None:
        policy.class_thresholds = np.asarray(clinical_thresholds, dtype=float)
        if int(policy.class_thresholds.shape[0]) != int(len(label_cols)):
            raise RuntimeError(
                f"policy_json class_thresholds length mismatch: got {policy.class_thresholds.shape[0]}, "
                f"expected {len(label_cols)} for label space {label_cols}"
            )
    else:
        raise RuntimeError(
            "policy_json missing class_thresholds. Re-run run_safety_policy_eval.py with threshold serialization enabled."
        )
    prevalence_recalibrator = None
    if decision_policy is not None and decision_policy.prevalence_action == "recalibrate":
        src_priors = policy_data.get("source_label_priors")
        if src_priors is None:
            if getattr(cfg.experiment, "tier", "dev") == "paper":
                raise RuntimeError(
                    "policy_json missing source_label_priors required for prevalence_action='recalibrate'. "
                    "Regenerate policy thresholds with run_safety_policy_eval.py."
                )
            print("[warn] source_label_priors missing; recalibrate falls back to no_eraser.")
        else:
            src = np.asarray(src_priors, dtype=float).reshape(-1)
            if int(src.shape[0]) != int(len(label_cols)):
                raise RuntimeError(
                    f"policy_json source_label_priors length mismatch: got {src.shape[0]}, "
                    f"expected {len(label_cols)} for label space {label_cols}"
                )
            src = np.where(np.isfinite(src), src, 0.5)
            src = np.clip(src, 1e-6, 1.0 - 1e-6)
            prevalence_recalibrator = LabelShiftRecalibrator(source_priors=src)

    rows = []
    decisions = []
    harm_no_hist = []
    harm_er_hist = []
    harm_rec_hist = []
    bootstrap_rows = []
    op_target_rows = []
    regret_vals_all = []
    regret_groups_all = []
    regret_vals_by_shift: dict[str, list[np.ndarray]] = {}
    regret_groups_by_shift: dict[str, list[np.ndarray]] = {}
    regret_vals_by_action: dict[str, list[np.ndarray]] = {}
    regret_groups_by_action: dict[str, list[np.ndarray]] = {}

    # reference features from calib split (no retuning)
    calib_df = data_engine.get_split_df(cfg.data.calib_split)
    calib_loader = data_engine.build_dataloaders(batch_size=cfg.train.batch_size, num_workers=cfg.train.num_workers, splits=[cfg.data.calib_split])[cfg.data.calib_split]
    feats_ref = []
    probs_ref = []
    labels_ref = []
    with torch.no_grad():
        for batch in calib_loader:
            x = batch["image"].to(device)
            y = batch["labels"].to(device)
            z = encoder.encode(x)
            logits = head(z)
            feats_ref.append(z.cpu().numpy())
            probs_ref.append(torch.sigmoid(logits).cpu().numpy())
            labels_ref.append(y.cpu().numpy())
    feats_ref = np.concatenate(feats_ref, axis=0)
    probs_ref = np.concatenate(probs_ref, axis=0)
    labels_ref = np.concatenate(labels_ref, axis=0)
    calib_target_defs = report_at_operating_points(
        labels_ref,
        probs_ref,
        label_names=label_cols,
        target_sensitivities=target_sensitivities,
        target_specificities=target_specificities,
    )
    label_to_idx = {name: idx for idx, name in enumerate(label_cols)}

    for shift_type in shift_types:
        if shift_type == "mixed":
            profiles = cfg.intervention.mixed_profiles or []
            scenarios = []
            for idx, profile in enumerate(profiles):
                name = profile.get("name", f"mixed_{idx}")
                if "intensities" in profile:
                    intens = profile["intensities"]
                elif "components" in profile:
                    intens = profile["components"]
                else:
                    intens = {k: v for k, v in profile.items() if k != "name"}
                scenarios.append((name, intens))
        else:
            grid = intensity_map.get(shift_type)
            if grid is None:
                raise RuntimeError(f"No calibrated intensity grid for shift_type={shift_type}")
            scenarios = [(float(intensity), None) for intensity in grid]
        for scenario in scenarios:
            intensity = scenario[0]
            probs = []
            probs_er = []
            labels = []
            feats = []
            meta_flag = False
            with torch.no_grad():
                eval_group_ids = []
                for batch in eval_loader:
                    x = batch["image"].to(device)
                    y = batch["labels"].to(device)
                    if "global_patient_id" in batch:
                        eval_group_ids.extend([str(v) for v in batch["global_patient_id"]])
                    else:
                        eval_group_ids.extend([str(v) for v in batch["patient_id"]])
                    if shift_type == "mixed":
                            sample_ids = sample_ids_from_batch(batch)
                            x_shift, _meta = shift_manager.apply_batch_scenario(x, scenario[1], sample_ids=sample_ids)
                            if _meta and _meta.get("summary", {}).get("destructive_count", 0) > 0:
                                meta_flag = True
                    else:
                        sample_ids = sample_ids_from_batch(batch)
                        x_shift, _meta = shift_manager.apply_batch_with_meta(x, shift_type, float(intensity), sample_ids=sample_ids)
                        if _meta and _meta.get("summary", {}).get("destructive_count", 0) > 0:
                            meta_flag = True
                    z = encoder.encode(x_shift)
                    logits = head(z)
                    feats.append(z.cpu().numpy())
                    probs.append(torch.sigmoid(logits).cpu().numpy())
                    logits_e = head(encode_erased(x_shift, shift_type))
                    probs_er.append(torch.sigmoid(logits_e).cpu().numpy())
                    labels.append(y.cpu().numpy())
            feats = np.concatenate(feats, axis=0)
            probs = np.concatenate(probs, axis=0)
            probs_er = np.concatenate(probs_er, axis=0)
            labels = np.concatenate(labels, axis=0)
            shift_info = shift_id.predict(feats_ref, probs_ref, feats, probs)
            destructive = meta_flag
            probs_rec = probs
            target_priors_est = None
            if prevalence_recalibrator is not None:
                probs_rec, target_priors_est = prevalence_recalibrator.adjust(probs)
            harm_no_all = expected_harm_multilabel(
                probs,
                labels,
                thresholds=clinical_thresholds,
                cost_fn=policy_cfg.cost_fn,
                cost_fp=policy_cfg.cost_fp,
                critical_idx=None,
            )
            harm_er_all = expected_harm_multilabel(
                probs_er,
                labels,
                thresholds=clinical_thresholds,
                cost_fn=policy_cfg.cost_fn,
                cost_fp=policy_cfg.cost_fp,
                critical_idx=None,
            )
            harm_rec_all = expected_harm_multilabel(
                probs_rec,
                labels,
                thresholds=clinical_thresholds,
                cost_fn=policy_cfg.cost_fn,
                cost_fp=policy_cfg.cost_fp,
                critical_idx=None,
            )
            if decision_policy is not None:
                pred_label = str(shift_info.get("label", "unknown"))
                label_harm = policy.shift_harm.get(pred_label, {})
                harm_no_decision = float(label_harm.get("no_eraser", policy.harm.get("no_eraser", float("nan"))))
                harm_er_decision = float(label_harm.get("eraser", policy.harm.get("eraser", float("nan"))))
                trace = decision_policy.decide(shift_info, harm_no=harm_no_decision, harm_er=harm_er_decision)
                decision = trace.action
                reason = trace.reason
            else:
                decision = policy.decide(shift_info, destructive=destructive)
                reason = "legacy_policy"

            sample_harm_no = sample_harm_multilabel(
                probs,
                labels,
                thresholds=clinical_thresholds,
                cost_fn=policy_cfg.cost_fn,
                cost_fp=policy_cfg.cost_fp,
                critical_idx=None,
            )
            sample_harm_er = sample_harm_multilabel(
                probs_er,
                labels,
                thresholds=clinical_thresholds,
                cost_fn=policy_cfg.cost_fn,
                cost_fp=policy_cfg.cost_fp,
                critical_idx=None,
            )
            sample_harm_rec = sample_harm_multilabel(
                probs_rec,
                labels,
                thresholds=clinical_thresholds,
                cost_fn=policy_cfg.cost_fn,
                cost_fp=policy_cfg.cost_fp,
                critical_idx=None,
            )
            if decision == "apply_eraser":
                sample_harm_policy = sample_harm_er
            elif decision == "recalibrate":
                sample_harm_policy = sample_harm_rec
            elif decision == "no_eraser":
                sample_harm_policy = sample_harm_no
            else:
                sample_harm_policy = np.full_like(sample_harm_no, float(policy_cfg.abstain_cost))
            sample_harm_oracle = np.minimum.reduce(
                [
                    sample_harm_no,
                    sample_harm_er,
                    sample_harm_rec,
                    np.full_like(sample_harm_no, float(policy_cfg.abstain_cost)),
                ]
            )
            sample_regret = sample_harm_policy - sample_harm_oracle
            if len(eval_group_ids) != len(sample_harm_policy):
                eval_group_ids = [str(i) for i in range(len(sample_harm_policy))]
            if shift_type == "mixed":
                scenario_name = str(intensity)
            else:
                scenario_name = f"{float(intensity):.6f}"
            scenario_key = f"{shift_type}|{scenario_name}"
            sample_groups = np.asarray([f"{scenario_key}|{g}" for g in eval_group_ids], dtype=object)
            regret_vals_all.append(np.asarray(sample_regret, dtype=float))
            regret_groups_all.append(sample_groups)
            regret_vals_by_shift.setdefault(str(shift_type), []).append(np.asarray(sample_regret, dtype=float))
            regret_groups_by_shift.setdefault(str(shift_type), []).append(sample_groups)
            regret_vals_by_action.setdefault(str(decision), []).append(np.asarray(sample_regret, dtype=float))
            regret_groups_by_action.setdefault(str(decision), []).append(sample_groups)
            boot_harm = cluster_bootstrap_mean(
                sample_harm_policy,
                np.asarray(eval_group_ids, dtype=object),
                n_boot=args.bootstrap_samples,
                seed=seed + 2009,
            )
            boot_regret = cluster_bootstrap_mean(
                sample_regret,
                np.asarray(eval_group_ids, dtype=object),
                n_boot=args.bootstrap_samples,
                seed=seed + 2011,
            )
            decision_row = {"shift": shift_type, "intensity": intensity, **shift_info, "decision": decision, "reason": reason}
            decision_row["decision_harm_source"] = "calib_policy_harm" if decision_policy is not None else "legacy_policy"
            if decision_policy is not None:
                decision_row["decision_harm_no_calib"] = float(harm_no_decision)
                decision_row["decision_harm_eraser_calib"] = float(harm_er_decision)
            decision_row.update(
                policy_regret(
                    decision,
                    harm_no_all,
                    harm_er_all,
                    abstain_cost=policy_cfg.abstain_cost,
                    harm_recalibrate=harm_rec_all,
                )
            )
            if target_priors_est is not None:
                decision_row["recalibrated_target_prevalence_mean"] = float(np.mean(target_priors_est))
            decision_row.update(
                {
                    "policy_harm_boot_mean": boot_harm["mean"],
                    "policy_harm_boot_ci_low": boot_harm["ci_low"],
                    "policy_harm_boot_ci_high": boot_harm["ci_high"],
                    "policy_regret_boot_mean": boot_regret["mean"],
                    "policy_regret_boot_ci_low": boot_regret["ci_low"],
                    "policy_regret_boot_ci_high": boot_regret["ci_high"],
                }
            )
            if shift_type == "mixed":
                decision_row["profile"] = intensity
            decisions.append(decision_row)
            harm_no_hist.append(harm_no_all)
            harm_er_hist.append(harm_er_all)
            harm_rec_hist.append(harm_rec_all)
            bootstrap_rows.append(
                {
                    "shift": shift_type,
                    "intensity": intensity,
                    "decision": decision,
                    "policy_harm_mean": boot_harm["mean"],
                    "policy_harm_ci_low": boot_harm["ci_low"],
                    "policy_harm_ci_high": boot_harm["ci_high"],
                    "policy_regret_mean": boot_regret["mean"],
                    "policy_regret_ci_low": boot_regret["ci_low"],
                    "policy_regret_ci_high": boot_regret["ci_high"],
                    "n": boot_harm["n"],
                    "n_clusters": boot_harm["n_clusters"],
                }
            )

            if decision == "apply_eraser":
                probs = probs_er
            elif decision == "recalibrate":
                probs = probs_rec

            accepted_probs = None
            accepted_labels = None
            if decision == "abstain":
                row = {"shift": shift_type, "intensity": intensity, "decision": decision, "coverage": 0.0}
            else:
                action = "eraser" if decision == "apply_eraser" else "no_eraser"
                mask = policy.apply_threshold(probs, action)
                if mask.sum() == 0:
                    row = {"shift": shift_type, "intensity": intensity, "decision": decision, "coverage": 0.0}
                else:
                    accepted_probs = probs[mask]
                    accepted_labels = labels[mask]
                    m = eval_metrics(
                        accepted_probs,
                        accepted_labels,
                        coverage=policy_cfg.coverage_budget,
                        thresholds=clinical_thresholds,
                    )
                    row = {"shift": shift_type, "intensity": intensity, "decision": decision, "coverage": float(mask.mean()), **m}
            for base in calib_target_defs:
                lname = str(base["label"])
                lidx = label_to_idx.get(lname)
                if lidx is None:
                    continue
                if accepted_probs is None or accepted_labels is None:
                    scen_stats = {
                        "n_valid": 0.0,
                        "threshold": float(base["threshold"]),
                        "sensitivity": float("nan"),
                        "specificity": float("nan"),
                        "ppv": float("nan"),
                        "npv": float("nan"),
                        "prevalence": float("nan"),
                    }
                else:
                    scen_stats = operating_point_stats(
                        accepted_labels[:, lidx],
                        accepted_probs[:, lidx],
                        threshold=float(base["threshold"]),
                    )
                rec = {
                    "shift": shift_type,
                    "intensity": intensity,
                    "decision": decision,
                    "label": lname,
                    "target_kind": str(base["target_kind"]),
                    "target_value": float(base["target_value"]),
                    "threshold": float(base["threshold"]),
                    "accepted_n_valid": float(scen_stats["n_valid"]),
                    "calib_n_valid": float(base.get("n_valid", float("nan"))),
                }
                for key in ["sensitivity", "specificity", "ppv", "npv", "prevalence"]:
                    s_val = float(scen_stats.get(key, float("nan")))
                    b_val = float(base.get(key, float("nan")))
                    rec[f"accepted_{key}"] = s_val
                    rec[f"calib_{key}"] = b_val
                    rec[f"delta_{key}_vs_calib"] = (
                        s_val - b_val if np.isfinite(s_val) and np.isfinite(b_val) else float("nan")
                    )
                op_target_rows.append(rec)
            if shift_type == "mixed":
                row["profile"] = intensity
            rows.append(row)

    import pandas as pd
    pd.DataFrame(rows).to_csv(out_dir / "external_policy_metrics.csv", index=False)
    pd.DataFrame(decisions).to_csv(out_dir / "external_policy_decisions.csv", index=False)
    if bootstrap_rows:
        pd.DataFrame(bootstrap_rows).to_csv(out_dir / "external_policy_bootstrap.csv", index=False)
    if op_target_rows:
        op_target_df = pd.DataFrame(op_target_rows)
        op_target_df.to_csv(out_dir / "external_policy_operating_points_targeted.csv", index=False)
        drift_cols = [
            "delta_sensitivity_vs_calib",
            "delta_specificity_vs_calib",
            "delta_ppv_vs_calib",
            "delta_npv_vs_calib",
        ]
        drift_summary = (
            op_target_df.groupby(["label", "target_kind", "target_value"], dropna=False)[drift_cols]
            .agg(["mean", "std", "max", "min"])
            .reset_index()
        )
        rows_target = []
        for _, r in drift_summary.iterrows():
            rec = {
                "label": str(r["label"]),
                "target_kind": str(r["target_kind"]),
                "target_value": float(r["target_value"]),
            }
            for col in drift_cols:
                for stat in ["mean", "std", "max", "min"]:
                    rec[f"{col}_{stat}"] = float(r[(col, stat)])
            rows_target.append(rec)
        (out_dir / "external_policy_operating_points_targeted_summary.json").write_text(
            json.dumps(
                {
                    "n_rows": int(len(op_target_df)),
                    "target_sensitivities": [float(v) for v in target_sensitivities],
                    "target_specificities": [float(v) for v in target_specificities],
                    "drift_stats": rows_target,
                },
                indent=2,
            )
        )
    summary = regret_by_shift(
        decisions,
        harm_no_hist,
        harm_er_hist,
        abstain_cost=policy_cfg.abstain_cost,
        harms_recalibrate=harm_rec_hist,
    )
    if regret_vals_all:
        boot_overall = cluster_bootstrap_mean(
            np.concatenate(regret_vals_all, axis=0),
            np.concatenate(regret_groups_all, axis=0),
            n_boot=args.bootstrap_samples,
            seed=seed + 4001,
        )
        boot_by_shift = {
            k: cluster_bootstrap_mean(
                np.concatenate(v, axis=0),
                np.concatenate(regret_groups_by_shift[k], axis=0),
                n_boot=args.bootstrap_samples,
                seed=seed + 4011 + i,
            )
            for i, (k, v) in enumerate(sorted(regret_vals_by_shift.items()))
        }
        boot_by_action = {
            k: cluster_bootstrap_mean(
                np.concatenate(v, axis=0),
                np.concatenate(regret_groups_by_action[k], axis=0),
                n_boot=args.bootstrap_samples,
                seed=seed + 4111 + i,
            )
            for i, (k, v) in enumerate(sorted(regret_vals_by_action.items()))
        }
    else:
        boot_overall = {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n": 0.0, "n_clusters": 0.0, "n_boot": float(args.bootstrap_samples)}
        boot_by_shift = {}
        boot_by_action = {}
    summary.update(
        {
            "regret_boot_mean": float(boot_overall["mean"]),
            "regret_boot_ci_low": float(boot_overall["ci_low"]),
            "regret_boot_ci_high": float(boot_overall["ci_high"]),
            "regret_boot_n": int(boot_overall["n"]),
            "regret_boot_n_clusters": int(boot_overall["n_clusters"]),
        }
    )
    (out_dir / "external_policy_regret_summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "external_policy_regret_decomposition.json").write_text(
        json.dumps(
            {
                "bootstrap_ci": {
                    "overall": boot_overall,
                    "by_shift": boot_by_shift,
                    "by_action": boot_by_action,
                }
            },
            indent=2,
        )
    )
    write_reprovenance_bundle(
        out_dir,
        cfg=cfg,
        config_path=args.config,
        artifact_manifest={
            "artifact_type": "external_policy_eval",
            "eval_sources": eval_sources,
            "shift_families": shift_types,
            "encoder_fingerprint": expected_calib_fp,
            "preprocess_signature": preprocess_signature(cfg),
            "head_sha256": sha256_file(Path(enc_cfg.head_path)) if enc_cfg.head_path and Path(enc_cfg.head_path).exists() else None,
            "realism_sha256": sha256_file(Path(cfg.intervention.realism_path))
            if cfg.intervention.realism_path is not None and Path(cfg.intervention.realism_path).exists()
            else None,
            "label_harmonization_sha256": sha256_file(out_dir / "label_harmonization_manifest.json")
            if (out_dir / "label_harmonization_manifest.json").exists()
            else None,
        },
        input_artifact_paths={
            "config": args.config,
            "eraser": args.eraser,
            "policy_json": args.policy_json,
            "head": enc_cfg.head_path,
            "realism": cfg.intervention.realism_path,
            "label_harmonization_manifest": out_dir / "label_harmonization_manifest.json",
        },
    )


if __name__ == "__main__":
    main()
