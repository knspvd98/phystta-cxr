#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

from phystta.config.loader import load_config
from phystta.data import DataEngine
from phystta.shifts import ShiftManager
from phystta.models import EncoderFactory, EncoderWrapper, build_encoder_fingerprint, resolve_head_contract
from phystta.intervention import (
    eraser_fingerprint_mismatches,
    load_eraser_collection,
    resolve_eraser_for_shift,
    TokenEraserWrapper,
    MixtureEraserAdaptor,
    FastSteeringAdaptor,
    EntropySteeringAdaptor,
    load_eraser_bank,
    SteeringEraser,
    TaskPreservingEraser,
)
from phystta.intervention import ProbeSuiteConfig, compute_erasure_certificate
from phystta.intervention import (
    probe_spuriousness_diagnostic,
    eraser_subspace_stability_diagnostic,
    extract_eraser_basis,
)
from phystta.safety.shift_id import rbf_mmd, energy_distance
from phystta.metrics import (
    expected_calibration_error,
    brier_score,
    reliability_curve,
    auroc_per_class,
    auprc_per_class,
    macro_average,
    selective_risk,
    invariance_metrics,
    aipw_from_potential_outcomes,
    worst_group_auc,
    underdiagnosis_gap,
    ece_gap,
    bin_age,
    optimal_thresholds,
    risk_coverage_curve_multilabel,
)
from phystta.viz import build_erasure_eval_figures
from phystta.shifts.realism import (
    calibrator_fingerprint_mismatches,
    intensity_grid_from_calibrator,
    load_calibrator,
    scenario_grid_from_calibrator,
)
from phystta.utils import (
    set_global_seed,
    tensor_stats,
    preprocess_signature,
    require_group_key,
    sample_ids_from_batch,
    load_torch_artifact,
    write_reprovenance_bundle,
    sha256_file,
    cluster_bootstrap_statistic,
)
from phystta.causal import paired_effect_ci, dose_response_slope
from phystta.metrics.risk import risk_coverage_curve, risk_coverage_inputs_multilabel


def parse_list(val: str) -> List[str]:
    return [v.strip() for v in val.split(",") if v.strip()]


def parse_floats(val: str) -> List[float]:
    return [float(v.strip()) for v in val.split(",") if v.strip()]


def eval_metrics(probs: np.ndarray, labels: np.ndarray, coverage: float = 0.8, label_cols=None, clinical_classes=None) -> Dict[str, float]:
    valid = labels >= 0
    probs_flat = probs[valid].reshape(-1)
    labels_flat = labels[valid].reshape(-1)
    preds = (probs_flat >= 0.5).astype(float)
    errors = preds != labels_flat
    conf = np.maximum(probs_flat, 1 - probs_flat)
    if len(conf) == 0:
        coverage_realized = float("nan")
        abstain_rate = float("nan")
    else:
        target_cov = float(np.clip(coverage, 0.0, 1.0))
        quant = np.quantile(conf, max(0.0, 1.0 - target_cov))
        accept = conf >= quant
        coverage_realized = float(np.mean(accept))
        abstain_rate = float(1.0 - coverage_realized)
    metrics = {
        "ece": expected_calibration_error(probs_flat, labels_flat, n_bins=15),
        "brier": brier_score(probs_flat, labels_flat),
        "risk_at_coverage": selective_risk(errors.astype(float), conf, coverage=coverage),
        "coverage_realized": coverage_realized,
        "abstain_rate": abstain_rate,
    }
    try:
        auroc = auroc_per_class(labels, probs)
        auprc = auprc_per_class(labels, probs)
        metrics["auroc_macro"] = macro_average(auroc)
        metrics["auprc_macro"] = macro_average(auprc)
        if clinical_classes and label_cols:
            for name in clinical_classes:
                if name in label_cols:
                    idx = label_cols.index(name)
                    metrics[f"auroc_{name}"] = auroc.get(idx, float("nan"))
                    metrics[f"auprc_{name}"] = auprc.get(idx, float("nan"))
    except Exception:
        metrics["auroc_macro"] = float("nan")
        metrics["auprc_macro"] = float("nan")
    return metrics


def eval_metrics_with_uncertainty(
    probs: np.ndarray,
    labels: np.ndarray,
    *,
    groups: np.ndarray,
    coverage: float = 0.8,
    label_cols=None,
    clinical_classes=None,
    thresholds: np.ndarray | None = None,
    cost_fn: float = 1.0,
    cost_fp: float = 1.0,
    bootstrap_samples: int = 200,
    risk_curve_points: int = 50,
    seed: int = 2024,
) -> tuple[Dict[str, float], Dict[str, np.ndarray]]:
    metrics = eval_metrics(
        probs,
        labels,
        coverage=coverage,
        label_cols=label_cols,
        clinical_classes=clinical_classes,
    )
    metrics["expected_harm"] = expected_harm(probs, labels, thresholds, cost_fn, cost_fp)
    curve_errors, curve_conf = risk_coverage_inputs_multilabel(
        probs,
        labels,
        threshold=thresholds if thresholds is not None else 0.5,
        cost_fn=cost_fn,
        cost_fp=cost_fp,
    )
    curve = risk_coverage_curve(curve_errors, curve_conf, points=risk_curve_points)
    metric_names = [
        "auroc_macro",
        "auprc_macro",
        "ece",
        "brier",
        "risk_at_coverage",
        "expected_harm",
        "coverage_realized",
        "abstain_rate",
    ]
    g = np.asarray(groups).reshape(-1).astype(str)
    if len(np.unique(g)) > 1 and int(bootstrap_samples) > 1:
        def _metric_stat(idx: np.ndarray) -> np.ndarray:
            sub = eval_metrics(
                probs[idx],
                labels[idx],
                coverage=coverage,
                label_cols=label_cols,
                clinical_classes=clinical_classes,
            )
            sub["expected_harm"] = expected_harm(probs[idx], labels[idx], thresholds, cost_fn, cost_fp)
            return np.asarray([sub.get(name, np.nan) for name in metric_names], dtype=float)

        boot = cluster_bootstrap_statistic(
            g,
            _metric_stat,
            n_boot=bootstrap_samples,
            seed=seed,
        )
        lo = np.asarray(boot["ci_low"], dtype=float)
        hi = np.asarray(boot["ci_high"], dtype=float)
        for idx, name in enumerate(metric_names):
            metrics[f"{name}_ci_low"] = float(lo[idx])
            metrics[f"{name}_ci_high"] = float(hi[idx])

        def _curve_stat(idx: np.ndarray) -> np.ndarray:
            sub_curve = risk_coverage_curve(
                curve_errors[idx],
                curve_conf[idx],
                points=risk_curve_points,
            )
            return np.concatenate(
                [
                    np.asarray(sub_curve["risk"], dtype=float),
                    np.asarray(sub_curve["coverage"], dtype=float),
                ],
                axis=0,
            )

        boot_curve = cluster_bootstrap_statistic(
            g,
            _curve_stat,
            n_boot=bootstrap_samples,
            seed=seed + 17,
        )
        curve_lo = np.asarray(boot_curve["ci_low"], dtype=float)
        curve_hi = np.asarray(boot_curve["ci_high"], dtype=float)
        n_points = int(risk_curve_points)
        curve["risk_ci_low"] = curve_lo[:n_points]
        curve["risk_ci_high"] = curve_hi[:n_points]
        curve["coverage_ci_low"] = curve_lo[n_points:]
        curve["coverage_ci_high"] = curve_hi[n_points:]
    else:
        for name in metric_names:
            metrics[f"{name}_ci_low"] = float("nan")
            metrics[f"{name}_ci_high"] = float("nan")
        curve["risk_ci_low"] = np.full_like(curve["risk"], np.nan, dtype=float)
        curve["risk_ci_high"] = np.full_like(curve["risk"], np.nan, dtype=float)
        curve["coverage_ci_low"] = np.full_like(curve["coverage"], np.nan, dtype=float)
        curve["coverage_ci_high"] = np.full_like(curve["coverage"], np.nan, dtype=float)
    return metrics, curve


def expected_harm(probs: np.ndarray, labels: np.ndarray, thresholds: np.ndarray, cost_fn: float, cost_fp: float) -> float:
    harms = []
    for i in range(probs.shape[1]):
        y = labels[:, i]
        p = probs[:, i]
        valid = y >= 0
        if valid.sum() == 0:
            continue
        th = float(thresholds[i]) if thresholds is not None else 0.5
        preds = (p[valid] >= th).astype(int)
        fn = ((preds == 0) & (y[valid] == 1)).astype(float)
        fp = ((preds == 1) & (y[valid] == 0)).astype(float)
        harms.append((cost_fn * fn + cost_fp * fp).mean())
    if not harms:
        return float("nan")
    return float(np.mean(harms))


def _sample_multilabel_error(probs: np.ndarray, labels: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    p = np.asarray(probs, dtype=float)
    y = np.asarray(labels, dtype=float)
    if p.shape != y.shape:
        raise ValueError("probs and labels shape mismatch")
    pred = (p >= threshold).astype(int)
    valid = y >= 0
    err = ((pred != y.astype(int)) & valid).astype(float)
    denom = np.maximum(valid.sum(axis=1), 1)
    return err.sum(axis=1) / denom


def compute_shift_scores(z_batch: torch.Tensor, ref_feats: np.ndarray, method: str = "mmd") -> torch.Tensor:
    if ref_feats is None:
        raise RuntimeError("Reference features required for online adaptor.")
    zb = z_batch.detach().cpu().numpy()
    if method == "mmd":
        score = rbf_mmd(zb, ref_feats)
    elif method == "energy":
        score = energy_distance(zb, ref_feats)
    elif method == "c2st":
        try:
            from sklearn.linear_model import LogisticRegression
        except Exception as e:
            raise RuntimeError("scikit-learn required for c2st scoring") from e
        X = np.concatenate([zb, ref_feats], axis=0)
        y = np.concatenate([np.ones(len(zb)), np.zeros(len(ref_feats))], axis=0)
        clf = LogisticRegression(max_iter=200)
        clf.fit(X, y)
        score = float(clf.score(X, y))
    else:
        raise RuntimeError(f"Unknown online score method: {method}")
    scores = torch.full((z_batch.shape[0], 1), float(score), device=z_batch.device, dtype=z_batch.dtype)
    return scores


def collect_feature_pairs(loader, encoder: EncoderWrapper, shift_manager: ShiftManager, shift_type: str, intensity: float, device: str, group_key: str):
    encoder.eval()
    feats_clean = []
    feats_shift = []
    groups = []
    meta_rows = []
    labels = []
    with torch.no_grad():
        for batch in loader:
            x = batch["image"].to(device)
            labels.append(batch["labels"].detach().cpu().numpy())
            sample_ids = sample_ids_from_batch(batch)
            x_shift = shift_manager.apply_batch(x, shift_type, float(intensity), sample_ids=sample_ids)
            z_clean = encoder.encode(x)
            z_shift = encoder.encode(x_shift)
            feats_clean.append(z_clean.detach().cpu().numpy())
            feats_shift.append(z_shift.detach().cpu().numpy())
            n = z_shift.shape[0]
            if group_key in batch:
                groups.extend(batch[group_key])
            else:
                groups.extend(["unknown"] * z_shift.shape[0])
            meta_rows.append(
                pd.DataFrame(
                    {
                        "view_family": list(batch.get("view_family", ["unknown"] * n)),
                        "procedure_context": list(batch.get("procedure_context", ["unknown"] * n)),
                        "orientation_context": list(batch.get("orientation_context", ["unknown"] * n)),
                    }
                )
            )
    meta_df = pd.concat(meta_rows, ignore_index=True) if meta_rows else pd.DataFrame()
    labels_arr = np.concatenate(labels, axis=0) if labels else np.empty((0, 0), dtype=float)
    return np.concatenate(feats_clean, axis=0), np.concatenate(feats_shift, axis=0), np.array(groups), meta_df, labels_arr


def collect_baseline(loader, encoder: EncoderWrapper, head, device: str, group_key: Optional[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    feats = []
    probs = []
    groups = []
    with torch.no_grad():
        for batch in loader:
            x = batch["image"].to(device)
            z = encoder.encode(x)
            logits = head(z)
            feats.append(z.detach().cpu().numpy())
            probs.append(torch.sigmoid(logits).detach().cpu().numpy())
            if group_key and group_key in batch:
                groups.extend(batch[group_key])
            else:
                groups.extend(["unknown"] * z.shape[0])
    return np.concatenate(feats, axis=0), np.concatenate(probs, axis=0), np.array(groups)


def _get_group_arrays(batch, group_names: List[str]) -> Dict[str, np.ndarray]:
    groups = {}
    for key in group_names:
        if key in batch:
            groups[key] = np.array(batch[key])
    return groups


def _normalize_groups(groups: Dict[str, np.ndarray], age_bins: List[int]) -> Dict[str, np.ndarray]:
    out = {}
    for key, arr in groups.items():
        if key == "age":
            try:
                arr_f = arr.astype(float)
            except Exception:
                arr_f = np.full(len(arr), np.nan, dtype=float)
            out[key] = bin_age(arr_f, age_bins)
        else:
            out[key] = arr.astype(str)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate representation surgery")
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default=None)
    parser.add_argument("--shift_type", default=None)
    parser.add_argument("--intensity_grid", default=None)
    parser.add_argument("--eraser", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--mode", default=None)
    parser.add_argument("--modes", default=None, help="comma-separated list")
    parser.add_argument("--scenario_mode", default=None, choices=["atomic_calibrated_eval", "mixed_stress_eval"])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--encoder_profile", default=None)
    parser.add_argument("--bootstrap_samples", type=int, default=None)
    parser.add_argument("--max_calib_batches", type=int, default=None)
    parser.add_argument("--max_eval_batches", type=int, default=None)
    parser.add_argument("--eval_batch_size", type=int, default=None)
    parser.add_argument("--eval_num_workers", type=int, default=None)
    parser.add_argument("--checkpoint_every_eval_batches", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else cfg.runtime.seed
    device = args.device or cfg.runtime.device
    set_global_seed(seed, deterministic=cfg.runtime.deterministic, benchmark=cfg.runtime.benchmark)

    split = args.split if args.split is not None else cfg.data.eval_split
    shift_types = [args.shift_type] if args.shift_type else cfg.erasure_eval.shift_types
    scenario_mode = (
        args.scenario_mode
        or getattr(cfg.erasure_eval, "scenario_mode", None)
        or ("mixed_stress_eval" if set(shift_types) == {"mixed"} else "atomic_calibrated_eval")
    )
    if scenario_mode == "atomic_calibrated_eval" and any(str(st) == "mixed" for st in shift_types):
        raise RuntimeError("scenario_mode=atomic_calibrated_eval forbids shift_type='mixed'. Use mixed_stress_eval instead.")
    if scenario_mode == "mixed_stress_eval" and any(str(st) != "mixed" for st in shift_types):
        raise RuntimeError("scenario_mode=mixed_stress_eval supports only shift_type='mixed'.")
    enc_cfg = EncoderFactory.resolve_encoder_config(cfg, profile=args.encoder_profile)
    expected_calib_fp = build_encoder_fingerprint(enc_cfg)
    expected_calib_fp["preprocess_signature"] = preprocess_signature(cfg)
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
        graded_only = scenario_mode == "atomic_calibrated_eval"
        intensity_map = intensity_grid_from_calibrator(calib, shift_types, graded_only=graded_only)
        scenario_map = scenario_grid_from_calibrator(calib, shift_types, graded_only=graded_only)
    else:
        grid = parse_floats(args.intensity_grid) if args.intensity_grid else cfg.erasure_eval.intensity_grid
        intensity_map = {st: grid for st in shift_types}
        scenario_map = {st: [(float("nan"), float(i)) for i in grid] for st in shift_types}
    modes = []
    if args.modes:
        modes = parse_list(args.modes)
    elif args.mode:
        modes = [args.mode]
    elif getattr(cfg.erasure_eval, "modes", None):
        modes = list(cfg.erasure_eval.modes)
    else:
        modes = [cfg.erasure_eval.mode]
    bootstrap_samples = int(
        args.bootstrap_samples
        if args.bootstrap_samples is not None
        else getattr(cfg.erasure_eval, "bootstrap_samples", 200)
    )

    data_engine = DataEngine(cfg.data).attach_runtime_contract(cfg)
    data_engine.build_index()
    data_engine.create_splits()
    splits = [split]
    available_splits = set(data_engine.index["split"].unique()) if data_engine.index is not None else set()
    if getattr(cfg.data, "calib_split", None) and cfg.data.calib_split in available_splits:
        splits.append(cfg.data.calib_split)
    eval_batch_size = int(args.eval_batch_size) if args.eval_batch_size is not None else int(cfg.train.batch_size)
    eval_num_workers = int(args.eval_num_workers) if args.eval_num_workers is not None else int(cfg.train.num_workers)
    loaders = data_engine.build_dataloaders(
        batch_size=eval_batch_size,
        num_workers=eval_num_workers,
        splits=splits,
    )
    loader = loaders[split]
    calib_loader = loaders.get(cfg.data.calib_split) if getattr(cfg.data, "calib_split", None) else None
    print(
        f"eval_runtime batch_size={eval_batch_size} "
        f"num_workers={eval_num_workers} "
        f"checkpoint_every_eval_batches={args.checkpoint_every_eval_batches}",
        flush=True,
    )
    causal_group_key = None
    if "causal_invariance" in modes or "all" in modes:
        try:
            split_df = data_engine.get_split_df(split)
            causal_group_key = require_group_key(split_df, preferred=cfg.intervention.group_key)
        except Exception as e:
            log_hint = f"causal_group_key_unavailable={e}"
            # logger is defined later; keep message for deferred logging
            causal_group_key = f"__UNAVAILABLE__:{log_hint}"

    shift_manager = ShiftManager(seed=cfg.shifts.seed)
    encoder = EncoderFactory.from_config(cfg, profile=args.encoder_profile)
    if hasattr(encoder, "model"):
        encoder.model.to(device)

    strict_meta = bool(cfg.runtime.freeze_artifacts or getattr(cfg.experiment, "tier", "dev") == "paper")
    head = None
    if enc_cfg.head_path:
        head_contract = resolve_head_contract(
            cfg,
            label_cols=data_engine.label_cols,
            encoder_profile=args.encoder_profile,
            require_head_path=False,
            strict_meta=strict_meta,
            load_artifact=True,
        )
        head = head_contract.head_artifact
    elif enc_cfg.encoder_source == "baseline_cnn":
        if enc_cfg.head_attr and hasattr(encoder.model, enc_cfg.head_attr):
            head = getattr(encoder.model, enc_cfg.head_attr)
        elif hasattr(encoder, "model") and hasattr(encoder.model, "head"):
            head = encoder.model.head
        elif hasattr(encoder, "model") and hasattr(encoder.model, "classifier"):
            head = encoder.model.classifier
        else:
            raise RuntimeError("Baseline encoder head not found. Provide head_path or head_attr.")
    else:
        raise RuntimeError("Non-baseline encoder requires head_path. Train with scripts/train_linear_head.py first.")
    if hasattr(head, "to"):
        head.to(device)
    if hasattr(head, "eval"):
        head.eval()

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
    token_wrappers: Dict[str, TokenEraserWrapper] = {}
    token_mode_enabled = bool(getattr(cfg, "eraser", None) and cfg.eraser.token_mode != "none")

    def _base_eraser(obj):
        if isinstance(obj, TaskPreservingEraser):
            return obj.eraser
        return obj

    def _get_eraser_for_shift(shift_family: str):
        return resolve_eraser_for_shift(eraser_collection, shift_family)

    online_adaptor = None
    entropy_adaptor = None
    ref_feats = None
    if getattr(cfg, "eraser", None) and cfg.eraser.online_enabled:
        if eraser_bundle_meta.get("kind") == "bank":
            raise RuntimeError("online_enabled is not compatible with eraser bank manifests. Use a single eraser artifact.")
        if cfg.eraser.token_mode != "none":
            raise RuntimeError("online_enabled is not compatible with token_mode. Disable token erasure or online adaptor.")
        eraser = _get_eraser_for_shift(str(shift_types[0]))
        if cfg.eraser.online_mode == "entropy":
            base = _base_eraser(eraser)
            if not isinstance(base, SteeringEraser):
                raise RuntimeError("online_mode=entropy requires SteeringEraser (eraser.type=steering).")
            entropy_adaptor = EntropySteeringAdaptor(
                base,
                head,
                steps=cfg.eraser.online_entropy_steps,
                lr=cfg.eraser.online_entropy_lr,
                guard_eps=cfg.eraser.online_guard_eps,
            ).to(device)
        else:
            if not cfg.eraser.online_bank or cfg.eraser.online_weights_path is None:
                raise RuntimeError("online_enabled requires eraser.online_bank and eraser.online_weights_path")
            bank = load_eraser_bank([str(p) for p in cfg.eraser.online_bank])
            for b in bank:
                if hasattr(b, "to"):
                    b.to(device)
            payload = load_torch_artifact(cfg.eraser.online_weights_path, map_location="cpu")
            if cfg.eraser.online_mode == "mixture":
                if not isinstance(payload, torch.nn.Module):
                    raise RuntimeError("online_weights_path must contain a torch.nn.Module for mixture mode.")
                online_adaptor = MixtureEraserAdaptor(bank, payload).to(device)
            elif cfg.eraser.online_mode == "fast":
                if not bank:
                    raise RuntimeError("online bank empty for fast mode.")
                probe_w = payload.get("probe_w") if isinstance(payload, dict) else None
                if probe_w is None:
                    raise RuntimeError("online_weights_path must contain probe_w for fast mode.")
                head_w = head.weight.detach() if hasattr(head, "weight") else None
                online_adaptor = FastSteeringAdaptor(bank[0], probe_w, head_w, ridge_lambda=cfg.eraser.online_ridge_lambda, guard_eps=cfg.eraser.online_guard_eps).to(device)
            else:
                raise RuntimeError(f"Unknown online_mode: {cfg.eraser.online_mode}")

        # build reference features from calib split
        if calib_loader is None:
            raise RuntimeError("online_enabled requires calib_split loader to build reference features.")
        feats = []
        with torch.no_grad():
            for batch in calib_loader:
                x = batch["image"].to(device)
                feats.append(encoder.encode(x).detach().cpu().numpy())
        ref_feats = np.concatenate(feats, axis=0) if feats else None

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = out_dir / "_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    scenario_bundle_dir = out_dir / "scenario_bundles"
    scenario_bundle_dir.mkdir(parents=True, exist_ok=True)
    stage_state_dir = out_dir.parent / ".stage_state"
    stage_state_dir.mkdir(parents=True, exist_ok=True)
    stage_state_path = stage_state_dir / "erasure_eval.json"
    log_path = out_dir / "log.txt"
    def log(msg: str):
        print(msg, flush=True)
        with log_path.open("a") as f:
            f.write(msg + "\n")

    def _json_default(obj):
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if hasattr(obj, "tolist"):
            try:
                return obj.tolist()
            except Exception:
                pass
        return str(obj)

    def _write_json(path: Path, payload):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, default=_json_default))

    def _write_stage(status: str, **extra):
        payload = {"status": status, "ts": time.time(), "out_dir": str(out_dir)}
        payload.update(extra)
        _write_json(stage_state_path, payload)

    def _scenario_key(shift_type, scenario_idx):
        return f"{str(shift_type)}__{int(scenario_idx):03d}"

    def _save_array(path: Path, arr):
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, arr)

    def _write_partial_scenario(
        shift_type,
        scenario_idx,
        eval_batch_idx,
        probs_b_list,
        probs_e_list,
        labels_b_list,
        labels_e_list,
        metric_groups_list,
        causal_groups_list,
    ):
        if not probs_b_list:
            return
        partial_dir = checkpoint_dir / "scenario_partials" / _scenario_key(shift_type, scenario_idx)
        partial_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            partial_dir / f"partial_{int(eval_batch_idx):05d}.npz",
            probs_b=np.concatenate(probs_b_list, axis=0),
            probs_e=np.concatenate(probs_e_list, axis=0),
            labels_b=np.concatenate(labels_b_list, axis=0),
            labels_e=np.concatenate(labels_e_list, axis=0),
            metric_groups=np.asarray(metric_groups_list).astype(str),
            causal_groups=np.asarray(causal_groups_list).astype(str),
        )
        _write_json(
            partial_dir / "progress.json",
            {
                "shift": str(shift_type),
                "scenario_idx": int(scenario_idx),
                "last_completed_batch": int(eval_batch_idx),
                "n_samples": int(sum(x.shape[0] for x in probs_b_list)),
            },
        )

    if isinstance(causal_group_key, str) and causal_group_key.startswith("__UNAVAILABLE__:"):
        log(causal_group_key.split(":", 1)[1])
        causal_group_key = None

    def encode_erased(x: torch.Tensor, shift_family: str, z_pre: torch.Tensor | None = None) -> torch.Tensor:
        if entropy_adaptor is not None:
            z = z_pre if z_pre is not None else encoder.encode(x)
            return entropy_adaptor(z)
        if online_adaptor is not None:
            z = z_pre if z_pre is not None else encoder.encode(x)
            scores = compute_shift_scores(z, ref_feats, method=cfg.eraser.online_scores)
            return online_adaptor(z, scores)
        eraser = _get_eraser_for_shift(shift_family)
        if token_mode_enabled:
            key = str(shift_family)
            if key not in token_wrappers:
                token_wrappers[key] = TokenEraserWrapper(
                    encoder,
                    eraser,
                    mode=cfg.eraser.token_mode,
                    token_layers=cfg.eraser.token_layers,
                ).to(device)
            return token_wrappers[key](x)
        z = z_pre if z_pre is not None else encoder.encode(x)
        return eraser(z)

    log(f"canonical_eval_path=encode->head vs encode->eraser->head")
    log(f"preprocess_signature={preprocess_signature(cfg)}")
    try:
        batch0 = next(iter(loader))
        raw_stats = tensor_stats(batch0["image"])
        adapted = encoder.preprocess(batch0["image"].to(device))
        adapted_stats = tensor_stats(adapted.detach().cpu())
        log(f"raw_stats={raw_stats}")
        log(f"adapter_stats={adapted_stats}")
    except Exception as e:
        log(f"stats_error={e}")

    # calibration thresholds for fairness/underdiagnosis on calib split
    thresholds = None
    if calib_loader is not None:
        probs_c = []
        labels_c = []
        calib_batches = 0
        calib_t0 = time.perf_counter()
        with torch.no_grad():
            for calib_batches, batch in enumerate(calib_loader, start=1):
                x = batch["image"].to(device)
                y = batch["labels"].to(device)
                z = encoder.encode(x)
                logits = head(z)
                probs_c.append(torch.sigmoid(logits).cpu().numpy())
                labels_c.append(y.cpu().numpy())
                if calib_batches == 1 or calib_batches % 100 == 0:
                    log(f"calib_pass_progress batches={calib_batches}")
                if args.max_calib_batches is not None and calib_batches >= int(args.max_calib_batches):
                    log(f"calib_pass_early_stop batches={calib_batches}")
                    break
        probs_c = np.concatenate(probs_c, axis=0)
        labels_c = np.concatenate(labels_c, axis=0)
        cost_fn = getattr(cfg.safety_policy, "cost_fn", 10.0) if hasattr(cfg, "safety_policy") else 10.0
        cost_fp = getattr(cfg.safety_policy, "cost_fp", 1.0) if hasattr(cfg, "safety_policy") else 1.0
        thresholds = optimal_thresholds(labels_c, probs_c, cost_fn=cost_fn, cost_fp=cost_fp)
        log(f"calib_thresholds=per-class (min_cost) using cost_fn={cost_fn} cost_fp={cost_fp}")
        log(f"calib_pass_done batches={calib_batches} samples={labels_c.shape[0]} sec={time.perf_counter() - calib_t0:.2f}")
        _save_array(checkpoint_dir / "calibration_probs.npy", probs_c)
        _save_array(checkpoint_dir / "calibration_labels.npy", labels_c)
        _save_array(checkpoint_dir / "calibration_thresholds.npy", np.asarray(thresholds))
        _write_json(
            checkpoint_dir / "calibration_state.json",
            {
                "status": "complete",
                "batches": int(calib_batches),
                "samples": int(labels_c.shape[0]),
                "cost_fn": float(cost_fn),
                "cost_fp": float(cost_fp),
            },
        )
        _write_stage("running", phase="calibration_complete", calib_batches=int(calib_batches), calib_samples=int(labels_c.shape[0]))
    else:
        log("calib_split not available; using threshold=0.5 for underdiagnosis metrics")

    rows = []
    fairness_rows = []
    causal_rows = []
    reliability_rows = []
    risk_curve_rows = []
    guarded_rows = []
    tradeoff_rows = []

    fairness_groups = getattr(cfg.erasure_eval, "fairness_groups", ["sex", "race", "age"])
    age_bins = getattr(cfg.erasure_eval, "age_bins", [40, 60])
    tradeoff_strengths = list(getattr(cfg.erasure_eval, "tradeoff_strengths", [0.0, 0.25, 0.5, 0.75, 1.0]))
    guarded_probe_strata = list(getattr(cfg.erasure_eval, "guarded_probe_strata", ["view_family", "procedure_context", "orientation_context"]))
    guarded_probe_min_samples = int(getattr(cfg.erasure_eval, "guarded_probe_min_samples", 24))
    stability_resamples = int(getattr(cfg.erasure_eval, "stability_resamples", 12))

    clean_probs_reference = []
    clean_ref_batches = 0
    clean_ref_t0 = time.perf_counter()
    with torch.no_grad():
        for clean_ref_batches, batch in enumerate(loader, start=1):
            x = batch["image"].to(device)
            z = encoder.encode(x)
            clean_probs_reference.append(torch.sigmoid(head(z)).cpu().numpy())
            if clean_ref_batches == 1 or clean_ref_batches % 100 == 0:
                log(f"clean_reference_progress batches={clean_ref_batches}")
            if args.max_eval_batches is not None and clean_ref_batches >= int(args.max_eval_batches):
                log(f"clean_reference_early_stop batches={clean_ref_batches}")
                break
    clean_probs_reference = np.concatenate(clean_probs_reference, axis=0) if clean_probs_reference else np.empty((0, 0), dtype=float)
    log(f"clean_reference_done batches={clean_ref_batches} samples={clean_probs_reference.shape[0] if clean_probs_reference.ndim > 0 else 0} sec={time.perf_counter() - clean_ref_t0:.2f}")
    _save_array(checkpoint_dir / "clean_reference_probs.npy", clean_probs_reference)
    _write_json(
        checkpoint_dir / "clean_reference_state.json",
        {
            "status": "complete",
            "batches": int(clean_ref_batches),
            "samples": int(clean_probs_reference.shape[0] if clean_probs_reference.ndim > 0 else 0),
        },
    )
    _write_stage(
        "running",
        phase="clean_reference_complete",
        clean_ref_batches=int(clean_ref_batches),
        clean_ref_samples=int(clean_probs_reference.shape[0] if clean_probs_reference.ndim > 0 else 0),
    )

    for shift_type in shift_types:
        probs_by_intensity_b = []
        probs_by_intensity_e = []
        probs_by_intensity_b_map = {}
        probs_by_intensity_e_map = {}
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
            grid = scenario_map.get(shift_type)
            if grid is None:
                raise RuntimeError(f"No calibrated intensity grid for shift_type={shift_type}")
            scenarios = [(float(intensity), None, float(level)) for level, intensity in grid]
        for scenario_idx, scenario in enumerate(scenarios, start=1):
            intensity = scenario[0]
            realism_level = scenario[2] if len(scenario) > 2 else float("nan")
            scenario_t0 = time.perf_counter()
            log(f"scenario_begin shift={shift_type} scenario_idx={scenario_idx} intensity={intensity} realism_level={realism_level}")
            feats_shift = []
            feats_erased = []
            probs_b = []
            labels_b = []
            probs_e = []
            labels_e = []
            group_cache = {k: [] for k in fairness_groups}
            causal_groups = []
            metric_groups = []
            with torch.no_grad():
                for eval_batch_idx, batch in enumerate(loader, start=1):
                    x = batch["image"].to(device)
                    y = batch["labels"].to(device)
                    if shift_type == "mixed":
                        sample_ids = sample_ids_from_batch(batch)
                        x_shift, _meta = shift_manager.apply_batch_scenario(x, scenario[1], sample_ids=sample_ids)
                    else:
                        sample_ids = sample_ids_from_batch(batch)
                        x_shift = shift_manager.apply_batch(x, shift_type, float(intensity), sample_ids=sample_ids)
                    z = encoder.encode(x_shift)
                    feats_shift.append(z.detach().cpu().numpy())
                    logits_b = head(z)
                    z_er = encode_erased(x_shift, shift_type, z_pre=z)
                    feats_erased.append(z_er.detach().cpu().numpy())
                    logits_e = head(z_er)
                    probs_b.append(torch.sigmoid(logits_b).cpu().numpy())
                    probs_e.append(torch.sigmoid(logits_e).cpu().numpy())
                    labels_b.append(y.cpu().numpy())
                    labels_e.append(y.cpu().numpy())
                    group_vals = _get_group_arrays(batch, fairness_groups)
                    for key, arr in group_vals.items():
                        group_cache[key].extend(arr)
                    if causal_group_key is not None and causal_group_key in batch:
                        causal_groups.extend(batch[causal_group_key])
                    else:
                        causal_groups.extend(["unknown"] * z.shape[0])
                    if "global_patient_id" in batch:
                        metric_groups.extend(batch["global_patient_id"])
                    elif "patient_id" in batch:
                        metric_groups.extend(batch["patient_id"])
                    else:
                        metric_groups.extend([f"sample_{len(metric_groups) + i}" for i in range(z.shape[0])])
                    if args.checkpoint_every_eval_batches is not None and eval_batch_idx % int(args.checkpoint_every_eval_batches) == 0:
                        _write_partial_scenario(
                            shift_type,
                            scenario_idx,
                            eval_batch_idx,
                            probs_b,
                            probs_e,
                            labels_b,
                            labels_e,
                            metric_groups,
                            causal_groups,
                        )
                    if eval_batch_idx == 1 or eval_batch_idx % 100 == 0:
                        log(f"scenario_progress shift={shift_type} scenario_idx={scenario_idx} batches={eval_batch_idx}")
                    if args.max_eval_batches is not None and eval_batch_idx >= int(args.max_eval_batches):
                        log(f"scenario_early_stop shift={shift_type} scenario_idx={scenario_idx} batches={eval_batch_idx}")
                        break
            feats_shift = np.concatenate(feats_shift, axis=0)
            feats_erased = np.concatenate(feats_erased, axis=0)
            probs_b = np.concatenate(probs_b, axis=0)
            probs_e = np.concatenate(probs_e, axis=0)
            labels_b = np.concatenate(labels_b, axis=0)
            labels_e = np.concatenate(labels_e, axis=0)
            causal_groups = np.asarray(causal_groups).astype(str)
            metric_groups = np.asarray(metric_groups).astype(str)
            log(f"scenario_done shift={shift_type} scenario_idx={scenario_idx} samples={probs_b.shape[0]} sec={time.perf_counter() - scenario_t0:.2f}")
            scenario_key = _scenario_key(shift_type, scenario_idx)
            scenario_dir = scenario_bundle_dir / scenario_key
            scenario_dir.mkdir(parents=True, exist_ok=True)
            _save_array(scenario_dir / "probs_no_eraser.npy", probs_b)
            _save_array(scenario_dir / "probs_eraser.npy", probs_e)
            _save_array(scenario_dir / "labels_no_eraser.npy", labels_b)
            _save_array(scenario_dir / "labels_eraser.npy", labels_e)
            _save_array(scenario_dir / "metric_groups.npy", metric_groups)
            _save_array(scenario_dir / "causal_groups.npy", causal_groups)
            _write_json(
                scenario_dir / "scenario_complete.json",
                {
                    "status": "complete",
                    "shift": str(shift_type),
                    "scenario_idx": int(scenario_idx),
                    "intensity": float(intensity) if not isinstance(intensity, str) else intensity,
                    "realism_level": float(realism_level) if np.isfinite(realism_level) else None,
                    "samples": int(probs_b.shape[0]),
                    "seconds": float(time.perf_counter() - scenario_t0),
                },
            )
            _write_stage(
                "running",
                phase="scenario_complete",
                shift=str(shift_type),
                scenario_idx=int(scenario_idx),
                samples=int(probs_b.shape[0]),
            )
            probs_by_intensity_b.append(probs_b)
            probs_by_intensity_e.append(probs_e)
            probs_by_intensity_b_map[float(intensity)] = probs_b
            probs_by_intensity_e_map[float(intensity)] = probs_e

            if "clinical" in modes or "all" in modes or "safety" in modes:
                cost_fn = getattr(cfg.safety_policy, "cost_fn", 10.0) if hasattr(cfg, "safety_policy") else 10.0
                cost_fp = getattr(cfg.safety_policy, "cost_fp", 1.0) if hasattr(cfg, "safety_policy") else 1.0
                mb, curve_b = eval_metrics_with_uncertainty(
                    probs_b,
                    labels_b,
                    groups=metric_groups,
                    coverage=cfg.erasure_eval.coverage,
                    label_cols=data_engine.label_cols,
                    clinical_classes=cfg.erasure_eval.clinical_classes,
                    thresholds=thresholds,
                    cost_fn=cost_fn,
                    cost_fp=cost_fp,
                    bootstrap_samples=bootstrap_samples,
                    risk_curve_points=int(cfg.metrics.risk_coverage_points),
                    seed=seed,
                )
                me, curve_e = eval_metrics_with_uncertainty(
                    probs_e,
                    labels_e,
                    groups=metric_groups,
                    coverage=cfg.erasure_eval.coverage,
                    label_cols=data_engine.label_cols,
                    clinical_classes=cfg.erasure_eval.clinical_classes,
                    thresholds=thresholds,
                    cost_fn=cost_fn,
                    cost_fp=cost_fp,
                    bootstrap_samples=bootstrap_samples,
                    risk_curve_points=int(cfg.metrics.risk_coverage_points),
                    seed=seed + 101,
                )
                row_b = {"method": "no_eraser", "shift": shift_type, "intensity": intensity, **mb}
                row_e = {"method": "eraser", "shift": shift_type, "intensity": intensity, **me}
                row_b["realism_level"] = realism_level
                row_e["realism_level"] = realism_level
                if shift_type == "mixed":
                    row_b["profile"] = intensity
                    row_e["profile"] = intensity
                rows.append(row_b)
                rows.append(row_e)
                scenario_label = (
                    f"{shift_type}:profile_{intensity}"
                    if shift_type == "mixed"
                    else (
                        f"{shift_type}:realism_{realism_level:.2f}"
                        if np.isfinite(realism_level)
                        else f"{shift_type}:intensity_{float(intensity):.2f}"
                    )
                )
                x_value = (
                    float(realism_level)
                    if np.isfinite(realism_level)
                    else (float(intensity) if shift_type != "mixed" else float("nan"))
                )
                for method_name, curve in [("no_eraser", curve_b), ("eraser", curve_e)]:
                    for coverage_val, risk_val, risk_lo, risk_hi, cov_lo, cov_hi in zip(
                        curve["coverage"],
                        curve["risk"],
                        curve["risk_ci_low"],
                        curve["risk_ci_high"],
                        curve["coverage_ci_low"],
                        curve["coverage_ci_high"],
                    ):
                        risk_curve_rows.append(
                            {
                                "method": method_name,
                                "shift": shift_type,
                                "scenario_label": scenario_label,
                                "intensity": intensity,
                                "realism_level": realism_level,
                                "profile": intensity if shift_type == "mixed" else None,
                                "x_value": x_value,
                                "coverage": float(coverage_val),
                                "risk": float(risk_val),
                                "risk_ci_low": float(risk_lo),
                                "risk_ci_high": float(risk_hi),
                                "coverage_ci_low": float(cov_lo),
                                "coverage_ci_high": float(cov_hi),
                            }
                        )
                if clean_probs_reference.shape == probs_b.shape and tradeoff_strengths:
                    delta = feats_erased - feats_shift
                    for alpha in tradeoff_strengths:
                        z_alpha = feats_shift + float(alpha) * delta
                        with torch.no_grad():
                            logits_alpha = head(torch.tensor(z_alpha, dtype=torch.float32, device=device))
                            probs_alpha = torch.sigmoid(logits_alpha).cpu().numpy()
                        m_alpha = eval_metrics(
                            probs_alpha,
                            labels_b,
                            coverage=cfg.erasure_eval.coverage,
                            label_cols=data_engine.label_cols,
                            clinical_classes=cfg.erasure_eval.clinical_classes,
                        )
                        m_alpha["expected_harm"] = expected_harm(probs_alpha, labels_b, thresholds, cost_fn, cost_fp)
                        invariance_delta = np.mean(np.abs(probs_alpha - clean_probs_reference), axis=1)
                        inv_boot = cluster_bootstrap_statistic(
                            metric_groups,
                            lambda idx: float(np.mean(invariance_delta[idx])),
                            n_boot=bootstrap_samples,
                            seed=seed + int(round(1000 * float(alpha))) + 307,
                        ) if len(np.unique(metric_groups)) > 1 and bootstrap_samples > 1 else {"ci_low": float("nan"), "ci_high": float("nan")}
                        tradeoff_rows.append(
                            {
                                "method": "eraser_tradeoff",
                                "shift": shift_type,
                                "scenario_label": scenario_label,
                                "intensity": intensity,
                                "realism_level": realism_level,
                                "profile": intensity if shift_type == "mixed" else None,
                                "strength": float(alpha),
                                "invariance_delta_vs_clean": float(np.mean(invariance_delta)),
                                "invariance_delta_vs_clean_ci_low": float(inv_boot.get("ci_low", float("nan"))),
                                "invariance_delta_vs_clean_ci_high": float(inv_boot.get("ci_high", float("nan"))),
                                **m_alpha,
                            }
                        )
                for method_name, probs_cur, labels_cur in [
                    ("no_eraser", probs_b, labels_b),
                    ("eraser", probs_e, labels_e),
                ]:
                    valid = labels_cur >= 0
                    probs_flat = probs_cur[valid].reshape(-1)
                    labels_flat = labels_cur[valid].reshape(-1)
                    if len(probs_flat) == 0:
                        continue
                    curve = reliability_curve(probs_flat, labels_flat, n_bins=12)
                    for conf_v, acc_v, count_v in zip(curve["confidence"], curve["accuracy"], curve["count"]):
                        reliability_rows.append(
                            {
                                "scenario_label": scenario_label,
                                "shift": shift_type,
                                "method": method_name,
                                "intensity": intensity,
                                "realism_level": realism_level,
                                "x_value": (
                                    float(realism_level)
                                    if np.isfinite(realism_level)
                                    else (float(intensity) if shift_type != "mixed" else float("nan"))
                                ),
                                "confidence": float(conf_v),
                                "accuracy": float(acc_v),
                                "count": int(count_v),
                            }
                        )
                log(f"{shift_type}@{intensity} no_eraser: {mb} eraser: {me}")

            if "causal_invariance" in modes or "all" in modes:
                try:
                    err_b = _sample_multilabel_error(probs_b, labels_b, threshold=0.5)
                    err_e = _sample_multilabel_error(probs_e, labels_e, threshold=0.5)
                    corr_b = 1.0 - err_b
                    corr_e = 1.0 - err_e
                    groups_dr = causal_groups if len(np.unique(causal_groups)) > 1 and set(np.unique(causal_groups)) != {"unknown"} else None
                    dr_corr = aipw_from_potential_outcomes(
                        feats_shift,
                        y0=corr_b,
                        y1=corr_e,
                        groups=groups_dr,
                        assign_prob=0.5,
                        n_splits=max(2, int(getattr(cfg.shift_id, "n_splits", 3))),
                        seed=seed,
                    )
                    dr_err = aipw_from_potential_outcomes(
                        feats_shift,
                        y0=err_b,
                        y1=err_e,
                        groups=groups_dr,
                        assign_prob=0.5,
                        n_splits=max(2, int(getattr(cfg.shift_id, "n_splits", 3))),
                        seed=seed + 17,
                    )
                    row_corr = {"method": "dr_effect", "shift": shift_type, "intensity": intensity, "effect_type": "aipw_randomized", "outcome": "correctness", **dr_corr}
                    row_err = {"method": "dr_effect", "shift": shift_type, "intensity": intensity, "effect_type": "aipw_randomized", "outcome": "error", **dr_err}
                    row_corr["realism_level"] = realism_level
                    row_err["realism_level"] = realism_level
                    if shift_type == "mixed":
                        row_corr["profile"] = intensity
                        row_err["profile"] = intensity
                    causal_rows.append(row_corr)
                    causal_rows.append(row_err)
                except Exception as e:
                    log(f"causal_dr_error={e}")

            if "fairness" in modes or "all" in modes:
                group_cache = _normalize_groups({k: np.array(v) for k, v in group_cache.items() if len(v) > 0}, age_bins)
                if not group_cache:
                    log("fairness skipped: metadata not available")
                else:
                    for gname, gvals in group_cache.items():
                        wb = worst_group_auc(labels_b, probs_b, gvals, min_group_size=cfg.metrics.min_group_size)
                        we = worst_group_auc(labels_e, probs_e, gvals, min_group_size=cfg.metrics.min_group_size)
                        fb = underdiagnosis_gap(labels_b, probs_b, gvals, threshold=thresholds if thresholds is not None else 0.5, min_group_size=cfg.metrics.min_group_size)
                        fe = underdiagnosis_gap(labels_e, probs_e, gvals, threshold=thresholds if thresholds is not None else 0.5, min_group_size=cfg.metrics.min_group_size)
                        eb = ece_gap(labels_b, probs_b, gvals, n_bins=cfg.metrics.calibration_bins, min_group_size=cfg.metrics.min_group_size)
                        ee = ece_gap(labels_e, probs_e, gvals, n_bins=cfg.metrics.calibration_bins, min_group_size=cfg.metrics.min_group_size)
                        row_b = {"method": "no_eraser", "shift": shift_type, "intensity": intensity, "group": gname, **wb, **fb, **eb}
                        row_e = {"method": "eraser", "shift": shift_type, "intensity": intensity, "group": gname, **we, **fe, **ee}
                        row_b["realism_level"] = realism_level
                        row_e["realism_level"] = realism_level
                        if shift_type == "mixed":
                            row_b["profile"] = intensity
                            row_e["profile"] = intensity
                        fairness_rows.append(row_b)
                        fairness_rows.append(row_e)

    if "causal_invariance" in modes or "all" in modes:
        try:
            probs_stack_b = np.stack(probs_by_intensity_b, axis=0)
            probs_stack_e = np.stack(probs_by_intensity_e, axis=0)
            cb = invariance_metrics(probs_stack_b)
            ce = invariance_metrics(probs_stack_e)
            row_b = {"method": "no_eraser", "shift": shift_type, **cb}
            row_e = {"method": "eraser", "shift": shift_type, **ce}
            row_b["group_key"] = causal_group_key if causal_group_key is not None else "unknown"
            row_e["group_key"] = causal_group_key if causal_group_key is not None else "unknown"
            row_b["realism_level"] = "aggregate"
            row_e["realism_level"] = "aggregate"
            if shift_type == "mixed":
                row_b["profile"] = "mixed_profiles"
                row_e["profile"] = "mixed_profiles"
            causal_rows.append(row_b)
            causal_rows.append(row_e)
        except Exception as e:
            log(f"causal_invariance_error={e}")

        # Paired intervention effect on invariance delta (no oracle treatment construction).
        try:
            if shift_type != "mixed":
                grid = intensity_map.get(shift_type)
                if grid is not None:
                    split_df = data_engine.get_split_df(split)
                    group_key = require_group_key(split_df, preferred=cfg.intervention.group_key)
                    groups0 = []
                    for batch in loader:
                        if group_key and group_key in batch:
                            groups0.extend(batch[group_key])
                        else:
                            n = int(batch["image"].shape[0])
                            groups0.extend(["unknown"] * n)
                    groups0 = np.asarray(groups0).astype(str)
                    p0 = clean_probs_reference
                    if p0.shape[0] != len(groups0):
                        raise RuntimeError(
                            f"clean_probs_reference length mismatch: probs={p0.shape[0]} groups={len(groups0)}"
                        )
                    slope_x = []
                    slope_y_no = []
                    slope_y_er = []
                    slope_g = []
                    p0_e = []
                    with torch.no_grad():
                        for batch in loader:
                            x = batch["image"].to(device)
                            logits_e = head(encode_erased(x, shift_type))
                            p0_e.append(torch.sigmoid(logits_e).cpu().numpy())
                    p0_e = np.concatenate(p0_e, axis=0)
                    if p0_e.shape[0] != len(groups0):
                        raise RuntimeError(
                            f"clean erased baseline length mismatch: probs={p0_e.shape[0]} groups={len(groups0)}"
                        )
                    for intensity in grid:
                        if float(intensity) == 0.0:
                            continue
                        p_shift = probs_by_intensity_b_map[float(intensity)]
                        p_shift_e = probs_by_intensity_e_map[float(intensity)]
                        # outcome: mean abs delta per sample against clean reference
                        y_delta = np.mean(np.abs(p_shift - p0), axis=1)
                        y_delta_e = np.mean(np.abs(p_shift_e - p0_e), axis=1)
                        res_b = paired_effect_ci(y_delta, groups0, seed=seed)
                        res_e = paired_effect_ci(y_delta_e, groups0, seed=seed)
                        realism_level_s = float("nan")
                        if cfg.intervention.use_realism and shift_type in scenario_map:
                            for lvl, inten in scenario_map[shift_type]:
                                if float(inten) == float(intensity):
                                    realism_level_s = float(lvl)
                                    break
                        causal_rows.append(
                            {
                                "method": "no_eraser",
                                "shift": shift_type,
                                "intensity": intensity,
                                "realism_level": realism_level_s,
                                "group_key": group_key,
                                "n_groups": int(len(np.unique(groups0.astype(str)))),
                                **res_b,
                                "effect_type": "paired_delta",
                            }
                        )
                        causal_rows.append(
                            {
                                "method": "eraser",
                                "shift": shift_type,
                                "intensity": intensity,
                                "realism_level": realism_level_s,
                                "group_key": group_key,
                                "n_groups": int(len(np.unique(groups0.astype(str)))),
                                **res_e,
                                "effect_type": "paired_delta",
                            }
                        )
                        slope_x.extend([float(intensity)] * len(y_delta))
                        slope_y_no.extend(y_delta.tolist())
                        slope_y_er.extend(y_delta_e.tolist())
                        slope_g.extend(groups0.astype(str).tolist())
                    if slope_x:
                        dr_no = dose_response_slope(np.array(slope_x), np.array(slope_y_no), np.array(slope_g))
                        dr_er = dose_response_slope(np.array(slope_x), np.array(slope_y_er), np.array(slope_g))
                        causal_rows.append(
                            {
                                "method": "no_eraser",
                                "shift": shift_type,
                                "intensity": "all",
                                "realism_level": "aggregate",
                                "group_key": group_key,
                                "n_groups": int(len(np.unique(np.array(slope_g, dtype=str)))),
                                **dr_no,
                                "effect_type": "dose_response_slope",
                            }
                        )
                        causal_rows.append(
                            {
                                "method": "eraser",
                                "shift": shift_type,
                                "intensity": "all",
                                "realism_level": "aggregate",
                                "group_key": group_key,
                                "n_groups": int(len(np.unique(np.array(slope_g, dtype=str)))),
                                **dr_er,
                                "effect_type": "dose_response_slope",
                            }
                        )
        except Exception as e:
            log(f"causal_effect_error={e}")

        if "probing" in modes or "all" in modes:
            try:
                if shift_type == "mixed":
                    log("probing skipped for mixed shift (use collect_probes with mixed_profiles)")
                else:
                    split_df = data_engine.get_split_df(split)
                    group_key = require_group_key(split_df, preferred=cfg.intervention.group_key)
                    grid = intensity_map.get(shift_type, cfg.erasure_eval.intensity_grid)
                    probe_intensity = float(grid[0])
                    feats_clean, feats_shift, groups, probe_meta_df, probe_labels = collect_feature_pairs(
                        loader,
                        encoder,
                        shift_manager,
                        shift_type,
                        probe_intensity,
                        device,
                        group_key,
                    )
                    active_eraser = _get_eraser_for_shift(shift_type)
                    if not token_mode_enabled and online_adaptor is None and entropy_adaptor is None:
                        z_clean_e = active_eraser(torch.tensor(feats_clean, device=device, dtype=torch.float32)).detach().cpu().numpy()
                        z_shift_e = active_eraser(torch.tensor(feats_shift, device=device, dtype=torch.float32)).detach().cpu().numpy()
                    else:
                        # recompute edited features via canonical encode->eraser path
                        feats_clean_e = []
                        feats_shift_e = []
                        with torch.no_grad():
                            for batch in loader:
                                x = batch["image"].to(device)
                                sample_ids = sample_ids_from_batch(batch)
                                x_shift = shift_manager.apply_batch(x, shift_type, probe_intensity, sample_ids=sample_ids)
                                feats_clean_e.append(encode_erased(x, shift_type).detach().cpu().numpy())
                                feats_shift_e.append(encode_erased(x_shift, shift_type).detach().cpu().numpy())
                        z_clean_e = np.concatenate(feats_clean_e, axis=0)
                        z_shift_e = np.concatenate(feats_shift_e, axis=0)
                    cert = compute_erasure_certificate(
                        feats_clean,
                        feats_shift,
                        z_clean_e,
                        z_shift_e,
                        groups=groups.astype(str),
                        target=cfg.intervention.probe_target,
                        probe_cfg=ProbeSuiteConfig(seed=seed),
                    )
                    row = {"method": "erasure_certificate", "shift": shift_type, "intensity": probe_intensity, **cert}
                    row["group_key"] = group_key
                    row["n_groups"] = int(len(np.unique(groups.astype(str))))
                    row["realism_level"] = (
                        float(scenario_map[shift_type][0][0])
                        if cfg.intervention.use_realism and shift_type in scenario_map and len(scenario_map[shift_type]) > 0
                        else float("nan")
                    )
                    rows.append(row)
                    spurious_summary, spurious_df = probe_spuriousness_diagnostic(
                        feats_clean,
                        feats_shift,
                        probe_labels,
                        groups=groups.astype(str),
                        metadata_df=probe_meta_df,
                        strata_keys=guarded_probe_strata,
                        min_samples=guarded_probe_min_samples,
                        seed=seed,
                        bootstrap_samples=bootstrap_samples,
                    )
                    stability_summary, stability_df = eraser_subspace_stability_diagnostic(
                        feats_shift,
                        z_shift_e,
                        groups=groups.astype(str),
                        base_basis=extract_eraser_basis(active_eraser),
                        n_resamples=stability_resamples,
                        k=int(getattr(cfg.eraser, "k", 8)),
                        seed=seed,
                    )
                    for _, grow in spurious_df.iterrows():
                        payload = grow.to_dict()
                        payload.update(
                            {
                                "shift": shift_type,
                                "intensity": probe_intensity,
                                "realism_level": row["realism_level"],
                            }
                        )
                        guarded_rows.append(payload)
                    for _, grow in stability_df.iterrows():
                        payload = grow.to_dict()
                        payload.update(
                            {
                                "shift": shift_type,
                                "intensity": probe_intensity,
                                "realism_level": row["realism_level"],
                            }
                        )
                        guarded_rows.append(payload)
                    guarded_rows.append(
                        {
                            "diagnostic": "probe_spuriousness_summary",
                            "shift": shift_type,
                            "intensity": probe_intensity,
                            "realism_level": row["realism_level"],
                            **spurious_summary,
                        }
                    )
                    guarded_rows.append(
                        {
                            "diagnostic": "eraser_stability_summary",
                            "shift": shift_type,
                            "intensity": probe_intensity,
                            "realism_level": row["realism_level"],
                            **stability_summary,
                        }
                    )
                    log(
                        f"probe-certificate {shift_type}@{probe_intensity} "
                        f"lin_auc {cert.get('detect_before_linear_auc', float('nan')):.3f}->{cert.get('detect_after_linear_auc', float('nan')):.3f} "
                        f"mlp_auc {cert.get('detect_before_mlp_auc', float('nan')):.3f}->{cert.get('detect_after_mlp_auc', float('nan')):.3f}"
                    )
            except Exception as e:
                log(f"probing_error={e}")
                guarded_rows.append(
                    {
                        "diagnostic": "guarded_probe_status",
                        "status": "error",
                        "shift": shift_type,
                        "intensity": float(probe_intensity) if "probe_intensity" in locals() else float("nan"),
                        "realism_level": float(realism_level) if "realism_level" in locals() else float("nan"),
                        "error": str(e),
                    }
                )

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "metrics.csv", index=False)
    (out_dir / "metrics.json").write_text(df.to_json(orient="records", indent=2))
    if fairness_rows:
        pd.DataFrame(fairness_rows).to_csv(out_dir / "fairness.csv", index=False)
    if causal_rows:
        pd.DataFrame(causal_rows).to_csv(out_dir / "causal_invariance.csv", index=False)
    if risk_curve_rows:
        pd.DataFrame(risk_curve_rows).to_csv(out_dir / "risk_coverage_curves.csv", index=False)
    guarded_df = pd.DataFrame(
        guarded_rows
        if guarded_rows
        else [
            {
                "diagnostic": "guarded_probe_status",
                "status": "empty",
                "shift": "",
                "intensity": float("nan"),
                "realism_level": float("nan"),
                "error": "",
            }
        ]
    )
    guarded_df.to_csv(out_dir / "guarded_erasure_diagnostics.csv", index=False)
    if tradeoff_rows:
        pd.DataFrame(tradeoff_rows).to_csv(out_dir / "erasure_tradeoff.csv", index=False)
    reliability_df = pd.DataFrame(reliability_rows)
    risk_curve_df = pd.DataFrame(risk_curve_rows)
    figure_outputs = build_erasure_eval_figures(
        df,
        out_dir=out_dir,
        scenario_mode=scenario_mode,
        reliability_df=reliability_df if not reliability_df.empty else None,
        risk_curve_df=risk_curve_df if not risk_curve_df.empty else None,
    )

    source_capabilities = {
        str(name): snap.as_dict() for name, snap in (data_engine.source_capabilities or {}).items()
    }
    write_reprovenance_bundle(
        out_dir,
        cfg=cfg,
        config_path=args.config,
        artifact_manifest={
            "artifact_type": "erasure_eval",
            "scenario_mode": scenario_mode,
            "shift_families": shift_types,
            "mixed_profiles": cfg.intervention.mixed_profiles,
            "source_capabilities": source_capabilities,
            "encoder_fingerprint": expected_calib_fp,
            "preprocess_signature": preprocess_signature(cfg),
            "figure_outputs": figure_outputs,
            "bootstrap_samples": bootstrap_samples,
            "eraser_selection_mode": eraser_bundle_meta.get("kind", "single"),
            "eraser_shift_families": sorted(str(v) for v in eraser_collection.keys() if str(v) != "__default__"),
            "realism_sha256": sha256_file(Path(cfg.intervention.realism_path))
            if cfg.intervention.realism_path is not None and Path(cfg.intervention.realism_path).exists()
            else None,
            "probe_sha256": None,
            "head_sha256": sha256_file(Path(enc_cfg.head_path)) if enc_cfg.head_path and Path(enc_cfg.head_path).exists() else None,
        },
        input_artifact_paths={
            "config": args.config,
            "eraser": args.eraser,
            "head": enc_cfg.head_path,
            "realism": cfg.intervention.realism_path,
        },
    )
    log(f"Saved results to {out_dir}")
    _write_stage(
        "completed",
        phase="finalize",
        metrics_rows=int(len(rows)),
        fairness_rows=int(len(fairness_rows)),
        causal_rows=int(len(causal_rows)),
        reliability_rows=int(len(reliability_rows)),
        risk_curve_rows=int(len(risk_curve_rows)),
    )


if __name__ == "__main__":
    main()
