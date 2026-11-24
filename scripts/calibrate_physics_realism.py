#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import pandas as pd

from phystta.config.loader import load_config, save_config
from phystta.data import DataEngine
from phystta.data.datasets import XRayDataset
from phystta.data.transforms import build_preprocess_transform
from phystta.models import EncoderFactory, build_encoder_fingerprint
from phystta.shifts import ShiftManager, shift_physics_semantics, physical_dependence_warnings, PARTIAL_REALISM_LIMITATIONS
from phystta.shifts.realism import RealismCalibrator, save_calibrator, build_realism_validity_summary
from phystta.viz import build_realism_calibration_figures
from phystta.utils import (
    set_global_seed,
    require_group_key,
    preprocess_signature,
    sample_ids_from_batch,
    write_reprovenance_bundle,
    sha256_file,
)
from phystta.causal import (
    annotate_pair_quality,
    PairQualityConfig,
    split_pairs_by_quality,
    ensure_pairs_group_key,
    summarize_pair_quality,
)


def _distances(z_a: np.ndarray, z_b: np.ndarray, metric: str = "l2") -> np.ndarray:
    if metric == "l2":
        return np.linalg.norm(z_a - z_b, axis=1)
    if metric == "cos":
        a = z_a / (np.linalg.norm(z_a, axis=1, keepdims=True) + 1e-8)
        b = z_b / (np.linalg.norm(z_b, axis=1, keepdims=True) + 1e-8)
        return 1.0 - np.sum(a * b, axis=1)
    raise ValueError(f"Unknown metric {metric}")


def _build_feature_map(df: pd.DataFrame, encoder, device: str, preprocess_cfg) -> Dict[str, np.ndarray]:
    label_cols = [c for c in df.columns if c not in {"image_path", "patient_id", "source", "study_id"}]
    for col in label_cols:
        df[col] = -1.0
    dataset = XRayDataset(df, label_cols=label_cols, transforms=build_preprocess_transform(preprocess_cfg), preprocess_cfg=preprocess_cfg)
    loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False)
    feats = {}
    with torch.no_grad():
        for batch in loader:
            x = batch["image"].to(device)
            z = encoder.encode(x).cpu().numpy()
            for path, feat in zip(batch["image_path"], z):
                feats[str(path)] = feat
    return feats


def _accepted_context_tokens(pairs_df: pd.DataFrame, stem: str) -> List[str]:
    values = []
    for suffix in ["_a", "_b"]:
        col = f"{stem}{suffix}"
        if col in pairs_df.columns:
            values.extend(pairs_df[col].fillna("unknown").astype(str).tolist())
    tokens = []
    for value in values:
        token = str(value).strip()
        if not token or token.lower() == "unknown":
            continue
        tokens.append(token)
    return sorted(set(tokens))


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate physics realism to real technical pairs")
    parser.add_argument("--config", required=True)
    parser.add_argument("--source", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--realism_levels", default="0.0,0.5,1.0")
    parser.add_argument("--shift_types", default=None)
    parser.add_argument("--max_pairs", type=int, default=500)
    parser.add_argument("--max_images", type=int, default=500)
    parser.add_argument("--metric", default="l2")
    parser.add_argument("--match_method", default="quantile", choices=["quantile", "mmd"])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--write_config", action="store_true")
    parser.add_argument("--encoder_profile", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else cfg.runtime.seed
    device = args.device or cfg.runtime.device
    set_global_seed(seed, deterministic=cfg.runtime.deterministic, benchmark=cfg.runtime.benchmark)

    data_engine = DataEngine(cfg.data).attach_runtime_contract(cfg)
    data_engine.build_index()
    data_engine.create_splits()

    source = args.source or (cfg.data.sources[0].domain_label or cfg.data.sources[0].name)
    calib_split = cfg.data.calib_split
    calib_df = data_engine.get_split_df(calib_split)
    group_key = require_group_key(calib_df, preferred=cfg.intervention.group_key)
    capability_snapshot = data_engine.get_source_capability_snapshot(source)
    source_columns_snapshot = data_engine.source_columns.get(str(source), [])
    index_sha256 = None
    if source in data_engine.index_paths and data_engine.index_paths[source].exists():
        index_sha256 = sha256_file(data_engine.index_paths[source])
    pairs_df = data_engine.build_paired_index(
        source,
        group_key=group_key,
        pair_mode=cfg.intervention.pair_mode,
    )
    pairs_df = ensure_pairs_group_key(pairs_df, group_key=group_key, default_source=source)
    allowed = set(calib_df[group_key].astype(str))
    n_allowed_groups = len(allowed)
    n_pairs_before = int(len(pairs_df))
    pairs_df = pairs_df[pairs_df[group_key].astype(str).isin(allowed)]
    if len(pairs_df) == 0:
        raise RuntimeError("No real technical pairs found for calibration split")
    pairs_df = pairs_df.sample(n=min(args.max_pairs, len(pairs_df)), random_state=seed)
    pq_cfg = PairQualityConfig(
        max_delta_hours=float(
            getattr(getattr(cfg.intervention, "pair_quality", None), "max_delta_hours", cfg.intervention.real_pairs_max_hours)
        ),
        label_change_weight=float(getattr(getattr(cfg.intervention, "pair_quality", None), "label_change_weight", 0.45)),
        view_change_weight=float(getattr(getattr(cfg.intervention, "pair_quality", None), "view_change_weight", 0.20)),
        context_change_weight=float(getattr(getattr(cfg.intervention, "pair_quality", None), "context_change_weight", 0.20)),
        sentinel_change_weight=float(getattr(getattr(cfg.intervention, "pair_quality", None), "sentinel_change_weight", 0.15)),
        technical_dominant_max_risk=float(
            getattr(getattr(cfg.intervention, "pair_quality", None), "technical_dominant_max_risk", 0.25)
        ),
    )
    pairs_df = annotate_pair_quality(
        pairs_df,
        index_df=data_engine.index,
        label_cols=data_engine.label_cols,
        cfg=pq_cfg,
    )
    pq = split_pairs_by_quality(pairs_df)
    pairs_td = pq["technical_dominant"]
    pairs_amb = pq["ambiguous"]
    pair_quality_summary = summarize_pair_quality(pairs_df)
    if len(pairs_td) == 0:
        raise RuntimeError("No technical-dominant pairs available for realism calibration.")
    # group-wise holdout split on technical-dominant pairs
    try:
        from sklearn.model_selection import GroupShuffleSplit
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        train_idx, hold_idx = next(gss.split(pairs_td, groups=pairs_td[group_key].astype(str)))
        pairs_train = pairs_td.iloc[train_idx]
        pairs_hold = pairs_td.iloc[hold_idx]
    except Exception:
        split = int(0.8 * len(pairs_td))
        pairs_train = pairs_td.iloc[:split]
        pairs_hold = pairs_td.iloc[split:]

    enc_cfg = EncoderFactory.resolve_encoder_config(cfg, profile=args.encoder_profile)
    encoder = EncoderFactory.from_config(cfg, profile=args.encoder_profile)
    if hasattr(encoder, "model"):
        encoder.model.to(device)
    encoder_fingerprint = build_encoder_fingerprint(enc_cfg)
    encoder_fingerprint["preprocess_signature"] = preprocess_signature(cfg)

    # Build features for unique images in pairs
    all_paths = pd.unique(pd.concat([pairs_td["image_path_a"], pairs_td["image_path_b"], pairs_hold["image_path_a"], pairs_hold["image_path_b"]]))
    df_images = pd.DataFrame({"image_path": all_paths, "patient_id": "unknown", "source": source})
    feats = _build_feature_map(df_images, encoder, device, cfg.data.preprocess)

    z_a = np.stack([feats[str(p)] for p in pairs_train["image_path_a"]])
    z_b = np.stack([feats[str(p)] for p in pairs_train["image_path_b"]])
    real_dists = _distances(z_a, z_b, metric=args.metric)
    z_a_h = np.stack([feats[str(p)] for p in pairs_hold["image_path_a"]])
    z_b_h = np.stack([feats[str(p)] for p in pairs_hold["image_path_b"]])
    real_dists_hold = _distances(z_a_h, z_b_h, metric=args.metric)

    # Synthetic distances by intensity
    shift_types = [s.strip() for s in args.shift_types.split(",") if s.strip()] if args.shift_types else cfg.intervention.shift_types
    if any(str(st) == "mixed" for st in shift_types):
        raise RuntimeError("calibrate_physics_realism.py only supports atomic shift families. Mixed stress profiles are not graded-calibrated.")
    intensity_grid = [float(v) for v in cfg.intervention.intensity_grid]
    family_intensity_grid = {
        "heel": intensity_grid,
        "scatter": intensity_grid,
        # Real fix: these families were saturating at the old shared ceiling 0.6.
        # Give them a denser and wider search space for graded realism calibration.
        "exposure_poisson": [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.60, 0.80, 1.00],
        "motion_blur":     [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.60, 0.80, 1.00],
    }

    # base images for synth
    synth_df = data_engine.get_split_df(cfg.data.eraser_split)
    synth_df = synth_df.sample(n=min(args.max_images, len(synth_df)), random_state=seed)
    dataset = XRayDataset(
        synth_df,
        label_cols=data_engine.label_cols,
        transforms=build_preprocess_transform(cfg.data.preprocess),
        model_representation=cfg.data.model_representation,
        preprocess_cfg=cfg.data.preprocess,
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False)

    shift_manager = ShiftManager(seed=cfg.shifts.seed)

    realism_levels = [float(v) for v in args.realism_levels.split(",")]
    calib = RealismCalibrator(distance_metric=args.metric)
    calib.metadata = {
        "pair_mode": cfg.intervention.pair_mode,
        "pair_filter_summary": {},
        "capability_snapshot": capability_snapshot.as_dict() if capability_snapshot is not None else {},
        "source_columns_snapshot": source_columns_snapshot,
        "index_sha256": index_sha256,
        "physics_semantics": {},
        "current_family_limitations": list(PARTIAL_REALISM_LIMITATIONS),
        "physical_dependence_warnings": [],
        "realism_validity_summary": {},
    }
    report = {
        "encoder_fingerprint": encoder_fingerprint,
        "group_key": group_key,
        "pair_mode": cfg.intervention.pair_mode,
        "capability_snapshot": capability_snapshot.as_dict() if capability_snapshot is not None else {},
        "source_columns_snapshot": source_columns_snapshot,
        "index_sha256": index_sha256,
        "n_allowed_groups": int(n_allowed_groups),
        "n_pairs_before_filter": int(n_pairs_before),
        "n_pairs_after_filter": int(len(pairs_df)),
        "n_unique_groups_after_filter": int(pairs_df[group_key].astype(str).nunique()),
        "pair_quality_config": {
            "max_delta_hours": float(pq_cfg.max_delta_hours),
            "label_change_weight": float(pq_cfg.label_change_weight),
            "view_change_weight": float(pq_cfg.view_change_weight),
            "context_change_weight": float(pq_cfg.context_change_weight),
            "sentinel_change_weight": float(pq_cfg.sentinel_change_weight),
            "technical_dominant_max_risk": float(pq_cfg.technical_dominant_max_risk),
        },
        "pair_quality_distribution": {
            **pair_quality_summary,
            "delta_hours_quantiles": {
                "all": {
                    "q50": float(np.quantile(pairs_df["delta_hours"], 0.5)),
                    "q90": float(np.quantile(pairs_df["delta_hours"], 0.9)),
                },
                "technical_dominant": {
                    "q50": float(np.quantile(pairs_td["delta_hours"], 0.5)) if len(pairs_td) else float("nan"),
                    "q90": float(np.quantile(pairs_td["delta_hours"], 0.9)) if len(pairs_td) else float("nan"),
                },
                "ambiguous": {
                    "q50": float(np.quantile(pairs_amb["delta_hours"], 0.5)) if len(pairs_amb) else float("nan"),
                    "q90": float(np.quantile(pairs_amb["delta_hours"], 0.9)) if len(pairs_amb) else float("nan"),
                },
            },
        },
        "real_quantiles": {"q50": float(np.quantile(real_dists, 0.5)), "q90": float(np.quantile(real_dists, 0.9))},
        "holdout_quantiles": {"q50": float(np.quantile(real_dists_hold, 0.5)), "q90": float(np.quantile(real_dists_hold, 0.9))},
        "pair_quality_counts": {
            "technical_dominant": int(len(pairs_td)),
            "ambiguous": int(len(pairs_amb)),
            "detailed": {
                str(k): int(v)
                for k, v in pairs_df["pair_quality"].fillna("unknown").value_counts().to_dict().items()
            },
        },
        "physics_semantics": {},
        "current_family_limitations": list(PARTIAL_REALISM_LIMITATIONS),
        "physical_dependence_warnings": [],
        "realism_validity_summary": {
            "is_graded": False,
        },
    }
    calib.metadata["pair_filter_summary"] = pair_quality_summary

    print(
        f"[realism] n_pairs_total={len(pairs_df)} n_pairs_td={len(pairs_td)} "
        f"n_pairs_train={len(pairs_train)} n_pairs_hold={len(pairs_hold)} "
        f"n_shift_types={len(shift_types)}",
        flush=True,
    )

    valid_mappings = 0
    for shift_type in shift_types:
        shift_grid = [float(v) for v in family_intensity_grid.get(shift_type, intensity_grid)]
        physics_meta = shift_physics_semantics(shift_type)
        dependence_warnings = physical_dependence_warnings([shift_type])
        report["physics_semantics"][shift_type] = physics_meta
        calib.metadata["physics_semantics"][shift_type] = physics_meta
        if dependence_warnings:
            report["physical_dependence_warnings"].extend(dependence_warnings)
            calib.metadata["physical_dependence_warnings"].extend(dependence_warnings)

        print(
            f"[realism] shift={shift_type} grid={shift_grid} "
            f"real_q50={float(np.quantile(real_dists, 0.5)):.6f} "
            f"hold_q50={float(np.quantile(real_dists_hold, 0.5)):.6f}",
            flush=True,
        )

        synth_dists_by_intensity: Dict[float, np.ndarray] = {}
        for intensity in shift_grid:
            dists = []
            with torch.no_grad():
                for batch_idx, batch in enumerate(loader, start=1):
                    x = batch["image"].to(device)
                    sample_ids = sample_ids_from_batch(batch)
                    x_shift = shift_manager.apply_batch(x, shift_type, float(intensity), sample_ids=sample_ids)
                    z = encoder.encode(x)
                    z_shift = encoder.encode(x_shift)
                    d = _distances(z.cpu().numpy(), z_shift.cpu().numpy(), metric=args.metric)
                    dists.append(d)
            synth = np.concatenate(dists, axis=0)
            synth_dists_by_intensity[float(intensity)] = synth
            print(
                f"[realism] shift={shift_type} intensity={float(intensity):.3f} "
                f"q50={float(np.quantile(synth, 0.5)):.6f} "
                f"q90={float(np.quantile(synth, 0.9)):.6f} "
                f"n={len(synth)}",
                flush=True,
            )

        mapping = calib.fit(
            shift_type,
            real_dists,
            synth_dists_by_intensity,
            realism_levels,
            method=args.match_method,
            n_pairs_total=int(len(pairs_df)),
            n_pairs_accepted=int(len(pairs_td)),
        )
        print(
            f"[realism] shift={shift_type} graded={mapping.is_graded} "
            f"flags={mapping.validity_flags} margin={mapping.level_separation_margin:.6f} "
            f"mapped_intensities={mapping.intensities}",
            flush=True,
        )

        if mapping.is_graded:
            valid_mappings += 1
        errors = {}
        for level, inten in zip(mapping.realism_levels, mapping.intensities):
            d = synth_dists_by_intensity.get(float(inten))
            if d is None:
                continue
            q50 = float(np.quantile(d, 0.5))
            q90 = float(np.quantile(d, 0.9))
            errors[str(level)] = {
                "q50_err": float(abs(q50 - report["holdout_quantiles"]["q50"]) / max(report["holdout_quantiles"]["q50"], 1e-8)),
                "q90_err": float(abs(q90 - report["holdout_quantiles"]["q90"]) / max(report["holdout_quantiles"]["q90"], 1e-8)),
                "mmd": float(RealismCalibrator._mmd_1d(real_dists_hold, d)),
            }
        report[shift_type] = {
            "mapping": {"levels": mapping.realism_levels, "intensities": mapping.intensities},
            "validity_flags": mapping.validity_flags,
            "is_graded": mapping.is_graded,
            "level_separation_margin": mapping.level_separation_margin,
            "n_pairs_total": mapping.n_pairs_total,
            "n_pairs_accepted": mapping.n_pairs_accepted,
            "quantile_errors": errors,
            "physics_semantics": physics_meta,
            "physical_dependence_warnings": dependence_warnings,
        }

    calib.encoder_fingerprint = encoder_fingerprint
    margins = [float(report[shift]["level_separation_margin"]) for shift in shift_types if shift in report]
    report["physical_dependence_warnings"] = sorted(set(report["physical_dependence_warnings"]))
    calib.metadata["physical_dependence_warnings"] = sorted(set(calib.metadata["physical_dependence_warnings"]))
    report["realism_validity_summary"] = build_realism_validity_summary(
        is_graded=bool(valid_mappings == len(shift_types)),
        level_separation_margin=float(min(margins)) if margins else float("nan"),
        n_pairs_total=int(len(pairs_df)),
        n_pairs_accepted=int(len(pairs_td)),
        pair_quality_distribution={
            str(k): int(v)
            for k, v in pairs_df["pair_quality"].fillna("unknown").value_counts().to_dict().items()
        },
        accepted_view_families=_accepted_context_tokens(pairs_td, "view_family"),
        accepted_procedure_contexts=_accepted_context_tokens(pairs_td, "procedure_context"),
        accepted_orientation_contexts=_accepted_context_tokens(pairs_td, "orientation_context"),
        pair_acceptance_funnel={
            "all_candidate_pairs": int(n_pairs_before),
            "valid_temporal_pairs": int(len(pairs_df)),
            "quality_accepted_pairs": int(len(pairs_td)),
            "calibrator_used_pairs": int(len(pairs_train)),
        },
    )
    calib.metadata["realism_validity_summary"] = dict(report["realism_validity_summary"])
    save_calibrator(args.out, calib)

    out_report = Path(args.out).with_suffix(".report.json")
    out_report.write_text(json.dumps(report, indent=2))
    pair_audit_dir = Path(args.out).parent
    (pair_audit_dir / "pair_quality_audit.json").write_text(json.dumps(pair_quality_summary, indent=2))
    pairs_df.to_csv(pair_audit_dir / "pair_quality_pairs.csv", index=False)
    figure_outputs = build_realism_calibration_figures(
        out_dir=Path(args.out).parent,
        report=report,
        pairs_df=pairs_df,
    )
    write_reprovenance_bundle(
        Path(args.out).parent,
        cfg=cfg,
        config_path=args.config,
        artifact_manifest={
            "artifact_type": "realism_calibrator",
            "index_sha256": index_sha256,
            "pair_mode": cfg.intervention.pair_mode,
            "pair_filter_stats": pair_quality_summary,
            "pair_quality_audit": pair_quality_summary,
            "source_capabilities": capability_snapshot.as_dict() if capability_snapshot is not None else {},
            "encoder_fingerprint": encoder_fingerprint,
            "preprocess_signature": preprocess_signature(cfg),
            "shift_families": shift_types,
            "mixed_profiles": cfg.intervention.mixed_profiles,
            "figure_outputs": figure_outputs,
            "realism_sha256": sha256_file(Path(args.out)) if Path(args.out).exists() else None,
        },
        input_artifact_paths={
            "config": args.config,
        },
    )

    if args.write_config:
        cfg.intervention.use_realism = True
        cfg.intervention.realism_path = Path(args.out)
        cfg.intervention.realism_levels = realism_levels
        save_config(args.config, cfg)
    if valid_mappings == 0:
        raise RuntimeError(
            "All calibrated shift families were marked non_graded or invalid_for_graded_eval. "
            "Widen the physical parameter range before running atomic graded evaluation."
        )

    print(f"Saved calibrator to {args.out}")


if __name__ == "__main__":
    main()
