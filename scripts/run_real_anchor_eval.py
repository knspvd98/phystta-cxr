#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import pandas as pd

from phystta.config.loader import load_config
from phystta.data import DataEngine
from phystta.data.datasets import XRayDataset
from phystta.data.transforms import build_preprocess_transform
from phystta.models import EncoderFactory, build_encoder_fingerprint
from phystta.shifts import ShiftManager
from phystta.shifts.realism import (
    calibrator_fingerprint_mismatches,
    load_calibrator,
    intensity_grid_from_calibrator,
)
from phystta.intervention import eraser_fingerprint_mismatches, load_eraser, TokenEraserWrapper
from phystta.utils import set_global_seed, preprocess_signature, require_group_key, sample_ids_from_batch
from phystta.causal import annotate_pair_quality, PairQualityConfig, split_pairs_by_quality, ensure_pairs_group_key


def _build_feature_map(df: pd.DataFrame, encoder, device: str, preprocess_cfg) -> dict:
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


def _subspace_from_deltas(deltas: np.ndarray, k: int) -> np.ndarray:
    _, _, vt = np.linalg.svd(deltas, full_matrices=False)
    return vt[:k].T


def _principal_angles(Va: np.ndarray, Vb: np.ndarray) -> np.ndarray:
    m = Va.T @ Vb
    s = np.linalg.svd(m, compute_uv=False)
    s = np.clip(s, -1.0, 1.0)
    return np.arccos(s) * 180 / np.pi


def _subspace_residual_norm(deltas: np.ndarray, V: np.ndarray) -> np.ndarray:
    if deltas.ndim != 2:
        raise ValueError("deltas must be [N,D]")
    if V.ndim != 2:
        raise ValueError("V must be [D,k]")
    proj = (deltas @ V) @ V.T
    res = deltas - proj
    return np.linalg.norm(res, axis=1)


def _cluster_bootstrap_mean(
    values: np.ndarray,
    clusters: np.ndarray,
    seed: int,
    n_boot: int = 200,
) -> dict:
    x = np.asarray(values, dtype=float).reshape(-1)
    g = np.asarray(clusters).reshape(-1)
    valid = np.isfinite(x)
    x = x[valid]
    g = g[valid]
    if x.size == 0:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n": 0, "n_clusters": 0}
    uniq = np.unique(g.astype(str))
    if uniq.size == 0:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n": int(x.size), "n_clusters": 0}
    by_group = {u: x[g.astype(str) == u] for u in uniq}
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(max(int(n_boot), 10)):
        sampled = rng.choice(uniq, size=len(uniq), replace=True)
        vals = np.concatenate([by_group[s] for s in sampled])
        boot.append(float(np.mean(vals)))
    boot = np.asarray(boot, dtype=float)
    return {
        "mean": float(np.mean(x)),
        "ci_low": float(np.quantile(boot, 0.025)),
        "ci_high": float(np.quantile(boot, 0.975)),
        "n": int(x.size),
        "n_clusters": int(len(uniq)),
        "samples": boot.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Real shift anchoring eval")
    parser.add_argument("--config", required=True)
    parser.add_argument("--shift_type", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--max_pairs", type=int, default=300)
    parser.add_argument("--max_images", type=int, default=300)
    parser.add_argument("--eraser", default=None)
    parser.add_argument("--encoder_profile", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--bootstrap_samples", type=int, default=200)
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else cfg.runtime.seed
    device = args.device or cfg.runtime.device
    set_global_seed(seed, deterministic=cfg.runtime.deterministic, benchmark=cfg.runtime.benchmark)

    data_engine = DataEngine(cfg.data)
    data_engine.build_index()
    data_engine.create_splits()

    # real pairs from eraser split
    source = cfg.data.sources[0].domain_label or cfg.data.sources[0].name
    eraser_df = data_engine.get_split_df(cfg.data.eraser_split)
    group_key = require_group_key(eraser_df, preferred=cfg.intervention.group_key)
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
    pairs_df = data_engine.build_paired_index(source, group_key=group_key)
    pairs_df = ensure_pairs_group_key(pairs_df, group_key=group_key, default_source=source)
    allowed = set(eraser_df[group_key].astype(str))
    before_n = len(pairs_df)
    pairs_df = pairs_df[pairs_df[group_key].astype(str).isin(allowed)]
    if len(pairs_df) == 0:
        raise RuntimeError(
            f"No real pairs left after filtering by group_key='{group_key}'. "
            f"allowed_groups={len(allowed)} pairs_before={before_n}"
        )
    pairs_df = pairs_df.sample(n=min(args.max_pairs, len(pairs_df)), random_state=seed)
    pairs_df = annotate_pair_quality(
        pairs_df,
        index_df=data_engine.index,
        label_cols=data_engine.label_cols,
        cfg=pq_cfg,
    )
    pq = split_pairs_by_quality(pairs_df)
    pairs_df = pq["technical_dominant"]
    ambiguous_df = pq["ambiguous"]
    if len(pairs_df) == 0:
        raise RuntimeError("No technical-dominant real pairs available for anchoring evaluation.")

    holdout_n = 0
    if len(pairs_df) >= max(12, 2 * args.k + 4):
        holdout_n = max(args.k + 2, int(round(0.2 * len(pairs_df))))
        holdout_n = min(holdout_n, len(pairs_df) - (args.k + 2))
    if holdout_n > 0:
        groups = pairs_df[group_key].astype(str)
        split_ok = False
        try:
            from sklearn.model_selection import GroupShuffleSplit
            uniq_groups = groups.nunique()
            if uniq_groups > 1:
                test_size = min(0.5, max(float(holdout_n) / float(len(pairs_df)), 1.0 / float(uniq_groups)))
                gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
                tr_idx, ho_idx = next(gss.split(pairs_df, groups=groups))
                train_pairs_df = pairs_df.iloc[tr_idx].reset_index(drop=True)
                holdout_df = pairs_df.iloc[ho_idx].reset_index(drop=True)
                split_ok = len(train_pairs_df) > 0 and len(holdout_df) > 0
        except Exception:
            split_ok = False

        if not split_ok:
            # deterministic group-wise fallback split to avoid patient leakage
            rng = np.random.default_rng(seed)
            uniq = groups.unique()
            perm_groups = rng.permutation(uniq)
            selected = []
            acc = 0
            for g in perm_groups:
                selected.append(str(g))
                acc += int((groups == g).sum())
                if acc >= holdout_n:
                    break
            hold_mask = groups.isin(selected)
            holdout_df = pairs_df[hold_mask].reset_index(drop=True)
            train_pairs_df = pairs_df[~hold_mask].reset_index(drop=True)
            if len(train_pairs_df) == 0 or len(holdout_df) == 0:
                holdout_df = pd.DataFrame(columns=pairs_df.columns)
                train_pairs_df = pairs_df.reset_index(drop=True)
    else:
        holdout_df = pd.DataFrame(columns=pairs_df.columns)
        train_pairs_df = pairs_df.reset_index(drop=True)

    overlap = set(train_pairs_df[group_key].astype(str)).intersection(set(holdout_df[group_key].astype(str)))
    if len(overlap) > 0:
        raise RuntimeError(
            f"Group leakage in real-anchor holdout split for group_key='{group_key}': "
            f"{len(overlap)} overlapping groups."
        )

    all_pair_frames = [train_pairs_df]
    if len(holdout_df) > 0:
        all_pair_frames.append(holdout_df)
    all_pairs = pd.concat(all_pair_frames, ignore_index=True)
    all_paths = pd.unique(pd.concat([all_pairs["image_path_a"], all_pairs["image_path_b"]]))
    df_images = pd.DataFrame({"image_path": all_paths, "patient_id": "unknown", "source": source})

    enc_cfg = EncoderFactory.resolve_encoder_config(cfg, profile=args.encoder_profile)
    encoder = EncoderFactory.from_config(cfg, profile=args.encoder_profile)
    if hasattr(encoder, "model"):
        encoder.model.to(device)
    expected_calib_fp = build_encoder_fingerprint(enc_cfg)
    expected_calib_fp["preprocess_signature"] = preprocess_signature(cfg)

    feats = _build_feature_map(df_images, encoder, device, cfg.data.preprocess)
    z_a_train = np.stack([feats[str(p)] for p in train_pairs_df["image_path_a"]])
    z_b_train = np.stack([feats[str(p)] for p in train_pairs_df["image_path_b"]])
    dz_real_train = z_b_train - z_a_train
    V_real_train = _subspace_from_deltas(dz_real_train, args.k)
    if len(holdout_df) > 0:
        z_a_hold = np.stack([feats[str(p)] for p in holdout_df["image_path_a"]])
        z_b_hold = np.stack([feats[str(p)] for p in holdout_df["image_path_b"]])
        dz_real_holdout = z_b_hold - z_a_hold
    else:
        dz_real_holdout = np.zeros((0, dz_real_train.shape[1]), dtype=dz_real_train.dtype)

    # synthetic subspace
    shift_manager = ShiftManager(seed=cfg.shifts.seed)
    if cfg.intervention.use_realism:
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
        intensity_grid = intensity_grid_from_calibrator(calib, [args.shift_type])[args.shift_type]
    else:
        intensity_grid = cfg.erasure_eval.intensity_grid

    synth_df = eraser_df.sample(n=min(args.max_images, len(eraser_df)), random_state=seed)
    dataset = XRayDataset(
        synth_df,
        label_cols=data_engine.label_cols,
        transforms=build_preprocess_transform(cfg.data.preprocess),
        model_representation=cfg.data.model_representation,
        preprocess_cfg=cfg.data.preprocess,
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False)
    dz_synth = []
    synth_norms_by_intensity = {float(i): [] for i in intensity_grid}
    with torch.no_grad():
        for batch in loader:
            x = batch["image"].to(device)
            sample_ids = sample_ids_from_batch(batch)
            z = encoder.encode(x)
            for intensity in intensity_grid:
                x_shift = shift_manager.apply_batch(x, args.shift_type, float(intensity), sample_ids=sample_ids)
                z_shift = encoder.encode(x_shift)
                delta = (z_shift - z).cpu().numpy()
                dz_synth.append(delta)
                synth_norms_by_intensity[float(intensity)].append(np.linalg.norm(delta, axis=1))
    dz_synth = np.concatenate(dz_synth, axis=0)
    V_synth = _subspace_from_deltas(dz_synth, args.k)

    angles_train_synth = _principal_angles(V_real_train, V_synth)
    median_train_synth = float(np.median(angles_train_synth))

    if len(dz_real_holdout) >= args.k:
        V_real_holdout = _subspace_from_deltas(dz_real_holdout, args.k)
        angles_train_holdout = _principal_angles(V_real_train, V_real_holdout)
        median_train_holdout = float(np.median(angles_train_holdout))
    else:
        angles_train_holdout = np.array([], dtype=float)
        median_train_holdout = float("nan")

    real_norm_train = np.linalg.norm(dz_real_train, axis=1)
    real_norm_holdout = np.linalg.norm(dz_real_holdout, axis=1) if len(dz_real_holdout) else np.array([], dtype=float)
    boot_train = _cluster_bootstrap_mean(
        real_norm_train,
        train_pairs_df[group_key].astype(str).values,
        seed=seed + 31,
        n_boot=args.bootstrap_samples,
    )
    if len(real_norm_holdout):
        boot_holdout = _cluster_bootstrap_mean(
            real_norm_holdout,
            holdout_df[group_key].astype(str).values,
            seed=seed + 37,
            n_boot=args.bootstrap_samples,
        )
    else:
        boot_holdout = {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n": 0, "n_clusters": 0, "samples": []}
    synth_mean_norm_by_intensity = {}
    for i, chunks in synth_norms_by_intensity.items():
        if len(chunks) == 0:
            synth_mean_norm_by_intensity[str(i)] = float("nan")
        else:
            synth_mean_norm_by_intensity[str(i)] = float(np.mean(np.concatenate(chunks, axis=0)))
    target_norm = float(np.mean(real_norm_train))
    best_intensity = None
    best_abs_error = float("inf")
    for k, v in synth_mean_norm_by_intensity.items():
        if not np.isfinite(v):
            continue
        err = abs(v - target_norm)
        if err < best_abs_error:
            best_abs_error = err
            best_intensity = float(k)

    holdout_projection = {}
    if len(dz_real_holdout) > 0:
        res_real = _subspace_residual_norm(dz_real_holdout, V_real_train)
        res_synth = _subspace_residual_norm(dz_real_holdout, V_synth)
        holdout_projection = {
            "residual_mean_real_subspace": float(np.mean(res_real)),
            "residual_mean_synth_subspace": float(np.mean(res_synth)),
            "residual_ratio_synth_over_real": float(np.mean(res_synth) / max(float(np.mean(res_real)), 1e-8)),
        }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "principal_angles.json").write_text(
        json.dumps(
            {
                "angles_train_vs_synth_deg": angles_train_synth.tolist(),
                "median_train_vs_synth_deg": median_train_synth,
                "angles_train_vs_holdout_deg": angles_train_holdout.tolist(),
                "median_train_vs_holdout_deg": median_train_holdout,
                "n_pairs_train": int(len(train_pairs_df)),
                "n_pairs_holdout": int(len(holdout_df)),
                "n_pairs_ambiguous": int(len(ambiguous_df)),
            },
            indent=2,
        )
    )
    falsification = {
        "group_key": group_key,
        "group_overlap_train_holdout": int(len(overlap)),
        "n_pairs_train": int(len(train_pairs_df)),
        "n_pairs_holdout": int(len(holdout_df)),
        "n_pairs_ambiguous": int(len(ambiguous_df)),
        "median_angle_train_vs_synth_deg": median_train_synth,
        "median_angle_train_vs_holdout_deg": median_train_holdout,
        "real_delta_norm_mean_train": float(np.mean(real_norm_train)),
        "real_delta_norm_mean_holdout": float(np.mean(real_norm_holdout)) if len(real_norm_holdout) else float("nan"),
        "synth_delta_norm_mean_by_intensity": synth_mean_norm_by_intensity,
        "best_intensity_by_train_norm": best_intensity,
        "best_intensity_abs_error_train": float(best_abs_error) if np.isfinite(best_abs_error) else float("nan"),
        **holdout_projection,
    }
    if best_intensity is not None and len(real_norm_holdout) > 0:
        synth_norm_best = synth_mean_norm_by_intensity.get(str(best_intensity), float("nan"))
        falsification["best_intensity_abs_error_holdout"] = (
            abs(float(synth_norm_best) - float(np.mean(real_norm_holdout)))
            if np.isfinite(float(synth_norm_best))
            else float("nan")
        )
    train_samples = np.asarray(boot_train.get("samples", []), dtype=float)
    hold_samples = np.asarray(boot_holdout.get("samples", []), dtype=float)
    if train_samples.size > 0 and hold_samples.size > 0:
        m = min(train_samples.size, hold_samples.size)
        gap_samples = train_samples[:m] - hold_samples[:m]
        boot_gap = {
            "mean_gap_train_minus_holdout": float(np.mean(gap_samples)),
            "ci_low": float(np.quantile(gap_samples, 0.025)),
            "ci_high": float(np.quantile(gap_samples, 0.975)),
            "n_boot": int(m),
        }
    else:
        boot_gap = {
            "mean_gap_train_minus_holdout": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "n_boot": 0,
        }
    # keep report compact; drop raw samples after deriving CIs
    boot_train.pop("samples", None)
    boot_holdout.pop("samples", None)
    falsification["bootstrap_ci"] = {
        "real_delta_norm_train": boot_train,
        "real_delta_norm_holdout": boot_holdout,
        "train_minus_holdout_gap": boot_gap,
        "n_bootstrap": int(args.bootstrap_samples),
    }
    (out_dir / "falsification_report.json").write_text(json.dumps(falsification, indent=2))

    pair_summary = pd.DataFrame(
        [
            {"subset": "technical_dominant_train", "n_pairs": int(len(train_pairs_df))},
            {"subset": "technical_dominant_holdout", "n_pairs": int(len(holdout_df))},
            {"subset": "ambiguous", "n_pairs": int(len(ambiguous_df))},
        ]
    )
    pair_summary.to_csv(out_dir / "pair_quality_summary.csv", index=False)

    if args.eraser:
        eraser = load_eraser(args.eraser)
        eraser_mismatches = eraser_fingerprint_mismatches(eraser, expected_calib_fp)
        if eraser_mismatches:
            msg = "Eraser fingerprint mismatch against active encoder/preprocess: " + "; ".join(eraser_mismatches)
            if getattr(cfg.experiment, "tier", "dev") == "paper":
                raise RuntimeError(msg)
            print(f"[warn] {msg}")
        if hasattr(eraser, "to"):
            eraser = eraser.to(device)
        if hasattr(eraser, "eval"):
            eraser.eval()
        dz_before_train = np.linalg.norm(dz_real_train, axis=1)
        if getattr(cfg, "eraser", None) and cfg.eraser.token_mode != "none":
            # token eraser operates on images; skip feature-space delta transfer
            report = {
                "delta_norm_before_train": float(np.mean(dz_before_train)),
                "delta_norm_after_train": None,
                "delta_norm_before_holdout": float(np.mean(real_norm_holdout)) if len(real_norm_holdout) else float("nan"),
                "delta_norm_after_holdout": None,
                "note": "token_mode enabled; feature-space delta transfer skipped",
            }
        else:
            dz_after_train = np.linalg.norm(
                eraser(torch.tensor(dz_real_train, dtype=torch.float32, device=device)).cpu().numpy(),
                axis=1,
            )
            report = {
                "delta_norm_before_train": float(np.mean(dz_before_train)),
                "delta_norm_after_train": float(np.mean(dz_after_train)),
            }
            if len(dz_real_holdout) > 0:
                dz_after_holdout = np.linalg.norm(
                    eraser(torch.tensor(dz_real_holdout, dtype=torch.float32, device=device)).cpu().numpy(),
                    axis=1,
                )
                report["delta_norm_before_holdout"] = float(np.mean(real_norm_holdout))
                report["delta_norm_after_holdout"] = float(np.mean(dz_after_holdout))
        (out_dir / "real_pair_transfer.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
