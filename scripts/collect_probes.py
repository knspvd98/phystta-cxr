#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import torch

from phystta.config.loader import load_config
from phystta.data import DataEngine
from phystta.shifts import ShiftManager
from phystta.shifts.realism import load_calibrator, intensity_grid_from_calibrator, calibrator_fingerprint_mismatches
from phystta.models import EncoderFactory, build_encoder_fingerprint
from phystta.intervention.probes import collect_probes, save_probes, build_meta, ProbeWriter
from phystta.utils import (
    set_global_seed,
    tensor_stats,
    preprocess_signature,
    require_group_key,
    sha256_file,
    write_reprovenance_bundle,
)
from phystta.causal import annotate_pair_quality, PairQualityConfig, split_pairs_by_quality, ensure_pairs_group_key


def parse_list(val: str) -> List[str]:
    return [v.strip() for v in val.split(",") if v.strip()]


def parse_floats(val: str) -> List[float]:
    return [float(v.strip()) for v in val.split(",") if v.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect probes for representation surgery")
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default=None)
    parser.add_argument("--shift_types", default=None, help="comma-separated list")
    parser.add_argument("--intensity_grid", default=None, help="comma-separated list")
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--encoder_profile", default=None)
    parser.add_argument("--real_pairs", action="store_true", help="collect probes from within-patient technical pairs")
    parser.add_argument("--include_ambiguous_pairs", action="store_true", help="include ambiguous real pairs (sensitivity mode only)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else cfg.runtime.seed
    device = args.device or cfg.runtime.device
    set_global_seed(seed, deterministic=cfg.runtime.deterministic, benchmark=cfg.runtime.benchmark)

    shift_types = parse_list(args.shift_types) if args.shift_types else cfg.intervention.shift_types
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
        intensity_map = intensity_grid_from_calibrator(calib, shift_types)
    else:
        intensity_map = {st: (parse_floats(args.intensity_grid) if args.intensity_grid else cfg.intervention.intensity_grid) for st in shift_types}
    max_images = args.max_images if args.max_images is not None else cfg.intervention.max_images
    mixed_profiles = cfg.intervention.mixed_profiles
    token_mode = cfg.eraser.token_mode
    token_budget = cfg.intervention.token_budget
    if token_mode != "none" and token_budget is None:
        raise RuntimeError("token_mode enabled but intervention.token_budget is not set. Define a token budget to avoid OOM.")

    data_engine = DataEngine(cfg.data).attach_runtime_contract(cfg)
    data_engine.build_index()
    data_engine.create_splits()
    split = args.split if args.split is not None else cfg.data.eraser_split
    loaders = data_engine.build_dataloaders(batch_size=args.batch_size, num_workers=cfg.train.num_workers, splits=[split])
    loader = loaders[split]
    split_df = data_engine.get_split_df(split)
    group_key = require_group_key(split_df, preferred=cfg.intervention.group_key)
    group_valid = bool(group_key) and split_df[group_key].nunique() > 1
    if getattr(cfg.experiment, "tier", "dev") == "paper" and not group_valid:
        raise RuntimeError(
            f"Paper-tier run requires valid group key '{cfg.intervention.group_key}'. "
            "Group key missing or degenerate; NOT CLINICALLY VALID."
        )

    shift_manager = ShiftManager(seed=cfg.shifts.seed)
    encoder = EncoderFactory.from_config(cfg, profile=args.encoder_profile)
    if hasattr(encoder, "model"):
        encoder.model.to(device)

    all_clean = []
    all_shifted = []
    all_intensity = []
    all_shift_type = []
    all_profiles = []
    patient_ids = []
    group_ids = []
    study_ids = []
    image_ids = []
    labels = []
    params_map_all = {}
    log_path = Path(args.out).parent / "log.txt"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    def log(msg: str):
        print(msg)
        log_path.open("a").write(msg + "\n")
    log(f"preprocess_signature={preprocess_signature(cfg)}")
    log(f"Collecting probes: shifts={shift_types} intensity_map={intensity_map} max_images={max_images}")
    try:
        batch = next(iter(loader))
        raw_stats = tensor_stats(batch["image"])
        adapted = encoder.preprocess(batch["image"].to(device))
        adapted_stats = tensor_stats(adapted.detach().cpu())
        log(f"raw_stats={raw_stats}")
        log(f"adapter_stats={adapted_stats}")
    except Exception as e:
        log(f"stats_error={e}")
    if args.real_pairs:
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
        source = cfg.data.sources[0].domain_label or cfg.data.sources[0].name
        pairs_df = data_engine.build_paired_index(
            source,
            max_hours=pq_cfg.max_delta_hours,
            group_key=group_key,
            pair_mode=cfg.intervention.pair_mode,
        )
        pairs_df = ensure_pairs_group_key(pairs_df, group_key=group_key, default_source=source)
        allowed = set(split_df[group_key].astype(str))
        pairs_df = pairs_df[pairs_df[group_key].astype(str).isin(allowed)]
        if len(pairs_df) == 0:
            raise RuntimeError("No real pairs found for real_pairs probes")
        pairs_df = annotate_pair_quality(
            pairs_df,
            index_df=data_engine.index,
            label_cols=data_engine.label_cols,
            cfg=pq_cfg,
        )
        quality_split = split_pairs_by_quality(pairs_df)
        pairs_main = quality_split["technical_dominant"]
        pairs_amb = quality_split["ambiguous"]
        if args.include_ambiguous_pairs:
            pairs_use = pairs_df
            pair_mode = "all_pairs"
        else:
            pairs_use = pairs_main
            pair_mode = "technical_dominant_only"
        if len(pairs_use) == 0:
            raise RuntimeError("No technical-dominant real pairs available after pair-quality filtering")
        # build minimal dataset for both sides
        import pandas as pd
        all_paths = pd.unique(pd.concat([pairs_use["image_path_a"], pairs_use["image_path_b"]]))
        df_images = pd.DataFrame({"image_path": all_paths, "patient_id": "unknown", "source": source})
        from phystta.data.datasets import XRayDataset
        from phystta.data.transforms import build_preprocess_transform
        dataset = XRayDataset(df_images, label_cols=data_engine.label_cols, transforms=build_preprocess_transform(cfg.data.preprocess), preprocess_cfg=cfg.data.preprocess)
        dl = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
        feats = {}
        with torch.no_grad():
            for batch in dl:
                x = batch["image"].to(device)
                if cfg.eraser.token_mode != "none":
                    tokens = encoder.encode_tokens(x, token_layers=cfg.eraser.token_layers)
                    if cfg.eraser.token_mode == "cls":
                        z = tokens[:, 0]
                    elif cfg.eraser.token_mode == "mean":
                        z = tokens.mean(dim=1)
                    elif cfg.eraser.token_mode == "patch":
                        z = tokens[:, 1:].reshape(tokens.shape[0], -1, tokens.shape[-1]).mean(dim=1)
                    elif cfg.eraser.token_mode == "all":
                        z = tokens.reshape(tokens.shape[0], -1, tokens.shape[-1]).mean(dim=1)
                    else:
                        raise RuntimeError(f"Unsupported token_mode={cfg.eraser.token_mode}")
                else:
                    z = encoder.encode(x)
                z = z.cpu()
                for path, feat in zip(batch["image_path"], z):
                    feats[str(path)] = feat
        for _, row in pairs_use.iterrows():
            z_a = feats[str(row["image_path_a"])]
            z_b = feats[str(row["image_path_b"])]
            all_clean.append(z_a.unsqueeze(0))
            all_shifted.append(z_b.unsqueeze(0))
            all_intensity.append(torch.tensor([1.0]))
            all_shift_type.append("real_pairs")
            all_profiles.append(str(row.get("pair_quality", "real_pairs")))
            patient_ids.append(str(row[group_key]))
            study_ids.append(row.get("study_id_a", "unknown"))
            image_ids.append(str(row.get("image_path_a", "unknown")))
        out = {
            "features_clean": torch.cat(all_clean, dim=0),
            "features_shifted": torch.cat(all_shifted, dim=0),
            "intensity": torch.cat(all_intensity, dim=0),
            "shift_type": all_shift_type,
            "profile": all_profiles,
            "patient_id": patient_ids,
            "group_id": patient_ids,
            "study_id": study_ids,
            "image_id": image_ids,
            "labels": None,
        }
        encoder_sig = {
            "encoder_source": enc_cfg.encoder_source,
            "encoder_name": enc_cfg.encoder_name,
            "encoder_path": str(enc_cfg.encoder_path) if enc_cfg.encoder_path else None,
            "model_id": enc_cfg.model_id,
        }
        meta = build_meta(
            cfg,
            split,
            "real_pairs",
            [1.0],
            enc_cfg.encoder_source,
            out["features_clean"].shape[0],
            extra={
                "preprocess_signature": str(preprocess_signature(cfg)),
                "encoder_signature": str(encoder_sig),
                "group_valid": "True",
                "physical_params_map": "real_pairs",
                "pair_mode": pair_mode,
                "pair_quality_counts": str({"technical_dominant": int(len(pairs_main)), "ambiguous": int(len(pairs_amb))}),
                "token_mode": str(cfg.eraser.token_mode),
                "token_layers": str(cfg.eraser.token_layers),
            },
        )
        save_probes(args.out, out, meta)
        log(f"Saved real_pairs probes to {args.out} with {out['features_clean'].shape[0]} samples.")
        return

    writer = None
    if token_mode != "none":
        writer = ProbeWriter(args.out, chunk_size=cfg.intervention.probe_chunk_size, token_budget=token_budget)

    for shift_type in shift_types:
        intensity_grid = intensity_map.get(shift_type)
        if shift_type != "mixed" and intensity_grid is None:
            raise RuntimeError(f"No calibrated intensity grid for shift_type={shift_type}")
        params_map = {}
        if shift_type == "mixed":
            for idx, profile in enumerate(mixed_profiles or []):
                name = profile.get("name", f"mixed_{idx}")
                params_map[name] = profile
        else:
            for intensity in intensity_grid:
                if shift_type in shift_manager.shifts and shift_manager.shifts[shift_type] is not None:
                    params_map[str(intensity)] = shift_manager.shifts[shift_type].physical_params(float(intensity))
        params_map_all[shift_type] = params_map
        probes = collect_probes(
            loader,
            shift_manager,
            shift_type=shift_type,
            intensity_grid=intensity_grid,
            encoder=encoder,
            max_images=max_images,
            device=device,
            seed=seed,
            mixed_profiles=mixed_profiles,
            token_mode=cfg.eraser.token_mode,
            token_layers=cfg.eraser.token_layers,
            token_sampling=cfg.intervention.token_sampling,
            token_sample_per_image=cfg.intervention.token_sample_per_image,
            token_budget=token_budget,
            writer=writer,
            group_key=group_key,
        )
        if writer is None:
            all_clean.append(probes["features_clean"])
            all_shifted.append(probes["features_shifted"])
            all_intensity.append(probes["intensity"])
            all_shift_type.extend([shift_type] * probes["features_shifted"].shape[0])
            if "profile" in probes:
                all_profiles.extend(probes["profile"].tolist())
            patient_ids.extend(probes["patient_id"].tolist())
            if probes.get("group_id") is not None:
                group_ids.extend(probes["group_id"].tolist())
            study_ids.extend(probes["study_id"].tolist())
            if probes.get("image_id") is not None:
                image_ids.extend(probes["image_id"].tolist())
            if probes["labels"] is not None:
                labels.append(probes["labels"])

    out = None
    if writer is None:
        out = {
            "features_clean": torch.cat(all_clean, dim=0),
            "features_shifted": torch.cat(all_shifted, dim=0),
            "intensity": torch.cat(all_intensity, dim=0),
            "shift_type": all_shift_type,
            "profile": all_profiles if all_profiles else None,
            "patient_id": patient_ids,
            "group_id": group_ids if group_ids else patient_ids,
            "study_id": study_ids,
            "image_id": image_ids,
            "labels": torch.cat(labels, dim=0) if labels else None,
        }
    encoder_sig = {
        "encoder_source": enc_cfg.encoder_source,
        "encoder_name": enc_cfg.encoder_name,
        "encoder_path": str(enc_cfg.encoder_path) if enc_cfg.encoder_path else None,
        "model_id": enc_cfg.model_id,
    }
    source_capabilities = {
        str(name): snap.as_dict() for name, snap in (data_engine.source_capabilities or {}).items()
    }
    index_hashes = {
        str(name): sha256_file(path) for name, path in data_engine.index_paths.items() if path.exists()
    }
    uniq_patients = set(group_ids if group_ids else patient_ids)
    if not group_valid and uniq_patients:
        group_valid = len(uniq_patients) > 1 and not (uniq_patients == {"unknown"} or uniq_patients == {"nan"})
    grid_meta = intensity_map.get(shift_types[0], []) if shift_types else []
    if not grid_meta and cfg.intervention.use_realism:
        grid_meta = cfg.intervention.realism_levels
    meta = build_meta(
        cfg,
        split,
        ",".join(shift_types),
        grid_meta,
        enc_cfg.encoder_source,
        (out["features_clean"].shape[0] if out is not None else writer.total_count),
        extra={
            "preprocess_signature": str(preprocess_signature(cfg)),
            "encoder_signature": str(encoder_sig),
            "group_valid": str(bool(group_valid)),
            "group_key": str(group_key),
            "source_split": str(split),
            "shift_family": ",".join(shift_types),
            "index_sha256": str(index_hashes.get(cfg.data.sources[0].domain_label or cfg.data.sources[0].name)),
            "source_capabilities": str(source_capabilities),
            "physical_params_map": str(params_map_all),
            "token_mode": str(cfg.eraser.token_mode),
            "token_layers": str(cfg.eraser.token_layers),
            "token_sampling": str(cfg.intervention.token_sampling),
            "token_sample_per_image": str(cfg.intervention.token_sample_per_image),
            "token_budget": str(token_budget),
        },
    )
    if writer is None:
        save_probes(args.out, out, meta)
    else:
        writer.finalize(meta)
    write_reprovenance_bundle(
        Path(args.out).parent,
        cfg=cfg,
        config_path=args.config,
        artifact_manifest={
            "artifact_type": "probes",
            "shift_families": shift_types,
            "mixed_profiles": mixed_profiles,
            "source_capabilities": source_capabilities,
            "index_sha256": index_hashes.get(cfg.data.sources[0].domain_label or cfg.data.sources[0].name),
            "encoder_fingerprint": encoder_sig,
            "preprocess_signature": preprocess_signature(cfg),
        },
        input_artifact_paths={
            "config": args.config,
        },
    )
    unique_patients = len(set(patient_ids)) if patient_ids else (int(split_df["patient_id"].nunique()) if "patient_id" in split_df.columns else 0)
    unique_studies = len(set(study_ids)) if study_ids else (int(split_df["study_id"].nunique()) if "study_id" in split_df.columns else 0)
    total_count = out["features_clean"].shape[0] if out is not None else writer.total_count
    log(f"Saved probes to {args.out} with {total_count} samples.")
    log(f"Unique patients={unique_patients} unique studies={unique_studies}")
    if not group_valid:
        log("WARNING: group key missing or degenerate. Marking probes as NOT CLINICALLY VALID.")


if __name__ == "__main__":
    main()
