#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from phystta.causal import PairQualityConfig, run_confounding_audit
from phystta.config.loader import load_config
from phystta.data import DataEngine
from phystta.models import EncoderFactory, build_encoder_fingerprint, load_head_meta, validate_head_meta
from phystta.utils import (
    load_torch_artifact,
    preprocess_signature,
    set_global_seed,
    sha256_file,
    write_reprovenance_bundle,
)
from phystta.viz import build_confounding_audit_figures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run confounding audit for acquisition-context mainline")
    parser.add_argument("--config", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--split", default=None)
    parser.add_argument("--source", default=None, help="Optional source for within-patient pair audit.")
    parser.add_argument("--bootstrap_samples", type=int, default=200)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--encoder_profile", default=None)
    return parser.parse_args()


def _batch_column(batch, key: str, n: int) -> list:
    if key not in batch:
        return [None] * n
    value = batch[key]
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return list(value)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    seed = int(args.seed if args.seed is not None else cfg.runtime.seed)
    device = str(args.device or cfg.runtime.device)
    set_global_seed(seed, deterministic=cfg.runtime.deterministic, benchmark=cfg.runtime.benchmark)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "log.txt"

    def log(msg: str) -> None:
        print(msg)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(msg + "\n")

    split = args.split or cfg.data.eval_split
    data_engine = DataEngine(cfg.data).attach_runtime_contract(cfg)
    data_engine.build_index()
    data_engine.create_splits()
    loader = data_engine.build_dataloaders(
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        splits=[split],
    )[split]

    enc_cfg = EncoderFactory.resolve_encoder_config(cfg, profile=args.encoder_profile)
    expected_fp = build_encoder_fingerprint(enc_cfg)
    expected_fp["preprocess_signature"] = preprocess_signature(cfg)
    encoder = EncoderFactory.from_config(cfg, profile=args.encoder_profile)
    if hasattr(encoder, "model"):
        encoder.model.to(device)

    if enc_cfg.head_path is None:
        raise RuntimeError("Confounding audit requires encoder.head_path for probability/severity diagnostics.")
    head = load_torch_artifact(enc_cfg.head_path, map_location="cpu")
    head.to(device)
    head.eval()
    meta = load_head_meta(
        Path(enc_cfg.head_path),
        explicit_meta=Path(enc_cfg.head_meta_path) if getattr(enc_cfg, "head_meta_path", None) is not None else None,
    )
    if meta is not None:
        errs = validate_head_meta(
            meta,
            enc_cfg=enc_cfg,
            expected_num_classes=cfg.model.num_classes,
            expected_label_cols=data_engine.label_cols,
        )
        if errs and bool(cfg.runtime.freeze_artifacts):
            raise RuntimeError("Head metadata validation failed: " + "; ".join(errs))
        if errs:
            log("head_meta_warning=" + "; ".join(errs))

    print("[script] start featurization_and_metadata_pass", flush=True)
    featurization_seen = 0
    features = []
    probs = []
    labels = []
    meta_rows = []
    encoder.eval()
    with torch.no_grad():
        for batch in loader:
            featurization_seen += 1
            if featurization_seen == 1 or featurization_seen % 25 == 0:
                print(f"[script] featurization_progress batch={featurization_seen}", flush=True)

            x = batch["image"].to(device)
            y = batch["labels"].cpu().numpy()
            z = encoder.encode(x)
            logits = head(z)
            p = torch.sigmoid(logits).cpu().numpy()
            features.append(z.detach().cpu().numpy())
            probs.append(p)
            labels.append(y)
            n = z.shape[0]
            row = pd.DataFrame(
                {
                    "patient_id": _batch_column(batch, "patient_id", n),
                    "global_patient_id": _batch_column(batch, "global_patient_id", n),
                    "study_id": _batch_column(batch, "study_id", n),
                    "source": _batch_column(batch, "source", n),
                    "image_path": _batch_column(batch, "image_path", n),
                    "view": _batch_column(batch, "view", n),
                    "view_family": _batch_column(batch, "view_family", n),
                    "procedure_context": _batch_column(batch, "procedure_context", n),
                    "orientation_context": _batch_column(batch, "orientation_context", n),
                    "resolution_bucket": _batch_column(batch, "resolution_bucket", n),
                    "rows": _batch_column(batch, "rows", n),
                    "columns": _batch_column(batch, "columns", n),
                }
            )
            meta_rows.append(row)
    features_np = np.concatenate(features, axis=0) if features else np.empty((0, 0), dtype=float)
    probs_np = np.concatenate(probs, axis=0) if probs else np.empty((0, 0), dtype=float)
    labels_np = np.concatenate(labels, axis=0) if labels else np.empty((0, 0), dtype=float)
    metadata_df = pd.concat(meta_rows, ignore_index=True) if meta_rows else pd.DataFrame()

    print(f"[script] done featurization_and_metadata_pass features_shape={features_np.shape} probs_shape={probs_np.shape} labels_shape={labels_np.shape} metadata_rows={len(metadata_df)}", flush=True)
    pair_source = args.source
    split_df = data_engine.get_split_df(split)
    if pair_source is None:
        split_sources = sorted({str(v) for v in split_df.get("source", pd.Series([], dtype=str)).dropna().astype(str).tolist()})
        if len(split_sources) == 1:
            pair_source = split_sources[0]
    print("[script] start pair_source_resolution", flush=True)
    pairs_df = None
    pair_summary_note = "pair_audit_skipped"
    if pair_source is not None and pair_source in set(data_engine.index["source"].astype(str).tolist()):
        try:
            pairs_df = data_engine.build_paired_index(
                pair_source,
                max_hours=float(cfg.intervention.real_pairs_max_hours),
                group_key=str(cfg.intervention.group_key),
                pair_mode=str(cfg.intervention.pair_mode),
            )
            pair_summary_note = f"pair_audit_source={pair_source}"
        except Exception as e:
            pair_summary_note = f"pair_audit_error={e}"
    print(f"[script] pair_build_result {pair_summary_note}", flush=True)
    cache_dir = out_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache_dir / "features.npy", features_np)
    np.save(cache_dir / "probs.npy", probs_np)
    np.save(cache_dir / "labels.npy", labels_np)
    metadata_df.to_csv(cache_dir / "metadata.csv.gz", index=False)
    if pairs_df is not None:
        pairs_df.to_csv(cache_dir / "pairs.csv.gz", index=False)
    print(f"[script] saved confounding cache dir={cache_dir}", flush=True)
    log(pair_summary_note)

    print("[script] start run_confounding_audit", flush=True)
    audit = run_confounding_audit(
        features=features_np,
        probs=probs_np,
        labels=labels_np,
        metadata_df=metadata_df,
        label_cols=data_engine.label_cols,
        index_df=data_engine.index,
        pairs_df=pairs_df,
        pair_quality_cfg=PairQualityConfig(**cfg.intervention.pair_quality.__dict__),
        shift_id_group_key=str(cfg.shift_id.group_key) if getattr(cfg, "shift_id", None) is not None else None,
        shift_id_score_family=list(cfg.shift_id.score_family) if getattr(cfg, "shift_id", None) is not None else None,
        bootstrap_samples=args.bootstrap_samples,
        seed=seed,
    )

    print("[script] done run_confounding_audit", flush=True)
    predictability_df = audit["predictability_df"]
    shift_id_df = audit["shift_id_proxy_df"]
    pair_delta_df = audit["technical_pair_delta_df"]
    predictability_df.to_csv(out_dir / "confounding_predictability.csv", index=False)
    shift_id_df.to_csv(out_dir / "shift_id_proxy_audit.csv", index=False)
    pair_delta_df.to_csv(out_dir / "technical_pair_pathology_deltas.csv", index=False)

    summary = {
        **audit["summary"],
        "shift_id_summary": audit["shift_id_summary"],
        "pair_quality_summary": audit["pair_quality_summary"],
        "config_path": str(args.config),
        "split": str(split),
        "pair_source": pair_source,
        "bootstrap_samples": int(args.bootstrap_samples),
    }
    (out_dir / "confounding_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    figure_outputs = build_confounding_audit_figures(
        out_dir=out_dir,
        predictability_df=predictability_df,
        shift_id_df=shift_id_df,
        pair_delta_df=pair_delta_df,
    )

    write_reprovenance_bundle(
        out_dir,
        cfg=cfg,
        config_path=args.config,
        artifact_manifest={
            "artifact_type": "confounding_audit",
            "split": split,
            "pair_source": pair_source,
            "label_cols": list(data_engine.label_cols),
            "bootstrap_samples": int(args.bootstrap_samples),
            "source_capabilities": {str(name): snap.as_dict() for name, snap in (data_engine.source_capabilities or {}).items()},
            "encoder_fingerprint": expected_fp,
            "preprocess_signature": preprocess_signature(cfg),
            "figure_outputs": figure_outputs,
            "head_sha256": sha256_file(Path(enc_cfg.head_path)) if Path(enc_cfg.head_path).exists() else None,
        },
        input_artifact_paths={
            "config": args.config,
            "head": enc_cfg.head_path,
        },
    )
    print("[script] saved all confounding outputs", flush=True)
    log(f"saved_confounding_audit={out_dir}")


if __name__ == "__main__":
    main()
