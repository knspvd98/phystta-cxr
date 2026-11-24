#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from phystta.config.loader import load_config
from phystta.data import DataEngine
from phystta.models import EncoderFactory, load_head_meta, validate_head_meta
from phystta.shifts import ShiftManager
from phystta.safety.shift_id import ShiftIdentifierV2
from phystta.monitoring import ShiftMonitor
from phystta.utils import set_global_seed, sample_ids_from_batch, load_torch_artifact
from phystta.viz import build_shift_monitor_figures


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline postmarket shift monitoring simulation")
    parser.add_argument("--config", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--encoder_profile", default=None)
    parser.add_argument("--shift_type", default="none")
    parser.add_argument("--intensity", type=float, default=0.0)
    parser.add_argument("--drift_start", type=int, default=20)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else cfg.runtime.seed
    device = args.device or cfg.runtime.device
    set_global_seed(seed, deterministic=cfg.runtime.deterministic, benchmark=cfg.runtime.benchmark)

    if not cfg.monitoring.enabled:
        raise RuntimeError("monitoring.enabled=false in config; enable monitoring block to run this script")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data_engine = DataEngine(cfg.data).attach_runtime_contract(cfg)
    data_engine.build_index()
    data_engine.create_splits()
    splits = [cfg.data.calib_split, cfg.data.eval_split]
    loaders = data_engine.build_dataloaders(batch_size=cfg.train.batch_size, num_workers=cfg.train.num_workers, splits=splits)
    calib_loader = loaders[cfg.data.calib_split]
    eval_loader = loaders[cfg.data.eval_split]

    enc = EncoderFactory.from_config(cfg, profile=args.encoder_profile)
    enc_cfg = EncoderFactory.resolve_encoder_config(cfg, profile=args.encoder_profile)
    if hasattr(enc, "model"):
        enc.model.to(device)

    if enc_cfg.head_path is None:
        raise RuntimeError("run_shift_monitor requires model.encoder.head_path")
    head = load_torch_artifact(enc_cfg.head_path, map_location="cpu")
    head.to(device)
    head.eval()
    strict_meta = bool(cfg.runtime.freeze_artifacts or getattr(cfg.experiment, "tier", "dev") == "paper")
    meta = load_head_meta(
        Path(enc_cfg.head_path),
        explicit_meta=Path(enc_cfg.head_meta_path) if getattr(enc_cfg, "head_meta_path", None) is not None else None,
    )
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
            expected_label_cols=data_engine.label_cols,
        )
        if errs and strict_meta:
            raise RuntimeError("Head metadata validation failed: " + "; ".join(errs))
        if errs:
            print("[warn] head metadata validation warnings: " + "; ".join(errs))

    shift_manager = ShiftManager(seed=cfg.shifts.seed)
    shift_id = ShiftIdentifierV2(score_family=cfg.monitoring.scores)

    # reference bank from calibration split
    feats_ref = []
    probs_ref = []
    with torch.no_grad():
        for batch in calib_loader:
            x = batch["image"].to(device)
            z = enc.encode(x)
            logits = head(z)
            feats_ref.append(z.detach().cpu().numpy())
            probs_ref.append(torch.sigmoid(logits).detach().cpu().numpy())
    feats_ref = np.concatenate(feats_ref, axis=0)
    probs_ref = np.concatenate(probs_ref, axis=0)

    # baseline score distribution (clean batches)
    baseline_scores = {k: [] for k in cfg.monitoring.scores}
    with torch.no_grad():
        for bidx, batch in enumerate(calib_loader):
            if bidx >= cfg.monitoring.warmup_batches:
                break
            x = batch["image"].to(device)
            z = enc.encode(x)
            logits = head(z)
            scores = shift_id.compute_scores(feats_ref, probs_ref, z.detach().cpu().numpy(), torch.sigmoid(logits).detach().cpu().numpy())
            for k in cfg.monitoring.scores:
                baseline_scores[k].append(float(scores.get(k, 0.0)))

    monitor = ShiftMonitor(alert_quantile=cfg.monitoring.alert_quantile, min_consecutive_alerts=cfg.monitoring.min_consecutive_alerts)
    thresholds = monitor.calibrate(baseline_scores)

    rows = []
    max_batches = cfg.monitoring.max_batches if cfg.monitoring.max_batches is not None else 10**9
    with torch.no_grad():
        for bidx, batch in enumerate(eval_loader):
            if bidx >= max_batches:
                break
            x = batch["image"].to(device)
            if bidx >= args.drift_start and args.shift_type != "none" and args.intensity > 0:
                sample_ids = sample_ids_from_batch(batch)
                x = shift_manager.apply_batch(x, args.shift_type, args.intensity, sample_ids=sample_ids)
            z = enc.encode(x)
            logits = head(z)
            scores = shift_id.compute_scores(feats_ref, probs_ref, z.detach().cpu().numpy(), torch.sigmoid(logits).detach().cpu().numpy())
            for k in cfg.monitoring.scores:
                ev = monitor.update(k, float(scores.get(k, 0.0)), bidx)
                rows.append(
                    {
                        "step": bidx,
                        "score_key": k,
                        "score": ev.score,
                        "threshold": ev.threshold,
                        "alert": ev.alert,
                        "reason": ev.reason,
                        "drift_active": bidx >= args.drift_start,
                    }
                )

    import pandas as pd

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "monitor_events.csv", index=False)
    figure_outputs = build_shift_monitor_figures(out_dir=out_dir, monitor_df=df, drift_start=int(args.drift_start))
    summary = {
        "thresholds": thresholds,
        "n_events": int(len(df)),
        "n_alerts": int(df["alert"].sum()) if not df.empty else 0,
        "first_alert_step": int(df[df["alert"]]["step"].min()) if not df[df["alert"]].empty else -1,
        "drift_start": int(args.drift_start),
        "figure_outputs": figure_outputs,
    }
    (out_dir / "monitor_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
