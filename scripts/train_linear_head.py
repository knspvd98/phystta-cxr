#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from phystta.config.loader import load_config
from phystta.data import DataEngine
from phystta.models import EncoderFactory, build_head_meta
from phystta.utils import (
    set_global_seed,
    tensor_stats,
    preprocess_signature,
    require_group_key,
    ensure_no_overlap,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train linear head on frozen encoder")
    parser.add_argument("--config", required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--out_head", required=True)
    parser.add_argument("--out_meta", required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--encoder_profile", default=None)
    parser.add_argument("--log_every", type=int, default=25)
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else cfg.runtime.seed
    device = args.device or cfg.runtime.device
    set_global_seed(seed, deterministic=cfg.runtime.deterministic, benchmark=cfg.runtime.benchmark)

    out_head = Path(args.out_head)
    out_head.parent.mkdir(parents=True, exist_ok=True)
    log_path = Path("/workspace/logs/train_linear_head_podrun.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(msg: str) -> None:
        print(msg, flush=True)
        with log_path.open("a") as f:
            f.write(msg + "\n")

    data_engine = DataEngine(cfg.data)
    data_engine.build_index()
    data_engine.create_splits()
    label_cols = list(data_engine.label_cols)

    batch_size = args.batch_size or cfg.train.batch_size
    train_split = cfg.data.head_split
    val_split = cfg.data.head_val_split or cfg.data.calib_split

    train_df = data_engine.get_split_df(train_split)
    val_df = data_engine.get_split_df(val_split)
    group_key = require_group_key(train_df, preferred=cfg.intervention.group_key)
    ensure_no_overlap(train_df, val_df, group_key)

    loaders = data_engine.build_dataloaders(
        batch_size=batch_size,
        num_workers=cfg.train.num_workers,
        splits=[train_split, val_split],
    )
    train_loader = loaders[train_split]
    val_loader = loaders[val_split]

    log(f"train_split={train_split} n_train={len(train_df)} batches={len(train_loader)}")
    log(f"val_split={val_split} n_val={len(val_df)} batches={len(val_loader)}")
    log(
        f"device={device} batch_size={batch_size} "
        f"num_workers={cfg.train.num_workers} epochs={args.epochs}"
    )

    encoder = EncoderFactory.from_config(cfg, profile=args.encoder_profile)
    enc_cfg = EncoderFactory.resolve_encoder_config(cfg, profile=args.encoder_profile)
    if hasattr(encoder, "model"):
        encoder.model.to(device)
        encoder.model.eval()

    amp_enabled = bool(
        str(device).startswith("cuda") and getattr(cfg.train, "mixed_precision", False)
    )

    def amp_ctx():
        if amp_enabled:
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return nullcontext()

    log(f"preprocess_signature={preprocess_signature(cfg)}")

    with torch.no_grad():
        batch = next(iter(train_loader))
        x_preview = batch["image"].to(device, non_blocking=True)
        raw_stats = tensor_stats(batch["image"])
        adapted = encoder.preprocess(batch["image"].to(device, non_blocking=True))
        adapted_stats = tensor_stats(adapted.detach().cpu())
        log(f"raw_stats={raw_stats}")
        log(f"adapter_stats={adapted_stats}")
        with amp_ctx():
            z_preview = encoder.encode(x_preview)
    feat_dim = z_preview.shape[1]

    head = nn.Linear(feat_dim, cfg.model.num_classes).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=1e-3)

    def compute_loss(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        loss_map = nn.functional.binary_cross_entropy_with_logits(
            logits.float(), y.float(), reduction="none"
        )
        mask = (y >= 0).float()
        denom = mask.sum().clamp_min(1.0)
        return (loss_map * mask).sum() / denom

    def step(loader, train: bool, epoch_idx: int) -> float:
        head.train() if train else head.eval()
        losses = []
        total_samples = 0
        n_batches = len(loader)
        start = time.time()

        for batch_idx, batch in enumerate(loader, start=1):
            x = batch["image"].to(device, non_blocking=True)
            y = batch["labels"].to(device, non_blocking=True)

            with torch.no_grad():
                with amp_ctx():
                    z = encoder.encode(x)

            if train:
                opt.zero_grad(set_to_none=True)
                with amp_ctx():
                    logits = head(z)
                loss = compute_loss(logits, y)
                loss.backward()
                opt.step()
            else:
                with torch.no_grad():
                    with amp_ctx():
                        logits = head(z)
                    loss = compute_loss(logits, y)

            losses.append(float(loss.item()))
            total_samples += int(x.shape[0])

            if batch_idx == 1 or batch_idx % args.log_every == 0 or batch_idx == n_batches:
                elapsed = max(time.time() - start, 1e-6)
                samples_per_sec = total_samples / elapsed
                sec_per_batch = elapsed / batch_idx
                eta_sec = max(n_batches - batch_idx, 0) * sec_per_batch
                phase = "train" if train else "val"
                log(
                    f"[{phase}] epoch={epoch_idx+1}/{args.epochs} "
                    f"batch={batch_idx}/{n_batches} "
                    f"loss={loss.item():.4f} avg_loss={np.mean(losses):.4f} "
                    f"samples={total_samples} samples_per_sec={samples_per_sec:.2f} "
                    f"eta_sec={eta_sec:.1f}"
                )

        return float(np.mean(losses) if losses else 0.0)

    for epoch in range(args.epochs):
        tr = step(train_loader, True, epoch)
        va = step(val_loader, False, epoch)
        log(f"epoch {epoch+1}/{args.epochs} train_loss={tr:.4f} val_loss={va:.4f}")

    torch.save(head, args.out_head)
    meta = build_head_meta(
        enc_cfg=enc_cfg,
        feat_dim=feat_dim,
        num_classes=cfg.model.num_classes,
        label_cols=label_cols,
        group_key=group_key,
        encoder_profile=args.encoder_profile,
        seed=seed,
    )
    meta["dataset"] = cfg.data.train_sources
    meta["preprocess_signature"] = preprocess_signature(cfg)
    Path(args.out_meta).write_text(json.dumps(meta, indent=2))
    log(f"Saved head to {args.out_head}")
    log(f"Saved head meta to {args.out_meta}")


if __name__ == "__main__":
    main()
