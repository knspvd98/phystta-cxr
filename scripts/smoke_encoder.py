#!/usr/bin/env python3
from __future__ import annotations

import argparse

import torch

from phystta.config.loader import load_config
from phystta.data import DataEngine
from phystta.models import EncoderFactory
from phystta.utils import set_global_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test encoder forward")
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--encoder_profile", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = args.device or cfg.runtime.device
    set_global_seed(cfg.runtime.seed, deterministic=cfg.runtime.deterministic, benchmark=cfg.runtime.benchmark)

    data_engine = DataEngine(cfg.data)
    data_engine.build_index()
    data_engine.create_splits()
    split = args.split if args.split is not None else cfg.data.eval_split
    loader = data_engine.build_dataloaders(batch_size=2, num_workers=0, splits=[split])[split]

    encoder = EncoderFactory.from_config(cfg, profile=args.encoder_profile)
    if hasattr(encoder, "model"):
        encoder.model.to(device)

    batch = next(iter(loader))
    x = batch["image"].to(device)
    with torch.no_grad():
        z = encoder.encode(x)
    print(f"smoke_ok: input={tuple(x.shape)} features={tuple(z.shape)}")


if __name__ == "__main__":
    main()
