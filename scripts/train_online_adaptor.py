#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn

from phystta.config.loader import load_config
from phystta.data import DataEngine
from phystta.models import EncoderFactory
from phystta.shifts import ShiftManager
from phystta.safety.shift_id import rbf_mmd, energy_distance
from phystta.utils import set_global_seed


def parse_list(val: str) -> List[str]:
    return [v.strip() for v in val.split(",") if v.strip()]


def compute_score(z_batch: np.ndarray, ref_feats: np.ndarray, method: str) -> float:
    if method == "mmd":
        return rbf_mmd(z_batch, ref_feats)
    if method == "energy":
        return energy_distance(z_batch, ref_feats)
    raise RuntimeError(f"Unknown score method: {method}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train online eraser adaptor")
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", default="mixture", choices=["mixture", "fast"])
    parser.add_argument("--shift_types", default=None)
    parser.add_argument("--shift_type", default=None)
    parser.add_argument("--intensity", type=float, default=0.4)
    parser.add_argument("--score_method", default="mmd")
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--encoder_profile", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else cfg.runtime.seed
    device = args.device or cfg.runtime.device
    set_global_seed(seed, deterministic=cfg.runtime.deterministic, benchmark=cfg.runtime.benchmark)

    shift_types = parse_list(args.shift_types) if args.shift_types else cfg.intervention.shift_types
    data_engine = DataEngine(cfg.data)
    data_engine.build_index()
    data_engine.create_splits()
    calib_split = cfg.data.calib_split
    loaders = data_engine.build_dataloaders(batch_size=cfg.train.batch_size, num_workers=cfg.train.num_workers, splits=[calib_split])
    loader = loaders[calib_split]

    encoder = EncoderFactory.from_config(cfg, profile=args.encoder_profile)
    if hasattr(encoder, "model"):
        encoder.model.to(device)

    # reference features on clean calib
    ref = []
    with torch.no_grad():
        for batch in loader:
            x = batch["image"].to(device)
            ref.append(encoder.encode(x).cpu().numpy())
    ref_feats = np.concatenate(ref, axis=0)

    shift_manager = ShiftManager(seed=cfg.shifts.seed)
    if args.mode == "mixture":
        scores = []
        labels = []
        with torch.no_grad():
            for batch in loader:
                x = batch["image"].to(device)
                for idx, st in enumerate(shift_types):
                    x_shift = shift_manager.apply_batch(x, st, args.intensity)
                    z = encoder.encode(x_shift).cpu().numpy()
                    sc = compute_score(z, ref_feats, args.score_method)
                    scores.append([sc])
                    labels.append(idx)

        X = torch.tensor(scores, dtype=torch.float32)
        y = torch.tensor(labels, dtype=torch.long)
        hidden = cfg.eraser.online_hidden if hasattr(cfg, "eraser") else 32
        weight_net = nn.Sequential(nn.Linear(X.shape[1], hidden), nn.ReLU(), nn.Linear(hidden, len(shift_types)))
        opt = torch.optim.Adam(weight_net.parameters(), lr=1e-3)
        for _ in range(200):
            opt.zero_grad()
            logits = weight_net(X)
            loss = nn.functional.cross_entropy(logits, y)
            loss.backward()
            opt.step()

        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        torch.save(weight_net, args.out)
        print(f"Saved online adaptor weights to {args.out}")
    else:
        # fast mode: save probe_w for steering adaptor
        if args.shift_type is None:
            raise RuntimeError("fast mode requires --shift_type")
        z_clean = []
        z_shift = []
        with torch.no_grad():
            for batch in loader:
                x = batch["image"].to(device)
                z_clean.append(encoder.encode(x).cpu().numpy())
                x_shift = shift_manager.apply_batch(x, args.shift_type, args.intensity)
                z_shift.append(encoder.encode(x_shift).cpu().numpy())
        z_clean = np.concatenate(z_clean, axis=0)
        z_shift = np.concatenate(z_shift, axis=0)
        X = np.concatenate([z_clean, z_shift], axis=0)
        y = np.concatenate([np.zeros(len(z_clean)), np.ones(len(z_shift))], axis=0)
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(max_iter=1000).fit(X, y)
        w = torch.tensor(clf.coef_.reshape(-1), dtype=torch.float32)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"probe_w": w}, args.out)
        print(f"Saved fast adaptor probe_w to {args.out}")


if __name__ == "__main__":
    main()
