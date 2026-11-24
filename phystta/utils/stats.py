from __future__ import annotations

from typing import Dict, Any

import torch


def tensor_stats(x: torch.Tensor) -> Dict[str, Any]:
    if x.dim() == 3:
        x = x.unsqueeze(0)
    if x.dim() != 4:
        raise ValueError("Expected tensor with shape [N,C,H,W] or [C,H,W]")
    stats = {}
    # per-channel stats
    mins = x.amin(dim=(0, 2, 3)).detach().cpu().tolist()
    maxs = x.amax(dim=(0, 2, 3)).detach().cpu().tolist()
    means = x.mean(dim=(0, 2, 3)).detach().cpu().tolist()
    stds = x.std(dim=(0, 2, 3)).detach().cpu().tolist()
    stats["min"] = mins
    stats["max"] = maxs
    stats["mean"] = means
    stats["std"] = stds
    stats["shape"] = list(x.shape)
    return stats


def preprocess_signature(cfg) -> Dict[str, Any]:
    adapter = cfg.model.encoder.input_adapter
    return {
        "dataset_output": getattr(cfg.data.preprocess, "dataset_output", "0_1"),
        "image_size": cfg.data.preprocess.image_size,
        "center_crop": cfg.data.preprocess.center_crop,
        "clip_min": cfg.data.preprocess.clip_min,
        "clip_max": cfg.data.preprocess.clip_max,
        "adapter_repeat_to_3": getattr(adapter, "repeat_to_3", None) if adapter else None,
        "adapter_image_size": getattr(adapter, "image_size", None) if adapter else None,
        "adapter_mean": getattr(adapter, "mean", None) if adapter else None,
        "adapter_std": getattr(adapter, "std", None) if adapter else None,
    }
