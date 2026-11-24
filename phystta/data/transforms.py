from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F

from ..config import PreprocessConfig


def build_preprocess_transform(cfg: PreprocessConfig) -> Callable[[torch.Tensor], torch.Tensor]:
    def _transform(x: torch.Tensor) -> torch.Tensor:
        if cfg.normalize_mean != 0.0 or cfg.normalize_std != 1.0:
            raise RuntimeError("Dataset-level normalization is запрещено. Move normalization to encoder input_adapter.")
        if x.dim() == 2:
            x = x.unsqueeze(0)
        if cfg.clip_min is not None or cfg.clip_max is not None:
            x = torch.clamp(x, cfg.clip_min, cfg.clip_max)
        _, h, w = x.shape
        target = cfg.image_size
        if h != target or w != target:
            x = x.unsqueeze(0)
            x = F.interpolate(x, size=(target, target), mode="bilinear", align_corners=False)
            x = x.squeeze(0)
        if cfg.center_crop is not None and cfg.center_crop < target:
            crop = cfg.center_crop
            start_h = (target - crop) // 2
            start_w = (target - crop) // 2
            x = x[:, start_h:start_h + crop, start_w:start_w + crop]
        return x
    return _transform
