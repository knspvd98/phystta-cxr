from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn.functional as F

from .base import DomainShift


class ProtocolShift(DomainShift):
    def __init__(self):
        super().__init__("protocol")

    def apply(self, x: torch.Tensor, intensity: float, rng=None, metadata=None) -> torch.Tensor:
        if rng is None:
            rng = np.random.default_rng()
        angle = (rng.random() - 0.5) * intensity * 10.0
        x = self._rotate(x, angle)
        scale = 1.0 - 0.1 * intensity
        x = self._rescale(x, scale)
        return x

    def physical_params(self, intensity: float):
        return {
            "protocol_deviation": float(intensity),
            "rotation_deg": float(intensity * 10.0),
            "scale_factor": float(1.0 - 0.1 * intensity),
        }

    def _rotate(self, x: torch.Tensor, angle: float) -> torch.Tensor:
        theta = torch.tensor([
            [math.cos(math.radians(angle)), -math.sin(math.radians(angle)), 0.0],
            [math.sin(math.radians(angle)), math.cos(math.radians(angle)), 0.0],
        ], device=x.device, dtype=x.dtype)
        theta = theta.unsqueeze(0)
        grid = F.affine_grid(theta, size=(1, 1, x.shape[1], x.shape[2]), align_corners=False)
        rotated = F.grid_sample(x.unsqueeze(0), grid, align_corners=False)
        return rotated.squeeze(0)

    def _rescale(self, x: torch.Tensor, scale: float) -> torch.Tensor:
        _, h, w = x.shape
        new_h = max(4, int(h * scale))
        new_w = max(4, int(w * scale))
        x_small = F.interpolate(x.unsqueeze(0), size=(new_h, new_w), mode="bilinear", align_corners=False)
        x_up = F.interpolate(x_small, size=(h, w), mode="bilinear", align_corners=False)
        return x_up.squeeze(0)
