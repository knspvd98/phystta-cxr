from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch


class DomainShift:
    def __init__(self, name: str):
        self.name = name

    def apply(self, x: torch.Tensor, intensity: float, rng: Optional[np.random.Generator] = None, metadata: Optional[Dict] = None) -> torch.Tensor:
        raise NotImplementedError

    def physical_params(self, intensity: float) -> Dict[str, float]:
        return {"intensity": float(intensity)}


class ShiftComposer:
    def __init__(self, shifts):
        self.shifts = shifts

    def apply(self, x: torch.Tensor, intensities: Dict[str, float], rng: Optional[np.random.Generator] = None, metadata: Optional[Dict] = None) -> torch.Tensor:
        out = x
        for shift in self.shifts:
            level = intensities.get(shift.name, 0.0)
            if level > 0:
                out = shift.apply(out, level, rng=rng, metadata=metadata)
        return out
