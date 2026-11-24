from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass
class ShiftScenario:
    name: str
    intensities: Dict[str, float]

    def id(self) -> str:
        parts = [f"{k}={v:.3f}" for k, v in sorted(self.intensities.items()) if v > 0]
        if not parts:
            return f"{self.name}:none"
        return f"{self.name}:" + ";".join(parts)
