from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelOutput:
    logits: torch.Tensor
    probs: torch.Tensor
    uncertainty: Optional[torch.Tensor] = None
    aleatoric: Optional[torch.Tensor] = None
    epistemic: Optional[torch.Tensor] = None
    features: Optional[torch.Tensor] = None


class BaseModel(nn.Module):
    def __init__(self, num_classes: int, in_chans: int = 1):
        super().__init__()
        self.num_classes = num_classes
        self.in_chans = in_chans

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward_with_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.forward_features(x)
        logits = self.forward(x)
        return logits, features

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x)
        return torch.sigmoid(logits)

    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        return (self.predict_proba(x) >= threshold).float()

    def enable_dropout(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()

    def mc_dropout_predict(self, x: torch.Tensor, passes: int = 20) -> Tuple[torch.Tensor, torch.Tensor]:
        self.enable_dropout()
        probs = []
        for _ in range(passes):
            probs.append(self.predict_proba(x).unsqueeze(0))
        probs = torch.cat(probs, dim=0)
        mean = probs.mean(dim=0)
        entropy = -(mean * torch.log(mean + 1e-8) + (1 - mean) * torch.log(1 - mean + 1e-8))
        return mean, entropy
