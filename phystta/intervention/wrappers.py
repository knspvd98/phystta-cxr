from __future__ import annotations

import torch
import torch.nn as nn


class EraserWrappedModel(nn.Module):
    def __init__(self, base_model: nn.Module, encoder, eraser: nn.Module, head_attr: str | None = None, head: nn.Module | None = None):
        super().__init__()
        self.base_model = base_model
        self.encoder = encoder
        self.eraser = eraser
        if head is not None:
            self.head = head
        else:
            if head_attr is not None and hasattr(base_model, head_attr):
                self.head = getattr(base_model, head_attr)
            elif hasattr(base_model, "classifier"):
                self.head = base_model.classifier
            elif hasattr(base_model, "head"):
                self.head = base_model.head
            elif hasattr(base_model, "fc"):
                self.head = base_model.fc
            else:
                raise RuntimeError("Head not found. Provide head_attr in config or pass explicit head.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder.encode(x)
        z_clean = self.eraser(z)
        return self.head(z_clean)
