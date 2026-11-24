from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .eraser import BaseEraser
from ..models.encoder_factory import EncoderWrapper


class TokenEraserWrapper(nn.Module):
    """
    Applies an eraser to token embeddings before pooling.
    Supports ViT/CLIP-style encoders with token outputs.
    """

    def __init__(self, encoder: EncoderWrapper, eraser: BaseEraser, mode: str = "cls", token_layers: int = 1):
        super().__init__()
        self.encoder = encoder
        self.eraser = eraser
        self.mode = mode
        self.token_layers = token_layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.encoder.encode_tokens(x, token_layers=self.token_layers)
        if tokens.dim() != 3:
            raise RuntimeError("TokenEraserWrapper expects [N,T,D] tokens.")
        if self.mode == "cls":
            cls = tokens[:, 0]
            cls = self.eraser(cls)
            tokens = tokens.clone()
            tokens[:, 0] = cls
        elif self.mode == "mean":
            mean_tok = tokens.mean(dim=1)
            mean_edit = self.eraser(mean_tok)
            delta = (mean_edit - mean_tok).unsqueeze(1)
            tokens = tokens + delta
        elif self.mode == "patch":
            if tokens.shape[1] < 2:
                raise RuntimeError("patch mode requires patch tokens.")
            patches = tokens[:, 1:].reshape(-1, tokens.shape[-1])
            patches = self.eraser(patches).reshape(tokens.shape[0], tokens.shape[1] - 1, tokens.shape[-1])
            tokens = tokens.clone()
            tokens[:, 1:] = patches
        elif self.mode == "all":
            flat = tokens.reshape(-1, tokens.shape[-1])
            flat = self.eraser(flat)
            tokens = flat.reshape(tokens.shape[0], tokens.shape[1], tokens.shape[-1])
        else:
            raise RuntimeError(f"Unknown token erasure mode: {self.mode}")
        pooled = self.encoder.pool_tokens(tokens)
        return pooled
