from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn

from .eraser import BaseEraser, SteeringEraser, load_eraser


class MixtureEraserAdaptor(nn.Module):
    def __init__(self, erasers: List[BaseEraser], weight_net: nn.Module):
        super().__init__()
        self.erasers = nn.ModuleList(erasers)
        self.weight_net = weight_net

    def forward(self, z: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        if scores is None:
            raise RuntimeError("MixtureEraserAdaptor requires shift scores.")
        weights = torch.softmax(self.weight_net(scores), dim=-1)
        outs = []
        for eraser in self.erasers:
            outs.append(eraser(z))
        stacked = torch.stack(outs, dim=-1)
        return (stacked * weights.unsqueeze(1)).sum(dim=-1)


class FastSteeringAdaptor(nn.Module):
    def __init__(
        self,
        steering_eraser: SteeringEraser,
        probe_w: torch.Tensor,
        head_w: Optional[torch.Tensor],
        ridge_lambda: float = 1e-2,
        guard_eps: float = 0.01,
    ):
        super().__init__()
        self.eraser = steering_eraser
        self.register_buffer("probe_w", probe_w)
        self.head_w = head_w
        self.ridge_lambda = ridge_lambda
        self.guard_eps = guard_eps

        V = steering_eraser.V.detach()
        w = probe_w.detach().unsqueeze(1)  # [D,1]
        A = V.T @ (w @ w.T) @ V
        if head_w is not None:
            H = head_w.detach()
            A = A + V.T @ (H.T @ H) @ V
        A = A + ridge_lambda * torch.eye(A.shape[0], device=A.device, dtype=A.dtype)
        self.register_buffer("A_inv", torch.inverse(A))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        V = self.eraser.V
        w = self.probe_w
        b = (z @ w)  # [N]
        vtw = (V.T @ w).view(1, -1)  # [1,K]
        rhs = b.view(-1, 1) * vtw  # [N,K]
        alpha = (self.A_inv @ rhs.T).T  # [N,K]
        z_edit = z - alpha @ V.T
        return z_edit


class EntropySteeringAdaptor(nn.Module):
    def __init__(self, steering_eraser: SteeringEraser, head: nn.Module, steps: int = 10, lr: float = 1e-2, guard_eps: float = 0.01):
        super().__init__()
        self.eraser = steering_eraser
        self.head = head
        self.steps = steps
        self.lr = lr
        self.guard_eps = guard_eps

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        V = self.eraser.V.detach()
        alpha = torch.zeros(z.shape[0], V.shape[1], device=z.device, dtype=z.dtype, requires_grad=True)
        opt = torch.optim.SGD([alpha], lr=self.lr)
        with torch.no_grad():
            logits0 = self.head(z).detach()
        for _ in range(self.steps):
            opt.zero_grad()
            z_edit = z - alpha @ V.T
            logits = self.head(z_edit)
            probs = torch.sigmoid(logits)
            entropy = -(probs * torch.log(probs + 1e-8) + (1 - probs) * torch.log(1 - probs + 1e-8)).mean()
            guard = (logits - logits0).pow(2).mean()
            loss = entropy + torch.relu(guard - self.guard_eps)
            loss.backward()
            opt.step()
        return z - alpha.detach() @ V.T


def load_eraser_bank(paths: List[str]) -> List[BaseEraser]:
    erasers = []
    for p in paths:
        erasers.append(load_eraser(p))
    return erasers
