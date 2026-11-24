from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, roc_auc_score, mean_absolute_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, ShuffleSplit


@dataclass
class ProbeSuiteConfig:
    seed: int = 2024
    test_size: float = 0.2
    rff_dim: int = 512
    hidden_dim: int = 128
    epochs: int = 200
    patience: int = 20
    lr: float = 1e-3


def build_presence_task(
    z_clean: np.ndarray,
    z_shifted: np.ndarray,
    groups: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    X = np.concatenate([z_clean, z_shifted], axis=0)
    y = np.concatenate([np.zeros(len(z_clean)), np.ones(len(z_shifted))], axis=0)
    g = None if groups is None else np.concatenate([groups, groups], axis=0)
    return X, y.astype(float), g


def build_intensity_task(
    z_clean: np.ndarray,
    z_shifted: np.ndarray,
    intensities: np.ndarray,
    groups: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    X = np.concatenate([z_clean, z_shifted], axis=0)
    y = np.concatenate([np.zeros(len(z_clean)), intensities.astype(float)], axis=0)
    g = None if groups is None else np.concatenate([groups, groups], axis=0)
    return X, y, g


def _split_indices(
    X: np.ndarray,
    y: np.ndarray,
    groups: Optional[np.ndarray],
    cfg: ProbeSuiteConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    if groups is not None and len(np.unique(groups)) > 1:
        gss = GroupShuffleSplit(n_splits=1, test_size=cfg.test_size, random_state=cfg.seed)
        tr, te = next(gss.split(X, y, groups=groups))
        return tr, te
    ss = ShuffleSplit(n_splits=1, test_size=cfg.test_size, random_state=cfg.seed)
    tr, te = next(ss.split(X, y))
    return tr, te


def _median_rff_sigma(X: np.ndarray) -> float:
    if len(X) > 1024:
        rng = np.random.default_rng(2024)
        X = X[rng.choice(len(X), 1024, replace=False)]
    d = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=-1)
    med = float(np.median(d))
    return max(med, 1e-3)


def _rff_transform(X: np.ndarray, dim: int, sigma: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    W = rng.normal(loc=0.0, scale=1.0 / sigma, size=(X.shape[1], dim)).astype(np.float32)
    b = rng.uniform(0.0, 2.0 * np.pi, size=(dim,)).astype(np.float32)
    return np.sqrt(2.0 / dim) * np.cos(X @ W + b)


class _MLPProbe(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _fit_mlp_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    cfg: ProbeSuiteConfig,
) -> Dict[str, float]:
    torch.manual_seed(cfg.seed)
    model = _MLPProbe(X_train.shape[1], cfg.hidden_dim, 1)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    best = None
    best_loss = float("inf")
    patience = 0
    xtr = torch.tensor(X_train, dtype=torch.float32)
    ytr = torch.tensor(y_train, dtype=torch.float32)
    xva = torch.tensor(X_val, dtype=torch.float32)
    yva = torch.tensor(y_val, dtype=torch.float32)
    for _ in range(cfg.epochs):
        model.train()
        opt.zero_grad()
        logits = model(xtr).squeeze(-1)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, ytr)
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            val_logits = model(xva).squeeze(-1)
            val_loss = nn.functional.binary_cross_entropy_with_logits(val_logits, yva).item()
        if val_loss < best_loss - 1e-6:
            best_loss = val_loss
            best = {k: v.detach().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= cfg.patience:
                break
    if best is not None:
        model.load_state_dict(best)
    model.eval()
    with torch.no_grad():
        logits = model(xva).squeeze(-1).numpy()
    probs = 1.0 / (1.0 + np.exp(-logits))
    pred = (probs >= 0.5).astype(float)
    auc = roc_auc_score(y_val, probs) if len(np.unique(y_val)) > 1 else float("nan")
    return {"acc": float(accuracy_score(y_val, pred)), "auc": float(auc)}


def _fit_mlp_regressor(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    cfg: ProbeSuiteConfig,
) -> Dict[str, float]:
    torch.manual_seed(cfg.seed)
    model = _MLPProbe(X_train.shape[1], cfg.hidden_dim, 1)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    best = None
    best_loss = float("inf")
    patience = 0
    xtr = torch.tensor(X_train, dtype=torch.float32)
    ytr = torch.tensor(y_train, dtype=torch.float32)
    xva = torch.tensor(X_val, dtype=torch.float32)
    yva = torch.tensor(y_val, dtype=torch.float32)
    for _ in range(cfg.epochs):
        model.train()
        opt.zero_grad()
        pred = model(xtr).squeeze(-1)
        loss = nn.functional.huber_loss(pred, ytr, delta=1.0)
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            val_pred = model(xva).squeeze(-1)
            val_loss = nn.functional.huber_loss(val_pred, yva, delta=1.0).item()
        if val_loss < best_loss - 1e-6:
            best_loss = val_loss
            best = {k: v.detach().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= cfg.patience:
                break
    if best is not None:
        model.load_state_dict(best)
    model.eval()
    with torch.no_grad():
        pred = model(xva).squeeze(-1).numpy()
    return {"mae": float(mean_absolute_error(y_val, pred)), "r2": float(r2_score(y_val, pred))}


def evaluate_probe_suite(
    X: np.ndarray,
    y: np.ndarray,
    groups: Optional[np.ndarray] = None,
    target: str = "presence",
    cfg: Optional[ProbeSuiteConfig] = None,
) -> Dict[str, float]:
    """
    Fits adaptive probes after erasure and returns residual detectability metrics.
    target: presence | intensity
    """
    if cfg is None:
        cfg = ProbeSuiteConfig()
    tr, te = _split_indices(X, y, groups, cfg)
    X_tr, X_te = X[tr], X[te]
    y_tr, y_te = y[tr], y[te]
    out: Dict[str, float] = {}
    if target == "presence":
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_tr, y_tr)
        p = clf.predict_proba(X_te)[:, 1]
        pred = (p >= 0.5).astype(float)
        out["linear_acc"] = float(accuracy_score(y_te, pred))
        out["linear_auc"] = float(roc_auc_score(y_te, p)) if len(np.unique(y_te)) > 1 else float("nan")

        sigma = _median_rff_sigma(X_tr)
        X_tr_phi = _rff_transform(X_tr, cfg.rff_dim, sigma=sigma, seed=cfg.seed)
        X_te_phi = _rff_transform(X_te, cfg.rff_dim, sigma=sigma, seed=cfg.seed)
        clf_phi = LogisticRegression(max_iter=1000)
        clf_phi.fit(X_tr_phi, y_tr)
        p_phi = clf_phi.predict_proba(X_te_phi)[:, 1]
        pred_phi = (p_phi >= 0.5).astype(float)
        out["kernel_acc"] = float(accuracy_score(y_te, pred_phi))
        out["kernel_auc"] = float(roc_auc_score(y_te, p_phi)) if len(np.unique(y_te)) > 1 else float("nan")

        out.update({f"mlp_{k}": v for k, v in _fit_mlp_classifier(X_tr, y_tr, X_te, y_te, cfg).items()})
    elif target == "intensity":
        reg = Ridge(alpha=1.0)
        reg.fit(X_tr, y_tr)
        p = reg.predict(X_te)
        out["linear_mae"] = float(mean_absolute_error(y_te, p))
        out["linear_r2"] = float(r2_score(y_te, p))

        sigma = _median_rff_sigma(X_tr)
        X_tr_phi = _rff_transform(X_tr, cfg.rff_dim, sigma=sigma, seed=cfg.seed)
        X_te_phi = _rff_transform(X_te, cfg.rff_dim, sigma=sigma, seed=cfg.seed)
        reg_phi = Ridge(alpha=1.0)
        reg_phi.fit(X_tr_phi, y_tr)
        p_phi = reg_phi.predict(X_te_phi)
        out["kernel_mae"] = float(mean_absolute_error(y_te, p_phi))
        out["kernel_r2"] = float(r2_score(y_te, p_phi))

        out.update({f"mlp_{k}": v for k, v in _fit_mlp_regressor(X_tr, y_tr, X_te, y_te, cfg).items()})
    else:
        raise ValueError(f"Unknown target {target}")
    out["n_train"] = int(len(tr))
    out["n_test"] = int(len(te))
    out["group_split"] = float(groups is not None)
    return out
