from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GroupShuffleSplit, ShuffleSplit


def _image_stats(x: torch.Tensor) -> np.ndarray:
    if x.dim() != 4:
        raise ValueError("x must be [N,C,H,W]")
    mins = x.amin(dim=(1, 2, 3)).cpu().numpy()
    maxs = x.amax(dim=(1, 2, 3)).cpu().numpy()
    means = x.mean(dim=(1, 2, 3)).cpu().numpy()
    stds = x.std(dim=(1, 2, 3)).cpu().numpy()
    q10 = torch.quantile(x.view(x.shape[0], -1), 0.10, dim=1).cpu().numpy()
    q50 = torch.quantile(x.view(x.shape[0], -1), 0.50, dim=1).cpu().numpy()
    q90 = torch.quantile(x.view(x.shape[0], -1), 0.90, dim=1).cpu().numpy()
    return np.stack([mins, maxs, means, stds, q10, q50, q90], axis=1).astype(np.float32)


def collect_dicom_leakage_features(
    loader,
    group_col: str = "manufacturer",
    fallback_col: str = "station_name",
    max_batches: int = 100,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    X = []
    y = []
    groups = []
    for bidx, batch in enumerate(loader):
        if bidx >= max_batches:
            break
        x = batch["image"].float()
        X.append(_image_stats(x))
        if group_col in batch:
            yy = np.asarray(batch[group_col]).astype(str)
        elif fallback_col in batch:
            yy = np.asarray(batch[fallback_col]).astype(str)
        else:
            yy = np.asarray(["unknown"] * x.shape[0])
        y.append(yy)
        if "global_patient_id" in batch:
            groups.append(np.asarray(batch["global_patient_id"]).astype(str))
        elif "patient_id" in batch:
            groups.append(np.asarray(batch["patient_id"]).astype(str))
        else:
            groups.append(np.asarray(["unknown"] * x.shape[0]))
    if not X:
        return np.zeros((0, 7), dtype=np.float32), np.zeros((0,), dtype=str), None
    Xn = np.concatenate(X, axis=0)
    yn = np.concatenate(y, axis=0)
    gn = np.concatenate(groups, axis=0) if groups else None
    return Xn, yn, gn


def audit_site_leakage(
    X: np.ndarray,
    y: np.ndarray,
    groups: Optional[np.ndarray] = None,
    seed: int = 2024,
) -> Dict[str, float]:
    valid = y != "unknown"
    X = X[valid]
    y = y[valid]
    g = groups[valid] if groups is not None else None
    if len(np.unique(y)) < 2 or len(X) < 20:
        return {"site_leakage_acc": float("nan"), "n": int(len(X)), "n_classes": int(len(np.unique(y)))}
    if g is not None and len(np.unique(g)) > 1:
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        tr, te = next(splitter.split(X, y, groups=g))
    else:
        splitter = ShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        tr, te = next(splitter.split(X, y))
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X[tr], y[tr])
    pred = clf.predict(X[te])
    return {
        "site_leakage_acc": float(accuracy_score(y[te], pred)),
        "n": int(len(X)),
        "n_classes": int(len(np.unique(y))),
    }
