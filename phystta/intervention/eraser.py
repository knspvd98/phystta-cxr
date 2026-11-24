from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn

from ..utils import load_torch_artifact


@dataclass
class EraserMeta:
    k: int
    dim: int
    method: str
    seed: int
    created_at: str
    note: Optional[str] = None
    explained_variance: Optional[list] = None
    probe_metric: Optional[Dict[str, float]] = None
    group_split: Optional[str] = None


class BaseEraser(nn.Module):
    eraser_type: str = "base"

    def __init__(self, meta: Optional[Dict] = None):
        super().__init__()
        self.meta = meta or {}

    def to_state(self) -> Dict:
        return {"type": self.eraser_type, "state_dict": self.state_dict(), "meta": self.meta}


def _orthonormalize_torch(V: torch.Tensor) -> torch.Tensor:
    if V.dim() != 2:
        raise ValueError("V must be 2D [D, k]")
    q, _ = torch.linalg.qr(V, mode="reduced")
    return q


def _orthonormalize_np(V: np.ndarray) -> np.ndarray:
    q, _ = np.linalg.qr(V)
    return q


def _project_out_subspace(V: torch.Tensor, U: Optional[torch.Tensor]) -> torch.Tensor:
    if U is None:
        return V
    if U.dim() != 2:
        raise ValueError("Task basis U must be 2D [D, r]")
    U = _orthonormalize_torch(U)
    proj = U @ U.T
    V = V - proj @ V
    return V


def task_subspace_from_head(head: nn.Module, critical_idx: Optional[List[int]] = None, k: Optional[int] = None) -> torch.Tensor:
    if not hasattr(head, "weight"):
        raise RuntimeError("Head must be linear to compute task subspace (requires .weight).")
    W = head.weight.detach().float()
    if critical_idx:
        W = W[critical_idx]
    if k is not None and k < W.shape[0]:
        W = W[:k]
    # Head weights are [num_classes, feat_dim]. The task basis must live in feature space [feat_dim, r],
    # so we use right singular vectors rather than class-space left singular vectors.
    _, _, Vh = torch.linalg.svd(W, full_matrices=False)
    basis = Vh.T
    return basis[:, : min(basis.shape[1], W.shape[0])]


class LinearSubspaceEraser(BaseEraser):
    eraser_type = "linear"
    def __init__(self, V: torch.Tensor, meta: Optional[Dict] = None):
        super().__init__(meta=meta)
        V = _orthonormalize_torch(V)
        self.register_buffer("V", V)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        V = self.V
        return x - (x @ V) @ V.T

    def validate_projection(self, x: torch.Tensor, atol: float = 1e-4) -> Dict[str, float]:
        V = self.V
        vt_v = V.T @ V
        ortho_err = float((vt_v - torch.eye(vt_v.shape[0], device=vt_v.device)).abs().max().item())
        x_proj = self.forward(x)
        leakage = float((x_proj @ V).abs().max().item())
        return {"orthonormal_err": ortho_err, "projection_leakage": leakage, "ok": float(ortho_err < atol and leakage < atol)}

    def state_dict(self, *args, **kwargs):
        sd = {"V": self.V}
        sd["_meta"] = self.meta
        return sd

    def load_state_dict(self, state_dict, strict: bool = True):
        meta = state_dict.pop("_meta", None)
        if meta is not None:
            self.meta = meta
        if "V" not in state_dict:
            if strict:
                raise RuntimeError("state_dict missing V")
            return
        V = state_dict["V"]
        V = _orthonormalize_torch(V)
        if hasattr(self, "V") and self.V.shape == V.shape:
            self.V.copy_(V)
        else:
            self.register_buffer("V", V)


class MatrixProjectionEraser(BaseEraser):
    eraser_type = "matrix_projection"

    def __init__(self, P: torch.Tensor, meta: Optional[Dict] = None):
        super().__init__(meta=meta)
        if P.dim() != 2 or P.shape[0] != P.shape[1]:
            raise ValueError("Projection matrix P must be square [D,D].")
        self.register_buffer("P", P)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.P

    def validate_projection(self, atol: float = 1e-4) -> Dict[str, float]:
        P = self.P
        idem = float((P @ P - P).abs().max().item())
        symm = float((P - P.T).abs().max().item())
        return {"idempotence_err": idem, "symmetry_err": symm, "ok": float(idem < atol and symm < atol)}

    def state_dict(self, *args, **kwargs):
        sd = {"P": self.P}
        sd["_meta"] = self.meta
        return sd

    def load_state_dict(self, state_dict, strict: bool = True):
        meta = state_dict.pop("_meta", None)
        if meta is not None:
            self.meta = meta
        if "P" not in state_dict:
            if strict:
                raise RuntimeError("state_dict missing P")
            return
        P = state_dict["P"]
        if hasattr(self, "P") and self.P.shape == P.shape:
            self.P.copy_(P)
        else:
            self.register_buffer("P", P)


class KernelEraser(BaseEraser):
    eraser_type = "kernel"

    def __init__(
        self,
        W: torch.Tensor,
        b: torch.Tensor,
        V_phi: torch.Tensor,
        decoder: torch.Tensor,
        meta: Optional[Dict] = None,
    ):
        super().__init__(meta=meta)
        self.register_buffer("W", W)
        self.register_buffer("b", b)
        self.register_buffer("V_phi", _orthonormalize_torch(V_phi))
        self.register_buffer("decoder", decoder)

    @staticmethod
    def _phi(x: torch.Tensor, W: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(torch.tensor(2.0 / W.shape[1], device=x.device, dtype=x.dtype)) * torch.cos(x @ W + b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        phi = self._phi(x, self.W, self.b)
        phi = phi - (phi @ self.V_phi) @ self.V_phi.T
        return phi @ self.decoder


class SteeringEraser(BaseEraser):
    eraser_type = "steering"

    def __init__(self, V: torch.Tensor, mlp: nn.Module, meta: Optional[Dict] = None):
        super().__init__(meta=meta)
        self.V = nn.Parameter(V)
        self.mlp = mlp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha = self.mlp(x)
        if alpha.dim() != 2:
            raise RuntimeError("SteeringEraser MLP must output [N,K]")
        return x - alpha @ self.V.T

    def state_dict(self, *args, **kwargs):
        return super().state_dict(*args, **kwargs)


class TaskPreservingEraser(BaseEraser):
    eraser_type = "task_preserve"

    def __init__(
        self,
        eraser: BaseEraser,
        task_basis: torch.Tensor,
        mode: str = "residualize",
        preserve_strength: float = 1.0,
        meta: Optional[Dict] = None,
    ):
        super().__init__(meta=meta or {})
        self.eraser = eraser
        self.register_buffer("task_basis", _orthonormalize_torch(task_basis))
        self.mode = mode
        self.preserve_strength = float(preserve_strength)
        if not (0.0 <= self.preserve_strength <= 1.0):
            raise ValueError("preserve_strength must be in [0.0, 1.0]")

    def _projected_basis(self) -> Optional[torch.Tensor]:
        if not hasattr(self.eraser, "V"):
            return None
        V = getattr(self.eraser, "V")
        if V is None or V.dim() != 2:
            return None
        Vp = _project_out_subspace(V, self.task_basis)
        if Vp.numel() == 0:
            return None
        keep = torch.norm(Vp, dim=0) > 1e-8
        if keep.sum() == 0:
            return None
        Vp = Vp[:, keep]
        return _orthonormalize_torch(Vp)

    def _apply_head_subspace(self, x: torch.Tensor) -> torch.Tensor:
        # Enforce erasure directions to be task-orthogonal and apply on full feature.
        Vp = self._projected_basis()
        if Vp is None:
            # Fallback for erasers without explicit basis: residual-only editing.
            z_task = x @ self.task_basis @ self.task_basis.T
            z_res = x - z_task
            z_res = self.eraser(z_res)
            return z_task + z_res
        if isinstance(self.eraser, LinearSubspaceEraser):
            return x - (x @ Vp) @ Vp.T
        if isinstance(self.eraser, SteeringEraser):
            alpha = self.eraser.mlp(x)
            return x - alpha @ Vp.T
        z_task = x @ self.task_basis @ self.task_basis.T
        z_res = x - z_task
        z_res = self.eraser(z_res)
        return z_task + z_res

    def _apply_head_subspace_soft(self, x: torch.Tensor) -> torch.Tensor:
        preserved = self._apply_head_subspace(x)
        raw = self.eraser(x)
        s = self.preserve_strength
        return s * preserved + (1.0 - s) * raw

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        U = self.task_basis
        z_task = x @ U @ U.T

        if self.mode == "head_subspace":
            return self._apply_head_subspace(x)

        if self.mode == "head_subspace_soft":
            return self._apply_head_subspace_soft(x)

        if self.mode == "residualize":
            z_res = x - z_task
            z_res = self.eraser(z_res)
            return z_task + z_res

        raise RuntimeError(f"Unknown task_preserve mode: {self.mode}")

    def to_state(self) -> Dict:
        payload = {
            "type": self.eraser_type,
            "state_dict": {"task_basis": self.task_basis},
            "meta": {
                **self.meta,
                "task_preserve": self.mode,
                "preserve_strength": float(self.preserve_strength),
            },
            "inner": self.eraser.to_state(),
        }
        return payload


class BiasSubspaceFitter:
    def __init__(self, seed: int = 2024):
        self.seed = seed
        np.random.seed(seed)
        torch.manual_seed(seed)

    @staticmethod
    def _get_groups(patient_ids, study_ids):
        def _valid(arr):
            if arr is None:
                return False
            if len(arr) == 0:
                return False
            uniq = set([str(x) for x in arr])
            if len(uniq) <= 1:
                return False
            if uniq == {"unknown"} or uniq == {"nan"}:
                return False
            return True
        if _valid(patient_ids):
            return patient_ids, "patient_id"
        if _valid(study_ids):
            return study_ids, "study_id"
        return None, "no-group-split"

    @staticmethod
    def _orthonormalize(vectors: np.ndarray) -> np.ndarray:
        return _orthonormalize_np(vectors)

    def fit_svd(self, z_clean: np.ndarray, z_shifted: np.ndarray, k: int) -> Tuple[np.ndarray, Dict[str, float]]:
        dmat = z_shifted - z_clean
        u, s, vt = np.linalg.svd(dmat, full_matrices=False)
        directions = vt[:k].T
        explained = (s ** 2) / max((s ** 2).sum(), 1e-8)
        meta = {
            "explained_variance": explained[:k].tolist(),
            "linearity_score": float(np.sum(explained[:k])),
            "method": "svd",
        }
        return self._orthonormalize(directions), meta

    def fit_leace(
        self,
        z_clean: np.ndarray,
        z_shifted: np.ndarray,
        intensities: np.ndarray,
        probe_target: str = "presence",
        ridge: float = 1e-4,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        LEACE-like closed-form linear erasure:
        Build linear projection P that removes linear predictability of nuisance S.
        """
        Z = np.concatenate([z_clean, z_shifted], axis=0).astype(np.float64)
        if probe_target == "presence":
            s = np.concatenate([np.zeros((len(z_clean), 1)), np.ones((len(z_shifted), 1))], axis=0).astype(np.float64)
        else:
            s = np.concatenate([np.zeros((len(z_clean), 1)), intensities.reshape(-1, 1)], axis=0).astype(np.float64)
        Zc = Z - Z.mean(axis=0, keepdims=True)
        Sc = s - s.mean(axis=0, keepdims=True)
        n = max(len(Zc), 1)
        Czz = (Zc.T @ Zc) / n
        Czs = (Zc.T @ Sc) / n
        D = Czz.shape[0]
        Czz = Czz + ridge * np.eye(D, dtype=np.float64)
        W = np.linalg.solve(Czz, Czs)  # [D,m]
        M = Czs.T @ W  # [m,m]
        M_pinv = np.linalg.pinv(M, rcond=1e-8)
        P = np.eye(D, dtype=np.float64) - W @ M_pinv @ Czs.T
        # Symmetrize for numerical stability.
        P = 0.5 * (P + P.T)
        # Empirical residual linear predictability proxy.
        Z_after = Zc @ P
        Czs_after = (Z_after.T @ Sc) / n
        leakage = float(np.linalg.norm(Czs_after))
        meta = {
            "method": "leace",
            "ridge": float(ridge),
            "leakage_norm": leakage,
            "target_dim": int(s.shape[1]),
        }
        return P.astype(np.float32), meta

    def fit_probe(
        self,
        z_clean: np.ndarray,
        z_shifted: np.ndarray,
        intensities: np.ndarray,
        patient_ids: Optional[np.ndarray],
        study_ids: Optional[np.ndarray],
        k: int,
        probe_target: str = "presence",
        require_group: bool = True,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        from sklearn.linear_model import LogisticRegression, Ridge
        from sklearn.model_selection import GroupShuffleSplit

        X = np.concatenate([z_clean, z_shifted], axis=0)
        if probe_target == "presence":
            y = np.concatenate([np.zeros(len(z_clean)), np.ones(len(z_shifted))], axis=0)
            task = "clf"
        else:
            y = np.concatenate([np.zeros(len(z_clean)), intensities], axis=0)
            task = "reg"
        groups, group_key = self._get_groups(patient_ids, study_ids)
        if groups is None:
            if require_group:
                raise RuntimeError("Group key missing for probe_clf. NOT CLINICALLY VALID.")
            idx = np.arange(len(X))
            split = int(0.8 * len(X))
            train_idx, val_idx = idx[:split], idx[split:]
            group_key = "no-group-split"
        else:
            gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=self.seed)
            train_idx, val_idx = next(gss.split(X, y, groups=np.concatenate([groups, groups], axis=0)))
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        if task == "clf":
            clf = LogisticRegression(max_iter=1000)
            clf.fit(X_train, y_train)
            acc = float(clf.score(X_val, y_val))
            coef = clf.coef_
            directions = coef.T
            meta = {"probe_accuracy": acc, "method": "probe_clf", "group_split": group_key}
        else:
            reg = Ridge(alpha=1.0)
            reg.fit(X_train, y_train)
            pred = reg.predict(X_val)
            mae = float(np.mean(np.abs(pred - y_val)))
            directions = reg.coef_.reshape(-1, 1)
            meta = {"probe_mae": mae, "method": "probe_reg", "group_split": group_key}
        if group_key == "no-group-split":
            meta["warning"] = "no-group-split"

        if directions.shape[1] < k:
            # Fill remaining directions from SVD of delta
            fallback, svd_meta = self.fit_svd(z_clean, z_shifted, k)
            directions = np.concatenate([directions, fallback[:, directions.shape[1]:k]], axis=1)
            meta["fallback_svd"] = svd_meta.get("explained_variance")
        else:
            directions = directions[:, :k]
        directions = self._orthonormalize(directions)
        return directions, meta

    def fit_probe_xfit(
        self,
        z_clean: np.ndarray,
        z_shifted: np.ndarray,
        intensities: np.ndarray,
        patient_ids: Optional[np.ndarray],
        study_ids: Optional[np.ndarray],
        k: int,
        probe_target: str = "presence",
        n_splits: int = 5,
        require_group: bool = True,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        from sklearn.linear_model import LogisticRegression, Ridge
        from sklearn.model_selection import GroupKFold

        X = np.concatenate([z_clean, z_shifted], axis=0)
        if probe_target == "presence":
            y = np.concatenate([np.zeros(len(z_clean)), np.ones(len(z_shifted))], axis=0)
            task = "clf"
        else:
            y = np.concatenate([np.zeros(len(z_clean)), intensities], axis=0)
            task = "reg"
        groups, group_key = self._get_groups(patient_ids, study_ids)
        if groups is None:
            if require_group:
                raise RuntimeError("Group key missing for probe_xfit. NOT CLINICALLY VALID.")
            group_key = "no-group-split"
            groups = np.arange(len(z_clean))
        groups_all = np.concatenate([groups, groups], axis=0)
        n_groups = len(np.unique(groups_all))
        n_splits = max(2, min(n_splits, n_groups))
        gkf = GroupKFold(n_splits=n_splits)
        fold_metrics = []
        V_folds = []
        for fold_idx, (tr, te) in enumerate(gkf.split(X, y, groups=groups_all)):
            X_train, X_val = X[tr], X[te]
            y_train, y_val = y[tr], y[te]
            if task == "clf":
                clf = LogisticRegression(max_iter=1000)
                clf.fit(X_train, y_train)
                acc = float(clf.score(X_val, y_val))
                coef = clf.coef_
                directions = coef.T
                fold_metrics.append({"fold": fold_idx, "probe_accuracy": acc})
            else:
                reg = Ridge(alpha=1.0)
                reg.fit(X_train, y_train)
                pred = reg.predict(X_val)
                mae = float(np.mean(np.abs(pred - y_val)))
                directions = reg.coef_.reshape(-1, 1)
                fold_metrics.append({"fold": fold_idx, "probe_mae": mae})
            if directions.shape[1] < k:
                fallback, _ = self.fit_svd(z_clean, z_shifted, k)
                directions = np.concatenate([directions, fallback[:, directions.shape[1]:k]], axis=1)
            else:
                directions = directions[:, :k]
            V_fold = self._orthonormalize(directions)
            V_folds.append(V_fold)
        V_stack = np.concatenate(V_folds, axis=1)
        V_final = self._orthonormalize(V_stack)[:, :k]
        # stability via principal angles
        sims = []
        for i in range(len(V_folds)):
            for j in range(i + 1, len(V_folds)):
                m = V_folds[i].T @ V_folds[j]
                s = np.linalg.svd(m, compute_uv=False)
                sims.append(float(np.mean(np.abs(s))))
        stability = float(np.mean(sims)) if sims else float("nan")
        # explained variance from SVD on delta
        dmat = z_shifted - z_clean
        _, s, _ = np.linalg.svd(dmat, full_matrices=False)
        explained = (s ** 2) / max((s ** 2).sum(), 1e-8)
        meta = {
            "method": "probe_xfit",
            "fold_metrics": fold_metrics,
            "stability": stability,
            "explained_variance": explained[:k].tolist(),
            "linearity_score": float(np.sum(explained[:k])),
            "group_split": group_key,
        }
        if group_key == "no-group-split":
            meta["warning"] = "no-group-split"
        return V_final, meta

    def fit(
        self,
        method: str,
        z_clean: np.ndarray,
        z_shifted: np.ndarray,
        intensities: np.ndarray,
        patient_ids: Optional[np.ndarray],
        study_ids: Optional[np.ndarray],
        k: int,
        probe_target: str = "presence",
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        if method == "svd":
            V, meta = self.fit_svd(z_clean, z_shifted, k)
        elif method == "leace":
            P, meta = self.fit_leace(z_clean, z_shifted, intensities, probe_target=probe_target)
            meta["dim"] = int(P.shape[0])
            meta["seed"] = self.seed
            meta["created_at"] = datetime.now(timezone.utc).isoformat()
            return P, meta
        elif method == "probe_clf":
            V, meta = self.fit_probe(z_clean, z_shifted, intensities, patient_ids, study_ids, k, probe_target, require_group=True)
        elif method == "probe_xfit":
            V, meta = self.fit_probe_xfit(z_clean, z_shifted, intensities, patient_ids, study_ids, k, probe_target, require_group=True)
        else:
            raise ValueError(f"Unknown fit method {method}")
        meta["k"] = k
        meta["dim"] = int(V.shape[0])
        meta["seed"] = self.seed
        meta["created_at"] = datetime.now(timezone.utc).isoformat()
        return V, meta


def principal_angle_overlap(Va: np.ndarray, Vb: np.ndarray) -> float:
    m = Va.T @ Vb
    s = np.linalg.svd(m, compute_uv=False)
    return float(np.mean(np.abs(s)))


def sensitivity_slope(intensities: np.ndarray, response: np.ndarray) -> float:
    """
    Linear sensitivity slope of response to intervention intensity.

    NOTE: this is not an ATE estimand and must not be reported as causal ATE.
    """
    x = intensities.reshape(-1)
    y = response.reshape(-1)
    if len(x) < 2 or np.std(x) < 1e-6:
        return float("nan")
    coef = np.polyfit(x, y, 1)[0]
    return float(coef)


def ate_intensity_effect(intensities: np.ndarray, z_proj: np.ndarray) -> float:
    # Backward-compatible alias. Kept to avoid checkpoint/script breakage.
    return sensitivity_slope(intensities, z_proj)


def save_eraser(path: str, eraser: BaseEraser) -> None:
    payload = eraser.to_state()
    torch.save(payload, path)


def load_eraser(obj) -> BaseEraser:
    if isinstance(obj, (str, bytes)):
        payload = load_torch_artifact(obj, map_location="cpu")
    elif isinstance(obj, dict):
        payload = obj
    else:
        payload = load_torch_artifact(obj, map_location="cpu")
    if "type" not in payload or "state_dict" not in payload:
        # fallback for legacy linear state_dict
        if "V" in payload:
            return LinearSubspaceEraser(payload["V"], meta=payload.get("_meta", {}))
        raise RuntimeError("Invalid eraser checkpoint: missing type/state_dict.")
    etype = payload["type"]
    state = payload["state_dict"]
    meta = payload.get("meta", {})
    if etype == "linear":
        eraser = LinearSubspaceEraser(state["V"], meta=meta)
        return eraser
    if etype == "matrix_projection":
        eraser = MatrixProjectionEraser(state["P"], meta=meta)
        return eraser
    if etype == "kernel":
        eraser = KernelEraser(state["W"], state["b"], state["V_phi"], state["decoder"], meta=meta)
        return eraser
    if etype == "steering":
        V = state["V"]
        hidden = 256
        if "mlp.0.weight" in state:
            hidden = state["mlp.0.weight"].shape[0]
        mlp = nn.Sequential(nn.Linear(V.shape[0], hidden), nn.ReLU(), nn.Linear(hidden, V.shape[1]), nn.Tanh())
        eraser = SteeringEraser(V, mlp, meta=meta)
        eraser.load_state_dict(state, strict=False)
        return eraser
    if etype == "task_preserve":
        inner = payload.get("inner")
        if inner is None:
            raise RuntimeError("TaskPreservingEraser requires inner eraser.")
        base = load_eraser(inner)
        task_basis = state["task_basis"]
        mode = meta.get("task_preserve", "residualize")
        preserve_strength = float(meta.get("preserve_strength", 1.0))
        eraser = TaskPreservingEraser(
            base,
            task_basis,
            mode=mode,
            preserve_strength=preserve_strength,
            meta=meta,
        )
        return eraser
    raise RuntimeError(f"Unknown eraser type: {etype}")


def load_eraser_collection(obj) -> tuple[Dict[str, BaseEraser], Dict[str, object]]:
    manifest = None
    manifest_path: Path | None = None
    if isinstance(obj, dict) and isinstance(obj.get("erasers"), list):
        manifest = obj
    else:
        path = Path(obj).expanduser() if isinstance(obj, (str, bytes, Path)) else None
        if path is not None and path.suffix == ".json" and path.exists():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                payload = None
            if isinstance(payload, dict) and isinstance(payload.get("erasers"), list):
                manifest = payload
                manifest_path = path

    if manifest is None:
        eraser = load_eraser(obj)
        supported = getattr(eraser, "meta", {}).get("supported_shift_families") or []
        return (
            {"__default__": eraser},
            {
                "kind": "single",
                "path": str(Path(obj).expanduser()) if isinstance(obj, (str, bytes, Path)) else None,
                "supported_shift_families": list(supported),
            },
        )

    erasers: Dict[str, BaseEraser] = {}
    for row in manifest.get("erasers", []):
        shift_family = str(row.get("shift_family", "")).strip()
        eraser_path = row.get("eraser_path")
        if not shift_family:
            raise RuntimeError("Invalid eraser bank manifest: missing shift_family.")
        if not eraser_path:
            raise RuntimeError(f"Invalid eraser bank manifest entry for shift_family='{shift_family}': missing eraser_path.")
        ep = Path(eraser_path).expanduser()
        if manifest_path is not None and not ep.is_absolute():
            ep = (manifest_path.parent / ep).resolve()
        erasers[shift_family] = load_eraser(ep)
    return (
        erasers,
        {
            "kind": "bank",
            "manifest_path": str(manifest_path) if manifest_path is not None else None,
            "shift_families": sorted(erasers.keys()),
        },
    )


def resolve_eraser_for_shift(eraser_collection: Dict[str, BaseEraser], shift_family: str) -> BaseEraser:
    key = str(shift_family)
    if key in eraser_collection:
        return eraser_collection[key]
    if "__default__" in eraser_collection:
        eraser = eraser_collection["__default__"]
        supported = getattr(eraser, "meta", {}).get("supported_shift_families") or []
        if supported and key not in {str(v) for v in supported}:
            raise RuntimeError(
                f"Eraser artifact does not declare support for shift_family='{key}'. "
                f"supported_shift_families={sorted(str(v) for v in supported)}"
            )
        return eraser
    available = sorted(str(v) for v in eraser_collection.keys())
    raise RuntimeError(f"No eraser available for shift_family='{key}'. Available families: {available}")


def eraser_fingerprint_mismatches(eraser: BaseEraser, expected_fingerprint: Dict[str, object]) -> List[str]:
    expected = expected_fingerprint or {}
    meta = getattr(eraser, "meta", {}) or {}
    actual = meta.get("encoder_fingerprint")
    if not isinstance(actual, dict) or not actual:
        return ["missing eraser.meta.encoder_fingerprint"]
    actual_preprocess = meta.get("preprocess_signature", actual.get("preprocess_signature"))
    actual_revision = actual.get("model_revision", actual.get("revision"))
    expected_preprocess = expected.get("preprocess_signature")
    expected_revision = expected.get("model_revision", expected.get("revision"))
    mismatches: List[str] = []
    keys = [
        "encoder_source",
        "architecture",
        "model_id",
        "encoder_name",
        "pretrained",
        "use_medical_pretrain",
        "effective_medical_pretrain",
        "encoder_path",
    ]
    for key in keys:
        if actual.get(key) != expected.get(key):
            mismatches.append(f"{key}: expected={expected.get(key)} got={actual.get(key)}")
    if actual_revision != expected_revision:
        mismatches.append(f"revision: expected={expected_revision} got={actual_revision}")
    if actual_preprocess != expected_preprocess:
        mismatches.append(f"preprocess_signature: expected={expected_preprocess} got={actual_preprocess}")
    if actual.get("input_adapter", {}) != expected.get("input_adapter", {}):
        mismatches.append("input_adapter mismatch")
    return mismatches
