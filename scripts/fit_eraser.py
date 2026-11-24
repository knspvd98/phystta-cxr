#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, Optional, List

import numpy as np
import torch
import torch.nn as nn

from phystta.config.loader import load_config
from phystta.models import EncoderFactory, build_encoder_fingerprint, resolve_head_contract
from phystta.intervention.eraser import (
    BaseEraser,
    BiasSubspaceFitter,
    LinearSubspaceEraser,
    MatrixProjectionEraser,
    KernelEraser,
    SteeringEraser,
    TaskPreservingEraser,
    principal_angle_overlap,
    sensitivity_slope,
    task_subspace_from_head,
    save_eraser,
)
from phystta.intervention.diagnostics import logit_recovery_diagnostic
from phystta.intervention.probes import load_probes
from phystta.utils import preprocess_signature, load_torch_artifact, repo_fingerprint, write_reprovenance_bundle


def _parse_range(val: Optional[str]) -> Optional[tuple]:
    if not val:
        return None
    lo, hi = [float(v) for v in val.split(",")]
    return lo, hi


def _median_sigma(x: np.ndarray, max_samples: int = 2048) -> float:
    rng = np.random.default_rng(0)
    if len(x) > max_samples:
        x = x[rng.choice(len(x), max_samples, replace=False)]
    dists = np.sum((x[:, None, :] - x[None, :, :]) ** 2, axis=-1)
    med = np.median(dists)
    return float(np.sqrt(max(med, 1e-6)))


def _rff_init(dim: int, rff_dim: int, sigma: float, seed: int) -> tuple:
    rng = np.random.default_rng(seed)
    W = rng.normal(scale=1.0 / sigma, size=(dim, rff_dim)).astype(np.float32)
    b = rng.uniform(0, 2 * np.pi, size=(rff_dim,)).astype(np.float32)
    return W, b


def _phi(z: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sqrt(2.0 / W.shape[1]) * np.cos(z @ W + b)


def _fit_decoder(phi: np.ndarray, z: np.ndarray, ridge_lambda: float) -> np.ndarray:
    # ridge regression: B = (Phi^T Phi + λI)^-1 Phi^T Z
    XtX = phi.T @ phi
    reg = ridge_lambda * np.eye(XtX.shape[0], dtype=np.float32)
    B = np.linalg.solve(XtX + reg, phi.T @ z)
    return B.astype(np.float32)


def _resolve_groups(group_ids, patient_ids, study_ids):
    def _valid(arr):
        if arr is None:
            return False
        arr = np.asarray(arr)
        if arr.size == 0:
            return False
        uniq = set([str(x) for x in arr.tolist()])
        if len(uniq) <= 1:
            return False
        if uniq == {"unknown"} or uniq == {"nan"}:
            return False
        return True
    if _valid(group_ids):
        return np.asarray(group_ids), "group_id"
    if _valid(patient_ids):
        return np.asarray(patient_ids), "patient_id"
    if _valid(study_ids):
        return np.asarray(study_ids), "study_id"
    return None, "no-group-split"


def _fit_kernel_eraser(
    z_clean,
    z_shifted,
    intensities,
    group_ids,
    patient_ids,
    study_ids,
    k,
    probe_target,
    rff_dim,
    sigma,
    ridge_lambda,
    seed,
    require_group: bool = False,
):
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.model_selection import GroupShuffleSplit, ShuffleSplit

    z = np.concatenate([z_clean, z_shifted], axis=0)
    if sigma is None:
        sigma = _median_sigma(z)
    W, b = _rff_init(z.shape[1], rff_dim, sigma, seed)
    phi_clean = _phi(z_clean, W, b)
    phi_shift = _phi(z_shifted, W, b)
    X = np.concatenate([phi_clean, phi_shift], axis=0)
    groups, group_key = _resolve_groups(group_ids, patient_ids, study_ids)
    if groups is not None:
        groups_all = np.concatenate([groups, groups], axis=0)
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        tr_idx, va_idx = next(splitter.split(X, groups=groups_all))
    else:
        if require_group:
            raise RuntimeError("Group key missing for kernel eraser fit. NOT CLINICALLY VALID.")
        splitter = ShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        tr_idx, va_idx = next(splitter.split(X))
    if probe_target == "presence":
        y = np.concatenate([np.zeros(len(phi_clean)), np.ones(len(phi_shift))], axis=0)
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X[tr_idx], y[tr_idx])
        coef = clf.coef_.T
        meta = {"probe_accuracy": float(clf.score(X[va_idx], y[va_idx])), "group_split": group_key}
    else:
        y = np.concatenate([np.zeros(len(phi_clean)), intensities], axis=0)
        reg = Ridge(alpha=1.0)
        reg.fit(X[tr_idx], y[tr_idx])
        coef = reg.coef_.reshape(-1, 1)
        pred = reg.predict(X[va_idx])
        meta = {"probe_mae": float(np.mean(np.abs(pred - y[va_idx]))), "group_split": group_key}
    if coef.shape[1] < k:
        # pad with SVD
        _, _, vt = np.linalg.svd((phi_shift - phi_clean), full_matrices=False)
        extra = vt.T[:, : k - coef.shape[1]]
        coef = np.concatenate([coef, extra], axis=1)
    V_phi = coef[:, :k]
    # decoder from phi to z
    phi_all = np.concatenate([phi_clean, phi_shift], axis=0)[tr_idx]
    z_all = np.concatenate([z_clean, z_shifted], axis=0)[tr_idx]
    decoder = _fit_decoder(phi_all, z_all, ridge_lambda)
    eraser = KernelEraser(
        torch.tensor(W),
        torch.tensor(b),
        torch.tensor(V_phi),
        torch.tensor(decoder),
        meta={"method": "kernel", "rff_dim": rff_dim, "sigma": sigma, **meta},
    )
    return eraser, meta


def _fit_steering_eraser(
    z_clean,
    z_shifted,
    k,
    hidden,
    steps,
    lr,
    eps,
    seed,
    head=None,
    probe_target="presence",
    intensities=None,
    logit_recovery_weight=0.0,
    prob_recovery_weight=0.0,
    clean_identity_weight=0.0,
    critical_idx=None,
):
    from sklearn.linear_model import LogisticRegression, Ridge

    z = np.concatenate([z_clean, z_shifted], axis=0)
    if probe_target == "presence":
        y = np.concatenate([np.zeros(len(z_clean)), np.ones(len(z_shifted))], axis=0)
        clf = LogisticRegression(max_iter=1000)
        clf.fit(z, y)
        w = clf.coef_.reshape(-1)
        b = float(clf.intercept_[0])
        task_kind = "classification"
    else:
        y = np.concatenate([np.zeros(len(z_clean)), intensities], axis=0).astype(np.float32)
        reg = Ridge(alpha=1.0)
        reg.fit(z, y)
        w = reg.coef_.reshape(-1)
        b = float(reg.intercept_)
        task_kind = "regression"
    # initial directions
    w = w / (np.linalg.norm(w) + 1e-8)
    V_init = np.stack([w] + [np.random.default_rng(seed+i).normal(size=w.shape) for i in range(k-1)], axis=1).astype(np.float32)
    V_init, _ = np.linalg.qr(V_init)

    V = torch.tensor(V_init, dtype=torch.float32, requires_grad=True)
    mlp = nn.Sequential(nn.Linear(z.shape[1], hidden), nn.ReLU(), nn.Linear(hidden, k), nn.Tanh())
    opt = torch.optim.Adam(list(mlp.parameters()) + [V], lr=lr)
    z_t = torch.tensor(z, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)
    w_t = torch.tensor(w, dtype=torch.float32)
    b_t = torch.tensor(b, dtype=torch.float32)

    head_w = None
    if head is not None and hasattr(head, "weight"):
        head_w = head.weight.detach().float()
        head = head.float()
        head.eval()
        for param in head.parameters():
            param.requires_grad_(False)
    ortho_lambda = 0.1

    n_clean = len(z_clean)
    for _ in range(steps):
        opt.zero_grad()
        alpha = mlp(z_t)
        z_edit = z_t - alpha @ V.T
        logits_probe = z_edit @ w_t + b_t
        if task_kind == "classification":
            loss_probe = nn.functional.binary_cross_entropy_with_logits(logits_probe, y_t)
        else:
            loss_probe = nn.functional.huber_loss(logits_probe, y_t, delta=1.0)
        z_clean_e = z_edit[:n_clean]
        z_shift_e = z_edit[n_clean:]
        loss_pair = torch.norm(z_shift_e - z_clean_e, dim=1).mean()
        loss = loss_pair - loss_probe  # minimize shift, maximize probe loss
        gram = V.T @ V
        I = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
        loss = loss + ortho_lambda * (gram - I).pow(2).mean()
        if clean_identity_weight > 0:
            loss_identity = nn.functional.mse_loss(z_clean_e, z_t[:n_clean])
            loss = loss + float(clean_identity_weight) * loss_identity
        if head is not None and (logit_recovery_weight > 0 or prob_recovery_weight > 0):
            z_clean_orig = z_t[:n_clean]
            z_shift_edit = z_edit[n_clean:]
            with torch.no_grad():
                logits_clean_target = head(z_clean_orig)
            logits_shift_edit = head(z_shift_edit)
            if critical_idx:
                crit = list(critical_idx)
                logits_clean_target = logits_clean_target[:, crit]
                logits_shift_edit = logits_shift_edit[:, crit]
            if logit_recovery_weight > 0:
                loss = loss + float(logit_recovery_weight) * nn.functional.mse_loss(
                    logits_shift_edit,
                    logits_clean_target,
                )
            if prob_recovery_weight > 0:
                probs_clean_target = torch.sigmoid(logits_clean_target)
                probs_shift_edit = torch.sigmoid(logits_shift_edit)
                loss = loss + float(prob_recovery_weight) * nn.functional.mse_loss(
                    probs_shift_edit,
                    probs_clean_target,
                )
        if head_w is not None:
            logits_orig = z_t @ head_w.T
            logits_edit = z_edit @ head_w.T
            diff = (logits_edit - logits_orig).pow(2).mean()
            loss = loss + torch.relu(diff - eps)
        loss.backward()
        opt.step()
        with torch.no_grad():
            q, _ = torch.linalg.qr(V, mode="reduced")
            V.copy_(q[:, :k])
    eraser = SteeringEraser(V.detach(), mlp, meta={"method": "steering", "k": k, "probe_target": probe_target, "probe_loss": task_kind})
    return eraser


def _critical_indices(meta_info, cfg, label_cols: List[str]) -> Optional[List[int]]:
    crit = None
    if cfg and cfg.eraser.critical_labels:
        crit = cfg.eraser.critical_labels
    elif cfg and cfg.safety_policy.critical_labels:
        crit = cfg.safety_policy.critical_labels
    if not crit:
        return None
    idx = [label_cols.index(c) for c in crit if c in label_cols]
    return idx if idx else None


def _resolve_shift_selection(probes, shift_spec: str, intensity_range: Optional[tuple]) -> tuple[np.ndarray, List[str]]:
    shift_values = probes.get("shift_type")
    if shift_values is None:
        shift_values = probes.get("profile")
    if shift_values is None:
        raise RuntimeError("Probe artifact is missing both shift_type and profile.")
    requested_shift_types = [s.strip() for s in str(shift_spec).split(",") if s.strip()]
    if len(requested_shift_types) == 1 and requested_shift_types[0].lower() == "all":
        shift_mask = np.array([str(s) not in {"none", "clean"} for s in shift_values], dtype=bool)
        selected_shift_types = sorted({str(s) for s in shift_values if str(s) not in {"none", "clean"}})
    else:
        requested = {str(s) for s in requested_shift_types}
        shift_mask = np.array([str(s) in requested for s in shift_values], dtype=bool)
        selected_shift_types = sorted(requested)
    if intensity_range:
        lo, hi = intensity_range
        intens = probes["intensity"].numpy()
        shift_mask = shift_mask & (intens >= lo) & (intens <= hi)
    return shift_mask, selected_shift_types


def _resolve_task_preserve_strength(task_preserve: Optional[str], preserve_strength: Optional[float]) -> float:
    """
    Resolve the effective preserve strength for task-preserving erasers.

    Backward-compatible behavior:
    - head_subspace_soft defaults to 0.75 if not explicitly provided
    - all other modes default to 1.0
    """
    if task_preserve == "head_subspace_soft":
        value = 0.75 if preserve_strength is None else float(preserve_strength)
    else:
        value = 1.0 if preserve_strength is None else float(preserve_strength)

    if not (0.0 <= value <= 1.0):
        raise ValueError("preserve_strength must be in [0.0, 1.0]")
    return value


def _wrap_task_preserving_eraser(
    eraser: BaseEraser,
    basis: torch.Tensor,
    *,
    task_preserve: str,
    preserve_strength: Optional[float],
    meta: Dict,
) -> TaskPreservingEraser:
    effective_strength = _resolve_task_preserve_strength(task_preserve, preserve_strength)
    wrapped_meta = {
        **meta,
        "task_preserve": task_preserve,
        "preserve_strength": float(effective_strength),
    }
    return TaskPreservingEraser(
        eraser,
        basis,
        mode=task_preserve,
        preserve_strength=effective_strength,
        meta=wrapped_meta,
    )


def _resolve_recovery_objective(
    *,
    enable_recovery_objective: bool,
    logit_recovery_weight: float,
    prob_recovery_weight: float,
    clean_identity_weight: float,
) -> Dict:
    """
    Normalize recovery-objective behavior into a single script-side payload.

    Backward-compatible behavior:
    - if not enabled and all weights are zero -> disabled path
    - if enabled and all weights are zero -> apply a conservative preset
    - if any explicit nonzero weights are already present -> enabled path
    """
    logit = float(logit_recovery_weight)
    prob = float(prob_recovery_weight)
    clean = float(clean_identity_weight)

    for name, value in {
        "logit_recovery_weight": logit,
        "prob_recovery_weight": prob,
        "clean_identity_weight": clean,
    }.items():
        if value < 0.0:
            raise ValueError(f"{name} must be >= 0.0")

    explicit_nonzero = (logit > 0.0) or (prob > 0.0) or (clean > 0.0)

    if enable_recovery_objective and not explicit_nonzero:
        # Conservative preset: enough to create classifier-visible pressure,
        # while keeping a strong clean anchor.
        logit = 0.25
        prob = 0.05
        clean = 1.0
        source = "flag_defaults"
        enabled = True
    elif explicit_nonzero:
        source = "explicit_weights"
        enabled = True
    else:
        source = "disabled"
        enabled = False

    return {
        "enabled": bool(enabled),
        "source": source,
        "weights": {
            "logit_recovery_weight": float(logit),
            "prob_recovery_weight": float(prob),
            "clean_identity_weight": float(clean),
        },
    }


def _apply_recovery_objective_to_args(args, recovery_objective: Dict) -> None:
    weights = recovery_objective["weights"]
    args.logit_recovery_weight = float(weights["logit_recovery_weight"])
    args.prob_recovery_weight = float(weights["prob_recovery_weight"])
    args.clean_identity_weight = float(weights["clean_identity_weight"])


def _fit_and_save_single(
    *,
    probes,
    meta_info,
    shift_mask: np.ndarray,
    selected_shift_types: List[str],
    args,
    cfg,
    out_path: Path,
    meta_path: Path,
    log,
) -> dict:
    if not bool(np.any(shift_mask)):
        raise RuntimeError(
            f"No probe rows matched shift_type='{','.join(selected_shift_types)}'"
            + (f" and intensity_range={args.intensity_range}" if args.intensity_range else "")
        )
    z_clean = probes["features_clean"][shift_mask].numpy()
    z_shifted = probes["features_shifted"][shift_mask].numpy()
    intensities = probes["intensity"][shift_mask].numpy()
    group_ids = np.array(probes["group_id"])[shift_mask] if probes.get("group_id") is not None else None
    patient_ids = np.array(probes["patient_id"])[shift_mask] if probes.get("patient_id") is not None else None
    study_ids = np.array(probes["study_id"])[shift_mask] if probes.get("study_id") is not None else None
    group_patient_ids = group_ids if group_ids is not None else patient_ids
    diagnostic_head = None
    recovery_objective = getattr(args, "recovery_objective", None)

    fitter = BiasSubspaceFitter(seed=args.seed)
    if args.fit_method in {"probe_clf", "probe_xfit"}:
        if str(meta_info.get("group_valid", "True")).lower() == "false":
            raise RuntimeError("Group key missing in probes. Cannot fit probe-based eraser (NOT CLINICALLY VALID).")

    if args.eraser_type == "linear":
        obj, meta = fitter.fit(
            args.fit_method,
            z_clean,
            z_shifted,
            intensities,
            group_patient_ids,
            study_ids,
            args.k,
            probe_target=args.probe_target,
        )
        if args.fit_method == "leace":
            eraser = MatrixProjectionEraser(torch.tensor(obj, dtype=torch.float32), meta=meta)
        else:
            eraser = LinearSubspaceEraser(torch.tensor(obj, dtype=torch.float32), meta=meta)
    elif args.eraser_type == "kernel":
        require_group = bool(cfg is not None and getattr(cfg.experiment, "tier", "dev") == "paper")
        eraser, meta = _fit_kernel_eraser(
            z_clean,
            z_shifted,
            intensities,
            group_ids,
            patient_ids,
            study_ids,
            args.k,
            args.probe_target,
            args.rff_dim,
            args.rff_sigma,
            args.ridge_lambda,
            args.seed,
            require_group=require_group,
        )
    elif args.eraser_type == "steering":
        head = None
        if args.task_preserve != "none":
            if args.head_path is None:
                raise RuntimeError("task_preserve requires --head_path or config.model.encoder.head_path")
            if cfg is not None:
                label_cols = meta_info.get("label_cols", "").split(",") if meta_info.get("label_cols") else None
                head = resolve_head_contract(
                    cfg,
                    label_cols=label_cols,
                    encoder_profile=None,
                    require_head_path=True,
                    strict_meta=True,
                    load_artifact=True,
                ).head_artifact
            else:
                head = load_torch_artifact(args.head_path, map_location="cpu")
        elif args.head_path is not None:
            head = load_torch_artifact(args.head_path, map_location="cpu")
        diagnostic_head = head
        critical_idx = None
        if cfg is not None:
            label_cols = meta_info.get("label_cols", "").split(",") if meta_info.get("label_cols") else []
            critical_idx = _critical_indices(meta_info, cfg, label_cols)
        eraser = _fit_steering_eraser(
            z_clean,
            z_shifted,
            args.k,
            args.steering_hidden,
            args.steering_steps,
            args.steering_lr,
            args.steering_eps,
            args.seed,
            head=head,
            probe_target=args.probe_target,
            intensities=intensities,
            logit_recovery_weight=args.logit_recovery_weight,
            prob_recovery_weight=args.prob_recovery_weight,
            clean_identity_weight=args.clean_identity_weight,
            critical_idx=critical_idx,
        )
        meta = eraser.meta
    else:
        raise RuntimeError(f"Unknown eraser_type: {args.eraser_type}")

    shift_label = selected_shift_types[0] if len(selected_shift_types) == 1 else ",".join(selected_shift_types)
    meta["eraser_type"] = args.eraser_type
    meta["task_preserve"] = args.task_preserve
    if args.task_preserve and args.task_preserve != "none":
        meta["preserve_strength"] = _resolve_task_preserve_strength(
            args.task_preserve,
            args.preserve_strength,
        )
    meta["recovery_objective"] = recovery_objective
    meta["shift_type"] = shift_label
    meta["shift_types"] = list(selected_shift_types)
    meta["supported_shift_families"] = list(selected_shift_types)
    meta["probe_file_sha256"] = hashlib.sha256(Path(args.probes).read_bytes()).hexdigest()
    meta["fit_pair_mode"] = meta_info.get("pair_mode")
    meta["pair_quality_summary"] = meta_info.get("pair_quality_counts")
    meta["probe_basis_id"] = hashlib.sha256(
        (meta["probe_file_sha256"] + "|" + "|".join(selected_shift_types) + "|" + str(args.fit_method)).encode()
    ).hexdigest()
    meta["index_sha256"] = meta_info.get("index_sha256")
    meta["probe_source_split"] = meta_info.get("source_split") or meta_info.get("split")
    meta["probe_shift_family"] = meta_info.get("shift_family") or shift_label
    meta["code_revision"] = repo_fingerprint().get("git_sha_or_repo_fingerprint")
    if cfg is not None:
        enc_cfg = EncoderFactory.resolve_encoder_config(cfg, profile=None)
        meta["encoder_fingerprint"] = build_encoder_fingerprint(enc_cfg)
        meta["preprocess_signature"] = preprocess_signature(cfg)
    if "encoder_signature" in meta_info:
        meta["probe_encoder_signature"] = str(meta_info["encoder_signature"])

    if args.task_preserve and args.task_preserve != "none":
        if args.head_path is None:
            raise RuntimeError("task_preserve requires --head_path or config.model.encoder.head_path")
        label_cols = meta_info.get("label_cols", "").split(",") if meta_info.get("label_cols") else []
        if cfg is not None:
            head = resolve_head_contract(
                cfg,
                label_cols=label_cols or None,
                encoder_profile=None,
                require_head_path=True,
                strict_meta=True,
                load_artifact=True,
            ).head_artifact
        else:
            head = load_torch_artifact(args.head_path, map_location="cpu")
        idx = _critical_indices(meta_info, cfg, label_cols)
        basis = task_subspace_from_head(
            head,
            critical_idx=idx,
            k=(cfg.eraser.task_subspace_k if cfg else None),
        )
        eraser = _wrap_task_preserving_eraser(
            eraser,
            basis,
            task_preserve=args.task_preserve,
            preserve_strength=args.preserve_strength,
            meta=meta,
        )
        diagnostic_head = head

    if diagnostic_head is not None:
        idx = None
        label_cols = meta_info.get("label_cols", "").split(",") if meta_info.get("label_cols") else []
        if cfg is not None:
            idx = _critical_indices(meta_info, cfg, label_cols)
        meta["recovery_diagnostic"] = logit_recovery_diagnostic(
            z_clean,
            z_shifted,
            eraser,
            diagnostic_head,
            critical_idx=idx,
            groups=group_patient_ids,
        )

    if args.anchor_real:
        if args.real_pairs is None:
            raise RuntimeError("anchor_real requires --real_pairs probes")
        real = load_torch_artifact(args.real_pairs, map_location="cpu")["probes"]
        z_a = real["features_clean"].numpy()
        z_b = real["features_shifted"].numpy()
        dmat = z_b - z_a
        _, _, vt = np.linalg.svd(dmat, full_matrices=False)
        V_real = vt[: args.k].T
        V_ref = V_real
        base = eraser.eraser if hasattr(eraser, "eraser") else eraser
        if isinstance(base, LinearSubspaceEraser):
            V_ref = base.V.detach().cpu().numpy()
        overlap = principal_angle_overlap(V_ref, V_real)
        if overlap < args.min_overlap:
            raise RuntimeError(f"Anchoring failed: overlap {overlap:.3f} < {args.min_overlap}")
        meta["real_overlap"] = overlap

    if args.causal_check:
        unique, counts = np.unique(intensities, return_counts=True)
        if np.any(counts < args.positivity_min_samples):
            raise RuntimeError("Positivity violated: some intensity levels below min_samples")
        with torch.no_grad():
            z_shift_t = torch.tensor(z_shifted, dtype=torch.float32)
            z_clean_t = torch.tensor(z_clean, dtype=torch.float32)
            z_post_shift = eraser(z_shift_t).numpy()
            z_post_clean = eraser(z_clean_t).numpy()
        diff_pre = np.linalg.norm(z_shifted - z_clean, axis=1)
        diff_post = np.linalg.norm(z_post_shift - z_post_clean, axis=1)
        slope_pre = sensitivity_slope(intensities, diff_pre)
        slope_post = sensitivity_slope(intensities, diff_post)
        meta["sensitivity_slope_pre"] = slope_pre
        meta["sensitivity_slope_post"] = slope_post
        meta["ate_pre"] = slope_pre
        meta["ate_post"] = slope_post
        if np.isfinite(slope_post) and abs(slope_post) > args.ate_threshold:
            raise RuntimeError(f"Sensitivity slope check failed: {slope_post:.4f} > {args.ate_threshold}")

    base = eraser.eraser if hasattr(eraser, "eraser") else eraser
    if args.self_check and isinstance(base, LinearSubspaceEraser):
        x = torch.randn(16, base.V.shape[0])
        check = base.validate_projection(x)
        meta["self_check"] = check
        log(f"Self-check[{shift_label}]: {check}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    save_eraser(str(out_path), eraser)
    meta_path.write_text(json.dumps(meta, indent=2))
    log(f"Saved eraser[{shift_label}] to {out_path} and meta to {meta_path}")
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit bias subspace eraser")
    parser.add_argument("--probes", required=True)
    parser.add_argument("--config", required=False)
    parser.add_argument(
        "--shift_type",
        required=True,
        help="Single shift type, comma-separated list, or 'all' for all non-clean shifts.",
    )
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--fit_method", choices=["svd", "leace", "probe_clf", "probe_xfit"], default=None)
    parser.add_argument("--probe_target", choices=["presence", "intensity"], default=None)
    parser.add_argument("--eraser_type", choices=["linear", "kernel", "steering"], default=None)
    parser.add_argument("--fit_mode", choices=["single", "one_eraser_per_shift_family"], default=None)
    parser.add_argument(
        "--task_preserve",
        choices=["none", "head_subspace", "head_subspace_soft", "residualize"],
        default=None,
    )
    parser.add_argument(
        "--preserve_strength",
        type=float,
        default=None,
        help=(
            "Optional preserve strength in [0,1] for task-preserving erasers. "
            "Defaults to 0.75 for head_subspace_soft and 1.0 otherwise."
        ),
    )
    parser.add_argument("--head_path", default=None)
    parser.add_argument("--intensity_range", default=None, help="min,max")
    parser.add_argument("--out", required=True)
    parser.add_argument("--meta", required=True)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--self_check", action="store_true")
    parser.add_argument("--anchor_real", action="store_true")
    parser.add_argument("--real_pairs", default=None, help="probes.pt from --real_pairs")
    parser.add_argument("--min_overlap", type=float, default=0.7)
    parser.add_argument("--causal_check", action="store_true")
    parser.add_argument("--ate_threshold", type=float, default=0.05)
    parser.add_argument("--positivity_min_samples", type=int, default=50)
    parser.add_argument("--rff_dim", type=int, default=None)
    parser.add_argument("--rff_sigma", type=float, default=None)
    parser.add_argument("--ridge_lambda", type=float, default=None)
    parser.add_argument("--steering_hidden", type=int, default=None)
    parser.add_argument("--steering_steps", type=int, default=None)
    parser.add_argument("--steering_lr", type=float, default=None)
    parser.add_argument("--steering_eps", type=float, default=None)
    parser.add_argument("--logit_recovery_weight", type=float, default=None)
    parser.add_argument("--prob_recovery_weight", type=float, default=None)
    parser.add_argument("--clean_identity_weight", type=float, default=None)
    parser.add_argument(
        "--enable_recovery_objective",
        action="store_true",
        help=(
            "Enable recovery-aware objective. If enabled while all recovery weights "
            "are zero, a conservative preset is applied."
        ),
    )
    args = parser.parse_args()

    cfg = load_config(args.config) if args.config else None

    # defaults from config
    if cfg:
        if args.k is None:
            args.k = cfg.eraser.k or cfg.intervention.k
        if args.fit_method is None:
            args.fit_method = cfg.intervention.fit_method
        if args.probe_target is None:
            args.probe_target = cfg.intervention.probe_target
        if args.eraser_type is None:
            args.eraser_type = cfg.eraser.type
        if args.fit_mode is None:
            args.fit_mode = cfg.eraser.fit_mode
        if args.task_preserve is None:
            args.task_preserve = cfg.eraser.task_preserve
        if args.rff_dim is None:
            args.rff_dim = cfg.eraser.rff_dim
        if args.rff_sigma is None:
            args.rff_sigma = cfg.eraser.rff_sigma
        if args.ridge_lambda is None:
            args.ridge_lambda = cfg.eraser.ridge_lambda
        if args.steering_hidden is None:
            args.steering_hidden = cfg.eraser.steering_hidden
        if args.steering_steps is None:
            args.steering_steps = cfg.eraser.steering_steps
        if args.steering_lr is None:
            args.steering_lr = cfg.eraser.steering_lr
        if args.steering_eps is None:
            args.steering_eps = cfg.eraser.steering_eps
        if args.logit_recovery_weight is None:
            args.logit_recovery_weight = cfg.eraser.logit_recovery_weight
        if args.prob_recovery_weight is None:
            args.prob_recovery_weight = cfg.eraser.prob_recovery_weight
        if args.clean_identity_weight is None:
            args.clean_identity_weight = cfg.eraser.clean_identity_weight
        if args.head_path is None and cfg.model.encoder.head_path is not None:
            args.head_path = str(cfg.model.encoder.head_path)
        if args.head_path is not None:
            cfg.model.encoder.head_path = Path(args.head_path)
    else:
        if args.logit_recovery_weight is None:
            args.logit_recovery_weight = 0.0
        if args.prob_recovery_weight is None:
            args.prob_recovery_weight = 0.0
        if args.clean_identity_weight is None:
            args.clean_identity_weight = 0.0

    recovery_objective = _resolve_recovery_objective(
        enable_recovery_objective=bool(args.enable_recovery_objective),
        logit_recovery_weight=float(args.logit_recovery_weight),
        prob_recovery_weight=float(args.prob_recovery_weight),
        clean_identity_weight=float(args.clean_identity_weight),
    )
    _apply_recovery_objective_to_args(args, recovery_objective)
    args.recovery_objective = recovery_objective

    if args.k is None:
        raise RuntimeError("k must be provided or specified in config.eraser.k")
    if args.fit_mode is None:
        args.fit_mode = "one_eraser_per_shift_family"

    log_path = Path(args.out).parent / "log.txt"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(msg: str):
        print(msg)
        log_path.open("a").write(msg + "\n")

    log(
        "recovery_objective "
        f"enabled={recovery_objective['enabled']} "
        f"source={recovery_objective['source']} "
        f"logit={args.logit_recovery_weight:.6g} "
        f"prob={args.prob_recovery_weight:.6g} "
        f"clean={args.clean_identity_weight:.6g}"
    )

    data = load_torch_artifact(args.probes, map_location="cpu")
    if "probes_chunks" in data:
        probes = load_probes(args.probes)
        meta_info = data.get("meta", {})
    else:
        probes = data["probes"]
        meta_info = data.get("meta", {})
    intensity_range = _parse_range(args.intensity_range)
    shift_mask, selected_shift_types = _resolve_shift_selection(probes, args.shift_type, intensity_range)

    if args.fit_mode == "one_eraser_per_shift_family" and len(selected_shift_types) > 1:
        out_root = Path(args.out)
        meta_root = Path(args.meta)
        out_dir = out_root if out_root.suffix == "" else out_root.with_suffix("")
        meta_dir = meta_root if meta_root.suffix == "" else meta_root.with_suffix("")
        bank_manifest_path = (meta_root / "eraser_bank_manifest.json") if meta_root.suffix == "" else meta_root
        out_dir.mkdir(parents=True, exist_ok=True)
        meta_dir.mkdir(parents=True, exist_ok=True)
        bank_rows = []
        for shift_type in selected_shift_types:
            child_mask, _ = _resolve_shift_selection(probes, shift_type, intensity_range)
            out_path = out_dir / f"{shift_type}.pt"
            meta_path = meta_dir / f"{shift_type}.json"
            meta = _fit_and_save_single(
                probes=probes,
                meta_info=meta_info,
                shift_mask=child_mask,
                selected_shift_types=[shift_type],
                args=args,
                cfg=cfg,
                out_path=out_path,
                meta_path=meta_path,
                log=log,
            )
            bank_rows.append(
                {
                    "shift_family": shift_type,
                    "eraser_path": str(out_path),
                    "meta_path": str(meta_path),
                    "probe_basis_id": str(meta.get("probe_basis_id")),
                }
            )
        bank_manifest_path.write_text(json.dumps({"fit_mode": args.fit_mode, "erasers": bank_rows}, indent=2))
        if cfg is not None:
            write_reprovenance_bundle(
                out_dir,
                cfg=cfg,
                config_path=args.config,
                artifact_manifest={
                    "artifact_type": "eraser_bank",
                    "fit_mode": args.fit_mode,
                    "shift_families": selected_shift_types,
                    "probe_sha256": hashlib.sha256(Path(args.probes).read_bytes()).hexdigest(),
                },
                input_artifact_paths={
                    "probes": args.probes,
                    "real_pairs": args.real_pairs,
                    "head": args.head_path,
                },
            )
        log(f"Saved eraser bank with shift_families={selected_shift_types}")
        return

    meta = _fit_and_save_single(
        probes=probes,
        meta_info=meta_info,
        shift_mask=shift_mask,
        selected_shift_types=selected_shift_types,
        args=args,
        cfg=cfg,
        out_path=Path(args.out),
        meta_path=Path(args.meta),
        log=log,
    )
    if cfg is not None:
        write_reprovenance_bundle(
            Path(args.out).parent,
            cfg=cfg,
            config_path=args.config,
            artifact_manifest={
                "artifact_type": "eraser",
                "fit_mode": args.fit_mode,
                "shift_families": selected_shift_types,
                "probe_sha256": hashlib.sha256(Path(args.probes).read_bytes()).hexdigest(),
                "probe_basis_id": meta.get("probe_basis_id"),
                "index_sha256": meta.get("index_sha256"),
                "source_capabilities": meta_info.get("source_capabilities"),
                "encoder_fingerprint": meta.get("encoder_fingerprint"),
                "preprocess_signature": meta.get("preprocess_signature"),
            },
            input_artifact_paths={
                "probes": args.probes,
                "real_pairs": args.real_pairs,
                "head": args.head_path,
            },
        )


if __name__ == "__main__":
    main()
