from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

from .adversary_eval import ProbeSuiteConfig, build_presence_task, build_intensity_task, evaluate_probe_suite
from ..metrics.calibration import expected_calibration_error, brier_score


@dataclass
class CertificateConfig:
    probe_cfg: ProbeSuiteConfig = field(default_factory=ProbeSuiteConfig)
    target: str = "presence"
    threshold: float = 0.5


def _distortion(z_before: np.ndarray, z_after: np.ndarray) -> Dict[str, float]:
    delta = z_after - z_before
    return {
        "z_l2_mean": float(np.linalg.norm(delta, axis=1).mean()),
        "z_l2_p95": float(np.quantile(np.linalg.norm(delta, axis=1), 0.95)),
        "z_cos_mean": float(
            1.0
            - np.mean(
                np.sum(z_before * z_after, axis=1)
                / (np.linalg.norm(z_before, axis=1) * np.linalg.norm(z_after, axis=1) + 1e-8)
            )
        ),
    }


def _prediction_distortion(
    probs_before: Optional[np.ndarray],
    probs_after: Optional[np.ndarray],
    labels: Optional[np.ndarray],
) -> Dict[str, float]:
    if probs_before is None or probs_after is None:
        return {}
    out = {
        "prob_abs_delta_mean": float(np.mean(np.abs(probs_after - probs_before))),
        "prob_abs_delta_p95": float(np.quantile(np.abs(probs_after - probs_before), 0.95)),
    }
    if labels is not None:
        valid = labels >= 0
        if valid.any():
            pb = probs_before[valid]
            pa = probs_after[valid]
            y = labels[valid]
            out["ece_before"] = float(expected_calibration_error(pb, y))
            out["ece_after"] = float(expected_calibration_error(pa, y))
            out["brier_before"] = float(brier_score(pb, y))
            out["brier_after"] = float(brier_score(pa, y))
    return out


def compute_erasure_certificate(
    z_clean_before: np.ndarray,
    z_shift_before: np.ndarray,
    z_clean_after: np.ndarray,
    z_shift_after: np.ndarray,
    groups: Optional[np.ndarray] = None,
    intensities: Optional[np.ndarray] = None,
    target: str = "presence",
    probe_cfg: Optional[ProbeSuiteConfig] = None,
    probs_before: Optional[np.ndarray] = None,
    probs_after: Optional[np.ndarray] = None,
    labels: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    if probe_cfg is None:
        probe_cfg = ProbeSuiteConfig()
    if target == "presence":
        X_b, y_b, g_b = build_presence_task(z_clean_before, z_shift_before, groups=groups)
        X_a, y_a, g_a = build_presence_task(z_clean_after, z_shift_after, groups=groups)
    elif target == "intensity":
        if intensities is None:
            raise ValueError("intensities are required for target='intensity'")
        X_b, y_b, g_b = build_intensity_task(z_clean_before, z_shift_before, intensities=intensities, groups=groups)
        X_a, y_a, g_a = build_intensity_task(z_clean_after, z_shift_after, intensities=intensities, groups=groups)
    else:
        raise ValueError(f"Unknown target {target}")

    before = evaluate_probe_suite(X_b, y_b, groups=g_b, target=target, cfg=probe_cfg)
    after = evaluate_probe_suite(X_a, y_a, groups=g_a, target=target, cfg=probe_cfg)
    cert: Dict[str, float] = {}
    for k, v in before.items():
        cert[f"detect_before_{k}"] = float(v)
    for k, v in after.items():
        cert[f"detect_after_{k}"] = float(v)
    if target == "presence":
        for m in ["linear_auc", "kernel_auc", "mlp_auc", "linear_acc", "kernel_acc", "mlp_acc"]:
            if m in before and m in after:
                cert[f"detect_drop_{m}"] = float(before[m] - after[m])
    else:
        for m in ["linear_mae", "kernel_mae", "mlp_mae"]:
            if m in before and m in after:
                cert[f"detect_mae_gain_{m}"] = float(after[m] - before[m])
    cert.update(_distortion(z_shift_before, z_shift_after))
    cert.update(_prediction_distortion(probs_before, probs_after, labels))
    return cert
