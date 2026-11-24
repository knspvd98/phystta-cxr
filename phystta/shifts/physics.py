from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .base import DomainShift


PARTIAL_REALISM_LIMITATIONS = [
    "blur / MTF degradation beyond the current low-pass and directional-motion proxies",
    "complex respiratory/cardiac motion beyond the current directional blur proxy",
    "rotation / positioning errors",
    "detector-accurate collimation geometry beyond the current crop-and-resize proxy",
    "grid / detector processing effects",
]

EXPOSURE_POISSON_DEPENDENCE_WARNING = (
    "Exposure and Poisson are not independent physical mechanisms in this implementation: "
    "both operate through an effective dose/noise approximation and should not be interpreted as orthogonal factors."
)

SHIFT_PHYSICS_SEMANTICS = {
    "heel": {
        "models": "A smooth field-intensity drop consistent with a heel-effect-like anode-side fluence gradient.",
        "does_not_model": "True device provenance, detector gain maps, or positioning-dependent beam geometry.",
        "approximation": "Implemented as a deterministic multiplicative field over normalized intensity.",
    },
    "scatter": {
        "models": "Low-frequency veiling/scatter contamination via an attenuation-informed low-pass proxy.",
        "does_not_model": "Grid failure modes, anti-scatter hardware behavior, or full Monte Carlo scatter transport.",
        "approximation": "Implemented as a scatter-to-primary-style additive low-pass component in image space.",
    },
    "exposure": {
        "models": "Dose reduction with AEC-like renormalization, approximating under/over-exposure effects after acquisition control.",
        "does_not_model": "Exact tube current-time settings, beam hardening, detector saturation, or full acquisition physics.",
        "approximation": "Implemented through dose scaling plus AEC normalization; noise is therefore only partially separable from Poisson.",
    },
    "poisson": {
        "models": "Quantum-limited shot noise with a small electronic-noise term under an effective dose proxy.",
        "does_not_model": "Independent detector-processing noise, correlated readout artifacts, or a fully separate non-dose noise process.",
        "approximation": "Implemented by Poisson sampling after dose scaling; this shares the same effective exposure axis as the exposure family.",
    },
    "exposure_poisson": {
        "models": "A coupled dose-reduction and quantum-noise stressor where reduced effective exposure directly increases shot-noise burden.",
        "does_not_model": "True tube-current trajectories, beam hardening, detector saturation, or detector-specific post-processing.",
        "approximation": "Implemented as a single dose-coupled AEC-normalized path rather than two orthogonal physical knobs.",
    },
    "motion_blur": {
        "models": "Directional motion blur as a proxy for respiratory, patient, or bedside motion during acquisition.",
        "does_not_model": "Full rigid-body repositioning, detector readout smear, or multi-directional motion trajectories.",
        "approximation": "Implemented as a fixed-axis line-spread blur in image space rather than a detector-specific motion simulator.",
    },
    "collimation_crop": {
        "models": "Electronic collimation / field-of-view variability with peripheral truncation and resized-back support.",
        "does_not_model": "True detector geometry, shutter mechanics, or positioning-aware beam collimation physics.",
        "approximation": "Implemented as an asymmetric crop-and-resize proxy that preserves tensor size while changing visible support region.",
    },
}


def shift_physics_semantics(shift_name: str) -> Dict[str, object]:
    semantics = dict(SHIFT_PHYSICS_SEMANTICS.get(str(shift_name), {}))
    if not semantics:
        return {}
    semantics["partial_realism_limitations"] = list(PARTIAL_REALISM_LIMITATIONS)
    if str(shift_name) in {"exposure", "poisson"}:
        semantics["physical_dependence_warning"] = EXPOSURE_POISSON_DEPENDENCE_WARNING
    return semantics


def physical_dependence_warnings(shift_names) -> list[str]:
    names = {str(name) for name in (shift_names or [])}
    warnings: list[str] = []
    if {"exposure", "poisson"} & names:
        warnings.append(EXPOSURE_POISSON_DEPENDENCE_WARNING)
    return warnings


def _gaussian_kernel(kernel_size: int, sigma: float, device=None) -> torch.Tensor:
    coords = torch.arange(kernel_size, device=device).float() - (kernel_size - 1) / 2
    grid = coords[None, :] ** 2 + coords[:, None] ** 2
    kernel = torch.exp(-grid / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()
    return kernel


def _lowpass(x: torch.Tensor, sigma: float) -> torch.Tensor:
    k = int(max(3, 2 * int(3 * sigma) + 1))
    kernel = _gaussian_kernel(k, sigma, device=x.device).view(1, 1, k, k)
    return F.conv2d(x.unsqueeze(0), kernel, padding=k // 2).squeeze(0)


def _heel_field(h: int, w: int, strength: float, axis: str = "x", device=None) -> torch.Tensor:
    # strength: relative drop across field (0..1)
    # use a concave profile so attenuation affects a larger fraction of the field
    # while preserving the same edge minimum 1 - strength
    gamma = 0.45
    if axis == "x":
        coord = torch.linspace(-1.0, 1.0, w, device=device)
        t = torch.clamp((coord + 1) / 2.0, 0.0, 1.0)
        field = 1.0 - strength * torch.pow(t, gamma)
        field = field.view(1, 1, w).repeat(1, h, 1)
    else:
        coord = torch.linspace(-1.0, 1.0, h, device=device)
        t = torch.clamp((coord + 1) / 2.0, 0.0, 1.0)
        field = 1.0 - strength * torch.pow(t, gamma)
        field = field.view(1, h, 1).repeat(1, 1, w)
    return torch.clamp(field, 0.0, 1.0)


def _attenuation_proxy(I: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return -torch.log(torch.clamp(I, eps, 1.0))


def _mtf_glare(I: torch.Tensor, mtf_sigma: float, glare_sigma: float, glare_weight: float) -> torch.Tensor:
    if mtf_sigma <= 0 and glare_weight <= 0:
        return I
    mtf_sigma = max(1e-3, mtf_sigma)
    blur = _lowpass(I, mtf_sigma)
    if glare_weight > 0:
        glare_sigma = max(glare_sigma, mtf_sigma)
        glare = _lowpass(I, glare_sigma)
        return (1.0 - glare_weight) * blur + glare_weight * glare
    return blur


def _motion_blur(I: torch.Tensor, kernel_px: float, axis: str = "x") -> torch.Tensor:
    length = int(max(1, round(float(kernel_px))))
    if length <= 1:
        return I
    if length % 2 == 0:
        length += 1
    if axis == "y":
        kernel = torch.ones((1, 1, length, 1), device=I.device, dtype=I.dtype) / float(length)
        padding = (length // 2, 0)
    else:
        kernel = torch.ones((1, 1, 1, length), device=I.device, dtype=I.dtype) / float(length)
        padding = (0, length // 2)
    return F.conv2d(I.unsqueeze(0), kernel, padding=padding).squeeze(0)


def _apply_noise(I: torch.Tensor, n0: float, electronic_std: float = 0.0) -> torch.Tensor:
    if n0 <= 0:
        return torch.clamp(I, 0.0, 1.0)
    lam = torch.clamp(I, 0.0, 1.0) * n0
    noisy = torch.poisson(lam) / max(n0, 1e-6)
    if electronic_std > 0:
        noisy = noisy + torch.randn_like(noisy) * electronic_std
    return torch.clamp(noisy, 0.0, 1.0)


def _aec_normalize(I: torch.Tensor, target_q: float = 0.5, clip_min: float = 0.0, clip_max: float = 1.0) -> torch.Tensor:
    q = torch.quantile(I.flatten(), target_q)
    scale = 0.5 / max(q.item(), 1e-6)
    return torch.clamp(I * scale, clip_min, clip_max)


def apply_physics_pipeline(
    x: torch.Tensor,
    heel_strength: float = 0.0,
    scatter_ratio: float = 0.0,
    scatter_sigma: float = 7.0,
    mtf_sigma: float = 0.0,
    glare_sigma: float = 8.0,
    glare_weight: float = 0.0,
    motion_kernel_px: float = 0.0,
    motion_axis: str = "x",
    dose_scale: float = 1.0,
    electronic_std: float = 0.0,
    aec: bool = False,
    aec_target_q: float = 0.5,
    n0_base: float = 1e4,
) -> torch.Tensor:
    # x is intensity in [0,1]
    I = torch.clamp(x, 0.0, 1.0)
    if heel_strength > 0:
        _, h, w = I.shape
        I0 = _heel_field(h, w, heel_strength, axis="x", device=I.device)
        I = I * I0
    # attenuation proxy for scatter
    A = _attenuation_proxy(I)
    if scatter_ratio > 0:
        scatter = scatter_ratio * _lowpass(A, scatter_sigma) * I
        I = I + scatter
    # MTF + glare
    I = _mtf_glare(I, mtf_sigma, glare_sigma, glare_weight)
    if motion_kernel_px > 1.0:
        I = _motion_blur(I, motion_kernel_px, axis=motion_axis)
    # dose/exposure
    I = torch.clamp(I * max(dose_scale, 1e-3), 0.0, 1.0)
    if aec:
        I = _aec_normalize(I, target_q=aec_target_q)
    # quantum + electronic noise
    I = _apply_noise(I, n0=n0_base * max(dose_scale, 1e-3), electronic_std=electronic_std)
    return torch.clamp(I, 0.0, 1.0)


def _crop_resize_back(
    I: torch.Tensor,
    *,
    left_frac: float,
    right_frac: float,
    top_frac: float,
    bottom_frac: float,
) -> torch.Tensor:
    _, h, w = I.shape
    x0 = int(round(np.clip(left_frac, 0.0, 0.45) * w))
    x1 = int(round(w - np.clip(right_frac, 0.0, 0.45) * w))
    y0 = int(round(np.clip(top_frac, 0.0, 0.45) * h))
    y1 = int(round(h - np.clip(bottom_frac, 0.0, 0.45) * h))
    x1 = max(x1, x0 + max(2, w // 4))
    y1 = max(y1, y0 + max(2, h // 4))
    x1 = min(x1, w)
    y1 = min(y1, h)
    crop = I[:, y0:y1, x0:x1]
    if crop.shape[-2:] == (h, w):
        return I
    out = F.interpolate(
        crop.unsqueeze(0),
        size=(h, w),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)
    return torch.clamp(out, 0.0, 1.0)


def _destructive_flag(mtf_sigma: float, glare_weight: float, dose_scale: float) -> bool:
    return (mtf_sigma >= 2.0) or (glare_weight >= 0.35) or (dose_scale <= 0.2)


class HeelEffectShift(DomainShift):
    def __init__(self):
        super().__init__("heel")

    def apply(self, x: torch.Tensor, intensity: float, rng=None, metadata=None) -> torch.Tensor:
        heel_strength = max(0.0, min(1.0, float(intensity) * 1.0))
        return apply_physics_pipeline(x, heel_strength=heel_strength)

    def physical_params(self, intensity: float):
        heel_strength = max(0.0, min(1.0, float(intensity) * 1.0))
        return {
            "heel_strength": heel_strength,  # relative drop across field (0..0.6)
            "unit": "relative",
            "destructive": False,
            **shift_physics_semantics(self.name),
        }


class ScatterShift(DomainShift):
    def __init__(self):
        super().__init__("scatter")

    def apply(self, x: torch.Tensor, intensity: float, rng=None, metadata=None) -> torch.Tensor:
        spr = max(0.0, min(1.2, float(intensity) * 1.2))
        sigma = 6.0 + 12.0 * spr
        return apply_physics_pipeline(x, scatter_ratio=spr, scatter_sigma=sigma, n0_base=0.0)

    def physical_params(self, intensity: float):
        spr = max(0.0, min(1.2, float(intensity) * 1.2))
        sigma = 6.0 + 12.0 * spr
        return {
            "scatter_ratio": spr,  # SPR (scatter-to-primary ratio)
            "scatter_sigma_px": sigma,
            "unit": "relative",
            "destructive": False,
            **shift_physics_semantics(self.name),
        }


class MTFGlareShift(DomainShift):
    def __init__(self):
        super().__init__("mtf")

    def apply(self, x: torch.Tensor, intensity: float, rng=None, metadata=None) -> torch.Tensor:
        mtf_sigma = 0.5 + 2.0 * float(intensity)
        glare_sigma = 6.0 + 10.0 * float(intensity)
        glare_weight = min(0.4, 0.4 * float(intensity))
        return apply_physics_pipeline(x, mtf_sigma=mtf_sigma, glare_sigma=glare_sigma, glare_weight=glare_weight)

    def physical_params(self, intensity: float):
        mtf_sigma = 0.5 + 2.0 * float(intensity)
        glare_sigma = 6.0 + 10.0 * float(intensity)
        glare_weight = min(0.4, 0.4 * float(intensity))
        return {
            "mtf_sigma_px": mtf_sigma,  # PSF sigma (px)
            "glare_sigma_px": glare_sigma,
            "glare_weight": glare_weight,
            "unit": "px/relative",
            "destructive": _destructive_flag(mtf_sigma, glare_weight, 1.0),
            "partial_realism_limitations": list(PARTIAL_REALISM_LIMITATIONS),
        }


class GlareShift(DomainShift):
    def __init__(self):
        super().__init__("glare")

    def apply(self, x: torch.Tensor, intensity: float, rng=None, metadata=None) -> torch.Tensor:
        glare_sigma = 6.0 + 12.0 * float(intensity)
        glare_weight = min(0.5, 0.5 * float(intensity))
        return apply_physics_pipeline(x, mtf_sigma=0.0, glare_sigma=glare_sigma, glare_weight=glare_weight, n0_base=0.0)

    def physical_params(self, intensity: float):
        glare_sigma = 6.0 + 12.0 * float(intensity)
        glare_weight = min(0.5, 0.5 * float(intensity))
        return {
            "glare_sigma_px": glare_sigma,
            "glare_weight": glare_weight,
            "unit": "px/relative",
            "destructive": _destructive_flag(0.0, glare_weight, 1.0),
            "partial_realism_limitations": list(PARTIAL_REALISM_LIMITATIONS),
        }


class ExposureShift(DomainShift):
    def __init__(self):
        super().__init__("exposure")

    def apply(self, x: torch.Tensor, intensity: float, rng=None, metadata=None) -> torch.Tensor:
        # intensity controls dose reduction
        dose_scale = max(0.1, 1.0 - 0.8 * float(intensity))
        return apply_physics_pipeline(x, dose_scale=dose_scale, aec=True, aec_target_q=0.5)

    def physical_params(self, intensity: float):
        dose_scale = max(0.1, 1.0 - 0.8 * float(intensity))
        return {
            "dose_scale": dose_scale,  # relative dose (N0 scale)
            "aec": True,
            "family_role": "legacy_separable_dose_proxy",
            "unit": "relative",
            "destructive": _destructive_flag(0.0, 0.0, dose_scale),
            **shift_physics_semantics(self.name),
        }


class PoissonNoiseShift(DomainShift):
    def __init__(self):
        super().__init__("poisson")

    def apply(self, x: torch.Tensor, intensity: float, rng=None, metadata=None) -> torch.Tensor:
        dose_scale = max(0.05, 1.0 - 0.9 * float(intensity))
        electronic_std = 0.002 * float(intensity)
        return apply_physics_pipeline(x, dose_scale=dose_scale, electronic_std=electronic_std)

    def physical_params(self, intensity: float):
        dose_scale = max(0.05, 1.0 - 0.9 * float(intensity))
        electronic_std = 0.002 * float(intensity)
        return {
            "dose_scale": dose_scale,
            "electronic_std": electronic_std,
            "unit": "relative",
            "destructive": _destructive_flag(0.0, 0.0, dose_scale),
            "family_role": "legacy_separable_noise_proxy",
            **shift_physics_semantics(self.name),
        }


class ExposurePoissonCoupledShift(DomainShift):
    def __init__(self):
        super().__init__("exposure_poisson")

    def apply(self, x: torch.Tensor, intensity: float, rng=None, metadata=None) -> torch.Tensor:
        dose_scale = max(0.08, 1.0 - 0.82 * float(intensity))
        electronic_std = 0.0015 + 0.003 * float(intensity)
        return apply_physics_pipeline(
            x,
            dose_scale=dose_scale,
            electronic_std=electronic_std,
            aec=True,
            aec_target_q=0.5,
        )

    def physical_params(self, intensity: float):
        dose_scale = max(0.08, 1.0 - 0.82 * float(intensity))
        electronic_std = 0.0015 + 0.003 * float(intensity)
        return {
            "dose_scale": dose_scale,
            "electronic_std": electronic_std,
            "aec": True,
            "noise_coupling": "dose_coupled",
            "family_role": "canonical_coupled_dose_noise",
            "unit": "relative",
            "destructive": _destructive_flag(0.0, 0.0, dose_scale),
            **shift_physics_semantics(self.name),
        }


class MotionBlurShift(DomainShift):
    def __init__(self):
        super().__init__("motion_blur")

    def apply(self, x: torch.Tensor, intensity: float, rng=None, metadata=None) -> torch.Tensor:
        kernel_px = 1.0 + 12.0 * float(intensity)
        return apply_physics_pipeline(
            x,
            motion_kernel_px=kernel_px,
            motion_axis="x",
            n0_base=0.0,
        )

    def physical_params(self, intensity: float):
        kernel_px = 1.0 + 12.0 * float(intensity)
        return {
            "motion_kernel_px": kernel_px,
            "motion_axis": "x",
            "unit": "px",
            "destructive": bool(kernel_px >= 7.0),
            **shift_physics_semantics(self.name),
        }


class CollimationCropShift(DomainShift):
    def __init__(self):
        super().__init__("collimation_crop")

    def _crop_params(self, intensity: float, rng=None) -> Dict[str, float]:
        crop_strength = float(np.clip(intensity, 0.0, 1.0))
        base = 0.03 + 0.18 * crop_strength
        asym = 0.02 + 0.10 * crop_strength
        if rng is not None:
            left = float(np.clip(base + rng.uniform(-0.01, asym), 0.0, 0.38))
            right = float(np.clip(base + rng.uniform(0.0, asym), 0.0, 0.42))
            top = float(np.clip(0.01 + 0.06 * crop_strength + rng.uniform(-0.01, 0.03), 0.0, 0.20))
            bottom = float(np.clip(base + rng.uniform(0.0, asym), 0.0, 0.42))
        else:
            left = float(np.clip(base, 0.0, 0.35))
            right = float(np.clip(base + 0.04 * crop_strength, 0.0, 0.40))
            top = float(np.clip(0.01 + 0.05 * crop_strength, 0.0, 0.16))
            bottom = float(np.clip(base + 0.06 * crop_strength, 0.0, 0.42))
        retained = float(max(1e-3, (1.0 - left - right) * (1.0 - top - bottom)))
        return {
            "left_frac": left,
            "right_frac": right,
            "top_frac": top,
            "bottom_frac": bottom,
            "retained_area_fraction": retained,
        }

    def apply(self, x: torch.Tensor, intensity: float, rng=None, metadata=None) -> torch.Tensor:
        params = self._crop_params(float(intensity), rng=rng)
        return _crop_resize_back(
            x,
            left_frac=params["left_frac"],
            right_frac=params["right_frac"],
            top_frac=params["top_frac"],
            bottom_frac=params["bottom_frac"],
        )

    def physical_params(self, intensity: float):
        params = self._crop_params(float(intensity), rng=None)
        return {
            **params,
            "unit": "relative_fraction",
            "field_of_view_proxy": "crop_and_resize",
            "destructive": bool(params["retained_area_fraction"] <= 0.58),
            **shift_physics_semantics(self.name),
        }
