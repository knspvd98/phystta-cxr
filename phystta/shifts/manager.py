from __future__ import annotations

from typing import Dict, Optional, List
import hashlib
import contextlib

import numpy as np
import torch

try:
    from .physics import (
        HeelEffectShift,
        ScatterShift,
        ExposureShift,
        PoissonNoiseShift,
        ExposurePoissonCoupledShift,
        MotionBlurShift,
        CollimationCropShift,
        MTFGlareShift,
        GlareShift,
        apply_physics_pipeline,
        physical_dependence_warnings,
    )
    _PHYSICS_IMPORT_ERROR: Optional[Exception] = None
except Exception as e:  # pragma: no cover - exercised in minimal envs without physics deps
    HeelEffectShift = None  # type: ignore[assignment]
    ScatterShift = None  # type: ignore[assignment]
    ExposureShift = None  # type: ignore[assignment]
    PoissonNoiseShift = None  # type: ignore[assignment]
    ExposurePoissonCoupledShift = None  # type: ignore[assignment]
    MotionBlurShift = None  # type: ignore[assignment]
    CollimationCropShift = None  # type: ignore[assignment]
    MTFGlareShift = None  # type: ignore[assignment]
    GlareShift = None  # type: ignore[assignment]
    apply_physics_pipeline = None  # type: ignore[assignment]
    physical_dependence_warnings = None  # type: ignore[assignment]
    _PHYSICS_IMPORT_ERROR = e

try:
    from .protocol import ProtocolShift
    _PROTOCOL_IMPORT_ERROR: Optional[Exception] = None
except Exception as e:  # pragma: no cover - exercised in minimal envs without protocol deps
    ProtocolShift = None  # type: ignore[assignment]
    _PROTOCOL_IMPORT_ERROR = e


class ShiftManager:
    def __init__(self, seed: int = 2024):
        self.rng = np.random.default_rng(seed)
        self._unavailable: Dict[str, str] = {}
        def _mk(name: str, cls):
            if cls is None:
                return None
            try:
                return cls()
            except Exception as e:
                self._unavailable[name] = f"{type(e).__name__}: {e}"
                return None

        self.shifts = {
            "none": None,
            "heel": _mk("heel", HeelEffectShift),
            "scatter": _mk("scatter", ScatterShift),
            "mtf": _mk("mtf", MTFGlareShift),
            "glare": _mk("glare", GlareShift),
            "exposure": _mk("exposure", ExposureShift),
            "poisson": _mk("poisson", PoissonNoiseShift),
            "exposure_poisson": _mk("exposure_poisson", ExposurePoissonCoupledShift),
            "motion_blur": _mk("motion_blur", MotionBlurShift),
            "collimation_crop": _mk("collimation_crop", CollimationCropShift),
            "protocol": _mk("protocol", ProtocolShift),
        }
        if _PHYSICS_IMPORT_ERROR is not None:
            msg = f"{type(_PHYSICS_IMPORT_ERROR).__name__}: {_PHYSICS_IMPORT_ERROR}"
            for n in ["heel", "scatter", "mtf", "glare", "exposure", "poisson", "exposure_poisson", "motion_blur", "collimation_crop"]:
                self._unavailable.setdefault(n, msg)
        if _PROTOCOL_IMPORT_ERROR is not None:
            msg = f"{type(_PROTOCOL_IMPORT_ERROR).__name__}: {_PROTOCOL_IMPORT_ERROR}"
            self._unavailable.setdefault("protocol", msg)
        self.drr_cfg = None

    @staticmethod
    def _quality_metrics(x: torch.Tensor) -> Dict[str, float]:
        xi = x.detach().float()
        if xi.dim() == 3 and xi.shape[0] > 1:
            xi = xi.mean(dim=0)
        elif xi.dim() == 3 and xi.shape[0] == 1:
            xi = xi[0]
        xi = xi.clamp(0.0, 1.0)
        mean = float(xi.mean().item())
        std = float(xi.std().item())
        q01 = float(torch.quantile(xi, 0.01).item())
        q99 = float(torch.quantile(xi, 0.99).item())
        clip_rate = float(((xi <= 0.01) | (xi >= 0.99)).float().mean().item())
        dx = (xi[:, 1:] - xi[:, :-1]).abs().mean() if xi.shape[1] > 1 else torch.tensor(0.0, device=xi.device)
        dy = (xi[1:, :] - xi[:-1, :]).abs().mean() if xi.shape[0] > 1 else torch.tensor(0.0, device=xi.device)
        edge_power = float((0.5 * (dx + dy)).item())
        snr_proxy = float(mean / max(std, 1e-6))
        dynamic_range = float(q99 - q01)
        destructive = bool(
            (clip_rate > 0.20)
            or (edge_power < 0.010)
            or (std < 0.030)
            or (dynamic_range < 0.10)
        )
        return {
            "mean": mean,
            "std": std,
            "clip_rate": clip_rate,
            "edge_power": edge_power,
            "snr_proxy": snr_proxy,
            "dynamic_range": dynamic_range,
            "destructive": destructive,
        }

    def apply(self, x: torch.Tensor, shift_name: str, intensity: float) -> torch.Tensor:
        if shift_name == "none" or intensity <= 0:
            return x
        shift = self.shifts.get(shift_name)
        if shift is None:
            if shift_name in self._unavailable:
                raise RuntimeError(
                    f"Shift '{shift_name}' is unavailable in this environment: {self._unavailable[shift_name]}"
                )
            raise KeyError(f"Unknown shift {shift_name}")
        return shift.apply(x, intensity, rng=self.rng)

    def apply_differentiable(self, x: torch.Tensor, shift_name: str, intensity: float) -> torch.Tensor:
        if shift_name == "none" or intensity <= 0:
            return x
        shift = self.shifts.get(shift_name)
        if shift is None:
            raise KeyError(f"Unknown shift {shift_name}")
        return shift.apply(x, intensity, rng=None)

    def _seed_from(self, sample_id: str, shift_key: str, intensity: float) -> int:
        msg = f"{sample_id}|{shift_key}|{float(intensity):.6f}"
        h = hashlib.md5(msg.encode()).hexdigest()
        return int(h[:8], 16)

    @contextlib.contextmanager
    def _seed_context(self, seed: Optional[int], device: torch.device):
        if seed is None:
            yield
            return
        devices = [device] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(int(seed))
            yield

    def apply_batch(self, x: torch.Tensor, shift_name: str, intensity: float, sample_ids: Optional[List[str]] = None) -> torch.Tensor:
        out, _ = self.apply_batch_with_meta(x, shift_name, intensity, sample_ids=sample_ids)
        return out

    def apply_batch_with_meta(self, x: torch.Tensor, shift_name: str, intensity: float, sample_ids: Optional[List[str]] = None):
        out = []
        per_sample = []
        destructive_count = 0
        for i in range(x.shape[0]):
            seed = None
            sid = None
            if sample_ids is not None:
                sid = sample_ids[i]
                seed = self._seed_from(str(sid), shift_name, intensity)
            with self._seed_context(seed, x.device):
                xi = self.apply(x[i], shift_name, intensity)
            qm = self._quality_metrics(xi)
            meta_i = {
                "shift": shift_name,
                "intensity": float(intensity),
                "sample_id": str(sid) if sid is not None else None,
                "seed": int(seed) if seed is not None else None,
                **qm,
            }
            if qm["destructive"]:
                destructive_count += 1
            out.append(xi)
            per_sample.append(meta_i)
        summary = {
            "scenario": "batch",
            "size": len(per_sample),
            "destructive_count": destructive_count,
        }
        return torch.stack(out, dim=0), {"per_sample": per_sample, "summary": summary}

    def apply_batch_differentiable(self, x: torch.Tensor, shift_name: str, intensity: float) -> torch.Tensor:
        out = []
        for i in range(x.shape[0]):
            out.append(self.apply_differentiable(x[i], shift_name, intensity))
        return torch.stack(out, dim=0)

    def apply_with_metadata(self, x: torch.Tensor, shift_name: str, intensity: float):
        shift = self.shifts.get(shift_name)
        if shift is None or shift_name == "none" or intensity <= 0:
            return x, {"scenario": "none", "intensity": 0.0}
        meta = shift.physical_params(intensity)
        meta.update({"shift": shift_name, "intensity": float(intensity)})
        if physical_dependence_warnings is not None:
            warnings = physical_dependence_warnings([shift_name])
            if warnings:
                meta["physical_dependence_warnings"] = warnings
        return shift.apply(x, intensity, rng=self.rng), meta

    def apply_scenario(self, x: torch.Tensor, intensities: Dict[str, float], seed: Optional[int] = None, sample_id: Optional[str] = None):
        meta = {"scenario": "composite", "components": {}, "destructive": False}
        # If composite uses physics shifts, apply a single pipeline pass
        physics_params = {
            "heel_strength": 0.0,
            "scatter_ratio": 0.0,
            "scatter_sigma": 7.0,
            "mtf_sigma": 0.0,
            "glare_sigma": 8.0,
            "glare_weight": 0.0,
            "motion_kernel_px": 0.0,
            "motion_axis": "x",
            "dose_scale": 1.0,
            "electronic_std": 0.0,
            "aec": False,
        }
        use_physics = False
        post_pipeline_ops = []
        for shift_name, intensity in intensities.items():
            if intensity <= 0 or shift_name == "none":
                continue
            shift = self.shifts.get(shift_name)
            if shift is None:
                continue
            meta["components"][shift_name] = shift.physical_params(intensity)
            if meta["components"][shift_name].get("destructive", False):
                meta["destructive"] = True
            if shift_name in {"heel", "scatter", "mtf", "glare", "exposure", "poisson", "exposure_poisson", "motion_blur"}:
                use_physics = True
                # map to pipeline params
                if shift_name == "heel":
                    physics_params["heel_strength"] = meta["components"][shift_name]["heel_strength"]
                elif shift_name == "scatter":
                    physics_params["scatter_ratio"] = meta["components"][shift_name]["scatter_ratio"]
                    physics_params["scatter_sigma"] = meta["components"][shift_name]["scatter_sigma_px"]
                elif shift_name == "mtf":
                    physics_params["mtf_sigma"] = meta["components"][shift_name]["mtf_sigma_px"]
                    physics_params["glare_sigma"] = meta["components"][shift_name]["glare_sigma_px"]
                    physics_params["glare_weight"] = meta["components"][shift_name]["glare_weight"]
                elif shift_name == "glare":
                    physics_params["glare_sigma"] = meta["components"][shift_name]["glare_sigma_px"]
                    physics_params["glare_weight"] = meta["components"][shift_name]["glare_weight"]
                elif shift_name == "exposure":
                    physics_params["dose_scale"] = meta["components"][shift_name]["dose_scale"]
                    physics_params["aec"] = True
                elif shift_name == "poisson":
                    physics_params["dose_scale"] = meta["components"][shift_name]["dose_scale"]
                    physics_params["electronic_std"] = meta["components"][shift_name]["electronic_std"]
                elif shift_name == "exposure_poisson":
                    physics_params["dose_scale"] = meta["components"][shift_name]["dose_scale"]
                    physics_params["electronic_std"] = meta["components"][shift_name]["electronic_std"]
                    physics_params["aec"] = True
                elif shift_name == "motion_blur":
                    physics_params["motion_kernel_px"] = meta["components"][shift_name]["motion_kernel_px"]
                    physics_params["motion_axis"] = meta["components"][shift_name]["motion_axis"]
            else:
                post_pipeline_ops.append((shift_name, float(intensity)))
        if use_physics:
            if apply_physics_pipeline is None:
                detail = self._unavailable.get("heel") or self._unavailable.get("scatter") or "physics backend import error"
                raise RuntimeError(
                    f"Composite physics scenario requested but physics pipeline is unavailable: {detail}"
                )
            with self._seed_context(seed, x.device):
                out = apply_physics_pipeline(x, **physics_params)
            if physical_dependence_warnings is not None:
                warnings = physical_dependence_warnings(intensities.keys())
                if warnings:
                    meta["physical_dependence_warnings"] = warnings
            for shift_name, intensity in post_pipeline_ops:
                shift = self.shifts.get(shift_name)
                if shift is None:
                    continue
                rng = np.random.default_rng(seed) if seed is not None else self.rng
                out = shift.apply(out, intensity, rng=rng)
        else:
            out = x
            for shift_name, intensity in intensities.items():
                if intensity <= 0 or shift_name == "none":
                    continue
                shift = self.shifts.get(shift_name)
                if shift is None:
                    continue
                rng = np.random.default_rng(seed) if seed is not None else self.rng
                out = shift.apply(out, intensity, rng=rng)
        if seed is not None:
            meta["seed"] = int(seed)
        if sample_id is not None:
            meta["sample_id"] = str(sample_id)
        return out, meta

    def apply_batch_scenario(self, x: torch.Tensor, intensities: Dict[str, float], sample_ids: Optional[List[str]] = None):
        out = []
        meta_list = []
        destructive_count = 0
        destructive_physics_count = 0
        for i in range(x.shape[0]):
            seed = None
            sid = None
            if sample_ids is not None:
                sid = sample_ids[i]
                key = "composite|" + ";".join(f"{k}:{float(v):.4f}" for k, v in sorted(intensities.items()))
                seed = self._seed_from(str(sid), key, 0.0)
            xi, meta = self.apply_scenario(x[i], intensities, seed=seed, sample_id=sid)
            qm = self._quality_metrics(xi)
            meta.update(qm)
            meta["destructive_physics"] = bool(meta.get("destructive", False))
            meta["destructive"] = bool(qm["destructive"])
            out.append(xi)
            meta_list.append(meta)
            if meta.get("destructive", False):
                destructive_count += 1
            if meta.get("destructive_physics", False):
                destructive_physics_count += 1
        summary = {
            "scenario": "batch",
            "size": len(meta_list),
            "destructive_count": destructive_count,
            "destructive_physics_count": destructive_physics_count,
        }
        return torch.stack(out, dim=0), {"per_sample": meta_list, "summary": summary}
