from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class RealismMapping:
    shift_type: str
    realism_levels: List[float]
    intensities: List[float]
    distance_metric: str = "l2"
    validity_flags: List[str] = field(default_factory=list)
    is_graded: bool = True
    level_separation_margin: float = 0.0
    n_pairs_total: int = 0
    n_pairs_accepted: int = 0


class RealismCalibrator:
    def __init__(self, distance_metric: str = "l2"):
        self.distance_metric = distance_metric
        self.mappings: Dict[str, RealismMapping] = {}
        self.real_quantiles: Dict[str, float] = {}
        self.encoder_fingerprint: Dict[str, object] = {}
        self.metadata: Dict[str, object] = {}

    @staticmethod
    def _mmd_1d(x: np.ndarray, y: np.ndarray) -> float:
        x = x.reshape(-1, 1)
        y = y.reshape(-1, 1)
        dists = np.linalg.norm(np.vstack([x, y])[:, None, :] - np.vstack([x, y])[None, :, :], axis=-1)
        med = np.median(dists)
        gamma = 1.0 / max(med ** 2, 1e-8)
        Kxx = np.exp(-gamma * np.linalg.norm(x[:, None, :] - x[None, :, :], axis=-1) ** 2)
        Kyy = np.exp(-gamma * np.linalg.norm(y[:, None, :] - y[None, :, :], axis=-1) ** 2)
        Kxy = np.exp(-gamma * np.linalg.norm(x[:, None, :] - y[None, :, :], axis=-1) ** 2)
        return float(Kxx.mean() + Kyy.mean() - 2 * Kxy.mean())

    def fit(
        self,
        shift_type: str,
        real_dists: np.ndarray,
        synth_dists_by_intensity: Dict[float, np.ndarray],
        realism_levels: List[float],
        method: str = "quantile",
        saturation_tolerance: float = 0.02,
        n_pairs_total: int = 0,
        n_pairs_accepted: int = 0,
    ) -> RealismMapping:
        sorted_items = sorted(((float(k), np.asarray(v)) for k, v in synth_dists_by_intensity.items()), key=lambda kv: kv[0])
        intensities_sorted = np.array([k for k, _ in sorted_items], dtype=np.float32)
        synth_medians = np.array([np.median(v) for _, v in sorted_items], dtype=np.float32)
        try:
            from sklearn.isotonic import IsotonicRegression
        except Exception as e:
            raise RuntimeError("scikit-learn is required for realism calibration (isotonic regression).") from e
        iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
        synth_medians_iso = iso.fit_transform(intensities_sorted, synth_medians)

        levels = np.array(realism_levels, dtype=np.float32)
        if levels.max() <= 1.0:
            targets = np.quantile(real_dists, levels)
        else:
            # normalize to [0,1] if levels are absolute
            levels = (levels - levels.min()) / max(levels.max() - levels.min(), 1e-6)
            targets = np.quantile(real_dists, levels)

        if method == "mmd":
            mmd_vals = np.array([self._mmd_1d(real_dists, v) for _, v in sorted_items], dtype=np.float32)
            # lower mmd = more realistic, map levels to mmd targets linearly
            mmd_iso = iso.fit_transform(intensities_sorted, mmd_vals)
            mmin, mmax = float(mmd_iso.min()), float(mmd_iso.max())
            target_mmd = mmin + (1.0 - levels) * (mmax - mmin)
            mapped = np.interp(target_mmd, mmd_iso, intensities_sorted)
        else:
            mapped = np.interp(targets, synth_medians_iso, intensities_sorted)

        mapped = [float(x) for x in mapped]
        diffs = np.diff(sorted(mapped)) if len(mapped) > 1 else np.array([], dtype=float)
        curve_diffs = np.diff(synth_medians_iso) if len(synth_medians_iso) > 1 else np.array([], dtype=float)
        mapped_margin = float(np.min(np.abs(diffs))) if diffs.size else float("inf")
        curve_margin = float(np.min(np.abs(curve_diffs))) if curve_diffs.size else float("inf")
        min_margin = float(min(mapped_margin, curve_margin))
        validity_flags: List[str] = []
        is_graded = True
        if diffs.size and min_margin <= float(saturation_tolerance):
            validity_flags.append("non_graded")
            is_graded = False
        if curve_diffs.size and curve_margin <= float(saturation_tolerance):
            if "non_graded" not in validity_flags:
                validity_flags.append("non_graded")
            validity_flags.append("saturated_reference_curve")
            is_graded = False
        if len(set(round(v, 6) for v in mapped)) < len(mapped):
            if "non_graded" not in validity_flags:
                validity_flags.append("non_graded")
            is_graded = False
        if not validity_flags:
            validity_flags.append("valid")

        mapping = RealismMapping(
            shift_type=shift_type,
            realism_levels=realism_levels,
            intensities=mapped,
            distance_metric=self.distance_metric,
            validity_flags=validity_flags,
            is_graded=is_graded,
            level_separation_margin=0.0 if not np.isfinite(min_margin) else float(min_margin),
            n_pairs_total=int(n_pairs_total),
            n_pairs_accepted=int(n_pairs_accepted),
        )
        self.mappings[shift_type] = mapping
        self.real_quantiles[shift_type] = float(np.quantile(real_dists, 0.9))
        return mapping

    def to_dict(self) -> Dict[str, object]:
        return {
            "distance_metric": self.distance_metric,
            "real_quantiles": self.real_quantiles,
            "encoder_fingerprint": self.encoder_fingerprint,
            "metadata": self.metadata,
            "mappings": {k: {
                "shift_type": v.shift_type,
                "realism_levels": v.realism_levels,
                "intensities": v.intensities,
                "distance_metric": v.distance_metric,
                "validity_flags": v.validity_flags,
                "is_graded": v.is_graded,
                "level_separation_margin": v.level_separation_margin,
                "n_pairs_total": v.n_pairs_total,
                "n_pairs_accepted": v.n_pairs_accepted,
            } for k, v in self.mappings.items()},
        }

    @staticmethod
    def from_dict(data: Dict[str, object]) -> "RealismCalibrator":
        calib = RealismCalibrator(distance_metric=str(data.get("distance_metric", "l2")))
        calib.real_quantiles = data.get("real_quantiles", {})
        fp = data.get("encoder_fingerprint", {})
        calib.encoder_fingerprint = fp if isinstance(fp, dict) else {}
        meta = data.get("metadata", {})
        calib.metadata = meta if isinstance(meta, dict) else {}
        for k, v in data.get("mappings", {}).items():
            calib.mappings[k] = RealismMapping(
                shift_type=v["shift_type"],
                realism_levels=v["realism_levels"],
                intensities=v["intensities"],
                distance_metric=v.get("distance_metric", "l2"),
                validity_flags=list(v.get("validity_flags", [])),
                is_graded=bool(v.get("is_graded", True)),
                level_separation_margin=float(v.get("level_separation_margin", 0.0)),
                n_pairs_total=int(v.get("n_pairs_total", 0)),
                n_pairs_accepted=int(v.get("n_pairs_accepted", 0)),
            )
        return calib


def calibrator_fingerprint_mismatches(
    calib: RealismCalibrator,
    expected_fingerprint: Dict[str, object],
) -> List[str]:
    expected = expected_fingerprint or {}
    actual = calib.encoder_fingerprint or {}
    if not actual:
        return ["missing calibrator.encoder_fingerprint"]
    mismatches: List[str] = []
    keys = [
        "encoder_source",
        "architecture",
        "model_id",
        "encoder_name",
        "revision",
        "pretrained",
        "use_medical_pretrain",
        "effective_medical_pretrain",
        "encoder_path",
        "preprocess_signature",
    ]
    for key in keys:
        if actual.get(key) != expected.get(key):
            mismatches.append(f"{key}: expected={expected.get(key)} got={actual.get(key)}")
    if actual.get("input_adapter", {}) != expected.get("input_adapter", {}):
        mismatches.append("input_adapter mismatch")
    return mismatches


def save_calibrator(path: str, calib: RealismCalibrator) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(calib.to_dict(), indent=2))


def load_calibrator(path: str) -> RealismCalibrator:
    data = json.loads(Path(path).read_text())
    return RealismCalibrator.from_dict(data)


def intensity_grid_from_calibrator(
    calib: RealismCalibrator,
    shift_types: List[str],
    graded_only: bool = False,
) -> Dict[str, List[float]]:
    out = {}
    for st in shift_types:
        if st in calib.mappings:
            mapping = calib.mappings[st]
            if graded_only and not mapping.is_graded:
                continue
            out[st] = mapping.intensities
    return out


def scenario_grid_from_calibrator(
    calib: RealismCalibrator,
    shift_types: List[str],
    graded_only: bool = False,
) -> Dict[str, List[Tuple[float, float]]]:
    """
    Return per-shift scenario grid as (realism_level, physical_intensity) pairs.
    """
    out: Dict[str, List[Tuple[float, float]]] = {}
    for st in shift_types:
        mapping = calib.mappings.get(st)
        if mapping is None:
            continue
        if graded_only and not mapping.is_graded:
            continue
        pairs = []
        for level, intensity in zip(mapping.realism_levels, mapping.intensities):
            pairs.append((float(level), float(intensity)))
        out[st] = pairs
    return out


def build_realism_validity_summary(
    *,
    is_graded: bool,
    level_separation_margin: float,
    n_pairs_total: int,
    n_pairs_accepted: int,
    pair_quality_distribution: Dict[str, int],
    accepted_view_families: List[str],
    accepted_procedure_contexts: List[str],
    accepted_orientation_contexts: List[str],
    pair_acceptance_funnel: Dict[str, int],
) -> Dict[str, object]:
    return {
        "is_graded": bool(is_graded),
        "level_separation_margin": float(level_separation_margin),
        "n_pairs_total": int(n_pairs_total),
        "n_pairs_accepted": int(n_pairs_accepted),
        "pair_quality_distribution": {str(k): int(v) for k, v in pair_quality_distribution.items()},
        "accepted_view_families": [str(v) for v in accepted_view_families],
        "accepted_procedure_contexts": [str(v) for v in accepted_procedure_contexts],
        "accepted_orientation_contexts": [str(v) for v in accepted_orientation_contexts],
        "pair_acceptance_funnel": {str(k): int(v) for k, v in pair_acceptance_funnel.items()},
    }
