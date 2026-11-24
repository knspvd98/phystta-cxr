from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


@dataclass
class DRRConfig:
    energies_keV: List[float]
    spectrum_weights: List[float]
    kvp: int = 120
    sid_mm: float = 1200.0
    sad_mm: float = 1000.0
    detector_pixels: int = 224
    pixel_spacing_mm: float = 1.0
    scatter_model: str = "thickness"  # thickness | none | monte_carlo
    clamp_min: float = 0.0
    clamp_max: float = 1.0
    output_representation: str = "intensity"  # intensity | mu


def kramers_spectrum(kvp: int, energies: List[float]) -> np.ndarray:
    # Simple Kramers law spectrum for x-ray tube
    energies = np.array(energies, dtype=np.float32)
    spectrum = (kvp - energies) * energies
    spectrum = np.clip(spectrum, 0.0, None)
    if spectrum.sum() > 0:
        spectrum = spectrum / spectrum.sum()
    return spectrum


def hu_to_mu(hu: np.ndarray, mu_water: float = 0.02) -> np.ndarray:
    # Linear attenuation coefficient approximation from Hounsfield Units
    return mu_water * (1.0 + hu / 1000.0)


def load_volume(path: Path) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    try:
        import SimpleITK as sitk
    except Exception as e:
        raise RuntimeError("SimpleITK is required to load CT volumes for DRR generation") from e
    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img).astype(np.float32)  # z,y,x
    spacing = img.GetSpacing()  # x,y,z
    return arr, (float(spacing[2]), float(spacing[1]), float(spacing[0]))


def project_mu(mu: np.ndarray, spacing: Tuple[float, float, float]) -> np.ndarray:
    dz = spacing[0]
    line_integral = np.sum(mu * dz, axis=0)
    line_integral = line_integral - line_integral.min()
    if line_integral.max() > 0:
        line_integral = line_integral / line_integral.max()
    return line_integral.astype(np.float32)


def project_drr(mu: np.ndarray, spacing: Tuple[float, float, float], energies: List[float], weights: List[float]) -> np.ndarray:
    # mu: z,y,x in 1/mm; spacing: z,y,x in mm
    dz = spacing[0]
    line_integral = np.sum(mu * dz, axis=0)
    energies = np.array(energies, dtype=np.float32)
    weights = np.array(weights, dtype=np.float32)
    if weights.sum() > 0:
        weights = weights / weights.sum()
    # Beer-Lambert over spectrum: I = sum w(E) * exp(-mu * path)
    att = np.exp(-line_integral[None, ...] * (energies[:, None, None] / energies.max()))
    intensity = np.sum(weights[:, None, None] * att, axis=0)
    intensity = intensity - intensity.min()
    if intensity.max() > 0:
        intensity = intensity / intensity.max()
    return intensity.astype(np.float32)


def torch_kramers_spectrum(kvp: int, energies: torch.Tensor) -> torch.Tensor:
    spectrum = (kvp - energies) * energies
    spectrum = torch.clamp(spectrum, 0.0)
    if spectrum.sum() > 0:
        spectrum = spectrum / spectrum.sum()
    return spectrum


def torch_project_drr_2d(mu_map: torch.Tensor, energies: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    # mu_map: (B,1,H,W) represents integrated attenuation (thickness map)
    if weights.sum() > 0:
        weights = weights / weights.sum()
    scale = energies / energies.max().clamp_min(1e-6)
    scale = scale.view(1, -1, 1, 1)
    att = torch.exp(-mu_map * scale)
    intensity = (weights.view(1, -1, 1, 1) * att).sum(dim=1, keepdim=True)
    intensity = intensity - intensity.amin(dim=(2, 3), keepdim=True)
    denom = intensity.amax(dim=(2, 3), keepdim=True).clamp_min(1e-6)
    intensity = intensity / denom
    return intensity


def thickness_scatter(mu_map: torch.Tensor, primary: torch.Tensor, strength: float = 0.2, kernel_size: int = 11) -> torch.Tensor:
    # Scatter proportional to low-pass of thickness map times primary
    if strength <= 0:
        return primary
    pad = kernel_size // 2
    kernel = torch.ones((1, 1, kernel_size, kernel_size), device=mu_map.device, dtype=mu_map.dtype)
    kernel = kernel / kernel.sum()
    thickness_lp = torch.nn.functional.conv2d(mu_map, kernel, padding=pad)
    scatter = strength * thickness_lp * primary
    out = primary + scatter
    out = out.clamp(0.0, 1.0)
    return out


class DRRGenerator:
    def __init__(self, cfg: DRRConfig):
        self.cfg = cfg

    def generate(self, volume_path: Path, out_path: Path) -> Path:
        vol, spacing = load_volume(volume_path)
        mu = hu_to_mu(vol)
        if self.cfg.output_representation == "mu":
            mu_map = project_mu(mu, spacing)
            out = mu_map
        else:
            intensity = project_drr(mu, spacing, self.cfg.energies_keV, self.cfg.spectrum_weights)
            out = intensity
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, out)
        return out_path


class DRRIndexBuilder:
    def __init__(self, root: Path, out_dir: Path, cfg: DRRConfig, metadata_csv: Optional[Path] = None, label_cols: Optional[List[str]] = None):
        self.root = Path(root)
        self.out_dir = Path(out_dir)
        self.cfg = cfg
        self.metadata_csv = metadata_csv
        self.label_cols = label_cols or []

    def build(self) -> "np.ndarray":
        import pandas as pd
        volumes = list(self.root.rglob("*.mhd")) + list(self.root.rglob("*.nii")) + list(self.root.rglob("*.nii.gz"))
        if not volumes:
            raise FileNotFoundError(f"No CT volumes found under {self.root}")
        label_map = {}
        if self.metadata_csv is not None and self.label_cols:
            meta = pd.read_csv(self.metadata_csv)
            if "volume_id" not in meta.columns:
                raise KeyError("LIDC metadata CSV must include volume_id for label mapping")
            for _, row in meta.iterrows():
                label_map[str(row["volume_id"])] = {col: float(row[col]) for col in self.label_cols if col in row}
        rows = []
        for vol_path in volumes:
            volume_id = vol_path.stem.replace(".nii", "")
            out_path = self.out_dir / f"{volume_id}_kvp{self.cfg.kvp}.npy"
            if not out_path.exists():
                self.generate_drr(vol_path, out_path)
            row = {
                "source": "lidc_drr",
                "patient_id": volume_id,
                "image_path": str(out_path),
            }
            if volume_id in label_map:
                row.update(label_map[volume_id])
            elif self.label_cols:
                for col in self.label_cols:
                    row[col] = -1.0
            rows.append(row)
        return pd.DataFrame(rows)

    def generate_drr(self, vol_path: Path, out_path: Path) -> None:
        gen = DRRGenerator(self.cfg)
        gen.generate(vol_path, out_path)
