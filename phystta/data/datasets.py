from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


class XRayDataset(Dataset):
    def __init__(
        self,
        df,
        label_cols: List[str],
        transforms=None,
        model_representation: str = "intensity",
        preprocess_cfg=None,
    ):
        self.df = df.reset_index(drop=True)
        self.label_cols = label_cols
        self.transforms = transforms
        self.model_representation = model_representation
        self.preprocess_cfg = preprocess_cfg

    def __len__(self) -> int:
        return len(self.df)

    @staticmethod
    def _clean_text(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, (float, np.floating)) and np.isnan(value):
            return ""
        return str(value)

    @staticmethod
    def _clean_float(value: Any) -> float:
        if value is None:
            return float("nan")
        if isinstance(value, (float, np.floating)) and np.isnan(value):
            return float("nan")
        try:
            return float(value)
        except Exception:
            return float("nan")

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        image = self._load_image(row["image_path"])
        source_type = row.get("source_type", "cxr")
        mu_max = getattr(self.preprocess_cfg, "mu_max", 6.0) if self.preprocess_cfg is not None else 6.0
        if self.model_representation == "mu":
            if source_type != "lidc_drr":
                image = self._intensity_to_mu(image, mu_max=mu_max)
        elif self.model_representation == "intensity":
            if source_type == "lidc_drr":
                image = self._mu_to_intensity(image, mu_max=mu_max)
        if self.transforms is not None:
            image = self.transforms(image)
        labels = torch.tensor(row[self.label_cols].values.astype(np.float32))
        sample = {
            "image": image,
            "labels": labels,
            "patient_id": self._clean_text(row["patient_id"]),
            "source": self._clean_text(row["source"]),
            "image_path": self._clean_text(row["image_path"]),
        }
        if "global_patient_id" in row:
            sample["global_patient_id"] = self._clean_text(row["global_patient_id"])
        if "global_image_id" in row:
            sample["global_image_id"] = self._clean_text(row["global_image_id"])
        if "age" in row:
            sample["age"] = self._clean_float(row["age"])
        if "sex" in row:
            sample["sex"] = self._clean_text(row["sex"])
        if "race" in row:
            sample["race"] = self._clean_text(row["race"])
        if "ethnicity" in row:
            sample["ethnicity"] = self._clean_text(row["ethnicity"])
        if "view" in row:
            sample["view"] = self._clean_text(row["view"])
        if "view_family" in row:
            sample["view_family"] = self._clean_text(row["view_family"])
        if "is_frontal" in row:
            sample["is_frontal"] = bool(row["is_frontal"])
        if "is_lateral" in row:
            sample["is_lateral"] = bool(row["is_lateral"])
        if "study_id" in row:
            sample["study_id"] = self._clean_text(row["study_id"])
        if "study_datetime" in row:
            sample["study_datetime"] = self._clean_text(row["study_datetime"])
        if "manufacturer" in row:
            sample["manufacturer"] = self._clean_text(row["manufacturer"])
        if "station_name" in row:
            sample["station_name"] = self._clean_text(row["station_name"])
        if "procedure_context" in row:
            sample["procedure_context"] = self._clean_text(row["procedure_context"])
        if "orientation_context" in row:
            sample["orientation_context"] = self._clean_text(row["orientation_context"])
        if "resolution_bucket" in row:
            sample["resolution_bucket"] = self._clean_text(row["resolution_bucket"])
        if "rows" in row:
            sample["rows"] = self._clean_float(row["rows"])
        if "columns" in row:
            sample["columns"] = self._clean_float(row["columns"])
        if "dicom_id" in row:
            sample["dicom_id"] = self._clean_text(row["dicom_id"])
        return sample

    def _load_image(self, path: str) -> torch.Tensor:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(p)
        if p.suffix.lower() in {".dcm", ".dicom"}:
            return self._load_dicom(p)
        if p.suffix.lower() in {".npy"}:
            return self._load_npy(p)
        return self._load_image_file(p)

    def _load_image_file(self, path: Path) -> torch.Tensor:
        try:
            from PIL import Image
        except Exception as e:
            raise RuntimeError("PIL is required for non-DICOM images") from e
        img = Image.open(path).convert("L")
        arr = np.array(img).astype(np.float32)
        arr = arr / 255.0
        return torch.from_numpy(arr).unsqueeze(0)

    def _load_npy(self, path: Path) -> torch.Tensor:
        arr = np.load(path).astype(np.float32)
        if arr.ndim == 2:
            arr = arr[None, ...]
        return torch.from_numpy(arr)

    @staticmethod
    def _intensity_to_mu(x: torch.Tensor, mu_max: float = 6.0) -> torch.Tensor:
        x = torch.clamp(x, min=1e-6)
        mu = -torch.log(x)
        mu = torch.clamp(mu, 0.0, mu_max)
        return mu / max(mu_max, 1e-6)

    @staticmethod
    def _mu_to_intensity(x: torch.Tensor, mu_max: float = 6.0) -> torch.Tensor:
        return torch.exp(-x * mu_max).clamp(0.0, 1.0)

    def _load_dicom(self, path: Path) -> torch.Tensor:
        try:
            import pydicom
        except Exception as e:
            raise RuntimeError("pydicom is required for DICOM images") from e
        ds = pydicom.dcmread(str(path))
        arr = ds.pixel_array.astype(np.float32)
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        arr = arr * slope + intercept
        if getattr(ds, "PhotometricInterpretation", "").upper() == "MONOCHROME1":
            arr = arr.max() - arr
        # fixed scaling (no per-image min-max)
        arr = self._normalize_dicom(arr, ds)
        return torch.from_numpy(arr).unsqueeze(0)

    def _normalize_dicom(self, arr: np.ndarray, ds) -> np.ndarray:
        # prefer window center/width if available
        wc = getattr(ds, "WindowCenter", None)
        ww = getattr(ds, "WindowWidth", None)
        if wc is not None and ww is not None:
            if isinstance(wc, (list, tuple)):
                wc = wc[0]
            if isinstance(ww, (list, tuple)):
                ww = ww[0]
            wc = float(wc)
            ww = float(ww)
            low = wc - ww / 2.0
            high = wc + ww / 2.0
            arr = np.clip(arr, low, high)
            arr = (arr - low) / max(high - low, 1e-6)
            return arr.astype(np.float32)
        # fixed min/max from preprocess config if provided
        if self.preprocess_cfg is not None:
            fixed_min = getattr(self.preprocess_cfg, "fixed_min", None)
            fixed_max = getattr(self.preprocess_cfg, "fixed_max", None)
            if fixed_min is not None and fixed_max is not None:
                arr = np.clip(arr, fixed_min, fixed_max)
                arr = (arr - fixed_min) / max(fixed_max - fixed_min, 1e-6)
                return arr.astype(np.float32)
            dicom_max = getattr(self.preprocess_cfg, "dicom_max", 4095.0)
        else:
            dicom_max = 4095.0
        arr = np.clip(arr, 0.0, dicom_max)
        arr = arr / max(dicom_max, 1e-6)
        return arr.astype(np.float32)
