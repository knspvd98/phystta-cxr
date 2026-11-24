from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from .schema import LabelSchema
from .drr_engine import DRRConfig, DRRIndexBuilder, kramers_spectrum
from .capabilities import enrich_index_frame
from ..config import DataConfig, DatasetSourceConfig


class DatasetIndexBuilder:
    def __init__(self, data_cfg: DataConfig):
        self.data_cfg = data_cfg

    @staticmethod
    def harmonize_frame(df: pd.DataFrame) -> pd.DataFrame:
        return enrich_index_frame(df)

    def build(self) -> pd.DataFrame:
        frames = []
        for src in self.data_cfg.sources:
            frames.append(self._build_source(src))
        index = pd.concat(frames, ignore_index=True)
        index = self.harmonize_frame(index)
        index["source"] = index["source"].astype(str)
        index["patient_id"] = index["patient_id"].astype(str)
        if "study_id" in index.columns:
            index["study_id"] = index["study_id"].astype(str)
        return index

    def _build_source(self, src: DatasetSourceConfig) -> pd.DataFrame:
        if src.source_type == "lidc_drr":
            drr_cfg = self.data_cfg.drr or DRRConfig()
            if drr_cfg.spectrum_weights is None:
                drr_cfg.spectrum_weights = list(kramers_spectrum(drr_cfg.kvp, drr_cfg.energies_keV))
            builder = DRRIndexBuilder(
                root=src.root,
                out_dir=self.data_cfg.cache_dir / "drr" / src.name,
                cfg=drr_cfg,
                metadata_csv=src.metadata_csv,
                label_cols=src.label_cols,
            )
            df = builder.build()
            if "source" not in df.columns:
                df["source"] = src.domain_label or src.name
            return df
        if src.metadata_csv is None:
            raise FileNotFoundError(f"{src.name}: metadata CSV not resolved. Provide metadata_csv or a known dataset root.")
        meta = pd.read_csv(src.metadata_csv)
        if src.extra_metadata_csv is not None:
            extra = pd.read_csv(src.extra_metadata_csv)
            join_cols = src.extra_metadata_join_cols or []
            if not join_cols:
                raise ValueError(f"{src.name}: extra_metadata_join_cols must be provided when extra_metadata_csv is set.")
            for col in join_cols:
                if col not in meta.columns or col not in extra.columns:
                    raise KeyError(f"{src.name}: join column {col} missing in metadata or extra metadata.")
            meta = meta.merge(extra, on=join_cols, how="left", suffixes=("", "_extra"))
        if src.path_col not in meta.columns:
            raise KeyError(f"{src.name}: missing path column {src.path_col}")
        if src.patient_id_col not in meta.columns and not src.patient_id_from_path_regex:
            raise KeyError(f"{src.name}: missing patient column {src.patient_id_col} and no path regex provided")
        for col in src.label_cols:
            if col not in meta.columns and not src.labels_from_findings_col:
                raise KeyError(f"{src.name}: missing label column {col}")

        label_schema = LabelSchema(labels=src.label_cols)
        if src.labels_from_findings_col and src.labels_from_findings_col in meta.columns:
            labels = self._expand_labels_from_findings(meta[src.labels_from_findings_col], src.label_cols, src.label_delim)
        else:
            labels = pd.DataFrame(
                {col: meta[col].map(label_schema.normalize_label) for col in src.label_cols},
                index=meta.index,
            )

        image_paths = meta[src.path_col].astype(str).apply(lambda p: self._resolve_path(src, p))
        if src.patient_id_col in meta.columns:
            patient_ids = meta[src.patient_id_col].astype(str)
        else:
            patient_ids = image_paths.apply(lambda p: self._extract_from_path(p, src.patient_id_from_path_regex))
        df = pd.DataFrame({
            "source": src.domain_label or src.name,
            "patient_id": patient_ids,
            "image_path": image_paths,
            "source_type": src.source_type,
        })
        if src.study_id_col and src.study_id_col in meta.columns:
            df["study_id"] = meta[src.study_id_col].astype(str)
        elif src.study_id_from_path_regex:
            df["study_id"] = image_paths.apply(lambda p: self._extract_from_path(p, src.study_id_from_path_regex))
        if src.age_col and src.age_col in meta.columns:
            df["age"] = pd.to_numeric(meta[src.age_col], errors="coerce")
        if src.sex_col and src.sex_col in meta.columns:
            df["sex"] = meta[src.sex_col].astype(str)
        if src.view_col and src.view_col in meta.columns:
            df["view"] = meta[src.view_col].astype(str)
        if src.race_col and src.race_col in meta.columns:
            df["race"] = meta[src.race_col].astype(str)
        if src.ethnicity_col and src.ethnicity_col in meta.columns:
            df["ethnicity"] = meta[src.ethnicity_col].astype(str)
        if src.manufacturer_col and src.manufacturer_col in meta.columns:
            df["manufacturer"] = meta[src.manufacturer_col].astype(str)
        if src.station_name_col and src.station_name_col in meta.columns:
            df["station_name"] = meta[src.station_name_col].astype(str)
        if src.study_time_col and src.study_time_col in meta.columns:
            df["study_time"] = meta[src.study_time_col].astype(str)
        if src.study_date_col and src.study_date_col in meta.columns:
            df["study_date"] = meta[src.study_date_col].astype(str)
        if src.study_datetime_col and src.study_datetime_col in meta.columns:
            df["study_datetime"] = meta[src.study_datetime_col]
        if src.dicom_id_col and src.dicom_id_col in meta.columns:
            df["dicom_id"] = meta[src.dicom_id_col].astype(str)
        if src.view_family_col and src.view_family_col in meta.columns:
            df["view_family"] = meta[src.view_family_col].astype(str)
        if src.performed_procedure_step_description_col and src.performed_procedure_step_description_col in meta.columns:
            df["performed_procedure_step_description"] = meta[src.performed_procedure_step_description_col].astype(str)
        if src.procedure_code_meaning_col and src.procedure_code_meaning_col in meta.columns:
            df["procedure_code_meaning"] = meta[src.procedure_code_meaning_col].astype(str)
        if src.view_code_meaning_col and src.view_code_meaning_col in meta.columns:
            df["view_code_meaning"] = meta[src.view_code_meaning_col].astype(str)
        if src.patient_orientation_code_meaning_col and src.patient_orientation_code_meaning_col in meta.columns:
            df["patient_orientation_code_meaning"] = meta[src.patient_orientation_code_meaning_col].astype(str)
        if src.rows_col and src.rows_col in meta.columns:
            df["rows"] = pd.to_numeric(meta[src.rows_col], errors="coerce")
        if src.columns_col and src.columns_col in meta.columns:
            df["columns"] = pd.to_numeric(meta[src.columns_col], errors="coerce")
        if src.labels_from_findings_col and src.labels_from_findings_col in meta.columns:
            df["labels_from_findings_raw"] = meta[src.labels_from_findings_col].fillna("").astype(str)

        for col in src.label_cols:
            df[col] = labels[col].astype(float)
        df = enrich_index_frame(df)
        if src.allowed_views:
            df = df[df.get("view", "").isin(src.allowed_views)]
        if src.require_exists:
            missing = df[~df["image_path"].apply(lambda p: Path(p).exists())]
            if len(missing) > 0:
                raise FileNotFoundError(f"{src.name}: {len(missing)} missing image files")
        return df

    def _resolve_path(self, src: DatasetSourceConfig, path_value: str) -> str:
        p = Path(path_value)
        if p.is_absolute():
            return str(p)
        base = src.image_root if src.image_root is not None else src.root
        return str(Path(base) / p)

    def _expand_labels_from_findings(self, findings_series, label_cols, delim: str):
        import pandas as pd
        labels = {col: [] for col in label_cols}
        for item in findings_series.fillna(""):
            findings = set(str(item).split(delim)) if item else set()
            for col in label_cols:
                labels[col].append(1.0 if col in findings else 0.0)
        return pd.DataFrame(labels)

    def _extract_from_path(self, path_value: str, pattern: Optional[str]) -> str:
        if not pattern:
            raise ValueError("No regex pattern provided for path extraction")
        m = re.search(pattern, path_value)
        if not m:
            raise ValueError(f"Pattern {pattern} not found in {path_value}")
        return m.group(1)


class LeakageDetector:
    @staticmethod
    def verify_patient_split(train_ids: List[str], val_ids: List[str], test_ids: List[str]) -> None:
        train = set(train_ids)
        val = set(val_ids)
        test = set(test_ids)
        if train & val or train & test or val & test:
            raise ValueError("Patient leakage detected across splits")
