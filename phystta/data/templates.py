from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from ..config import DatasetSourceConfig


OPEN_CXR_CANONICAL_LABELS = [
    "No Finding",
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Pneumothorax",
    "Pleural Effusion",
    "Pneumonia",
]


CHEXPERT_LABELS = [
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
]

NIH_LABELS = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia",
]

VINDR_LABELS = [
    "Aortic enlargement",
    "Atelectasis",
    "Calcification",
    "Cardiomegaly",
    "Consolidation",
    "ILD",
    "Infiltration",
    "Lung Opacity",
    "Nodule/Mass",
    "Other lesion",
    "Pleural effusion",
    "Pleural thickening",
    "Pneumothorax",
    "Pulmonary fibrosis",
]

PADCHEST_LABELS = list(OPEN_CXR_CANONICAL_LABELS)


def chexpert_source(root: str, csv_path: str, label_cols: Optional[List[str]] = None) -> DatasetSourceConfig:
    labels = label_cols or CHEXPERT_LABELS
    return DatasetSourceConfig(
        name="chexpert",
        root=Path(root),
        metadata_csv=Path(csv_path),
        path_col="Path",
        patient_id_col="PatientID",
        study_id_col="Study",
        label_cols=labels,
        age_col="Age",
        sex_col="Sex",
        view_col="AP/PA",
        view_family_col="Frontal/Lateral",
        patient_id_from_path_regex=r"patient(\d+)",
        study_id_from_path_regex=r"study(\d+)",
        require_exists=True,
        supports_temporal_pairs=False,
        supports_view_metadata=True,
        supports_demographics=True,
        supports_orientation_metadata=False,
        supports_resolution_metadata=False,
        supports_boxes=False,
        supports_device_metadata=False,
        supports_procedure_metadata=False,
    )


def mimic_source(root: str, csv_path: str, label_cols: Optional[List[str]] = None) -> DatasetSourceConfig:
    labels = label_cols or CHEXPERT_LABELS
    return DatasetSourceConfig(
        name="mimic",
        root=Path(root),
        metadata_csv=Path(csv_path),
        extra_metadata_csv=Path(root) / "mimic-cxr-2.0.0-metadata.csv",
        extra_metadata_join_cols=["subject_id", "study_id", "dicom_id"],
        path_col="path",
        patient_id_col="subject_id",
        study_id_col="study_id",
        label_cols=labels,
        view_col="ViewPosition",
        study_time_col="StudyTime",
        study_date_col="StudyDate",
        dicom_id_col="dicom_id",
        performed_procedure_step_description_col="PerformedProcedureStepDescription",
        procedure_code_meaning_col="ProcedureCodeSequence_CodeMeaning",
        view_code_meaning_col="ViewCodeSequence_CodeMeaning",
        patient_orientation_code_meaning_col="PatientOrientationCodeSequence_CodeMeaning",
        rows_col="Rows",
        columns_col="Columns",
        require_exists=True,
        supports_temporal_pairs=True,
        supports_view_metadata=True,
        supports_demographics=False,
        supports_orientation_metadata=True,
        supports_resolution_metadata=True,
        supports_boxes=False,
        supports_device_metadata=False,
        supports_procedure_metadata=True,
    )


def nih_source(root: str, csv_path: str, label_cols: Optional[List[str]] = None) -> DatasetSourceConfig:
    labels = label_cols or NIH_LABELS
    return DatasetSourceConfig(
        name="nih_chestxray14",
        root=Path(root),
        metadata_csv=Path(csv_path),
        image_root=Path(root) / "images",
        path_col="Image Index",
        patient_id_col="Patient ID",
        label_cols=labels,
        labels_from_findings_col="Finding Labels",
        age_col="Patient Age",
        sex_col="Patient Gender",
        view_col="View Position",
        require_exists=True,
        supports_temporal_pairs=False,
        supports_view_metadata=True,
        supports_demographics=True,
        supports_orientation_metadata=False,
        supports_resolution_metadata=False,
        supports_boxes=False,
        supports_device_metadata=False,
        supports_procedure_metadata=False,
    )


def vindr_source(root: str, csv_path: str, label_cols: Optional[List[str]] = None) -> DatasetSourceConfig:
    labels = label_cols or VINDR_LABELS
    return DatasetSourceConfig(
        name="vindr",
        root=Path(root),
        metadata_csv=Path(csv_path),
        image_root=Path(root),
        path_col="image_id",
        patient_id_col="image_id",
        label_cols=labels,
        require_exists=True,
        supports_temporal_pairs=False,
        supports_view_metadata=False,
        supports_demographics=False,
        supports_orientation_metadata=False,
        supports_resolution_metadata=False,
        supports_boxes=True,
        supports_device_metadata=False,
        supports_procedure_metadata=False,
    )


def padchest_source(root: str, csv_path: str, label_cols: Optional[List[str]] = None) -> DatasetSourceConfig:
    root_path = Path(root)
    labels = label_cols or PADCHEST_LABELS
    return DatasetSourceConfig(
        name="padchest",
        root=root_path,
        metadata_csv=Path(csv_path),
        image_root=_prefer_existing_dir(root_path, ["images", "images-224", "images-512", "PNG"]) or root_path,
        path_col="path",
        patient_id_col="patient_id",
        study_id_col="study_id",
        label_cols=labels,
        age_col="age",
        sex_col="sex",
        view_col="view",
        require_exists=True,
    )


def lidc_drr_source(root: str, label_cols: Optional[List[str]] = None, metadata_csv: Optional[str] = None) -> DatasetSourceConfig:
    return DatasetSourceConfig(
        name="lidc_drr",
        root=Path(root),
        source_type="lidc_drr",
        metadata_csv=Path(metadata_csv) if metadata_csv else None,
        label_cols=label_cols or [],
        require_exists=True,
    )


def resolve_dataset_source(name: str, root: str, label_cols: Optional[List[str]] = None) -> DatasetSourceConfig:
    root_path = Path(root).expanduser()
    if not root_path.exists():
        raise FileNotFoundError(f"Dataset root not found: {root_path}. {expected_files_message(name)}")
    name_lower = name.lower()
    if name_lower in {"nih", "nih_chestxray14", "chestxray14"}:
        candidates = ["Data_Entry_2017.csv", "data_entry_2017.csv"]
        csv_path = _find_first(root_path, candidates)
        if csv_path is None:
            raise FileNotFoundError("NIH ChestX-ray14 requires Data_Entry_2017.csv and images/ directory in the root.")
        return nih_source(str(root_path), str(csv_path), label_cols=label_cols)
    if name_lower in {"chexpert"}:
        candidates = ["train.csv", "CheXpert-v1.0/train.csv", "CheXpert-v1.0-small/train.csv"]
        csv_path = _find_first(root_path, candidates)
        if csv_path is None:
            raise FileNotFoundError("CheXpert requires train.csv in root (and image folders).")
        return chexpert_source(str(root_path), str(csv_path), label_cols=label_cols)
    if name_lower in {"padchest"}:
        candidates = [
            "padchest_metadata.csv",
            "padchest_normalized.csv",
            "metadata/padchest_metadata.csv",
            "metadata/padchest_normalized.csv",
        ]
        csv_path = _find_first(root_path, candidates)
        if csv_path is None:
            raise FileNotFoundError(
                "PadChest requires a normalized metadata CSV such as padchest_metadata.csv in the dataset root."
            )
        return padchest_source(str(root_path), str(csv_path), label_cols=label_cols)
    if name_lower in {"mimic", "mimic-cxr", "mimic-cxr-jpg"}:
        candidates = ["mimic-cxr-2.0.0-chexpert.csv", "mimic-cxr-2.0.0/mimic-cxr-2.0.0-chexpert.csv"]
        csv_path = _find_first(root_path, candidates)
        if csv_path is None:
            raise FileNotFoundError("MIMIC-CXR-JPG requires mimic-cxr-2.0.0-chexpert.csv in root.")
        return mimic_source(str(root_path), str(csv_path), label_cols=label_cols)
    if name_lower in {"lidc", "lidc-idri", "lidc_drr"}:
        return lidc_drr_source(str(root_path), label_cols=label_cols, metadata_csv=None)
    if name_lower in {"vindr", "vindr-cxr"}:
        candidates = ["train.csv", "vinDr-CXR/train.csv"]
        csv_path = _find_first(root_path, candidates)
        if csv_path is None:
            raise FileNotFoundError("VinDr-CXR requires train.csv in root.")
        return vindr_source(str(root_path), str(csv_path), label_cols=label_cols)
    raise ValueError(f"Unknown dataset name: {name}")


def _find_first(root: Path, candidates: List[str]) -> Optional[Path]:
    for c in candidates:
        p = root / c
        if p.exists():
            return p
    return None


def _prefer_existing_dir(root: Path, candidates: List[str]) -> Optional[Path]:
    for candidate in candidates:
        path = root / candidate
        if path.exists() and path.is_dir():
            return path
    return None


def expected_files_message(name: str) -> str:
    name_lower = name.lower()
    if name_lower in {"nih", "nih_chestxray14", "chestxray14"}:
        return "Expected files: Data_Entry_2017.csv and images/ directory."
    if name_lower in {"chexpert"}:
        return "Expected files: train.csv (and image folders) in the dataset root."
    if name_lower in {"padchest"}:
        return "Expected files: padchest_metadata.csv (normalized) plus the image folders under the dataset root."
    if name_lower in {"mimic", "mimic-cxr", "mimic-cxr-jpg"}:
        return "Expected files: mimic-cxr-2.0.0-chexpert.csv and JPG image tree in the root."
    if name_lower in {"lidc", "lidc-idri", "lidc_drr"}:
        return "Expected files: CT volumes (.mhd/.nii/.nii.gz) under dataset root."
    if name_lower in {"vindr", "vindr-cxr"}:
        return "Expected files: train.csv and image folders in the root."
    return "Expected dataset metadata CSV and image directory."
