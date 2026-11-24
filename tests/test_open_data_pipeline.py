from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from phystta.config.schema import DataConfig, DatasetSourceConfig
from phystta.data import DataEngine
from phystta.data.templates import OPEN_CXR_CANONICAL_LABELS, PADCHEST_LABELS, resolve_dataset_source
from scripts.prepare_open_cxr_metadata import (
    prepare_chexpert_metadata,
    prepare_nih_metadata,
    prepare_padchest_metadata,
)


def _write_image(path: Path) -> None:
    arr = np.full((16, 16), 128, dtype=np.uint8)
    Image.fromarray(arr, mode="L").save(path)


def test_padchest_source_resolution_uses_normalized_template(tmp_path: Path):
    root = tmp_path / "padchest"
    (root / "metadata").mkdir(parents=True)
    (root / "images").mkdir()
    (root / "metadata" / "padchest_metadata.csv").write_text("path,patient_id,study_id\n", encoding="utf-8")

    resolved = resolve_dataset_source("padchest", str(root))

    assert resolved.name == "padchest"
    assert resolved.path_col == "path"
    assert resolved.patient_id_col == "patient_id"
    assert resolved.study_id_col == "study_id"
    assert resolved.image_root == root / "images"
    assert resolved.label_cols == PADCHEST_LABELS


def test_prepare_padchest_metadata_maps_labels_and_preserves_missing_as_minus_one(tmp_path: Path):
    root = tmp_path / "padchest_raw"
    root.mkdir()
    raw_csv = root / "PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv"
    out_csv = root / "metadata" / "padchest_metadata.csv"
    pd.DataFrame(
        [
            {
                "ImageID": "img1.png",
                "PatientID": "p1",
                "Projection": "PA",
                "PatientAge": "57",
                "PatientSex_DICOM": "M",
                "Labels": "['cardiomegaly']",
                "Report": "Mild cardiomegaly.",
            },
            {
                "ImageID": "img2.png",
                "PatientID": "p2",
                "Projection": "AP",
                "PatientAge": "63",
                "PatientSex_DICOM": "F",
                "Labels": "",
                "Report": "No structured labels.",
            },
        ]
    ).to_csv(raw_csv, index=False)

    normalized = prepare_padchest_metadata(root, out_csv)

    assert out_csv.exists()
    assert set(OPEN_CXR_CANONICAL_LABELS).issubset(normalized.columns)
    assert normalized.loc[0, "Cardiomegaly"] == 1.0
    assert normalized.loc[0, "No Finding"] == 0.0
    assert all(normalized.loc[1, label] == -1.0 for label in OPEN_CXR_CANONICAL_LABELS)


def test_prepare_chexpert_and_nih_metadata_emit_canonical_columns(tmp_path: Path):
    chexpert_root = tmp_path / "chexpert"
    nih_root = tmp_path / "nih"
    chexpert_root.mkdir()
    nih_root.mkdir()

    chexpert_raw = chexpert_root / "train.csv"
    nih_raw = nih_root / "Data_Entry_2017.csv"
    chexpert_out = chexpert_root / "metadata" / "chexpert_metadata.csv"
    nih_out = nih_root / "metadata" / "nih_metadata.csv"

    pd.DataFrame(
        [
            {
                "Path": "train/patient0001/study1/view1_frontal.jpg",
                "Sex": "Male",
                "Age": 45,
                "AP/PA": "PA",
                "No Finding": 0,
                "Atelectasis": 1,
                "Edema": -1,
            }
        ]
    ).to_csv(chexpert_raw, index=False)
    pd.DataFrame(
        [
            {
                "Image Index": "0001.png",
                "Patient ID": 1,
                "Finding Labels": "Effusion|Pneumonia",
                "View Position": "PA",
                "Patient Age": 60,
                "Patient Gender": "F",
            }
        ]
    ).to_csv(nih_raw, index=False)

    chexpert_df = prepare_chexpert_metadata(chexpert_root, chexpert_out)
    nih_df = prepare_nih_metadata(nih_root, nih_out)

    assert set(OPEN_CXR_CANONICAL_LABELS).issubset(chexpert_df.columns)
    assert set(OPEN_CXR_CANONICAL_LABELS).issubset(nih_df.columns)
    assert chexpert_df.loc[0, "Atelectasis"] == 1.0
    assert chexpert_df.loc[0, "Edema"] == -1.0
    assert nih_df.loc[0, "Pleural Effusion"] == 1.0
    assert nih_df.loc[0, "Pneumonia"] == 1.0
    assert nih_df.loc[0, "No Finding"] == 0.0


def test_open_data_index_has_canonical_labels_and_fills_missing_values(tmp_path: Path):
    labels = list(OPEN_CXR_CANONICAL_LABELS)
    pad_root = tmp_path / "padchest"
    nih_root = tmp_path / "nih"
    pad_img_dir = pad_root / "images"
    nih_img_dir = nih_root / "images"
    pad_img_dir.mkdir(parents=True)
    nih_img_dir.mkdir(parents=True)
    _write_image(pad_img_dir / "pad_1.png")
    _write_image(nih_img_dir / "nih_1.png")

    pad_csv = pad_root / "padchest_metadata.csv"
    nih_csv = nih_root / "nih_metadata.csv"
    pad_row = {
        "path": "pad_1.png",
        "patient_id": "p1",
        "study_id": "pad_study_1",
        "view": "PA",
        "age": 57,
        "sex": "M",
        "report_text": "",
        "No Finding": 0.0,
        "Atelectasis": 0.0,
        "Cardiomegaly": 1.0,
        "Consolidation": 0.0,
        "Edema": 0.0,
        "Pneumothorax": 0.0,
        "Pleural Effusion": np.nan,
        "Pneumonia": np.nan,
    }
    nih_row = {
        "path": "nih_1.png",
        "patient_id": "n1",
        "study_id": "nih_study_1",
        "view": "AP",
        "age": 61,
        "sex": "F",
        "report_text": "",
        "No Finding": 1.0,
        "Atelectasis": 0.0,
        "Cardiomegaly": 0.0,
        "Consolidation": 0.0,
        "Edema": 0.0,
        "Pneumothorax": 0.0,
        "Pleural Effusion": np.nan,
        "Pneumonia": 0.0,
    }
    pd.DataFrame([pad_row]).to_csv(pad_csv, index=False)
    pd.DataFrame([nih_row]).to_csv(nih_csv, index=False)

    cfg = DataConfig(
        sources=[
            DatasetSourceConfig(
                name="padchest",
                root=pad_root,
                metadata_csv=pad_csv,
                image_root=pad_img_dir,
                path_col="path",
                patient_id_col="patient_id",
                study_id_col="study_id",
                view_col="view",
                age_col="age",
                sex_col="sex",
                label_cols=labels,
            ),
            DatasetSourceConfig(
                name="nih_chestxray14",
                root=nih_root,
                metadata_csv=nih_csv,
                image_root=nih_img_dir,
                path_col="path",
                patient_id_col="patient_id",
                study_id_col="study_id",
                view_col="view",
                age_col="age",
                sex_col="sex",
                label_cols=labels,
            ),
        ],
        label_cols=labels,
        split_group_key="global_patient_id",
        index_dir=tmp_path / "indexes",
        cache_dir=tmp_path / "cache",
    ).resolve()

    index = DataEngine(cfg).build_index(force=True)

    assert set(labels).issubset(index.columns)
    assert "global_patient_id" in index.columns
    pad_row_out = index[index["source"] == "padchest"].iloc[0]
    nih_row_out = index[index["source"] == "nih_chestxray14"].iloc[0]
    assert pad_row_out["Pleural Effusion"] == -1.0
    assert pad_row_out["Pneumonia"] == -1.0
    assert nih_row_out["Pleural Effusion"] == -1.0


def test_data_label_cols_override_source_label_union(tmp_path: Path):
    labels = list(OPEN_CXR_CANONICAL_LABELS)
    root = tmp_path / "chexpert"
    image_root = root / "images"
    image_root.mkdir(parents=True)
    _write_image(image_root / "sample.png")

    meta_csv = root / "meta.csv"
    pd.DataFrame(
        [
            {
                "path": "sample.png",
                "patient_id": "p1",
                "study_id": "s1",
                "view": "PA",
                "age": 55,
                "sex": "F",
                "report_text": "",
                **{label: 0.0 for label in labels},
            }
        ]
    ).to_csv(meta_csv, index=False)

    cfg = DataConfig(
        sources=[
            DatasetSourceConfig(
                name="chexpert",
                root=root,
                metadata_csv=meta_csv,
                image_root=image_root,
                path_col="path",
                patient_id_col="patient_id",
                study_id_col="study_id",
                view_col="view",
                age_col="age",
                sex_col="sex",
                label_cols=labels + ["Support Devices"],
            )
        ],
        label_cols=labels,
        index_dir=tmp_path / "indexes",
        cache_dir=tmp_path / "cache",
    ).resolve()

    engine = DataEngine(cfg)
    engine.build_index(force=True)

    assert engine.label_cols == labels
    assert "Support Devices" not in engine.label_cols
