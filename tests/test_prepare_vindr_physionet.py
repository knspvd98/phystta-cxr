from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pydicom
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import ImplicitVRLittleEndian


def _write_minimal_dicom(path: Path, *, patient_id: str, age: str, sex: str, view_position: str) -> None:
    file_meta = Dataset()
    file_meta.TransferSyntaxUID = ImplicitVRLittleEndian
    ds = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.is_little_endian = True
    ds.is_implicit_VR = True
    ds.PatientID = patient_id
    ds.PatientAge = age
    ds.PatientSex = sex
    ds.ViewPosition = view_position
    ds.save_as(str(path))


def test_prepare_vindr_physionet_emits_root_level_train_csv(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    root = tmp_path / "vindr_physionet"
    train_dir = root / "Train"
    annotations_dir = root / "Annotations"
    train_dir.mkdir(parents=True)
    annotations_dir.mkdir(parents=True)

    _write_minimal_dicom(
        train_dir / "img_001.dicom",
        patient_id="patient-1",
        age="063Y",
        sex="F",
        view_position="PA",
    )
    _write_minimal_dicom(
        train_dir / "img_002.dicom",
        patient_id="patient-2",
        age="057Y",
        sex="M",
        view_position="AP",
    )

    pd.DataFrame(
        [
            {"image_id": "img_001", "rad_id": "r1", "labels": "Atelectasis|Pleural effusion"},
            {"image_id": "img_001", "rad_id": "r2", "labels": "Cardiomegaly"},
            {"image_id": "img_002", "rad_id": "r1", "labels": "No finding"},
        ]
    ).to_csv(annotations_dir / "image_labels_train.csv", index=False)

    out_csv = root / "train.csv"
    manifest_json = root / "train.label_harmonization_manifest.json"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    subprocess.run(
        [
            sys.executable,
            str(repo / "scripts" / "prepare_vindr_physionet.py"),
            "--root",
            str(root),
            "--out",
            str(out_csv),
            "--manifest_out",
            str(manifest_json),
        ],
        check=True,
        cwd=str(repo),
        env=env,
    )

    df = pd.read_csv(out_csv).sort_values("image_id").reset_index(drop=True)
    assert list(df["image_id"]) == ["Train/img_001.dicom", "Train/img_002.dicom"]
    assert list(df["patient_id"]) == ["patient-1", "patient-2"]
    assert list(df["age"].astype(str)) == ["63", "57"]
    assert list(df["sex"]) == ["F", "M"]
    assert list(df["view_position"]) == ["PA", "AP"]

    row1 = df.iloc[0]
    assert row1["Atelectasis"] == 1.0
    assert row1["Cardiomegaly"] == 1.0
    assert row1["Pleural effusion"] == 1.0
    assert row1["Pneumothorax"] == 0.0

    row2 = df.iloc[1]
    assert row2["Atelectasis"] == 0.0
    assert row2["Cardiomegaly"] == 0.0
    assert row2["Pleural effusion"] == 0.0
    manifest = json.loads(manifest_json.read_text(encoding="utf-8"))
    assert manifest["dataset"] == "vindr"
    assert manifest["projection"]["source"] == "vindr"
    assert "Lung Lesion" in manifest["projection"]["approximate_labels"]
    assert "No Finding" in manifest["projection"]["unsupported_labels"]
