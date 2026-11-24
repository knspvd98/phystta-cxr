from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from phystta.config.schema import DataConfig, DatasetSourceConfig
from phystta.data import DataEngine


def test_min_patient_images_uses_split_group_key_for_prebuilt_index(tmp_path: Path):
    csv_path = tmp_path / "index.csv"
    df = pd.DataFrame(
        [
            {"source": "A", "patient_id": "p1", "global_patient_id": "A:p1", "image_path": "/tmp/a1.png"},
            {"source": "B", "patient_id": "p1", "global_patient_id": "B:p1", "image_path": "/tmp/b1.png"},
            {"source": "A", "patient_id": "p2", "global_patient_id": "A:p2", "image_path": "/tmp/a2.png"},
            {"source": "B", "patient_id": "p2", "global_patient_id": "B:p2", "image_path": "/tmp/b2.png"},
        ]
    )
    df.to_csv(csv_path, index=False)
    digest = hashlib.sha256(csv_path.read_bytes()).hexdigest()

    cfg = DataConfig(
        sources=[
            DatasetSourceConfig(name="A", root=tmp_path),
            DatasetSourceConfig(name="B", root=tmp_path),
        ],
        require_labels=False,
        split_group_key="global_patient_id",
        prebuilt_index_csv=csv_path,
        prebuilt_index_sha256=digest,
        min_patient_images=2,
        max_patient_images=None,
    ).resolve()

    engine = DataEngine(cfg)
    index = engine.build_index()
    # Each global patient appears once; with min_patient_images=2 all rows must be filtered out.
    assert len(index) == 0


def test_prebuilt_index_derives_global_image_id_when_missing(tmp_path: Path):
    csv_path = tmp_path / "index.csv"
    df = pd.DataFrame(
        [
            {"source": "A", "patient_id": "p1", "global_patient_id": "A:p1", "study_id": "s1", "image_path": "/tmp/a1.png"},
            {"source": "A", "patient_id": "p2", "global_patient_id": "A:p2", "study_id": "s2", "image_path": "/tmp/a2.png"},
        ]
    )
    df.to_csv(csv_path, index=False)
    digest = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    cfg = DataConfig(
        sources=[DatasetSourceConfig(name="A", root=tmp_path)],
        require_labels=False,
        split_group_key="global_patient_id",
        prebuilt_index_csv=csv_path,
        prebuilt_index_sha256=digest,
    ).resolve()
    index = DataEngine(cfg).build_index()
    assert "global_image_id" in index.columns
    assert index["global_image_id"].astype(str).str.contains(r"^A\|p[12]\|s[12]\|a[12]\.png$").all()
