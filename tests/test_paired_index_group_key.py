from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from phystta.config.loader import load_config
from phystta.data import DataEngine


def _write_pairs_fixture(tmp_path: Path) -> tuple[Path, Path]:
    img_dir = tmp_path / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(123)
    rows = []
    for i in range(4):
        pid = f"p{i:03d}"
        for t in (0, 1):
            arr = rng.uniform(0.1, 0.9, size=(32, 32)).astype(np.float32)
            p = img_dir / f"{pid}_t{t}.png"
            Image.fromarray((arr * 255.0).astype(np.uint8), mode="L").save(p)
            rows.append(
                {
                    "path": p.name,
                    "patient_id": pid,
                    "study_id": f"s{i:03d}_{t}",
                    "view": "PA",
                    "manufacturer": "GE" if t == 0 else "Philips",
                    "station_name": "StationA" if t == 0 else "StationB",
                    "study_date": "20240101",
                    "study_time": "080000" if t == 0 else "083000",
                    "No Finding": 0.0,
                }
            )
    meta = tmp_path / "meta.csv"
    pd.DataFrame(rows).to_csv(meta, index=False)
    return img_dir, meta


def _write_cfg(tmp_path: Path, img_dir: Path, meta_csv: Path) -> Path:
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        f"""
data:
  sources:
    - name: mini
      root: {tmp_path}
      metadata_csv: {meta_csv}
      image_root: {img_dir}
      path_col: path
      patient_id_col: patient_id
      study_id_col: study_id
      view_col: view
      manufacturer_col: manufacturer
      station_name_col: station_name
      study_date_col: study_date
      study_time_col: study_time
      label_cols: ["No Finding"]
  preprocess:
    image_size: 32
    normalize_mean: 0.0
    normalize_std: 1.0
  index_dir: {tmp_path / "indexes"}
  cache_dir: {tmp_path / "cache"}
  split_fractions:
    train_head: 0.5
    train_eraser: 0.2
    calib: 0.1
    eval: 0.2
  head_split: train_head
  eraser_split: train_eraser
  calib_split: calib
  eval_split: eval
runtime:
  device: cpu
  seed: 7
""".strip(),
        encoding="utf-8",
    )
    return cfg_path


def test_build_paired_index_includes_global_patient_id(tmp_path):
    img_dir, meta = _write_pairs_fixture(tmp_path)
    cfg_path = _write_cfg(tmp_path, img_dir, meta)
    cfg = load_config(str(cfg_path))

    engine = DataEngine(cfg.data)
    engine.build_index()
    pairs = engine.build_paired_index("mini", group_key="global_patient_id")

    assert len(pairs) > 0
    assert "global_patient_id" in pairs.columns
    expected = set("mini:" + x for x in pairs["patient_id"].astype(str))
    actual = set(pairs["global_patient_id"].astype(str))
    assert actual == expected


def test_build_paired_index_rejects_unknown_group_key(tmp_path):
    img_dir, meta = _write_pairs_fixture(tmp_path)
    cfg_path = _write_cfg(tmp_path, img_dir, meta)
    cfg = load_config(str(cfg_path))

    engine = DataEngine(cfg.data)
    engine.build_index()
    with pytest.raises(RuntimeError, match="requested group_key"):
        engine.build_paired_index("mini", group_key="unknown_key")
