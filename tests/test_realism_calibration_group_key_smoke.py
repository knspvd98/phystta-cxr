import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


def _write_pairs_dataset(tmp_path: Path, n_patients: int = 10) -> tuple[Path, Path]:
    img_dir = tmp_path / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(77)
    rows = []
    for i in range(n_patients):
        pid = f"p{i:03d}"
        for t in (0, 1):
            arr = rng.uniform(0.1, 0.9, size=(48, 48)).astype(np.float32)
            arr = np.clip(arr + 0.02 * t, 0.0, 1.0)
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
                }
            )
    meta = tmp_path / "meta.csv"
    pd.DataFrame(rows).to_csv(meta, index=False)
    return img_dir, meta


def test_calibrate_realism_respects_global_group_key(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    img_dir, meta_csv = _write_pairs_dataset(tmp_path, n_patients=10)
    cfg_path = tmp_path / "cfg.yaml"
    out_json = tmp_path / "realism_calibrator.json"

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
      label_cols: []
  preprocess:
    image_size: 48
    normalize_mean: 0.0
    normalize_std: 1.0
  index_dir: {tmp_path / "indexes"}
  cache_dir: {tmp_path / "cache"}
  split_fractions:
    train_head: 0.4
    train_eraser: 0.3
    calib: 0.2
    eval: 0.1
  head_split: train_head
  eraser_split: train_eraser
  calib_split: calib
  eval_split: eval
  require_labels: false
model:
  architecture: densenet121
  family: cnn
  num_classes: 1
  in_chans: 1
  use_medical_pretrain: false
  encoder:
    encoder_source: baseline_cnn
    architecture: densenet121
    pretrained: false
    input_adapter:
      repeat_to_3: true
      image_size: 48
      mean: [0.5, 0.5, 0.5]
      std: [0.5, 0.5, 0.5]
train:
  batch_size: 4
  num_workers: 0
shifts:
  seed: 2024
intervention:
  shift_types: [heel]
  intensity_grid: [0.0, 0.2, 0.4]
  group_key: global_patient_id
  pair_mode: strict_device_change
runtime:
  device: cpu
  seed: 13
  deterministic: true
  benchmark: false
  allow_download: false
""".strip(),
        encoding="utf-8",
    )

    cmd = [
        sys.executable,
        str(repo / "scripts" / "calibrate_physics_realism.py"),
        "--config",
        str(cfg_path),
        "--out",
        str(out_json),
        "--shift_types",
        "heel",
        "--max_pairs",
        "30",
        "--max_images",
        "20",
        "--seed",
        "13",
        "--device",
        "cpu",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    proc = subprocess.run(cmd, check=False, cwd=str(repo), env=env, capture_output=True, text=True)

    assert out_json.exists()
    report = out_json.with_suffix(".report.json")
    assert report.exists()
    report_data = json.loads(report.read_text())
    assert report_data["group_key"] == "global_patient_id"
    assert report_data["n_pairs_after_filter"] > 0
    assert report_data["n_unique_groups_after_filter"] > 0
    if proc.returncode != 0:
        assert "non_graded" in proc.stderr or "Widen the physical parameter range" in proc.stderr
