import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


def _write_minicxr(tmp_path: Path, n_patients: int = 20) -> tuple[Path, Path]:
    img_dir = tmp_path / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    rng = np.random.default_rng(17)
    for i in range(n_patients):
        pid = f"p{i:03d}"
        study_base = 100000 + i
        for t in (0, 1):
            arr = rng.uniform(0.1, 0.9, size=(48, 48)).astype(np.float32)
            arr = np.clip(arr + (0.03 * t), 0.0, 1.0)
            img_path = img_dir / f"{pid}_t{t}.png"
            Image.fromarray((arr * 255.0).astype(np.uint8), mode="L").save(img_path)
            rows.append(
                {
                    "path": img_path.name,
                    "patient_id": pid,
                    "study_id": f"s{study_base+t}",
                    "view": "PA",
                    "manufacturer": "GE" if t == 0 else "Philips",
                    "station_name": "StationA" if t == 0 else "StationB",
                    "study_date": "20240101",
                    "study_time": "080000" if t == 0 else "083000",
                }
            )
    meta_csv = tmp_path / "meta.csv"
    pd.DataFrame(rows).to_csv(meta_csv, index=False)
    return img_dir, meta_csv


def test_run_real_anchor_eval_smoke(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    img_dir, meta_csv = _write_minicxr(tmp_path, n_patients=20)
    out_dir = tmp_path / "anchor_out"
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
      label_cols: []
  preprocess:
    image_size: 48
    normalize_mean: 0.0
    normalize_std: 1.0
  index_dir: {tmp_path / "indexes"}
  cache_dir: {tmp_path / "cache"}
  split_fractions:
    train_head: 0.1
    train_eraser: 0.7
    calib: 0.1
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
  intensity_grid: [0.2]
  use_realism: false
  group_key: global_patient_id
erasure_eval:
  split: eval
  shift_types: [heel]
  intensity_grid: [0.2]
  modes: [clinical]
runtime:
  device: cpu
  seed: 7
  deterministic: true
  benchmark: false
  allow_download: false
  freeze_artifacts: false
output:
  root: {tmp_path / "outputs"}
  run_name: smoke
""".strip(),
        encoding="utf-8",
    )
    cmd = [
        sys.executable,
        str(repo / "scripts" / "run_real_anchor_eval.py"),
        "--config",
        str(cfg_path),
        "--shift_type",
        "heel",
        "--out_dir",
        str(out_dir),
        "--k",
        "2",
        "--max_pairs",
        "20",
        "--max_images",
        "20",
        "--bootstrap_samples",
        "20",
        "--device",
        "cpu",
        "--seed",
        "7",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    subprocess.run(cmd, check=True, cwd=str(repo), env=env)

    principal = out_dir / "principal_angles.json"
    falsification = out_dir / "falsification_report.json"
    summary = out_dir / "pair_quality_summary.csv"
    assert principal.exists()
    assert falsification.exists()
    assert summary.exists()

    data = json.loads(falsification.read_text(encoding="utf-8"))
    assert "bootstrap_ci" in data
    assert data["bootstrap_ci"]["n_bootstrap"] == 20
    assert data["n_pairs_train"] > 0
    assert data["group_key"] == "global_patient_id"
    assert data["group_overlap_train_holdout"] == 0
