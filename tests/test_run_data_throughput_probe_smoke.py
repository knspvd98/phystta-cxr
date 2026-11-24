from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image


def _write_minicxr_with_labels(tmp_path: Path, n_patients: int = 24) -> tuple[Path, Path]:
    img_dir = tmp_path / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    rng = np.random.default_rng(17)
    for i in range(n_patients):
        pid = f"p{i:03d}"
        label = float((i % 4) == 0)
        for t in (0, 1):
            arr = rng.uniform(0.1, 0.9, size=(64, 64)).astype(np.float32)
            arr = np.clip(arr + (0.01 * t), 0.0, 1.0)
            img_path = img_dir / f"{pid}_t{t}.png"
            Image.fromarray((arr * 255.0).astype(np.uint8), mode="L").save(img_path)
            rows.append(
                {
                    "path": img_path.name,
                    "patient_id": pid,
                    "study_id": f"s{i:04d}_{t}",
                    "view": "PA" if t == 0 else "AP",
                    "sex": "M" if i % 2 == 0 else "F",
                    "age": 40 + (i % 20),
                    "race": "group_a" if (i % 3 == 0) else "group_b",
                    "manufacturer": "GE" if (i % 2 == 0) else "Philips",
                    "station_name": "StationA" if t == 0 else "StationB",
                    "study_date": "20240101",
                    "study_time": "080000" if t == 0 else "083000",
                    "No Finding": label,
                }
            )
    meta_csv = tmp_path / "meta.csv"
    pd.DataFrame(rows).to_csv(meta_csv, index=False)
    return img_dir, meta_csv


def _write_config(cfg_path: Path, *, tmp_path: Path, img_dir: Path, meta_csv: Path, head_path: Path) -> None:
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
      sex_col: sex
      age_col: age
      race_col: race
      manufacturer_col: manufacturer
      station_name_col: station_name
      study_date_col: study_date
      study_time_col: study_time
      label_cols: ["No Finding"]
  preprocess:
    image_size: 64
    normalize_mean: 0.0
    normalize_std: 1.0
  index_dir: {tmp_path / "indexes"}
  cache_dir: {tmp_path / "cache"}
  split_fractions:
    train_head: 0.6
    train_eraser: 0.2
    calib: 0.1
    eval: 0.1
  head_split: train_head
  eraser_split: train_eraser
  calib_split: calib
  eval_split: eval
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
    head_path: {head_path}
    input_adapter:
      repeat_to_3: true
      image_size: 64
      mean: [0.5, 0.5, 0.5]
      std: [0.5, 0.5, 0.5]
train:
  batch_size: 8
  num_workers: 0
shifts:
  seed: 2024
intervention:
  shift_types: [heel]
  use_realism: false
  group_key: global_patient_id
erasure_eval:
  split: eval
  shift_types: [heel]
  intensity_grid: [0.2]
safety_policy:
  use_decision_policy: false
shift_id:
  score_family: [mmd]
runtime:
  device: cpu
  seed: 9
  deterministic: true
  benchmark: false
  allow_download: false
output:
  root: {tmp_path / "outputs"}
  run_name: throughput_probe
""".strip(),
        encoding="utf-8",
    )


def test_run_data_throughput_probe_smoke(tmp_path: Path):
    repo = Path(__file__).resolve().parents[1]
    img_dir, meta_csv = _write_minicxr_with_labels(tmp_path)
    head_path = tmp_path / "head.pt"
    torch.save(torch.nn.Linear(1024, 1), head_path)
    cfg_path = tmp_path / "throughput_probe_config.yaml"
    _write_config(cfg_path, tmp_path=tmp_path, img_dir=img_dir, meta_csv=meta_csv, head_path=head_path)

    out_dir = tmp_path / "out"
    cmd = [
        sys.executable,
        str(repo / "scripts" / "run_data_throughput_probe.py"),
        "--config",
        str(cfg_path),
        "--split",
        "eval",
        "--batch_sizes",
        "2,4",
        "--num_workers_grid",
        "0,1",
        "--max_batches",
        "2",
        "--device",
        "cpu",
        "--out_dir",
        str(out_dir),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    subprocess.run(cmd, check=True, cwd=str(repo), env=env)

    rows_path = out_dir / "throughput_probe_rows.csv"
    summary_path = out_dir / "throughput_probe_summary.json"
    md_path = out_dir / "throughput_probe.md"
    assert rows_path.exists()
    assert summary_path.exists()
    assert md_path.exists()

    rows = pd.read_csv(rows_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["status"] == "ok"
    assert set(rows["mode"].astype(str)) == {"loader_only", "encoder_forward"}
    assert summary["recommended_batch_size"] in {2, 4}
    assert summary["recommended_num_workers"] in {0, 1}
