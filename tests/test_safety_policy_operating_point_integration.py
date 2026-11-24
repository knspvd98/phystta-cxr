import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from phystta.intervention.eraser import LinearSubspaceEraser, save_eraser


def _write_dataset(tmp_path: Path, n_patients: int = 28) -> tuple[Path, Path]:
    img_dir = tmp_path / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    rng = np.random.default_rng(17)
    for i in range(n_patients):
        pid = f"p{i:03d}"
        label = float(i % 2 == 0)
        for t in (0, 1):
            base = 0.60 if label == 1.0 else 0.40
            arr = rng.normal(loc=base + (0.01 * t), scale=0.03, size=(32, 32)).astype(np.float32)
            arr = np.clip(arr, 0.0, 1.0)
            img_path = img_dir / f"{pid}_t{t}.png"
            Image.fromarray((arr * 255.0).astype(np.uint8), mode="L").save(img_path)
            rows.append(
                {
                    "path": img_path.name,
                    "patient_id": pid,
                    "study_id": f"s{i:04d}_{t}",
                    "view": "PA" if t == 0 else "AP",
                    "manufacturer": "GE" if (i % 2 == 0) else "Philips",
                    "station_name": "A" if t == 0 else "B",
                    "study_date": "20240101",
                    "study_time": "080000" if t == 0 else "083000",
                    "No Finding": label,
                }
            )
    meta_csv = tmp_path / "meta.csv"
    pd.DataFrame(rows).to_csv(meta_csv, index=False)
    return img_dir, meta_csv


def _write_encoder_checkpoint(path: Path) -> None:
    model = torch.nn.Module()
    model.backbone = torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d((1, 1)), torch.nn.Flatten())
    torch.save(model, path)


def _write_config(
    cfg_path: Path,
    *,
    tmp_path: Path,
    img_dir: Path,
    meta_csv: Path,
    encoder_path: Path,
    head_path: Path,
) -> None:
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
    train_head: 0.6
    train_eraser: 0.2
    calib: 0.1
    eval: 0.1
  split_group_key: global_patient_id
  head_split: train_head
  eraser_split: train_eraser
  calib_split: calib
  eval_split: eval
model:
  architecture: tiny
  family: transformer
  num_classes: 1
  in_chans: 1
  use_medical_pretrain: false
  encoder:
    encoder_source: eva_x
    encoder_path: {encoder_path}
    head_path: {head_path}
    input_adapter:
      repeat_to_3: false
      image_size: 32
      mean: [0.0]
      std: [1.0]
train:
  batch_size: 8
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
  modes: [clinical, safety]
  coverage: 1.0
safety_policy:
  apply_on_covariate: false
  abstain_on_mixed: false
  abstain_on_unknown: false
  use_decision_policy: false
  cost_fn: 10.0
  cost_fp: 1.0
  coverage_budget: 1.0
  critical_labels: ["No Finding"]
  conformal: false
  shared_threshold: true
  stratify_technical: false
shift_id:
  score_family: [mmd, ks]
  batch_sizes: [20]
  n_splits: 1
  calibration: temperature
runtime:
  device: cpu
  seed: 11
  deterministic: true
  benchmark: false
  allow_download: false
  freeze_artifacts: false
output:
  root: {tmp_path / "outputs"}
  run_name: op_test
""".strip(),
        encoding="utf-8",
    )


def _run_eval(repo: Path, cfg: Path, eraser: Path, out_dir: Path, policy_path: Path, sens: float) -> None:
    cmd = [
        sys.executable,
        str(repo / "scripts" / "run_safety_policy_eval.py"),
        "--config",
        str(cfg),
        "--eraser",
        str(eraser),
        "--out_dir",
        str(out_dir),
        "--save_policy",
        str(policy_path),
        "--clinical_threshold_source",
        "target_sensitivity",
        "--target_sensitivities",
        str(sens),
        "--target_specificities",
        "0.90",
        "--bootstrap_samples",
        "20",
        "--device",
        "cpu",
        "--seed",
        "11",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    subprocess.run(cmd, check=True, cwd=str(repo), env=env)


def _finite_harm_mean(path: Path) -> float:
    df = pd.read_csv(path)
    vals = pd.to_numeric(df.get("expected_harm"), errors="coerce").dropna()
    if len(vals) == 0:
        raise AssertionError("No finite expected_harm values in safety.csv")
    return float(vals.mean())


def test_operating_point_changes_harm_and_serializes_thresholds(tmp_path):
    try:
        import iterstrat  # noqa: F401
    except Exception:
        pytest.skip("iterstrat is required for split stratification in safety policy integration test")

    repo = Path(__file__).resolve().parents[1]
    img_dir, meta_csv = _write_dataset(tmp_path)

    encoder_path = tmp_path / "encoder.pt"
    _write_encoder_checkpoint(encoder_path)

    head = torch.nn.Linear(1, 1)
    with torch.no_grad():
        head.weight[:] = torch.tensor([[12.0]])
        head.bias[:] = torch.tensor([-6.0])
    head_path = tmp_path / "head.pt"
    torch.save(head, head_path)

    eraser_path = tmp_path / "eraser.pt"
    eraser = LinearSubspaceEraser(torch.zeros((1, 1), dtype=torch.float32))
    save_eraser(str(eraser_path), eraser)

    cfg_path = tmp_path / "cfg.yaml"
    _write_config(
        cfg_path,
        tmp_path=tmp_path,
        img_dir=img_dir,
        meta_csv=meta_csv,
        encoder_path=encoder_path,
        head_path=head_path,
    )

    out_lo = tmp_path / "out_sens_060"
    out_hi = tmp_path / "out_sens_095"
    pol_lo = tmp_path / "policy_060.json"
    pol_hi = tmp_path / "policy_095.json"

    _run_eval(repo, cfg_path, eraser_path, out_lo, pol_lo, sens=0.60)
    _run_eval(repo, cfg_path, eraser_path, out_hi, pol_hi, sens=0.95)

    meta_lo = json.loads(pol_lo.read_text(encoding="utf-8"))
    meta_hi = json.loads(pol_hi.read_text(encoding="utf-8"))
    assert meta_lo["class_thresholds_meta"]["source"] == "target_sensitivity"
    assert meta_hi["class_thresholds_meta"]["source"] == "target_sensitivity"
    assert float(meta_lo["class_thresholds_meta"]["target_value"]) == pytest.approx(0.60)
    assert float(meta_hi["class_thresholds_meta"]["target_value"]) == pytest.approx(0.95)

    th_lo = float(meta_lo["class_thresholds"][0])
    th_hi = float(meta_hi["class_thresholds"][0])
    assert th_hi <= th_lo + 1e-12

    harm_lo = _finite_harm_mean(out_lo / "safety.csv")
    harm_hi = _finite_harm_mean(out_hi / "safety.csv")
    assert harm_hi <= harm_lo + 1e-6

    auto_thresholds = json.loads((out_hi / "policy_thresholds.json").read_text(encoding="utf-8"))
    assert auto_thresholds["class_thresholds_meta"]["source"] == "target_sensitivity"
