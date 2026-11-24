import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import pytest
from PIL import Image

from phystta.intervention.eraser import LinearSubspaceEraser, save_eraser


def _write_minicxr_with_labels(tmp_path: Path, n_patients: int = 30) -> tuple[Path, Path]:
    img_dir = tmp_path / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    rng = np.random.default_rng(23)
    for i in range(n_patients):
        pid = f"p{i:03d}"
        label = float((i % 3) == 0)
        for t in (0, 1):
            arr = rng.uniform(0.1, 0.9, size=(64, 64)).astype(np.float32)
            arr = np.clip(arr + (0.02 * t), 0.0, 1.0)
            img_path = img_dir / f"{pid}_t{t}.png"
            Image.fromarray((arr * 255.0).astype(np.uint8), mode="L").save(img_path)
            rows.append(
                {
                    "path": img_path.name,
                    "patient_id": pid,
                    "study_id": f"s{i:04d}_{t}",
                    "view": "PA" if t == 0 else "AP",
                    "sex": "M" if (i % 2 == 0) else "F",
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


def test_safety_policy_artifacts_smoke(tmp_path):
    try:
        import iterstrat  # noqa: F401
    except Exception:
        pytest.skip("iterstrat is required for split stratification in safety policy smoke test")
    repo = Path(__file__).resolve().parents[1]
    img_dir, meta_csv = _write_minicxr_with_labels(tmp_path, n_patients=30)
    out_dir = tmp_path / "safety_out"
    cfg_path = tmp_path / "cfg.yaml"
    head_path = tmp_path / "head.pt"
    eraser_path = tmp_path / "eraser.pt"
    eraser_manifest = tmp_path / "eraser_bank_manifest.json"
    shift_id_path = tmp_path / "shift_id.pkl"
    policy_path = tmp_path / "policy.json"

    # DenseNet121 feature dim=1024 in this codebase.
    head = torch.nn.Linear(1024, 1)
    torch.save(head, head_path)
    eraser = LinearSubspaceEraser(torch.zeros((1024, 1), dtype=torch.float32))
    save_eraser(str(eraser_path), eraser)
    eraser_manifest.write_text(
        json.dumps(
            {
                "fit_mode": "one_eraser_per_shift_family",
                "erasers": [
                    {
                        "shift_family": "heel",
                        "eraser_path": str(eraser_path),
                    }
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

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
  intensity_grid: [0.2]
  use_realism: false
  group_key: global_patient_id
erasure_eval:
  split: eval
  shift_types: [heel]
  intensity_grid: [0.2]
  modes: [clinical, safety, fairness, causal_invariance]
  coverage: 0.8
  fairness_groups: [sex, race, age]
safety_policy:
  apply_on_covariate: true
  abstain_on_mixed: true
  abstain_on_unknown: true
  use_decision_policy: true
  uncertainty_abstain_threshold: 0.35
  mixed_abstain_threshold: 0.40
  prevalence_action: recalibrate
  cost_fn: 10.0
  cost_fp: 1.0
  coverage_budget: 0.8
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
  seed: 9
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
        str(repo / "scripts" / "run_safety_policy_eval.py"),
        "--config",
        str(cfg_path),
        "--eraser",
        str(eraser_manifest),
        "--out_dir",
        str(out_dir),
        "--save_shift_id",
        str(shift_id_path),
        "--save_policy",
        str(policy_path),
        "--bootstrap_samples",
        "50",
        "--device",
        "cpu",
        "--seed",
        "9",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    subprocess.run(cmd, check=True, cwd=str(repo), env=env)

    required = [
        "safety.csv",
        "policy_decisions.csv",
        "scenario_trace.csv",
        "policy_bootstrap.csv",
        "policy_operating_points.csv",
        "policy_operating_points_summary.json",
        "policy_operating_points_targeted.csv",
        "policy_operating_points_targeted_summary.json",
        "policy_regret_summary.json",
        "policy_regret_decomposition.json",
        "shift_id_report.json",
    ]
    for name in required:
        assert (out_dir / name).exists(), f"missing artifact: {name}"
    assert shift_id_path.exists()
    assert policy_path.exists()

    boot = pd.read_csv(out_dir / "policy_bootstrap.csv")
    assert "policy_regret_ci_low" in boot.columns
    assert "policy_regret_ci_high" in boot.columns
    trace = pd.read_csv(out_dir / "scenario_trace.csv")
    assert {"reason", "p_none", "p_covariate", "p_prevalence", "p_mixed", "uncertainty"}.issubset(trace.columns)
    decomp = json.loads((out_dir / "policy_regret_decomposition.json").read_text(encoding="utf-8"))
    assert "bootstrap_ci" in decomp
