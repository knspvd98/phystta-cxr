from __future__ import annotations

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


def _write_minicxr_with_labels(tmp_path: Path, n_patients: int = 24) -> tuple[Path, Path]:
    img_dir = tmp_path / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    rng = np.random.default_rng(77)
    for i in range(n_patients):
        pid = f"p{i:03d}"
        label = float((i % 4) == 0)
        for t in (0, 1):
            arr = rng.uniform(0.1, 0.9, size=(64, 64)).astype(np.float32)
            arr = np.clip(arr + (0.015 * t), 0.0, 1.0)
            img_path = img_dir / f"{pid}_t{t}.png"
            Image.fromarray((arr * 255.0).astype(np.uint8), mode="L").save(img_path)
            rows.append(
                {
                    "path": img_path.name,
                    "patient_id": pid,
                    "study_id": f"s{i:04d}_{t}",
                    "view": "PA" if t == 0 else "AP",
                    "sex": "M" if (i % 2 == 0) else "F",
                    "age": 45 + (i % 15),
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


def _write_config(
    cfg_path: Path,
    *,
    tmp_path: Path,
    img_dir: Path,
    meta_csv: Path,
    head_path: Path,
    decision_label_source: str,
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
  modes: [clinical, safety]
  coverage: 0.8
  fairness_groups: [sex, race, age]
safety_policy:
  apply_on_covariate: true
  abstain_on_mixed: true
  abstain_on_unknown: true
  use_decision_policy: true
  policy_variant: expected_risk_v2
  decision_label_source: {decision_label_source}
  uncertainty_abstain_threshold: 0.35
  mixed_abstain_threshold: 0.40
  prevalence_action: recalibrate
  recalibration_mode: stratified
  stratified_recalibration_min_samples: 1
  abstain_cost: 0.25
  uncertainty_lambda: 0.1
  route_gamma: 0.1
  cost_fn: 10.0
  cost_fp: 1.0
  class_cost_fn: [12.0]
  class_cost_fp: [0.5]
  coverage_budget: 0.8
  critical_labels: ["No Finding"]
  conformal: false
  shared_threshold: true
  stratify_technical: true
  strata_keys: [view_family, procedure_context]
shift_id:
  score_family: [mmd, ks]
  batch_sizes: [20]
  n_splits: 1
  calibration: temperature
  label_space: physics
  report_calibration_metrics: true
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


def _run_eval(repo: Path, cfg_path: Path, eraser_manifest: Path, out_dir: Path, policy_path: Path) -> subprocess.CompletedProcess[str]:
    cmd = [
        sys.executable,
        str(repo / "scripts" / "run_safety_policy_eval.py"),
        "--config",
        str(cfg_path),
        "--eraser",
        str(eraser_manifest),
        "--out_dir",
        str(out_dir),
        "--save_policy",
        str(policy_path),
        "--bootstrap_samples",
        "20",
        "--device",
        "cpu",
        "--seed",
        "9",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return subprocess.run(cmd, cwd=str(repo), env=env, capture_output=True, text=True)


def test_run_safety_policy_eval_decision_label_source_contract(tmp_path: Path):
    try:
        import iterstrat  # noqa: F401
    except Exception:
        pytest.skip("iterstrat is required for split stratification in safety policy label-source contract test")

    repo = Path(__file__).resolve().parents[1]
    img_dir, meta_csv = _write_minicxr_with_labels(tmp_path)
    head = torch.nn.Linear(1024, 1)
    head_path = tmp_path / "head.pt"
    torch.save(head, head_path)

    eraser_path = tmp_path / "eraser.pt"
    eraser = LinearSubspaceEraser(torch.zeros((1024, 1), dtype=torch.float32))
    save_eraser(str(eraser_path), eraser)
    eraser_manifest = tmp_path / "eraser_bank_manifest.json"
    eraser_manifest.write_text(
        json.dumps(
            {
                "fit_mode": "one_eraser_per_shift_family",
                "erasers": [{"shift_family": "heel", "eraser_path": str(eraser_path)}],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    cfg_pred = tmp_path / "cfg_predicted.yaml"
    out_pred = tmp_path / "out_predicted"
    policy_pred = tmp_path / "policy_predicted.json"
    _write_config(
        cfg_pred,
        tmp_path=tmp_path,
        img_dir=img_dir,
        meta_csv=meta_csv,
        head_path=head_path,
        decision_label_source="predicted",
    )
    proc_pred = _run_eval(repo, cfg_pred, eraser_manifest, out_pred, policy_pred)
    assert proc_pred.returncode == 0, proc_pred.stderr
    policy_payload_pred = json.loads(policy_pred.read_text(encoding="utf-8"))
    assert policy_payload_pred["decision_label_source"] == "predicted"
    assert policy_payload_pred["decision_label_diagnostic_only"] is False
    assert policy_payload_pred["class_cost_fn"] == [12.0]
    assert policy_payload_pred["class_cost_fp"] == [0.5]
    assert policy_payload_pred["recalibration_mode"] == "stratified"
    assert policy_payload_pred["target_priors_by_stratum_present"] is True
    assert policy_payload_pred["shift_id_calibration_report_present"] is True
    trace_pred = pd.read_csv(out_pred / "scenario_trace.csv")
    assert "decision_label_source" in trace_pred.columns
    assert set(trace_pred["decision_label_source"].astype(str)) == {"predicted"}

    cfg_oracle = tmp_path / "cfg_oracle.yaml"
    out_oracle = tmp_path / "out_oracle"
    policy_oracle = tmp_path / "policy_oracle.json"
    _write_config(
        cfg_oracle,
        tmp_path=tmp_path,
        img_dir=img_dir,
        meta_csv=meta_csv,
        head_path=head_path,
        decision_label_source="oracle_shift_family",
    )
    proc_oracle = _run_eval(repo, cfg_oracle, eraser_manifest, out_oracle, policy_oracle)
    assert proc_oracle.returncode == 0, proc_oracle.stderr
    policy_payload_oracle = json.loads(policy_oracle.read_text(encoding="utf-8"))
    assert policy_payload_oracle["decision_label_source"] == "oracle_shift_family"
    assert policy_payload_oracle["decision_label_diagnostic_only"] is True
    assert policy_payload_oracle["class_cost_fn"] == [12.0]
    assert policy_payload_oracle["class_cost_fp"] == [0.5]
    assert policy_payload_oracle["recalibration_mode"] == "stratified"
    trace_oracle = pd.read_csv(out_oracle / "scenario_trace.csv")
    assert {"decision_label", "predicted_shift_label", "oracle_shift_family", "decision_label_source"}.issubset(trace_oracle.columns)
    assert set(trace_oracle["decision_label_source"].astype(str)) == {"oracle_shift_family"}

    cfg_bad = tmp_path / "cfg_bad.yaml"
    out_bad = tmp_path / "out_bad"
    policy_bad = tmp_path / "policy_bad.json"
    _write_config(
        cfg_bad,
        tmp_path=tmp_path,
        img_dir=img_dir,
        meta_csv=meta_csv,
        head_path=head_path,
        decision_label_source="bad_value",
    )
    proc_bad = _run_eval(repo, cfg_bad, eraser_manifest, out_bad, policy_bad)
    assert proc_bad.returncode != 0
