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

from phystta.config.loader import load_config
from phystta.intervention.eraser import LinearSubspaceEraser, save_eraser
from phystta.models import build_head_meta
from phystta.models.encoder_factory import EncoderFactory


def _write_minicxr(tmp_path: Path, n_patients: int = 16) -> tuple[Path, Path]:
    img_dir = tmp_path / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    rng = np.random.default_rng(42)
    for i in range(n_patients):
        patient_id = f"p{i:03d}"
        arr = rng.uniform(0.05, 0.95, size=(64, 64)).astype(np.float32)
        img_path = img_dir / f"{patient_id}.png"
        Image.fromarray((arr * 255.0).astype(np.uint8), mode="L").save(img_path)
        rows.append(
            {
                "path": img_path.name,
                "patient_id": patient_id,
                "study_id": f"s{i:04d}",
                "view": "PA",
                "No Finding": float(i % 2 == 0),
            }
        )
    meta_csv = tmp_path / "meta.csv"
    pd.DataFrame(rows).to_csv(meta_csv, index=False)
    return img_dir, meta_csv


def _write_config(
    path: Path,
    *,
    root: Path,
    meta_csv: Path,
    img_dir: Path,
    head_path: Path,
    head_meta_path: Path,
    output_root: Path,
    task_subspace_k: int | None,
) -> None:
    task_subspace_line = f"  task_subspace_k: {task_subspace_k}\n" if task_subspace_k is not None else ""
    path.write_text(
        f"""
data:
  sources:
    - name: mini
      root: {root}
      metadata_csv: {meta_csv}
      image_root: {img_dir}
      path_col: path
      patient_id_col: patient_id
      study_id_col: study_id
      view_col: view
      label_cols: ["No Finding"]
  preprocess:
    image_size: 64
    normalize_mean: 0.0
    normalize_std: 1.0
  index_dir: {root / "indexes"}
  cache_dir: {root / "cache"}
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
    head_meta_path: {head_meta_path}
    input_adapter:
      repeat_to_3: true
      image_size: 64
      mean: [0.5, 0.5, 0.5]
      std: [0.5, 0.5, 0.5]
train:
  batch_size: 4
  num_workers: 0
shifts:
  seed: 2024
intervention:
  shift_types: [heel, scatter]
  intensity_grid: [0.2]
  use_realism: false
  group_key: global_patient_id
eraser:
  fit_mode: one_eraser_per_shift_family
{task_subspace_line}erasure_eval:
  split: eval
  shift_types: [heel, scatter]
  intensity_grid: [0.2]
  modes: [clinical]
runtime:
  device: cpu
  seed: 42
  deterministic: true
  benchmark: false
  allow_download: false
output:
  root: {output_root}
  run_name: {path.stem}
""".strip(),
        encoding="utf-8",
    )


def _write_manifest(manifest_path: Path, heel_path: Path, scatter_path: Path) -> None:
    manifest_path.write_text(
        json.dumps(
            {
                "fit_mode": "one_eraser_per_shift_family",
                "erasers": [
                    {"shift_family": "heel", "eraser_path": str(heel_path)},
                    {"shift_family": "scatter", "eraser_path": str(scatter_path)},
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def test_k_sweep_diag_smoke(tmp_path):
    try:
        import iterstrat  # noqa: F401
    except Exception:
        pytest.skip("iterstrat is required for split stratification in k_sweep_diag smoke test")

    repo = Path(__file__).resolve().parents[1]
    img_dir, meta_csv = _write_minicxr(tmp_path)

    head_path = tmp_path / "head.pt"
    head_meta_path = tmp_path / "head_meta.json"
    legacy_cfg_path = tmp_path / "legacy.yaml"
    k2_cfg_path = tmp_path / "k2.yaml"
    k4_cfg_path = tmp_path / "k4.yaml"
    k8_cfg_path = tmp_path / "k8.yaml"
    out_dir = tmp_path / "out"

    _write_config(
        legacy_cfg_path,
        root=tmp_path,
        meta_csv=meta_csv,
        img_dir=img_dir,
        head_path=head_path,
        head_meta_path=head_meta_path,
        output_root=tmp_path / "outputs_legacy",
        task_subspace_k=None,
    )
    _write_config(
        k2_cfg_path,
        root=tmp_path,
        meta_csv=meta_csv,
        img_dir=img_dir,
        head_path=head_path,
        head_meta_path=head_meta_path,
        output_root=tmp_path / "outputs_k2",
        task_subspace_k=2,
    )
    _write_config(
        k4_cfg_path,
        root=tmp_path,
        meta_csv=meta_csv,
        img_dir=img_dir,
        head_path=head_path,
        head_meta_path=head_meta_path,
        output_root=tmp_path / "outputs_k4",
        task_subspace_k=4,
    )
    _write_config(
        k8_cfg_path,
        root=tmp_path,
        meta_csv=meta_csv,
        img_dir=img_dir,
        head_path=head_path,
        head_meta_path=head_meta_path,
        output_root=tmp_path / "outputs_k8",
        task_subspace_k=8,
    )

    head = torch.nn.Linear(1024, 1)
    torch.save(head, head_path)

    legacy_cfg = load_config(str(legacy_cfg_path))
    resolved_enc = EncoderFactory.resolve_encoder_config(legacy_cfg)
    head_meta = build_head_meta(
        enc_cfg=resolved_enc,
        feat_dim=1024,
        num_classes=1,
        label_cols=["No Finding"],
        group_key="global_patient_id",
        encoder_profile=None,
        seed=42,
    )
    head_meta_path.write_text(json.dumps(head_meta, indent=2), encoding="utf-8")

    manifest_paths = {}
    for variant, rank in [("legacy", 1), ("k2", 2), ("k4", 4), ("k8", 8)]:
        heel_path = tmp_path / f"{variant}_heel.pt"
        scatter_path = tmp_path / f"{variant}_scatter.pt"
        manifest_path = tmp_path / f"{variant}_manifest.json"
        if variant == "legacy":
            heel_eraser = LinearSubspaceEraser(torch.zeros((1024, 1), dtype=torch.float32))
            scatter_eraser = LinearSubspaceEraser(torch.zeros((1024, 1), dtype=torch.float32))
        else:
            heel_eraser = LinearSubspaceEraser(torch.eye(1024, rank, dtype=torch.float32))
            scatter_eraser = LinearSubspaceEraser(torch.eye(1024, rank, dtype=torch.float32))
        save_eraser(str(heel_path), heel_eraser)
        save_eraser(str(scatter_path), scatter_eraser)
        _write_manifest(manifest_path, heel_path, scatter_path)
        manifest_paths[variant] = manifest_path

    cmd = [
        sys.executable,
        str(repo / "scripts" / "k_sweep_diag.py"),
        "--legacy_config",
        str(legacy_cfg_path),
        "--legacy_eraser",
        str(manifest_paths["legacy"]),
        "--k2_config",
        str(k2_cfg_path),
        "--k2_eraser",
        str(manifest_paths["k2"]),
        "--k4_config",
        str(k4_cfg_path),
        "--k4_eraser",
        str(manifest_paths["k4"]),
        "--k8_config",
        str(k8_cfg_path),
        "--k8_eraser",
        str(manifest_paths["k8"]),
        "--out_dir",
        str(out_dir),
        "--families",
        "heel,scatter",
        "--levels",
        "0.2",
        "--max_batches",
        "1",
        "--device",
        "cpu",
        "--seed",
        "42",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    subprocess.run(cmd, check=True, cwd=str(repo), env=env)

    required = [
        out_dir / "scenario_rows.csv",
        out_dir / "k_sweep_summary.csv",
        out_dir / "k_sweep_summary.json",
        out_dir / "metadata.json",
        out_dir / "winner.txt",
    ]
    for path in required:
        assert path.exists()

    summary_df = pd.read_csv(out_dir / "k_sweep_summary.csv")
    assert set(summary_df["variant"]) == {"legacy", "k2", "k4", "k8"}
    for col in [
        "mean_prob_recovery_fraction",
        "mean_prob_repair_gain",
        "mean_logit_repair_gain",
        "clean_disturbance_prob_mae",
        "clean_disturbance_logit_mae",
        "winner_score",
    ]:
        assert col in summary_df.columns
    assert summary_df["clean_disturbance_prob_mae"].notna().all()
    assert summary_df["clean_disturbance_logit_mae"].notna().all()

    scenario_df = pd.read_csv(out_dir / "scenario_rows.csv")
    assert "none" in set(scenario_df["shift_family"].astype(str))
    assert {"heel", "scatter"} & set(scenario_df["shift_family"].astype(str))
    for col in [
        "prob_clean_repair_gain_mean",
        "logit_clean_repair_gain_mean",
        "prob_delta_clean_afterclean_mae_mean",
        "logit_delta_clean_afterclean_mae_mean",
    ]:
        assert col in scenario_df.columns
