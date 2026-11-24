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

from phystta.causal import run_confounding_audit
from phystta.cli.main import PIPELINE_NOTES, SCRIPT_MAP
from phystta.data import build_label_projection_manifest, canonical_external_label_space
from phystta.intervention import probe_spuriousness_diagnostic, eraser_subspace_stability_diagnostic
from phystta.intervention.eraser import LinearSubspaceEraser, save_eraser
from phystta.shifts import ShiftManager
from phystta.viz import build_confounding_audit_figures, build_erasure_eval_figures


def _write_minicxr(tmp_path: Path, n_patients: int = 12) -> tuple[Path, Path]:
    img_dir = tmp_path / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    rng = np.random.default_rng(41)
    for i in range(n_patients):
        pid = f"p{i:03d}"
        label = float((i % 3) == 0)
        for t in (0, 1):
            arr = rng.uniform(0.05, 0.95, size=(64, 64)).astype(np.float32)
            arr[:, 8:16] += 0.05 * float(i % 2)
            arr = np.clip(arr, 0.0, 1.0)
            img_path = img_dir / f"{pid}_t{t}.png"
            Image.fromarray((arr * 255.0).astype(np.uint8), mode="L").save(img_path)
            rows.append(
                {
                    "path": img_path.name,
                    "patient_id": pid,
                    "study_id": f"s{i:04d}_{t}",
                    "view": "PA",
                    "view_family": "frontal",
                    "procedure_context": "portable_like" if (i % 2) else "upright_pa",
                    "orientation_context": "upright",
                    "study_date": "20240101",
                    "study_time": "080000" if t == 0 else "081000",
                    "No Finding": label,
                }
            )
    meta_csv = tmp_path / "meta.csv"
    pd.DataFrame(rows).to_csv(meta_csv, index=False)
    return img_dir, meta_csv


def test_collimation_crop_shift_is_registered_and_changes_support():
    manager = ShiftManager(seed=7)
    assert "collimation_crop" in manager.shifts
    shift = manager.shifts["collimation_crop"]
    assert shift is not None
    img = torch.linspace(0.0, 1.0, steps=64 * 64, dtype=torch.float32).reshape(1, 64, 64)
    out = shift.apply(img, 0.7)
    assert out.shape == img.shape
    assert not torch.allclose(out, img)
    assert float(torch.mean(torch.abs(out - img))) > 1e-3
    params = shift.physical_params(0.7)
    assert params["retained_area_fraction"] < 1.0
    assert params["field_of_view_proxy"] == "crop_and_resize"


def test_confounding_audit_returns_predictability_and_shift_proxy_outputs():
    rng = np.random.default_rng(13)
    n = 72
    patients = np.asarray([f"p{i // 3:03d}" for i in range(n)], dtype=str)
    view_family = np.where(np.arange(n) % 2 == 0, "frontal", "lateral")
    procedure_context = np.where(np.arange(n) % 3 == 0, "portable_like", "upright_pa")
    orientation_context = np.where(np.arange(n) % 4 == 0, "supine", "upright")
    resolution_bucket = np.where(np.arange(n) % 5 == 0, "high", "standard")
    latent = (
        (view_family == "frontal").astype(float)[:, None] * np.array([[1.0, 0.0, 0.3]])
        + (procedure_context == "portable_like").astype(float)[:, None] * np.array([[0.2, 1.0, 0.4]])
        + rng.normal(scale=0.08, size=(n, 3))
    )
    features = np.concatenate([latent, rng.normal(scale=0.05, size=(n, 5))], axis=1)
    label_a = (latent[:, 0] + rng.normal(scale=0.1, size=n) > 0.6).astype(float)
    label_b = (latent[:, 1] + rng.normal(scale=0.1, size=n) > 0.6).astype(float)
    labels = np.stack([label_a, label_b], axis=1)
    probs = np.clip(labels * 0.75 + 0.15 + rng.normal(scale=0.05, size=labels.shape), 0.01, 0.99)
    metadata_df = pd.DataFrame(
        {
            "global_patient_id": patients,
            "patient_id": patients,
            "view_family": view_family,
            "procedure_context": procedure_context,
            "orientation_context": orientation_context,
            "resolution_bucket": resolution_bucket,
            "view": np.where(view_family == "frontal", "PA", "LAT"),
        }
    )

    audit = run_confounding_audit(
        features=features,
        probs=probs,
        labels=labels,
        metadata_df=metadata_df,
        label_cols=["No Finding", "Edema"],
        bootstrap_samples=24,
        seed=17,
    )

    predictability = audit["predictability_df"]
    assert not predictability.empty
    assert {"context_from_embeddings", "context_from_labels", "labels_from_context", "severity_from_context"} <= set(
        predictability["audit_family"].astype(str)
    )
    shift_proxy = audit["shift_id_proxy_df"]
    assert not shift_proxy.empty
    assert audit["summary"]["active_shift_id_group_key"] in {"view_family", "procedure_context", "orientation_context", "resolution_bucket", "view"}


def test_guarded_diagnostics_return_expected_summary_keys():
    rng = np.random.default_rng(23)
    n = 48
    groups = np.asarray([f"p{i // 2:03d}" for i in range(n)], dtype=str)
    view_family = np.where(np.arange(n) % 2 == 0, "frontal", "lateral")
    shift_vec = (view_family == "frontal").astype(float)[:, None] * np.array([[0.8, 0.0, 0.4, 0.2]])
    z_clean = rng.normal(size=(n, 16)).astype(np.float32)
    z_shift = z_clean + np.pad(shift_vec, ((0, 0), (0, 12)), mode="constant")
    z_after = z_shift - 0.6 * np.pad(shift_vec, ((0, 0), (0, 12)), mode="constant")
    labels = np.stack(
        [
            (view_family == "frontal").astype(float),
            (np.arange(n) % 3 == 0).astype(float),
        ],
        axis=1,
    )
    meta_df = pd.DataFrame(
        {
            "view_family": view_family,
            "procedure_context": np.where(np.arange(n) % 3 == 0, "portable_like", "upright_pa"),
            "orientation_context": np.where(np.arange(n) % 4 == 0, "supine", "upright"),
        }
    )

    spurious_summary, spurious_df = probe_spuriousness_diagnostic(
        z_clean,
        z_shift,
        labels,
        groups=groups,
        metadata_df=meta_df,
        bootstrap_samples=24,
        seed=19,
    )
    stability_summary, stability_df = eraser_subspace_stability_diagnostic(
        z_shift,
        z_after,
        groups=groups,
        base_basis=np.eye(16, 4, dtype=float),
        n_resamples=6,
        k=4,
        seed=19,
    )

    assert "severity_shortcut_auroc" in spurious_summary
    assert "max_strata_gap" in spurious_summary
    assert not spurious_df.empty
    assert "mean_pairwise_overlap" in stability_summary
    assert not stability_df.empty


def test_atomic_figure_pack_writes_risk_coverage_ci_outputs(tmp_path: Path):
    metrics_df = pd.DataFrame(
        [
            {
                "method": method,
                "shift": shift,
                "intensity": realism,
                "realism_level": realism,
                "auroc_macro": 0.82 - 0.05 * realism + (0.02 if method == "eraser" else 0.0),
                "auprc_macro": 0.61 - 0.04 * realism + (0.01 if method == "eraser" else 0.0),
                "ece": 0.10 + 0.04 * realism - (0.01 if method == "eraser" else 0.0),
                "brier": 0.18 + 0.03 * realism - (0.01 if method == "eraser" else 0.0),
                "risk_at_coverage": 0.12 + 0.05 * realism - (0.01 if method == "eraser" else 0.0),
                "expected_harm": 0.25 + 0.06 * realism - (0.02 if method == "eraser" else 0.0),
                "coverage_realized": 0.8,
                "abstain_rate": 0.2,
                "risk_at_coverage_ci_low": 0.10,
                "risk_at_coverage_ci_high": 0.18,
            }
            for method in ["no_eraser", "eraser"]
            for shift in ["heel", "collimation_crop"]
            for realism in [0.0, 0.5, 1.0]
        ]
    )
    risk_curve_df = pd.DataFrame(
        [
            {
                "method": method,
                "shift": shift,
                "coverage": cov,
                "risk": 0.18 - 0.05 * cov + (0.01 if method == "eraser" else 0.0),
                "risk_ci_low": 0.15 - 0.05 * cov,
                "risk_ci_high": 0.22 - 0.05 * cov,
            }
            for method in ["no_eraser", "eraser"]
            for shift in ["heel", "collimation_crop"]
            for cov in [0.2, 0.5, 0.8, 1.0]
        ]
    )

    outputs = build_erasure_eval_figures(
        metrics_df,
        out_dir=tmp_path,
        scenario_mode="atomic_calibrated_eval",
        reliability_df=None,
        risk_curve_df=risk_curve_df,
    )

    fig_dir = tmp_path / "figures" / "atomic_calibrated"
    assert (fig_dir / "atomic_risk_coverage_curves.png").exists()
    assert (fig_dir / "atomic_risk_coverage_tidy.csv").exists()
    assert "atomic_risk_coverage_curves" in outputs


def test_confounding_figure_pack_preserves_existing_outputs(tmp_path: Path):
    predictability_df = pd.DataFrame(
        [
            {"audit_family": "context_from_embeddings", "target": "view_family", "auroc_macro": 0.82, "auroc_macro_ci_low": 0.78, "auroc_macro_ci_high": 0.86},
            {"audit_family": "context_from_labels", "target": "procedure_context", "auroc_macro": 0.76, "auroc_macro_ci_low": 0.72, "auroc_macro_ci_high": 0.81},
            {"audit_family": "labels_from_context", "target": "Edema", "auroc": 0.68},
            {"audit_family": "severity_from_context", "target": "severity", "balanced_accuracy": 0.64},
        ]
    )
    shift_id_df = pd.DataFrame(
        [
            {"scenario_kind": "heel:0.5", "p_none": 0.1, "p_covariate": 0.7, "p_prevalence": 0.1, "p_mixed": 0.1, "uncertainty": 0.2},
            {"scenario_kind": "scatter:1.0", "p_none": 0.2, "p_covariate": 0.2, "p_prevalence": 0.2, "p_mixed": 0.4, "uncertainty": 0.5},
        ]
    )
    pair_delta_df = pd.DataFrame(
        [
            {"label": "Edema", "mean_abs_delta": 0.12, "mean_abs_delta_ci_low": 0.08, "mean_abs_delta_ci_high": 0.15},
            {"label": "Pneumothorax", "mean_abs_delta": 0.09, "mean_abs_delta_ci_low": 0.05, "mean_abs_delta_ci_high": 0.13},
        ]
    )

    outputs = build_confounding_audit_figures(
        out_dir=tmp_path,
        predictability_df=predictability_df,
        shift_id_df=shift_id_df,
        pair_delta_df=pair_delta_df,
    )

    fig_dir = tmp_path / "figures" / "confounding"
    assert (fig_dir / "confounding_predictability_summary.png").exists()
    assert (fig_dir / "shift_id_proxy_audit.png").exists()
    assert (fig_dir / "technical_pair_pathology_deltas.png").exists()
    assert (fig_dir / "technical_pair_pathology_deltas_tidy.csv").exists()
    assert "technical_pair_pathology_deltas" in outputs


def test_external_projection_manifests_remain_serializable_sidecars(tmp_path: Path):
    target = canonical_external_label_space()
    nih_manifest = build_label_projection_manifest("nih", target_labels=target)
    vindr_manifest = build_label_projection_manifest("vindr", target_labels=target)
    nih_path = tmp_path / "nih.label_harmonization_manifest.json"
    vindr_path = tmp_path / "vindr.label_harmonization_manifest.json"
    nih_path.write_text(json.dumps(nih_manifest, indent=2), encoding="utf-8")
    vindr_path.write_text(json.dumps(vindr_manifest, indent=2), encoding="utf-8")
    assert json.loads(nih_path.read_text(encoding="utf-8"))["source"] == "nih"
    assert json.loads(vindr_path.read_text(encoding="utf-8"))["source"] == "vindr"
    assert "Lung Lesion" in vindr_manifest["approximate_labels"]


def test_confounding_audit_cli_is_registered():
    assert SCRIPT_MAP["confounding-audit"] == "scripts/run_confounding_audit.py"
    assert "confounding audit" in PIPELINE_NOTES["confounding-audit"]


def test_run_erasure_eval_writes_bootstrap_ci_and_curve_outputs(tmp_path: Path):
    try:
        import iterstrat  # noqa: F401
    except Exception:
        pytest.skip("iterstrat is required for erasure-eval smoke test")

    repo = Path(__file__).resolve().parents[1]
    img_dir, meta_csv = _write_minicxr(tmp_path)
    cfg_path = tmp_path / "cfg.yaml"
    out_dir = tmp_path / "eval_out"
    eraser_path = tmp_path / "eraser.pt"
    eraser_manifest = tmp_path / "eraser_bank_manifest.json"

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
      view_family_col: view_family
      study_date_col: study_date
      study_time_col: study_time
      performed_procedure_step_description_col: procedure_context
      patient_orientation_code_meaning_col: orientation_context
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
    input_adapter:
      repeat_to_3: true
      image_size: 64
      mean: [0.5, 0.5, 0.5]
      std: [0.5, 0.5, 0.5]
train:
  batch_size: 6
  num_workers: 0
metrics:
  risk_coverage_points: 12
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
  bootstrap_samples: 24
  tradeoff_strengths: [0.0, 0.5, 1.0]
runtime:
  device: cpu
  seed: 42
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
        str(repo / "scripts" / "run_erasure_eval.py"),
        "--config",
        str(cfg_path),
        "--eraser",
        str(eraser_manifest),
        "--out_dir",
        str(out_dir),
        "--device",
        "cpu",
        "--seed",
        "42",
        "--bootstrap_samples",
        "24",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    subprocess.run(cmd, check=True, cwd=str(repo), env=env)

    metrics_df = pd.read_csv(out_dir / "metrics.csv")
    assert "auroc_macro_ci_low" in metrics_df.columns
    assert "expected_harm_ci_high" in metrics_df.columns
    curves_df = pd.read_csv(out_dir / "risk_coverage_curves.csv")
    assert {"coverage", "risk", "risk_ci_low", "risk_ci_high"} <= set(curves_df.columns)
    tradeoff_df = pd.read_csv(out_dir / "erasure_tradeoff.csv")
    assert "strength" in tradeoff_df.columns
    assert set(np.round(tradeoff_df["strength"].unique(), 2)) >= {0.0, 0.5, 1.0}
    assert (out_dir / "figures" / "atomic_calibrated" / "atomic_risk_coverage_curves.png").exists()
