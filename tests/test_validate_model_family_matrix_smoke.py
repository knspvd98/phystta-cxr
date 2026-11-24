import json
import os
import subprocess
import sys
from pathlib import Path

import yaml


def _write_base_config(path: Path) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "model": {
                    "encoder_profiles": [
                        {"name": "baseline_cnn", "encoder": {"encoder_source": "baseline_cnn"}},
                        {"name": "baseline_cnn_resnet50", "encoder": {"encoder_source": "baseline_cnn"}},
                        {"name": "hf_vit", "encoder": {"encoder_source": "hf_clip_or_vit"}},
                        {"name": "hf_clip_or_vit", "encoder": {"encoder_source": "hf_clip_or_vit"}},
                    ]
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def _write_head_meta(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "feat_dim": 512,
                "num_classes": 14,
                "label_cols": [
                    "No Finding",
                    "Enlarged Cardiomediastinum",
                    "Cardiomegaly",
                    "Lung Opacity",
                    "Lung Lesion",
                    "Edema",
                    "Consolidation",
                    "Pneumonia",
                    "Atelectasis",
                    "Pneumothorax",
                    "Pleural Effusion",
                    "Pleural Other",
                    "Fracture",
                    "Support Devices",
                ],
                "group_key": "global_patient_id",
                "fingerprint": {
                    "encoder_source": "hf_clip_or_vit",
                    "architecture": None,
                    "model_id": "openai/clip-vit-base-patch32",
                    "feat_dim": 512,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _write_restart_validation(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "status": "ok",
                "branch_purity": "matched",
                "reusable_archived_head_contract": {
                    "status": "ok",
                    "fingerprint": {"model_id": "openai/clip-vit-base-patch32"},
                    "family": "clip_base_patch32_mainline",
                },
                "historical_repair_branch_contract": {
                    "status": "warning",
                    "variants": [
                        {"variant": "w0_pathfix", "head_lineage_status": "meta_only_reference"},
                        {"variant": "k2", "head_lineage_status": "meta_only_reference"},
                    ],
                    "historical_public_atomic_head_reference_status": "meta_only",
                },
                "head_contract": {"status": "ok"},
                "index_contract": {"status": "matched"},
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _write_future_inputs(path: Path, head_meta_path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "status": "ok",
                "local_sources": {
                    "bundle_root": "/tmp/fake_bundle",
                    "repo_root_local": "/tmp/fake_repo",
                    "source_tgz_local": "/tmp/fake_bundle/repo/phystta_active_20260321_source.tgz",
                    "config_paths_local": {
                        "w0_pathfix": "/tmp/fake_bundle/w0.yaml",
                        "k2": "/tmp/fake_bundle/k2.yaml",
                        "k4": "/tmp/fake_bundle/k4.yaml",
                        "k8": "/tmp/fake_bundle/k8.yaml",
                    },
                    "eraser_archives_local": {
                        "repair3": "/tmp/fake_bundle/repair3.tgz",
                        "k2": "/tmp/fake_bundle/k2.tgz",
                        "k4": "/tmp/fake_bundle/k4.tgz",
                        "k8": "/tmp/fake_bundle/k8.tgz",
                    },
                    "head_path_local": "/tmp/fake_bundle/head.pt",
                    "head_meta_path_local": str(head_meta_path.resolve()),
                    "index_path_local": "/tmp/fake_bundle/index.csv",
                },
                "canonical_pod_restore": {
                    "repo_root": "/workspace/releases/phystta_active_20260321",
                    "expected_dataset_root": "/workspace/data/mimic-cxr-jpg",
                    "historical_repair_branch_configs": {
                        "w0_pathfix": "/workspace/outputs/bounded_mimic_atomic_repair1/public_mainline.effective.headsubspace_w0.pathfix.yaml",
                        "k2": "/workspace/outputs/bounded_mimic_atomic_repair1/public_mainline.effective.headsubspace_k2.yaml",
                        "k4": "/workspace/outputs/bounded_mimic_atomic_repair1/public_mainline.effective.headsubspace_k4.yaml",
                        "k8": "/workspace/outputs/bounded_mimic_atomic_repair1/public_mainline.effective.headsubspace_k8.yaml",
                    },
                    "reusable_archived_head": {
                        "path": "/workspace/outputs/paperpilot_mimic_artifacts/heads/head.pt",
                        "meta_path": "/workspace/outputs/paperpilot_mimic_artifacts/heads/head_meta.json",
                        "family": "clip_base_patch32_mainline",
                    },
                    "historical_public_atomic_head_reference": {
                        "path": "/workspace/outputs/heads/mimic_public_atomic_head.pt",
                        "meta_path": "/workspace/outputs/heads/mimic_public_atomic_head.meta.json",
                        "availability": "meta_only_or_missing",
                        "status": "meta_only",
                    },
                    "restored_eraser_manifests": {
                        "repair3": "/workspace/outputs/bounded_mimic_atomic_headsubspace_repair3/eraser_bank_manifest.json",
                        "k2": "/workspace/outputs/bounded_mimic_atomic_headsubspace_k2/eraser_bank_manifest.json",
                        "k4": "/workspace/outputs/bounded_mimic_atomic_headsubspace_k4/eraser_bank_manifest.json",
                        "k8": "/workspace/outputs/bounded_mimic_atomic_headsubspace_k8/eraser_bank_manifest.json",
                    },
                    "restored_head_path": "/workspace/outputs/paperpilot_mimic_artifacts/heads/head.pt",
                    "restored_head_meta_path": "/workspace/outputs/paperpilot_mimic_artifacts/heads/head_meta.json",
                    "restored_index_path": "/workspace/outputs/bounded_mimic_atomic_headsubspace_repair3/index_pathfix/mimic_index_fullresolved_enriched.workspacepaths.csv",
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _write_matrix(path: Path) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "shared_contract": {
                    "dataset_id": "mimic_public_acquisition",
                    "label_space": "chexpert14",
                    "group_key": "global_patient_id",
                    "shift_protocol": "bounded_headsubspace_atomic",
                    "atomic_families": ["heel", "scatter"],
                    "runtime_allow_download": False,
                    "reference_family": "clip_base_patch32_mainline",
                },
                "families": {
                    "clip_base_patch32_mainline": {
                        "enabled": True,
                        "wave": 1,
                        "comparison_bucket": "clip_like",
                        "encoder_source": "hf_clip_or_vit",
                        "encoder_profile": "hf_clip_or_vit",
                        "model_id": "openai/clip-vit-base-patch32",
                        "revision": "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268",
                        "runtime_mode": "classifier",
                        "head_policy": "reuse_resolved_head",
                        "requires_new_head": False,
                        "requires_classifier_head": True,
                        "requires_local_encoder_path": False,
                        "eligible_for_intervention_mainline": True,
                        "eligible_for_shift_benchmark": True,
                    },
                    "vit_base_patch16_in21k": {
                        "enabled": True,
                        "wave": 1,
                        "comparison_bucket": "vit_like",
                        "encoder_source": "hf_clip_or_vit",
                        "encoder_profile": "hf_vit",
                        "model_id": "google/vit-base-patch16-224-in21k",
                        "revision": "b4569560a39a0f1af58e3ddaf17facf20ab919b0",
                        "runtime_mode": "classifier",
                        "head_policy": "train_new_head",
                        "requires_new_head": True,
                        "requires_classifier_head": True,
                        "requires_local_encoder_path": False,
                        "eligible_for_intervention_mainline": True,
                        "eligible_for_shift_benchmark": True,
                    },
                    "cnn_densenet121": {
                        "enabled": True,
                        "wave": 1,
                        "comparison_bucket": "cnn",
                        "encoder_source": "baseline_cnn",
                        "encoder_profile": "baseline_cnn",
                        "runtime_mode": "classifier",
                        "head_policy": "train_new_head",
                        "requires_new_head": True,
                        "requires_classifier_head": True,
                        "requires_local_encoder_path": True,
                        "bootstrap_source": "existing_cache_bootstrap",
                        "eligible_for_intervention_mainline": True,
                        "eligible_for_shift_benchmark": True,
                    },
                    "cnn_resnet50": {
                        "enabled": True,
                        "wave": 1,
                        "comparison_bucket": "cnn",
                        "encoder_source": "baseline_cnn",
                        "encoder_profile": "baseline_cnn_resnet50",
                        "runtime_mode": "classifier",
                        "head_policy": "train_new_head",
                        "requires_new_head": True,
                        "requires_classifier_head": True,
                        "requires_local_encoder_path": True,
                        "bootstrap_source": "existing_cache_bootstrap",
                        "eligible_for_intervention_mainline": True,
                        "eligible_for_shift_benchmark": True,
                    },
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def test_validate_model_family_matrix_smoke(tmp_path: Path):
    repo = Path(__file__).resolve().parents[1]
    matrix_path = tmp_path / "model_family_matrix.yaml"
    restart_validation_path = tmp_path / "restart_bundle_validation.json"
    future_inputs_path = tmp_path / "future_pod_inputs.json"
    base_config_path = tmp_path / "base_config.yaml"
    head_meta_path = tmp_path / "head_meta.json"
    out_dir = tmp_path / "out"

    _write_base_config(base_config_path)
    _write_head_meta(head_meta_path)
    _write_restart_validation(restart_validation_path)
    _write_future_inputs(future_inputs_path, head_meta_path)
    _write_matrix(matrix_path)

    cmd = [
        sys.executable,
        str(repo / "scripts" / "validate_model_family_matrix.py"),
        "--matrix_path",
        str(matrix_path),
        "--restart_bundle_validation_json",
        str(restart_validation_path),
        "--future_pod_inputs_json",
        str(future_inputs_path),
        "--base_config",
        str(base_config_path),
        "--out_dir",
        str(out_dir),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    subprocess.run(cmd, check=True, cwd=str(repo), env=env)

    payload = json.loads((out_dir / "model_family_validation.json").read_text(encoding="utf-8"))
    runplan = json.loads((out_dir / "future_model_family_runplan.json").read_text(encoding="utf-8"))

    assert payload["status"] == "ok"
    assert runplan["reusable_archived_head_available"] is True
    assert runplan["reusable_archived_head_family"] == "clip_base_patch32_mainline"
    assert runplan["historical_repair_branch_present"] is True
    assert runplan["historical_public_atomic_head_reference_status"] == "meta_only"
    assert runplan["family_head_strategy"]["clip_base_patch32_mainline"] == "reuse_archived_head"
    assert runplan["family_head_strategy"]["vit_base_patch16_in21k"] == "materialize_config_only_no_head_reuse"
    assert runplan["family_head_strategy"]["cnn_densenet121"] == "materialize_config_only_no_head_reuse"
