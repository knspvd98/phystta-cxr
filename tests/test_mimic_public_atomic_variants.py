from __future__ import annotations

from pathlib import Path

from phystta.config.loader import load_config


def _cfg(repo: Path, rel: str):
    return load_config(str(repo / rel))


def test_canonical_bounded_atomic_clip_variants_load_and_narrow_to_claim_core():
    repo = Path(__file__).resolve().parents[1]

    rel_paths = [
        "configs/mimic_public_acquisition_atomic_clip_bounded_nopreserve.yaml",
        "configs/mimic_public_acquisition_atomic_clip_headsubspace_k2.yaml",
        "configs/mimic_public_acquisition_atomic_clip_headsubspace_k4.yaml",
        "configs/mimic_public_acquisition_atomic_clip_headsubspace_soft_k2.yaml",
        "configs/mimic_public_acquisition_atomic_clip_headsubspace_soft_recovery_k2.yaml",
        "configs/mimic_public_acquisition_atomic_clip_headsubspace_soft_recovery_k4.yaml",
    ]

    cfgs = [_cfg(repo, p) for p in rel_paths]

    for cfg in cfgs:
        assert list(cfg.intervention.shift_types) == ["heel", "scatter"]
        assert list(cfg.erasure_eval.shift_types) == ["heel", "scatter"]
        assert cfg.intervention.pair_mode == "temporal_same_view"
        assert cfg.intervention.group_key == "global_patient_id"
        assert list(cfg.intervention.realism_levels) == [0.0, 0.5, 1.0]
        assert cfg.erasure_eval.scenario_mode == "atomic_calibrated_eval"


def test_canonical_bounded_atomic_clip_variants_share_clip_lineage():
    repo = Path(__file__).resolve().parents[1]

    rel_paths = [
        "configs/mimic_public_acquisition_atomic_clip_bounded_nopreserve.yaml",
        "configs/mimic_public_acquisition_atomic_clip_headsubspace_k2.yaml",
        "configs/mimic_public_acquisition_atomic_clip_headsubspace_k4.yaml",
        "configs/mimic_public_acquisition_atomic_clip_headsubspace_soft_k2.yaml",
        "configs/mimic_public_acquisition_atomic_clip_headsubspace_soft_recovery_k2.yaml",
        "configs/mimic_public_acquisition_atomic_clip_headsubspace_soft_recovery_k4.yaml",
    ]

    cfgs = [_cfg(repo, p) for p in rel_paths]

    for cfg in cfgs:
        assert cfg.model.family == "transformer"
        assert cfg.model.architecture == "clip-vit-base-patch32"
        assert cfg.model.encoder.encoder_source == "hf_clip_or_vit"
        assert cfg.model.encoder.model_id == "openai/clip-vit-base-patch32"
        assert cfg.model.encoder.revision == "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"
        assert cfg.model.encoder.head_path == "./outputs/paperpilot_mimic_artifacts/heads/head.pt"
        assert cfg.model.encoder.head_meta_path == "./outputs/paperpilot_mimic_artifacts/heads/head_meta.json"
        assert str(cfg.intervention.realism_path) == "outputs/clip_base_patch32_mainline/realism/realism_calibrator.json"


def test_canonical_bounded_atomic_clip_variant_modes_and_recovery_weights():
    repo = Path(__file__).resolve().parents[1]

    bounded_nopreserve = _cfg(repo, "configs/mimic_public_acquisition_atomic_clip_bounded_nopreserve.yaml")
    hard_k2 = _cfg(repo, "configs/mimic_public_acquisition_atomic_clip_headsubspace_k2.yaml")
    hard_k4 = _cfg(repo, "configs/mimic_public_acquisition_atomic_clip_headsubspace_k4.yaml")
    soft_k2 = _cfg(repo, "configs/mimic_public_acquisition_atomic_clip_headsubspace_soft_k2.yaml")
    soft_rec_k2 = _cfg(repo, "configs/mimic_public_acquisition_atomic_clip_headsubspace_soft_recovery_k2.yaml")
    soft_rec_k4 = _cfg(repo, "configs/mimic_public_acquisition_atomic_clip_headsubspace_soft_recovery_k4.yaml")

    assert bounded_nopreserve.eraser.task_preserve == "none"
    assert bounded_nopreserve.eraser.task_subspace_k == 0
    assert bounded_nopreserve.eraser.logit_recovery_weight == 0.0
    assert bounded_nopreserve.eraser.prob_recovery_weight == 0.0
    assert bounded_nopreserve.eraser.clean_identity_weight == 0.0

    assert hard_k2.eraser.task_preserve == "head_subspace"
    assert hard_k2.eraser.task_subspace_k == 2
    assert hard_k2.eraser.logit_recovery_weight == 0.0
    assert hard_k2.eraser.prob_recovery_weight == 0.0
    assert hard_k2.eraser.clean_identity_weight == 0.0

    assert hard_k4.eraser.task_preserve == "head_subspace"
    assert hard_k4.eraser.task_subspace_k == 4
    assert hard_k4.eraser.logit_recovery_weight == 0.0
    assert hard_k4.eraser.prob_recovery_weight == 0.0
    assert hard_k4.eraser.clean_identity_weight == 0.0

    assert soft_k2.eraser.task_preserve == "head_subspace_soft"
    assert soft_k2.eraser.task_subspace_k == 2
    assert soft_k2.eraser.logit_recovery_weight == 0.0
    assert soft_k2.eraser.prob_recovery_weight == 0.0
    assert soft_k2.eraser.clean_identity_weight == 0.0

    assert soft_rec_k2.eraser.task_preserve == "head_subspace_soft"
    assert soft_rec_k2.eraser.task_subspace_k == 2
    assert soft_rec_k2.eraser.logit_recovery_weight == 0.25
    assert soft_rec_k2.eraser.prob_recovery_weight == 0.05
    assert soft_rec_k2.eraser.clean_identity_weight == 1.0

    assert soft_rec_k4.eraser.task_preserve == "head_subspace_soft"
    assert soft_rec_k4.eraser.task_subspace_k == 4
    assert soft_rec_k4.eraser.logit_recovery_weight == 0.25
    assert soft_rec_k4.eraser.prob_recovery_weight == 0.05
    assert soft_rec_k4.eraser.clean_identity_weight == 1.0


def test_canonical_bounded_atomic_clip_variant_outputs_are_unique():
    repo = Path(__file__).resolve().parents[1]

    rel_paths = [
        "configs/mimic_public_acquisition_atomic_clip_bounded_nopreserve.yaml",
        "configs/mimic_public_acquisition_atomic_clip_headsubspace_k2.yaml",
        "configs/mimic_public_acquisition_atomic_clip_headsubspace_k4.yaml",
        "configs/mimic_public_acquisition_atomic_clip_headsubspace_soft_k2.yaml",
        "configs/mimic_public_acquisition_atomic_clip_headsubspace_soft_recovery_k2.yaml",
        "configs/mimic_public_acquisition_atomic_clip_headsubspace_soft_recovery_k4.yaml",
    ]

    cfgs = [_cfg(repo, p) for p in rel_paths]
    output_roots = [cfg.output.root for cfg in cfgs]
    run_names = [cfg.output.run_name for cfg in cfgs]

    assert len(output_roots) == len(set(output_roots))
    assert len(run_names) == len(set(run_names))


def test_legacy_bounded_baseline_cnn_and_broad_baseline_are_not_canonical_clip_set():
    repo = Path(__file__).resolve().parents[1]

    legacy_broad = _cfg(repo, "configs/mimic_public_acquisition_atomic.yaml")
    legacy_bounded_cnn = _cfg(repo, "configs/mimic_public_acquisition_atomic_bounded_nopreserve.yaml")

    assert list(legacy_broad.intervention.shift_types) != ["heel", "scatter"]
    assert legacy_bounded_cnn.model.encoder.encoder_source == "baseline_cnn"
    assert legacy_bounded_cnn.model.family == "cnn"
