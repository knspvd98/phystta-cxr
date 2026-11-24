from pathlib import Path

import pytest

from phystta.config.loader import load_config


def _base_cfg():
    cfg = load_config("configs/representation_surgery_2026.yaml")
    cfg.experiment.tier = "paper"
    cfg.intervention.use_realism = True
    cfg.intervention.realism_path = Path("outputs/representation_surgery/realism_calibrator.json")
    cfg.runtime.freeze_artifacts = True
    cfg.data.split_fractions = {
        cfg.data.head_split: 0.5,
        cfg.data.eraser_split: 0.2,
        cfg.data.calib_split: 0.15,
        cfg.data.eval_split: 0.15,
    }
    cfg.shift_id.calibration = "platt"
    cfg.erasure_eval.modes = ["clinical", "causal_invariance", "safety", "probing", "fairness"]
    cfg.erasure_eval.fairness_groups = ["sex", "race", "age"]
    return cfg


def test_paper_tier_requires_freeze_artifacts():
    cfg = _base_cfg()
    cfg.runtime.freeze_artifacts = False
    with pytest.raises(ValueError, match="freeze_artifacts"):
        cfg.validate()


def test_paper_tier_requires_required_modes():
    cfg = _base_cfg()
    cfg.erasure_eval.modes = ["clinical", "probing"]
    with pytest.raises(ValueError, match="erasure_eval.modes"):
        cfg.validate()


def test_paper_tier_requires_fairness_groups():
    cfg = _base_cfg()
    cfg.erasure_eval.fairness_groups = []
    with pytest.raises(ValueError, match="fairness_groups"):
        cfg.validate()


def test_paper_tier_requires_calibrated_shift_id():
    cfg = _base_cfg()
    cfg.shift_id.calibration = "none"
    with pytest.raises(ValueError, match="shift_id.calibration"):
        cfg.validate()


def test_shift_id_label_space_must_be_known():
    cfg = _base_cfg()
    cfg.shift_id.label_space = "physical_kind"
    with pytest.raises(ValueError, match="shift_id.label_space"):
        cfg.validate()


def test_paper_tier_rejects_patient_group_key_for_shift_id_nature_mode():
    cfg = _base_cfg()
    cfg.shift_id.label_space = "nature"
    cfg.shift_id.group_key = "global_patient_id"
    with pytest.raises(ValueError, match="shift_id.group_key"):
        cfg.validate()


def test_paper_tier_requires_no_download():
    cfg = _base_cfg()
    cfg.runtime.allow_download = True
    with pytest.raises(ValueError, match="allow_download=false"):
        cfg.validate()


def test_paper_tier_requires_deterministic_index_contract():
    cfg = _base_cfg()
    cfg.data.prebuilt_index_csv = None
    for src in cfg.data.sources:
        src.metadata_csv = None
    with pytest.raises(ValueError, match="deterministic indexing"):
        cfg.validate()


def test_paper_tier_requires_prebuilt_index_hash_when_index_is_prebuilt():
    cfg = _base_cfg()
    cfg.data.prebuilt_index_csv = Path("data_indexes/mimic_index.csv")
    cfg.data.prebuilt_index_sha256 = None
    with pytest.raises(ValueError, match="prebuilt_index_sha256"):
        cfg.validate()


def test_paper_tier_requires_pinned_encoder_revision():
    cfg = _base_cfg()
    cfg.model.encoder.revision = None
    with pytest.raises(ValueError, match="pinned revision"):
        cfg.validate()


def test_paper_tier_rejects_mutable_encoder_revision():
    cfg = _base_cfg()
    cfg.model.encoder.revision = "main"
    with pytest.raises(ValueError, match="immutable pinned revision"):
        cfg.validate()


def test_paper_tier_rejects_manual_intensity_grid_under_realism():
    cfg = _base_cfg()
    cfg.intervention.intensity_grid = [0.0, 0.3]
    with pytest.raises(ValueError, match="forbids manual intervention.intensity_grid"):
        cfg.validate()

    cfg = _base_cfg()
    cfg.erasure_eval.intensity_grid = [0.0, 0.3]
    with pytest.raises(ValueError, match="forbids manual erasure_eval.intensity_grid"):
        cfg.validate()


def test_paper_tier_rejects_baseline_cnn_without_offline_contract():
    cfg = _base_cfg()
    prof_map = {p.name: p.encoder for p in (cfg.model.encoder_profiles or [])}
    assert "baseline_cnn" in prof_map
    prof_map["baseline_cnn"].encoder_path = None
    with pytest.raises(ValueError, match="baseline_cnn requires explicit encoder_path"):
        cfg.validate()


def test_paper_tier_rejects_baseline_cnn_placeholder_path():
    cfg = _base_cfg()
    prof_map = {p.name: p.encoder for p in (cfg.model.encoder_profiles or [])}
    prof_map["baseline_cnn"].encoder_path = Path("/path/to/local/baseline_cnn.pt")
    with pytest.raises(ValueError, match="placeholder encoder_path"):
        cfg.validate()


def test_paper_tier_rejects_openclip_without_offline_contract():
    cfg = _base_cfg()
    prof_map = {p.name: p.encoder for p in (cfg.model.encoder_profiles or [])}
    # repurpose one profile to simulate openclip entry in study matrix
    prof_map["baseline_cnn"].encoder_source = "openclip"
    prof_map["baseline_cnn"].encoder_path = None
    prof_map["baseline_cnn"].pretrained = None
    with pytest.raises(ValueError, match="openclip.*requires encoder_path or encoder.pretrained=false"):
        cfg.validate()


def test_paper_tier_rejects_vlm_placeholder_path():
    cfg = _base_cfg()
    cfg.vlm.enabled = False
    cfg.vlm.model_id = None
    cfg.vlm.model_path = Path("/path/to/local/llava")
    with pytest.raises(ValueError, match="placeholder vlm.model_path"):
        cfg.validate()


def test_paper_tier_allows_local_vlm_without_revision():
    cfg = _base_cfg()
    cfg.vlm.enabled = True
    cfg.vlm.model_id = None
    cfg.vlm.model_path = Path("outputs/vlm/local_llava")
    cfg.vlm.revision = None
    cfg.validate()


def test_paper_tier_requires_head_meta_sidecar_when_head_exists(tmp_path):
    cfg = _base_cfg()
    # Create a real head artifact path to trigger sidecar enforcement.
    head_path = tmp_path / "head.pt"
    head_path.write_bytes(b"dummy")
    cfg.model.encoder.head_path = head_path
    cfg.model.encoder.head_meta_path = None
    with pytest.raises(ValueError, match="head metadata sidecar"):
        cfg.validate()

    # Add sidecar and validation should pass this guard.
    meta_path = head_path.with_suffix(".meta.json")
    meta_path.write_text("{}", encoding="utf-8")
    cfg.validate()
