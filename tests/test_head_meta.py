from pathlib import Path

from phystta.config.loader import load_config
from phystta.models.head_meta import (
    build_head_meta,
    build_encoder_fingerprint,
    infer_head_meta_paths,
    validate_head_meta,
)


def test_build_head_meta_contains_fingerprint():
    cfg = load_config("configs/representation_surgery.yaml")
    enc_cfg = cfg.model.encoder
    meta = build_head_meta(
        enc_cfg=enc_cfg,
        feat_dim=1024,
        num_classes=cfg.model.num_classes,
        label_cols=["No Finding"],
        group_key="global_patient_id",
        encoder_profile=None,
        seed=7,
    )
    assert meta["feat_dim"] == 1024
    assert "fingerprint" in meta
    assert meta["fingerprint"]["encoder_source"] == enc_cfg.encoder_source


def test_validate_head_meta_detects_encoder_mismatch():
    cfg = load_config("configs/representation_surgery.yaml")
    enc_cfg = cfg.model.encoder
    meta = {
        "feat_dim": 1024,
        "num_classes": cfg.model.num_classes,
        "label_cols": ["No Finding"],
        "fingerprint": build_encoder_fingerprint(enc_cfg),
    }
    # mutate expected encoder to force mismatch
    enc_cfg_mismatch = load_config("configs/representation_surgery.yaml").model.encoder
    enc_cfg_mismatch.encoder_source = "baseline_cnn"
    errs = validate_head_meta(
        meta,
        enc_cfg=enc_cfg_mismatch,
        expected_num_classes=cfg.model.num_classes,
        expected_label_cols=["No Finding"],
    )
    assert any("encoder fingerprint mismatch" in e for e in errs)


def test_infer_head_meta_paths():
    head = Path("/tmp/foo/bar/clip_head.pt")
    candidates = infer_head_meta_paths(head)
    assert str(candidates[0]).endswith("clip_head.meta.json")
    assert str(candidates[1]).endswith("clip_head_meta.json")
    assert str(candidates[2]).endswith("head_meta.json")
