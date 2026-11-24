from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from phystta.config.loader import load_config
from phystta.models import build_head_meta
from phystta.models.encoder_factory import EncoderFactory
from phystta.models.head_contract import resolve_head_contract


def _base_cfg(tmp_path: Path):
    cfg = load_config("configs/representation_surgery.yaml")
    cfg.model.num_classes = 1
    cfg.model.encoder.head_path = tmp_path / "head.pt"
    cfg.model.encoder.head_meta_path = tmp_path / "head_meta.json"
    return cfg


def _save_head(path: Path, in_features: int, out_features: int):
    head = torch.nn.Linear(in_features, out_features, bias=False)
    torch.save(head, path)
    return head


def test_resolve_head_contract_accepts_matching_meta(tmp_path: Path):
    cfg = _base_cfg(tmp_path)
    _save_head(Path(cfg.model.encoder.head_path), in_features=8, out_features=1)

    meta = build_head_meta(
        enc_cfg=cfg.model.encoder,
        feat_dim=8,
        num_classes=1,
        label_cols=["No Finding"],
        group_key="global_patient_id",
        encoder_profile=None,
        seed=7,
    )
    meta.setdefault("fingerprint", {})
    meta["fingerprint"].pop("use_medical_pretrain", None)
    meta["fingerprint"].pop("effective_medical_pretrain", None)
    Path(cfg.model.encoder.head_meta_path).write_text(json.dumps(meta), encoding="utf-8")

    contract = resolve_head_contract(
        cfg,
        label_cols=["No Finding"],
        require_head_path=True,
        strict_meta=True,
        load_artifact=True,
    )
    assert contract.head_path == Path(cfg.model.encoder.head_path)
    assert contract.head_meta["feat_dim"] == 8
    assert contract.head_artifact is not None


def test_resolve_head_contract_rejects_missing_meta(tmp_path: Path):
    cfg = _base_cfg(tmp_path)
    _save_head(Path(cfg.model.encoder.head_path), in_features=8, out_features=1)

    with pytest.raises(RuntimeError, match="Head metadata not found"):
        resolve_head_contract(
            cfg,
            label_cols=["No Finding"],
            require_head_path=True,
            strict_meta=True,
            load_artifact=False,
        )


def test_resolve_head_contract_rejects_fingerprint_mismatch(tmp_path: Path):
    cfg = _base_cfg(tmp_path)
    _save_head(Path(cfg.model.encoder.head_path), in_features=8, out_features=1)

    meta = build_head_meta(
        enc_cfg=cfg.model.encoder,
        feat_dim=8,
        num_classes=1,
        label_cols=["No Finding"],
        group_key="global_patient_id",
        encoder_profile=None,
        seed=7,
    )
    Path(cfg.model.encoder.head_meta_path).write_text(json.dumps(meta), encoding="utf-8")

    cfg.model.encoder.model_id = "wrong/model-id"

    with pytest.raises(RuntimeError, match="encoder fingerprint mismatch"):
        resolve_head_contract(
            cfg,
            label_cols=["No Finding"],
            require_head_path=True,
            strict_meta=True,
            load_artifact=False,
        )


def test_resolve_head_contract_rejects_feat_dim_mismatch(tmp_path: Path):
    cfg = _base_cfg(tmp_path)
    _save_head(Path(cfg.model.encoder.head_path), in_features=16, out_features=1)

    meta = build_head_meta(
        enc_cfg=cfg.model.encoder,
        feat_dim=8,
        num_classes=1,
        label_cols=["No Finding"],
        group_key="global_patient_id",
        encoder_profile=None,
        seed=7,
    )
    Path(cfg.model.encoder.head_meta_path).write_text(json.dumps(meta), encoding="utf-8")

    with pytest.raises(RuntimeError, match="head artifact feat_dim mismatch"):
        resolve_head_contract(
            cfg,
            label_cols=["No Finding"],
            require_head_path=True,
            strict_meta=True,
            load_artifact=True,
        )


def test_resolve_head_contract_rejects_missing_head_path_when_required(tmp_path: Path):
    cfg = _base_cfg(tmp_path)
    cfg.model.encoder.head_path = None

    with pytest.raises(RuntimeError, match="requires an explicit trained head artifact"):
        resolve_head_contract(
            cfg,
            label_cols=["No Finding"],
            require_head_path=True,
            strict_meta=True,
            load_artifact=False,
        )


def test_resolve_head_contract_rejects_missing_legacy_bool_when_expected_true_for_baseline_cnn(tmp_path: Path):
    cfg = _base_cfg(tmp_path)

    cfg.model.encoder.encoder_source = "baseline_cnn"
    cfg.model.encoder.architecture = "densenet121"
    cfg.model.architecture = "densenet121"
    cfg.model.use_medical_pretrain = True
    cfg.runtime.allow_download = True
    cfg.model.encoder.pretrained = True

    _save_head(Path(cfg.model.encoder.head_path), in_features=8, out_features=1)

    resolved_enc = EncoderFactory.resolve_encoder_config(cfg)
    assert resolved_enc.use_medical_pretrain is True
    assert resolved_enc.effective_medical_pretrain is True

    meta = build_head_meta(
        enc_cfg=resolved_enc,
        feat_dim=8,
        num_classes=1,
        label_cols=["No Finding"],
        group_key="global_patient_id",
        encoder_profile=None,
        seed=7,
    )
    meta.setdefault("fingerprint", {})
    meta["fingerprint"].pop("use_medical_pretrain", None)

    Path(cfg.model.encoder.head_meta_path).write_text(json.dumps(meta), encoding="utf-8")

    with pytest.raises(RuntimeError, match="use_medical_pretrain"):
        resolve_head_contract(
            cfg,
            label_cols=["No Finding"],
            require_head_path=True,
            strict_meta=True,
            load_artifact=False,
        )


def test_resolve_head_contract_accepts_missing_legacy_bool_for_nonbaseline_expected_false(tmp_path: Path):
    cfg = _base_cfg(tmp_path)
    _save_head(Path(cfg.model.encoder.head_path), in_features=8, out_features=1)

    resolved_enc = EncoderFactory.resolve_encoder_config(cfg)
    assert resolved_enc.encoder_source == "hf_clip_or_vit"
    assert resolved_enc.use_medical_pretrain is False
    assert resolved_enc.effective_medical_pretrain is False

    meta = build_head_meta(
        enc_cfg=resolved_enc,
        feat_dim=8,
        num_classes=1,
        label_cols=["No Finding"],
        group_key="global_patient_id",
        encoder_profile=None,
        seed=7,
    )
    meta.setdefault("fingerprint", {})
    meta["fingerprint"].pop("use_medical_pretrain", None)
    meta["fingerprint"].pop("effective_medical_pretrain", None)

    Path(cfg.model.encoder.head_meta_path).write_text(json.dumps(meta), encoding="utf-8")

    contract = resolve_head_contract(
        cfg,
        label_cols=["No Finding"],
        require_head_path=True,
        strict_meta=True,
        load_artifact=True,
    )

    assert contract.head_meta is not None
    assert contract.head_meta["fingerprint"]["use_medical_pretrain"] is False
    assert contract.head_meta["fingerprint"]["effective_medical_pretrain"] is False


def test_resolve_head_contract_accepts_legacy_none_bool_fields(tmp_path: Path):
    cfg = _base_cfg(tmp_path)
    _save_head(Path(cfg.model.encoder.head_path), in_features=8, out_features=1)

    meta = build_head_meta(
        enc_cfg=cfg.model.encoder,
        feat_dim=8,
        num_classes=1,
        label_cols=["No Finding"],
        group_key="global_patient_id",
        encoder_profile=None,
        seed=7,
    )
    meta.setdefault("fingerprint", {})
    meta["fingerprint"]["use_medical_pretrain"] = None
    meta["fingerprint"]["effective_medical_pretrain"] = None
    Path(cfg.model.encoder.head_meta_path).write_text(json.dumps(meta), encoding="utf-8")

    contract = resolve_head_contract(
        cfg,
        label_cols=["No Finding"],
        require_head_path=True,
        strict_meta=True,
        load_artifact=True,
    )
    assert contract.head_artifact is not None


def test_resolve_head_contract_accepts_legacy_encoder_fingerprint_alias(tmp_path: Path):
    cfg = _base_cfg(tmp_path)
    _save_head(Path(cfg.model.encoder.head_path), in_features=8, out_features=1)

    meta = build_head_meta(
        enc_cfg=cfg.model.encoder,
        feat_dim=8,
        num_classes=1,
        label_cols=["No Finding"],
        group_key="global_patient_id",
        encoder_profile=None,
        seed=7,
    )
    fp = dict(meta.pop("fingerprint"))
    fp.pop("use_medical_pretrain", None)
    fp.pop("effective_medical_pretrain", None)
    meta["encoder_fingerprint"] = fp

    Path(cfg.model.encoder.head_meta_path).write_text(json.dumps(meta), encoding="utf-8")

    contract = resolve_head_contract(
        cfg,
        label_cols=["No Finding"],
        require_head_path=True,
        strict_meta=True,
        load_artifact=True,
    )
    assert contract.head_artifact is not None
    assert contract.head_meta is not None
