from __future__ import annotations

import torch

from phystta.intervention.eraser import LinearSubspaceEraser, eraser_fingerprint_mismatches
from phystta.shifts.realism import (
    RealismCalibrator,
    RealismMapping,
    calibrator_fingerprint_mismatches,
)


def test_realism_calibrator_roundtrip_keeps_encoder_fingerprint():
    calib = RealismCalibrator(distance_metric="l2")
    calib.encoder_fingerprint = {
        "encoder_source": "hf_clip_or_vit",
        "model_id": "openai/clip-vit-base-patch32",
        "revision": "abc123",
        "preprocess_signature": "sig",
        "input_adapter": {"repeat_to_3": True},
    }
    calib.mappings["heel"] = RealismMapping(
        shift_type="heel",
        realism_levels=[0.0, 0.5, 1.0],
        intensities=[0.0, 0.2, 0.4],
    )
    loaded = RealismCalibrator.from_dict(calib.to_dict())
    assert loaded.encoder_fingerprint == calib.encoder_fingerprint
    assert "heel" in loaded.mappings


def test_realism_calibrator_fingerprint_mismatch_detection():
    calib = RealismCalibrator()
    calib.encoder_fingerprint = {
        "encoder_source": "hf_clip_or_vit",
        "model_id": "openai/clip-vit-base-patch32",
        "revision": "abc123",
        "preprocess_signature": "sigA",
        "input_adapter": {"repeat_to_3": True},
    }
    expected = {
        "encoder_source": "hf_clip_or_vit",
        "model_id": "openai/clip-vit-base-patch32",
        "revision": "abc123",
        "preprocess_signature": "sigB",
        "input_adapter": {"repeat_to_3": True},
    }
    mismatches = calibrator_fingerprint_mismatches(calib, expected)
    assert any("preprocess_signature" in m for m in mismatches)


def test_eraser_fingerprint_accepts_top_level_preprocess_signature():
    eraser = LinearSubspaceEraser(
        torch.zeros((4, 1), dtype=torch.float32),
        meta={
            "encoder_fingerprint": {
                "encoder_source": "baseline_cnn",
                "architecture": "densenet121",
                "model_id": None,
                "encoder_name": None,
                "revision": None,
                "pretrained": False,
                "use_medical_pretrain": False,
                "effective_medical_pretrain": False,
                "encoder_path": None,
                "input_adapter": {"repeat_to_3": True},
            },
            "preprocess_signature": {"dataset_output": "0_1", "image_size": 224},
        },
    )
    expected = {
        "encoder_source": "baseline_cnn",
        "architecture": "densenet121",
        "model_id": None,
        "encoder_name": None,
        "revision": None,
        "pretrained": False,
        "use_medical_pretrain": False,
        "effective_medical_pretrain": False,
        "encoder_path": None,
        "input_adapter": {"repeat_to_3": True},
        "preprocess_signature": {"dataset_output": "0_1", "image_size": 224},
    }
    assert eraser_fingerprint_mismatches(eraser, expected) == []
