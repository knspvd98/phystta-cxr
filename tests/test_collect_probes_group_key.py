import numpy as np
import torch

from phystta.intervention.probes import collect_probes


class _DummyShiftManager:
    def apply_batch(self, x, shift_type, intensity, sample_ids=None):
        return x + float(intensity) * 0.0

    def apply_batch_scenario(self, x, intens, sample_ids=None):
        return x, {"summary": {"destructive_count": 0}}


class _DummyEncoder:
    def eval(self):
        return self

    def encode(self, x):
        # [N, C, H, W] -> [N, 2]
        flat = x.reshape(x.shape[0], -1)
        mean = flat.mean(dim=1, keepdim=True)
        std = flat.std(dim=1, keepdim=True)
        return torch.cat([mean, std], dim=1)


def test_collect_probes_uses_group_key_when_present():
    batch = {
        "image": torch.rand(2, 1, 8, 8),
        "labels": torch.tensor([[1.0], [0.0]], dtype=torch.float32),
        "patient_id": ["p1", "p1"],  # intentionally colliding raw patient ids
        "global_patient_id": ["srcA:p1", "srcB:p1"],
        "study_id": ["s1", "s2"],
        "image_path": ["a.png", "b.png"],
    }
    loader = [batch]
    out = collect_probes(
        loader=loader,
        shift_manager=_DummyShiftManager(),
        shift_type="heel",
        intensity_grid=[0.2],
        encoder=_DummyEncoder(),
        max_images=None,
        device="cpu",
        seed=7,
        token_mode="none",
        group_key="global_patient_id",
    )
    assert out["features_clean"].shape[0] == 2
    assert out["features_shifted"].shape[0] == 2
    assert out["group_id"].tolist() == ["srcA:p1", "srcB:p1"]
    # patient_id is also promoted to resolved group key for split-safe fitting
    assert out["patient_id"].tolist() == ["srcA:p1", "srcB:p1"]


def test_collect_probes_falls_back_to_patient_id():
    batch = {
        "image": torch.rand(2, 1, 8, 8),
        "labels": torch.tensor([[1.0], [0.0]], dtype=torch.float32),
        "patient_id": ["p1", "p2"],
        "study_id": ["s1", "s2"],
        "image_path": ["a.png", "b.png"],
    }
    loader = [batch]
    out = collect_probes(
        loader=loader,
        shift_manager=_DummyShiftManager(),
        shift_type="heel",
        intensity_grid=[0.2],
        encoder=_DummyEncoder(),
        max_images=None,
        device="cpu",
        seed=7,
        token_mode="none",
        group_key=None,
    )
    assert np.array_equal(out["group_id"], out["patient_id"])
