import numpy as np
import torch

from phystta.shifts.manager import ShiftManager


def _make_batch(seed: int = 0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.05, 0.95, size=(3, 1, 32, 32)).astype(np.float32)
    return torch.from_numpy(x)


def test_apply_batch_is_deterministic_with_sample_ids():
    manager = ShiftManager(seed=123)
    x = _make_batch(seed=11)
    sample_ids = ["img_a", "img_b", "img_c"]

    y1 = manager.apply_batch(x, "poisson", 0.4, sample_ids=sample_ids)
    y2 = manager.apply_batch(x, "poisson", 0.4, sample_ids=sample_ids)
    assert torch.allclose(y1, y2)

    # Order invariance: each sample gets a deterministic seed from sample_id and shift spec.
    perm = [2, 0, 1]
    x_perm = x[perm]
    ids_perm = [sample_ids[i] for i in perm]
    y_perm = manager.apply_batch(x_perm, "poisson", 0.4, sample_ids=ids_perm)
    inv = np.argsort(perm)
    assert torch.allclose(y1, y_perm[inv])


def test_apply_batch_scenario_is_deterministic_with_sample_ids():
    manager = ShiftManager(seed=321)
    x = _make_batch(seed=7)
    sample_ids = ["img_x", "img_y", "img_z"]
    scenario = {"scatter": 0.3, "heel": 0.2, "poisson": 0.4}

    y1, meta1 = manager.apply_batch_scenario(x, scenario, sample_ids=sample_ids)
    y2, meta2 = manager.apply_batch_scenario(x, scenario, sample_ids=sample_ids)
    assert torch.allclose(y1, y2)
    seeds1 = [m["seed"] for m in meta1["per_sample"]]
    seeds2 = [m["seed"] for m in meta2["per_sample"]]
    assert seeds1 == seeds2
