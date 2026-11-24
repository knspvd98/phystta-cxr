import numpy as np
import torch
import torch.nn.functional as F

from scripts.fit_eraser import _fit_steering_eraser


def test_steering_intensity_does_not_use_bce(monkeypatch):
    z_clean = np.random.randn(32, 8).astype(np.float32)
    intens = np.random.uniform(0.0, 3.0, size=(32,)).astype(np.float32)
    z_shift = z_clean + 0.1 * np.random.randn(32, 8).astype(np.float32)

    def _raise(*args, **kwargs):
        raise RuntimeError("BCE should not be used for continuous intensity target")

    monkeypatch.setattr(F, "binary_cross_entropy_with_logits", _raise)
    eraser = _fit_steering_eraser(
        z_clean,
        z_shift,
        k=3,
        hidden=16,
        steps=5,
        lr=1e-3,
        eps=0.01,
        seed=0,
        head=None,
        probe_target="intensity",
        intensities=intens,
    )
    x = torch.randn(4, 8)
    y = eraser(x)
    assert y.shape == x.shape
    assert eraser.meta.get("probe_loss") == "regression"
