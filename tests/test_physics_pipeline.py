import numpy as np
import torch

from phystta.shifts.physics import (
    HeelEffectShift,
    ScatterShift,
    MTFGlareShift,
    PoissonNoiseShift,
    ExposureShift,
    ExposurePoissonCoupledShift,
    MotionBlurShift,
)


def _psd_energy(x: torch.Tensor):
    # x: [1,H,W]
    x = x.squeeze(0).numpy()
    f = np.fft.fftshift(np.fft.fft2(x))
    psd = np.abs(f) ** 2
    return psd


def _band_energy(psd, low: bool = True):
    h, w = psd.shape
    cy, cx = h // 2, w // 2
    r = min(h, w) // 8
    yy, xx = np.ogrid[:h, :w]
    dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    if low:
        mask = dist <= r
    else:
        mask = dist >= 3 * r
    return psd[mask].mean()


def test_physics_no_nan():
    img = torch.rand(1, 128, 128)
    for shift in [
        HeelEffectShift(),
        ScatterShift(),
        MTFGlareShift(),
        ExposureShift(),
        PoissonNoiseShift(),
        ExposurePoissonCoupledShift(),
        MotionBlurShift(),
    ]:
        y = shift.apply(img, 0.6)
        assert torch.isfinite(y).all()
        assert y.min() >= 0 and y.max() <= 1


def test_mtf_monotonic_highfreq():
    img = torch.zeros(1, 128, 128)
    img[:, :, 64:] = 1.0
    shift = MTFGlareShift()
    y1 = shift.apply(img, 0.1)
    y2 = shift.apply(img, 0.6)
    psd1 = _psd_energy(y1)
    psd2 = _psd_energy(y2)
    assert _band_energy(psd2, low=False) < _band_energy(psd1, low=False)


def test_scatter_increases_lowfreq():
    img = torch.zeros(1, 128, 128)
    img[:, :, 64:] = 1.0
    shift = ScatterShift()
    y1 = shift.apply(img, 0.1)
    y2 = shift.apply(img, 0.6)
    psd1 = _psd_energy(y1)
    psd2 = _psd_energy(y2)
    assert _band_energy(psd2, low=True) >= _band_energy(psd1, low=True)


def test_dose_noise_variance_increases():
    img = torch.full((1, 128, 128), 0.5)
    shift = PoissonNoiseShift()
    y1 = shift.apply(img, 0.1)
    y2 = shift.apply(img, 0.8)
    var1 = y1.var().item()
    var2 = y2.var().item()
    assert var2 >= var1


def test_motion_blur_reduces_high_frequency_energy():
    img = torch.zeros(1, 128, 128)
    img[:, :, ::4] = 1.0
    shift = MotionBlurShift()
    y1 = shift.apply(img, 0.1)
    y2 = shift.apply(img, 0.8)
    psd1 = _psd_energy(y1)
    psd2 = _psd_energy(y2)
    assert _band_energy(psd2, low=False) < _band_energy(psd1, low=False)


def test_coupled_exposure_poisson_increases_noise_under_lower_dose():
    img = torch.full((1, 128, 128), 0.5)
    shift = ExposurePoissonCoupledShift()
    y1 = shift.apply(img, 0.1)
    y2 = shift.apply(img, 0.8)
    assert y2.var().item() >= y1.var().item()
