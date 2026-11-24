import numpy as np

from phystta.metrics.selective import selective_risk_multilabel
from phystta.metrics.risk import risk_coverage_curve_multilabel


def test_selective_risk_multilabel_ignores_unknown_labels():
    probs = np.array(
        [
            [0.90, 0.20],
            [0.60, 0.40],
            [0.70, 0.20],
        ],
        dtype=float,
    )
    labels = np.array(
        [
            [1.0, 0.0],
            [1.0, -1.0],  # second label is unknown and must be ignored
            [0.0, 1.0],
        ],
        dtype=float,
    )
    risk = selective_risk_multilabel(probs, labels, coverage=2.0 / 3.0, threshold=0.5, cost_fn=10.0, cost_fp=1.0)
    assert np.isfinite(risk)
    assert risk > 0.0


def test_risk_coverage_curve_multilabel_shapes_and_bounds():
    rng = np.random.default_rng(0)
    probs = rng.uniform(0.0, 1.0, size=(32, 4))
    labels = rng.integers(0, 2, size=(32, 4)).astype(float)
    labels[:5, 2] = -1.0
    curve = risk_coverage_curve_multilabel(probs, labels, points=17, cost_fn=10.0, cost_fp=1.0)
    assert curve["risk"].shape == (17,)
    assert curve["coverage"].shape == (17,)
    assert np.all(curve["coverage"] >= 0.0)
    assert np.all(curve["coverage"] <= 1.0)
