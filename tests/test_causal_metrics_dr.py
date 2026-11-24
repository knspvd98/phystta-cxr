import numpy as np

from phystta.metrics.causal import aipw_from_potential_outcomes


def test_aipw_from_potential_outcomes_recovers_direction():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(128, 8))
    y0 = rng.normal(loc=0.2, scale=0.3, size=128)
    y1 = y0 + 0.15  # positive treatment effect
    groups = np.array([f"p{i//2}" for i in range(128)])
    out = aipw_from_potential_outcomes(X, y0, y1, groups=groups, assign_prob=0.5, n_splits=3, seed=42)
    assert np.isfinite(out["ate"])
    assert np.isfinite(out["ate_true"])
    assert out["ate_true"] > 0
