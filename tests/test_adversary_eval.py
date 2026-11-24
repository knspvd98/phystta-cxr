import numpy as np

from phystta.intervention.adversary_eval import (
    ProbeSuiteConfig,
    build_presence_task,
    build_intensity_task,
    evaluate_probe_suite,
)


def test_adversary_eval_presence():
    rng = np.random.default_rng(0)
    z_clean = rng.normal(size=(120, 16)).astype(np.float32)
    z_shift = z_clean + 0.4 * rng.normal(size=(120, 16)).astype(np.float32)
    groups = np.array([f"p{i}" for i in range(120)])
    X, y, g = build_presence_task(z_clean, z_shift, groups=groups)
    out = evaluate_probe_suite(X, y, groups=g, target="presence", cfg=ProbeSuiteConfig(seed=0, epochs=50, patience=5))
    assert "linear_auc" in out and "mlp_auc" in out and "kernel_auc" in out


def test_adversary_eval_intensity():
    rng = np.random.default_rng(1)
    z_clean = rng.normal(size=(100, 12)).astype(np.float32)
    intens = rng.uniform(0.0, 1.0, size=(100,)).astype(np.float32)
    z_shift = z_clean + intens[:, None] * rng.normal(size=(100, 12)).astype(np.float32)
    groups = np.array([f"p{i}" for i in range(100)])
    X, y, g = build_intensity_task(z_clean, z_shift, intensities=intens, groups=groups)
    out = evaluate_probe_suite(X, y, groups=g, target="intensity", cfg=ProbeSuiteConfig(seed=1, epochs=50, patience=5))
    assert "linear_mae" in out and "mlp_mae" in out and "kernel_mae" in out
