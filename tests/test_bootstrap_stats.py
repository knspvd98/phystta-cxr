import numpy as np

from phystta.utils import cluster_bootstrap_mean


def test_cluster_bootstrap_mean_is_deterministic_for_seed():
    values = np.array([1.0, 1.2, 2.0, 2.2, 3.0, 3.2], dtype=float)
    groups = np.array(["a", "a", "b", "b", "c", "c"], dtype=object)
    out1 = cluster_bootstrap_mean(values, groups, n_boot=100, seed=7)
    out2 = cluster_bootstrap_mean(values, groups, n_boot=100, seed=7)
    assert out1 == out2


def test_cluster_bootstrap_mean_ci_contains_observed_mean():
    values = np.array([0.0, 0.1, 0.8, 0.9, 0.2, 0.3], dtype=float)
    groups = np.array(["p1", "p1", "p2", "p2", "p3", "p3"], dtype=object)
    out = cluster_bootstrap_mean(values, groups, n_boot=120, seed=11)
    assert np.isfinite(out["mean"])
    assert np.isfinite(out["ci_low"])
    assert np.isfinite(out["ci_high"])
    assert out["ci_low"] <= out["mean"] <= out["ci_high"]
    assert out["n"] == float(len(values))
    assert out["n_clusters"] == 3.0
