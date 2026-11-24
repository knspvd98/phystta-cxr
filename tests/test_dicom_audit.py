import numpy as np

from phystta.data.dicom_audit import audit_site_leakage


def test_dicom_audit_runs():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 7)).astype(np.float32)
    y = np.array(["m1"] * 100 + ["m2"] * 100)
    y[:20] = "unknown"
    groups = np.array([f"p{i}" for i in range(200)])
    rep = audit_site_leakage(X, y, groups=groups, seed=0)
    assert "site_leakage_acc" in rep
    assert rep["n"] > 0
