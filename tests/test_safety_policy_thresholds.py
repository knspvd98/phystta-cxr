import numpy as np

from phystta.config.schema import SafetyPolicyConfig
from phystta.safety.policy import SafetyPolicy


def test_apply_threshold_returns_row_mask_for_multilabel_probs():
    cfg = SafetyPolicyConfig(coverage_budget=0.5, conformal=False, shared_threshold=True)
    pol = SafetyPolicy(cfg)
    probs_no = np.array(
        [
            [0.9, 0.1, 0.3],
            [0.7, 0.6, 0.2],
            [0.2, 0.8, 0.9],
            [0.4, 0.4, 0.4],
        ],
        dtype=float,
    )
    probs_er = probs_no.copy()
    labels = np.array(
        [
            [1.0, 0.0, 0.0],
            [1.0, 1.0, -1.0],
            [0.0, 1.0, 1.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    pol.calibrate(probs_no, probs_er, labels, coverage=0.5, critical_idx=[0, 1, 2])
    mask = pol.apply_threshold(probs_no, "no_eraser")
    assert mask.ndim == 1
    assert mask.shape[0] == probs_no.shape[0]
    assert mask.dtype == bool


def test_conformal_threshold_calibration_runs_on_multilabel():
    cfg = SafetyPolicyConfig(coverage_budget=0.8, conformal=True, risk_target=0.2, conformal_alpha=0.1)
    pol = SafetyPolicy(cfg)
    rng = np.random.default_rng(1)
    probs_no = rng.uniform(0.05, 0.95, size=(64, 3))
    probs_er = np.clip(probs_no + rng.normal(0.0, 0.05, size=(64, 3)), 0.01, 0.99)
    labels = rng.integers(0, 2, size=(64, 3)).astype(float)
    labels[:8, 1] = -1.0
    pol.calibrate(probs_no, probs_er, labels, coverage=0.8, critical_idx=[0, 1, 2])
    assert np.isfinite(pol.thresholds["no_eraser"])
    assert np.isfinite(pol.thresholds["eraser"])


def test_calibrate_uses_clinical_class_thresholds_for_harm():
    cfg = SafetyPolicyConfig(coverage_budget=1.0, conformal=False, shared_threshold=True, cost_fn=10.0, cost_fp=1.0)
    pol = SafetyPolicy(cfg)
    probs_no = np.array(
        [
            [0.7, 0.3],
            [0.6, 0.4],
            [0.2, 0.8],
            [0.1, 0.9],
        ],
        dtype=float,
    )
    probs_er = probs_no.copy()
    labels = np.array(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ],
        dtype=float,
    )
    class_th = np.array([0.8, 0.2], dtype=float)
    pol.calibrate(
        probs_no,
        probs_er,
        labels,
        coverage=1.0,
        critical_idx=[0, 1],
        class_thresholds=class_th,
    )
    assert np.allclose(pol.class_thresholds, class_th)

    # With class_th[0]=0.8 positives in class 0 are mostly FN -> higher harm than default 0.5.
    harm_custom = pol.harm["no_eraser"]
    harm_default = pol._expected_harm(probs_no, labels, critical_idx=[0, 1], thresholds=np.array([0.5, 0.5]))
    assert np.isfinite(harm_custom)
    assert harm_custom > harm_default


def test_apply_eraser_action_alias_uses_thresholds():
    cfg = SafetyPolicyConfig(coverage_budget=0.5, conformal=False, shared_threshold=True)
    pol = SafetyPolicy(cfg)
    probs_no = np.array(
        [
            [0.9, 0.2],
            [0.8, 0.1],
            [0.7, 0.3],
            [0.6, 0.4],
        ],
        dtype=float,
    )
    probs_er = np.array(
        [
            [0.9, 0.2],
            [0.4, 0.1],
            [0.3, 0.3],
            [0.2, 0.4],
        ],
        dtype=float,
    )
    labels = np.array(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=float,
    )
    pol.calibrate(probs_no, probs_er, labels, coverage=0.5, critical_idx=[0, 1], class_thresholds=np.array([0.5, 0.5]))
    mask_alias = pol.apply_threshold(probs_er, "apply_eraser")
    mask_canonical = pol.apply_threshold(probs_er, "eraser")
    assert np.array_equal(mask_alias, mask_canonical)
    assert 0 < int(mask_alias.sum()) < len(mask_alias)
