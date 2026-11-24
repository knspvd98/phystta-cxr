from phystta.safety.policy_harm import sample_harm_multilabel, expected_harm_multilabel, policy_regret, regret_by_shift, regret_decomposition
import numpy as np


def test_expected_harm_and_regret():
    probs_no = np.array([[0.9, 0.1], [0.2, 0.8], [0.6, 0.3]])
    probs_er = np.array([[0.8, 0.1], [0.1, 0.7], [0.55, 0.35]])
    labels = np.array([[1, 0], [0, 1], [1, 0]], dtype=float)
    h0 = expected_harm_multilabel(probs_no, labels, cost_fn=10.0, cost_fp=1.0)
    h1 = expected_harm_multilabel(probs_er, labels, cost_fn=10.0, cost_fp=1.0)
    reg = policy_regret("apply_eraser", h0, h1, abstain_cost=1.0)
    assert np.isfinite(h0) and np.isfinite(h1)
    assert "policy_regret" in reg
    summary = regret_by_shift(
        [{"decision": "apply_eraser"}, {"decision": "no_eraser"}, {"decision": "abstain"}],
        [h0, h0, h0],
        [h1, h1, h1],
        abstain_cost=1.0,
    )
    assert summary["n"] == 3
    decomp = regret_decomposition(
        [
            {"decision": "apply_eraser", "shift": "covariate"},
            {"decision": "no_eraser", "shift": "prevalence"},
            {"decision": "abstain", "shift": "mixed"},
        ],
        [h0, h0, h0],
        [h1, h1, h1],
        abstain_cost=1.0,
    )
    assert "by_shift" in decomp and "by_action" in decomp
    assert "covariate" in decomp["by_shift"]


def test_sample_harm_multilabel_matches_expected_mean():
    probs = np.array([[0.9, 0.1], [0.2, 0.8], [0.6, 0.3]])
    labels = np.array([[1, 0], [0, 1], [1, 0]], dtype=float)
    thresholds = np.array([0.7, 0.4], dtype=float)
    per_sample = sample_harm_multilabel(
        probs,
        labels,
        thresholds=thresholds,
        cost_fn=10.0,
        cost_fp=1.0,
        critical_idx=[0, 1],
    )
    mean_harm = expected_harm_multilabel(
        probs,
        labels,
        thresholds=thresholds,
        cost_fn=10.0,
        cost_fp=1.0,
        critical_idx=[0, 1],
    )
    assert per_sample.shape == (3,)
    assert np.isfinite(per_sample).all()
    assert np.isclose(float(np.mean(per_sample)), mean_harm)


def test_policy_regret_accepts_eraser_alias():
    h0 = 0.4
    h1 = 0.2
    reg_apply = policy_regret("apply_eraser", h0, h1, abstain_cost=1.0)
    reg_alias = policy_regret("eraser", h0, h1, abstain_cost=1.0)
    assert reg_apply == reg_alias


def test_classwise_costs_change_harm_and_none_preserves_scalar_behavior():
    probs = np.array([[0.2, 0.9], [0.8, 0.1]], dtype=float)
    labels = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=float)
    thresholds = np.array([0.5, 0.5], dtype=float)

    scalar_sample = sample_harm_multilabel(
        probs,
        labels,
        thresholds=thresholds,
        cost_fn=10.0,
        cost_fp=1.0,
        class_cost_fn=None,
        class_cost_fp=None,
    )
    legacy_sample = sample_harm_multilabel(
        probs,
        labels,
        thresholds=thresholds,
        cost_fn=10.0,
        cost_fp=1.0,
    )
    assert np.allclose(scalar_sample, legacy_sample)

    weighted_sample = sample_harm_multilabel(
        probs,
        labels,
        thresholds=thresholds,
        cost_fn=10.0,
        cost_fp=1.0,
        class_cost_fn=[20.0, 5.0],
        class_cost_fp=[0.5, 3.0],
    )
    weighted_expected = expected_harm_multilabel(
        probs,
        labels,
        thresholds=thresholds,
        cost_fn=10.0,
        cost_fp=1.0,
        class_cost_fn=[20.0, 5.0],
        class_cost_fp=[0.5, 3.0],
    )
    assert not np.allclose(weighted_sample, scalar_sample)
    assert np.isclose(float(np.mean(weighted_sample)), weighted_expected)
