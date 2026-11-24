from __future__ import annotations

from types import SimpleNamespace

import pytest

from phystta.safety.decision import (
    DecisionPolicy,
    ExpectedRiskDecisionPolicy,
    build_decision_policy_from_config,
    decision_policy_contract_payload,
)


def test_legacy_decision_policy_keeps_high_uncertainty_abstain():
    policy = DecisionPolicy(
        uncertainty_abstain_threshold=0.35,
        mixed_abstain_threshold=0.40,
        prevalence_action="recalibrate",
        apply_on_covariate=True,
    )
    trace = policy.decide(
        {
            "label": "covariate",
            "p_none": 0.1,
            "p_covariate": 0.8,
            "p_prevalence": 0.1,
            "p_mixed": 0.0,
            "uncertainty": 0.9,
        },
        harm_no=0.5,
        harm_er=0.1,
    )
    assert trace.action == "abstain"
    assert trace.reason == "high_uncertainty"
    assert trace.policy_variant == "legacy_v1"


def test_expected_risk_policy_can_choose_apply_eraser_under_high_uncertainty():
    policy = ExpectedRiskDecisionPolicy(
        abstain_cost=0.25,
        uncertainty_lambda=0.1,
        route_gamma=0.1,
    )
    trace = policy.decide(
        {
            "label": "covariate",
            "p_none": 0.1,
            "p_covariate": 1.0,
            "p_prevalence": 0.0,
            "p_mixed": 0.0,
            "uncertainty": 0.9,
        },
        harm_no=0.5,
        harm_er=0.1,
        harm_rec=float("nan"),
    )
    assert trace.action == "apply_eraser"
    assert trace.reason == "min_expected_risk"
    assert trace.policy_variant == "expected_risk_v2"


def test_expected_risk_policy_can_choose_recalibrate():
    policy = ExpectedRiskDecisionPolicy(
        abstain_cost=0.25,
        uncertainty_lambda=0.1,
        route_gamma=0.1,
    )
    trace = policy.decide(
        {
            "label": "prevalence",
            "p_none": 0.1,
            "p_covariate": 0.0,
            "p_prevalence": 1.0,
            "p_mixed": 0.0,
            "uncertainty": 0.2,
        },
        harm_no=0.4,
        harm_er=0.3,
        harm_rec=0.05,
    )
    assert trace.action == "recalibrate"
    assert trace.reason == "min_expected_risk"
    assert trace.policy_variant == "expected_risk_v2"


def test_build_decision_policy_from_config_variants_and_contract_payload():
    disabled_cfg = SimpleNamespace(
        use_decision_policy=False,
        policy_variant="legacy_v1",
        apply_on_covariate=False,
        uncertainty_abstain_threshold=0.35,
        mixed_abstain_threshold=0.40,
        prevalence_action="recalibrate",
        abstain_cost=0.25,
        uncertainty_lambda=0.0,
        route_gamma=0.0,
    )
    policy, variant = build_decision_policy_from_config(disabled_cfg)
    assert policy is None
    assert variant == "legacy_policy_disabled"

    legacy_cfg = SimpleNamespace(
        use_decision_policy=True,
        policy_variant="legacy_v1",
        apply_on_covariate=True,
        uncertainty_abstain_threshold=0.35,
        mixed_abstain_threshold=0.40,
        prevalence_action="recalibrate",
        abstain_cost=0.25,
        uncertainty_lambda=0.0,
        route_gamma=0.0,
    )
    legacy_policy, legacy_variant = build_decision_policy_from_config(legacy_cfg, covariate_labels=["heel"])
    assert isinstance(legacy_policy, DecisionPolicy)
    assert legacy_variant == "legacy_v1"

    v2_cfg = SimpleNamespace(
        use_decision_policy=True,
        policy_variant="expected_risk_v2",
        apply_on_covariate=True,
        uncertainty_abstain_threshold=0.35,
        mixed_abstain_threshold=0.40,
        prevalence_action="recalibrate",
        abstain_cost=0.25,
        uncertainty_lambda=0.1,
        route_gamma=0.1,
    )
    v2_policy, v2_variant = build_decision_policy_from_config(v2_cfg)
    assert isinstance(v2_policy, ExpectedRiskDecisionPolicy)
    assert v2_variant == "expected_risk_v2"
    payload = decision_policy_contract_payload(v2_policy, v2_variant)
    assert payload["decision_policy_variant"] == "expected_risk_v2"
    assert payload["decision_policy_active"] is True
    assert payload["decision_policy_class"] == "ExpectedRiskDecisionPolicy"
    assert payload["uncertainty_lambda"] == pytest.approx(0.1)
    assert payload["route_gamma"] == pytest.approx(0.1)
