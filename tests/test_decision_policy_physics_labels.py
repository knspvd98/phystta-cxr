from phystta.safety.decision import DecisionPolicy


def test_decision_policy_treats_physics_labels_as_covariate_actions():
    p = DecisionPolicy(covariate_labels=["heel", "scatter"], apply_on_covariate=True)
    info = {
        "label": "heel",
        "uncertainty": 0.05,
        "p_none": 0.05,
        "p_covariate": 0.10,
        "p_prevalence": 0.05,
        "p_mixed": 0.05,
    }
    tr = p.decide(info, harm_no=0.8, harm_er=0.4)
    assert tr.action == "apply_eraser"
    assert tr.reason == "lower_expected_harm"
