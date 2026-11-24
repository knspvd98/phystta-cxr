from phystta.safety import DecisionPolicy, misidentification_cost


def test_decision_policy_covariate_apply_eraser():
    p = DecisionPolicy()
    info = {"label": "covariate", "uncertainty": 0.1, "p_covariate": 0.8, "p_none": 0.1, "p_prevalence": 0.05, "p_mixed": 0.05}
    tr = p.decide(info, harm_no=0.3, harm_er=0.2)
    assert tr.action == "apply_eraser"


def test_decision_policy_mixed_abstain():
    p = DecisionPolicy(mixed_abstain_threshold=0.3)
    info = {"label": "mixed", "uncertainty": 0.1, "p_covariate": 0.2, "p_none": 0.1, "p_prevalence": 0.2, "p_mixed": 0.6}
    tr = p.decide(info, harm_no=0.2, harm_er=0.1)
    assert tr.action == "abstain"


def test_misidentification_cost_asymmetric():
    labels = ["none", "covariate", "prevalence", "mixed"]
    cm = {
        "none": {"none": 0, "covariate": 1, "prevalence": 1, "mixed": 2},
        "covariate": {"none": 1, "covariate": 0, "prevalence": 2, "mixed": 3},
        "prevalence": {"none": 1, "covariate": 2, "prevalence": 0, "mixed": 2},
        "mixed": {"none": 2, "covariate": 5, "prevalence": 2, "mixed": 0},
    }
    cost = misidentification_cost(["mixed", "none"], ["covariate", "none"], labels, cm)
    assert cost > 0
