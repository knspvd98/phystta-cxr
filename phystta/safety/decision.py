from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class DecisionTrace:
    action: str
    reason: str
    p_none: float
    p_covariate: float
    p_prevalence: float
    p_mixed: float
    uncertainty: float
    policy_variant: str = "legacy_v1"


class DecisionPolicy:
    """Decision-theoretic policy layer on top of shift-ID probabilities."""

    def __init__(
        self,
        uncertainty_abstain_threshold: float = 0.35,
        mixed_abstain_threshold: float = 0.40,
        prevalence_action: str = "recalibrate",
        apply_on_covariate: bool = True,
        covariate_labels: Optional[List[str]] = None,
    ):
        self.uncertainty_abstain_threshold = float(uncertainty_abstain_threshold)
        self.mixed_abstain_threshold = float(mixed_abstain_threshold)
        self.prevalence_action = prevalence_action
        self.apply_on_covariate = bool(apply_on_covariate)
        labels = covariate_labels or []
        self.covariate_labels = {str(v) for v in labels}
        self.covariate_labels.add("covariate")

    def decide(
        self,
        shift_info: Dict[str, float | str],
        harm_no: float,
        harm_er: float,
        harm_rec: float = float("nan"),
    ) -> DecisionTrace:
        p_none = float(shift_info.get("p_none", 0.0))
        p_cov = float(shift_info.get("p_covariate", 0.0))
        p_prev = float(shift_info.get("p_prevalence", 0.0))
        p_mix = float(shift_info.get("p_mixed", 0.0))
        unc = float(shift_info.get("uncertainty", 1.0))

        if unc >= self.uncertainty_abstain_threshold:
            return DecisionTrace("abstain", "high_uncertainty", p_none, p_cov, p_prev, p_mix, unc)
        if p_mix >= self.mixed_abstain_threshold:
            return DecisionTrace("abstain", "mixed_shift_risk", p_none, p_cov, p_prev, p_mix, unc)

        label = str(shift_info.get("label", "unknown"))
        if label == "prevalence":
            return DecisionTrace(self.prevalence_action, "prevalence_shift", p_none, p_cov, p_prev, p_mix, unc)
        if label in self.covariate_labels:
            if not self.apply_on_covariate:
                return DecisionTrace("no_eraser", "covariate_disabled", p_none, p_cov, p_prev, p_mix, unc)
            if np.isfinite(harm_no) and np.isfinite(harm_er):
                if harm_er <= harm_no:
                    return DecisionTrace("apply_eraser", "lower_expected_harm", p_none, p_cov, p_prev, p_mix, unc)
                return DecisionTrace("no_eraser", "eraser_harm_higher", p_none, p_cov, p_prev, p_mix, unc)
            return DecisionTrace("no_eraser", "harm_unavailable", p_none, p_cov, p_prev, p_mix, unc)
        if label == "none":
            return DecisionTrace("no_eraser", "no_shift", p_none, p_cov, p_prev, p_mix, unc)
        return DecisionTrace("abstain", "unknown_shift", p_none, p_cov, p_prev, p_mix, unc)


def _cfg_get(cfg: Any, name: str, default: Any = None) -> Any:
    if isinstance(cfg, dict):
        return cfg.get(name, default)
    return getattr(cfg, name, default)


def _route_penalty(prob_value: float, route_gamma: float) -> float:
    if not np.isfinite(prob_value):
        return 0.0
    return float(route_gamma * (1.0 - prob_value))


class ExpectedRiskDecisionPolicy:
    def __init__(
        self,
        abstain_cost: float,
        uncertainty_lambda: float = 0.0,
        route_gamma: float = 0.0,
        covariate_labels: Optional[List[str]] = None,
    ):
        self.abstain_cost = float(abstain_cost)
        self.uncertainty_lambda = float(uncertainty_lambda)
        self.route_gamma = float(route_gamma)
        labels = covariate_labels or []
        self.covariate_labels = {str(v) for v in labels}
        self.covariate_labels.add("covariate")

    def decide(
        self,
        shift_info: Dict[str, float | str],
        harm_no: float,
        harm_er: float,
        harm_rec: float = float("nan"),
    ) -> DecisionTrace:
        p_none = float(shift_info.get("p_none", 0.0))
        p_cov = float(shift_info.get("p_covariate", 0.0))
        p_prev = float(shift_info.get("p_prevalence", 0.0))
        p_mix = float(shift_info.get("p_mixed", 0.0))
        unc = float(shift_info.get("uncertainty", 1.0))
        premium = self.uncertainty_lambda * unc
        action_costs = {
            "abstain": float(self.abstain_cost),
            "no_eraser": float(harm_no + premium) if np.isfinite(harm_no) else float("nan"),
            "apply_eraser": (
                float(harm_er + premium + _route_penalty(p_cov, self.route_gamma))
                if np.isfinite(harm_er)
                else float("nan")
            ),
            "recalibrate": (
                float(harm_rec + premium + _route_penalty(p_prev, self.route_gamma))
                if np.isfinite(harm_rec)
                else float("nan")
            ),
        }
        non_abstain_finite = any(np.isfinite(action_costs[name]) for name in ["no_eraser", "apply_eraser", "recalibrate"])
        if not non_abstain_finite:
            return DecisionTrace(
                "abstain",
                "no_finite_action_cost",
                p_none,
                p_cov,
                p_prev,
                p_mix,
                unc,
                policy_variant="expected_risk_v2",
            )
        best_action = "abstain"
        best_cost = float(action_costs["abstain"])
        for action in ["no_eraser", "apply_eraser", "recalibrate"]:
            value = float(action_costs[action])
            if not np.isfinite(value):
                continue
            if value < best_cost:
                best_action = action
                best_cost = value
        return DecisionTrace(
            best_action,
            "min_expected_risk",
            p_none,
            p_cov,
            p_prev,
            p_mix,
            unc,
            policy_variant="expected_risk_v2",
        )


def build_decision_policy_from_config(
    cfg: Any,
    covariate_labels: Optional[List[str]] = None,
) -> tuple[Optional[object], str]:
    policy_cfg = _cfg_get(cfg, "safety_policy", cfg)
    use_decision_policy = bool(_cfg_get(policy_cfg, "use_decision_policy", False))
    policy_variant = str(_cfg_get(policy_cfg, "policy_variant", "legacy_v1"))
    if not use_decision_policy:
        if policy_variant == "expected_risk_v2":
            raise RuntimeError("safety_policy.policy_variant='expected_risk_v2' requires use_decision_policy=true.")
        return None, "legacy_policy_disabled"
    if policy_variant == "legacy_v1":
        return (
            DecisionPolicy(
                uncertainty_abstain_threshold=float(_cfg_get(policy_cfg, "uncertainty_abstain_threshold", 0.35)),
                mixed_abstain_threshold=float(_cfg_get(policy_cfg, "mixed_abstain_threshold", 0.40)),
                prevalence_action=str(_cfg_get(policy_cfg, "prevalence_action", "recalibrate")),
                apply_on_covariate=bool(_cfg_get(policy_cfg, "apply_on_covariate", True)),
                covariate_labels=covariate_labels,
            ),
            "legacy_v1",
        )
    if policy_variant == "expected_risk_v2":
        return (
            ExpectedRiskDecisionPolicy(
                abstain_cost=float(_cfg_get(policy_cfg, "abstain_cost", 1.0)),
                uncertainty_lambda=float(_cfg_get(policy_cfg, "uncertainty_lambda", 0.0)),
                route_gamma=float(_cfg_get(policy_cfg, "route_gamma", 0.0)),
                covariate_labels=covariate_labels,
            ),
            "expected_risk_v2",
        )
    raise RuntimeError(f"Unknown safety_policy.policy_variant={policy_variant!r}.")


def decision_policy_contract_payload(decision_policy: Optional[object], variant: str) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "decision_policy_variant": str(variant),
        "decision_policy_active": bool(decision_policy is not None),
        "decision_policy_class": None if decision_policy is None else decision_policy.__class__.__name__,
        "decision_path_source": "decision_policy" if decision_policy is not None else "legacy_policy",
    }
    if variant == "expected_risk_v2" and decision_policy is not None:
        payload["abstain_cost"] = float(getattr(decision_policy, "abstain_cost"))
        payload["uncertainty_lambda"] = float(getattr(decision_policy, "uncertainty_lambda"))
        payload["route_gamma"] = float(getattr(decision_policy, "route_gamma"))
    return payload


def misidentification_cost(
    y_true: List[str],
    y_pred: List[str],
    labels: List[str],
    cost_matrix: Dict[str, Dict[str, float]] | None = None,
) -> float:
    if not y_true:
        return float("nan")
    if cost_matrix is None:
        # default asymmetric costs: mixed->covariate is most dangerous.
        cost_matrix = {l: {k: (0.0 if l == k else 1.0) for k in labels} for l in labels}
        if "mixed" in labels and "covariate" in labels:
            cost_matrix["mixed"]["covariate"] = 3.0
    total = 0.0
    n = 0
    for t, p in zip(y_true, y_pred):
        if t not in cost_matrix:
            continue
        total += float(cost_matrix[t].get(p, 1.0))
        n += 1
    return float(total / max(n, 1))


def action_cost(
    action: str,
    expected_harm: Dict[str, float],
    abstain_cost: float = 1.0,
) -> float:
    a = str(action)
    if a == "apply_eraser":
        a = "eraser"
    if a == "eraser":
        return float(expected_harm.get("eraser", np.nan))
    if a == "recalibrate":
        return float(expected_harm.get("recalibrate", expected_harm.get("no_eraser", np.nan)))
    if a == "no_eraser":
        return float(expected_harm.get("no_eraser", np.nan))
    if a == "abstain":
        return float(abstain_cost)
    return float("nan")
