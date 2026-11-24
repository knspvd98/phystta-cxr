from __future__ import annotations

from scripts.fit_eraser import (
    _apply_recovery_objective_to_args,
    _resolve_recovery_objective,
)


class _Args:
    def __init__(self):
        self.logit_recovery_weight = 0.0
        self.prob_recovery_weight = 0.0
        self.clean_identity_weight = 0.0


def test_recovery_objective_disabled_path_is_backward_compatible():
    payload = _resolve_recovery_objective(
        enable_recovery_objective=False,
        logit_recovery_weight=0.0,
        prob_recovery_weight=0.0,
        clean_identity_weight=0.0,
    )
    assert payload["enabled"] is False
    assert payload["source"] == "disabled"
    assert payload["weights"]["logit_recovery_weight"] == 0.0
    assert payload["weights"]["prob_recovery_weight"] == 0.0
    assert payload["weights"]["clean_identity_weight"] == 0.0


def test_recovery_objective_flag_applies_conservative_defaults():
    payload = _resolve_recovery_objective(
        enable_recovery_objective=True,
        logit_recovery_weight=0.0,
        prob_recovery_weight=0.0,
        clean_identity_weight=0.0,
    )
    assert payload["enabled"] is True
    assert payload["source"] == "flag_defaults"
    assert abs(payload["weights"]["logit_recovery_weight"] - 0.25) < 1e-12
    assert abs(payload["weights"]["prob_recovery_weight"] - 0.05) < 1e-12
    assert abs(payload["weights"]["clean_identity_weight"] - 1.0) < 1e-12


def test_recovery_objective_explicit_weights_enable_without_flag():
    payload = _resolve_recovery_objective(
        enable_recovery_objective=False,
        logit_recovery_weight=0.3,
        prob_recovery_weight=0.07,
        clean_identity_weight=0.9,
    )
    assert payload["enabled"] is True
    assert payload["source"] == "explicit_weights"
    assert abs(payload["weights"]["logit_recovery_weight"] - 0.3) < 1e-12
    assert abs(payload["weights"]["prob_recovery_weight"] - 0.07) < 1e-12
    assert abs(payload["weights"]["clean_identity_weight"] - 0.9) < 1e-12


def test_recovery_objective_rejects_negative_weights():
    bad_cases = [
        (-0.1, 0.0, 0.0),
        (0.0, -0.1, 0.0),
        (0.0, 0.0, -0.1),
    ]
    for logit_w, prob_w, clean_w in bad_cases:
        try:
            _resolve_recovery_objective(
                enable_recovery_objective=False,
                logit_recovery_weight=logit_w,
                prob_recovery_weight=prob_w,
                clean_identity_weight=clean_w,
            )
        except ValueError as exc:
            assert "must be >= 0.0" in str(exc)
        else:
            raise AssertionError("Expected ValueError for negative recovery weight")


def test_apply_recovery_objective_to_args_updates_effective_weights():
    args = _Args()
    payload = _resolve_recovery_objective(
        enable_recovery_objective=True,
        logit_recovery_weight=0.0,
        prob_recovery_weight=0.0,
        clean_identity_weight=0.0,
    )
    _apply_recovery_objective_to_args(args, payload)

    assert abs(args.logit_recovery_weight - 0.25) < 1e-12
    assert abs(args.prob_recovery_weight - 0.05) < 1e-12
    assert abs(args.clean_identity_weight - 1.0) < 1e-12
