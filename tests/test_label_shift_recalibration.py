from __future__ import annotations

import numpy as np

from phystta.safety.policy_harm import policy_regret
from phystta.safety.recalibration import LabelShiftRecalibrator, StratifiedLabelShiftRecalibrator


def test_label_shift_recalibration_changes_prevalence_when_shifted():
    labels = np.zeros((100, 1), dtype=float)
    labels[:20, 0] = 1.0
    recal = LabelShiftRecalibrator.fit_from_labels(labels)

    p_same = np.full((100, 1), 0.2, dtype=float)
    adj_same, tgt_same = recal.adjust(p_same)
    assert np.isclose(float(adj_same.mean()), 0.2, atol=1e-6)
    assert np.isclose(float(tgt_same[0]), 0.2, atol=1e-6)

    p_shifted = np.full((100, 1), 0.5, dtype=float)
    adj_shifted, tgt_shifted = recal.adjust(p_shifted)
    assert float(adj_shifted.mean()) > 0.9
    assert float(tgt_shifted[0]) > 0.9


def test_policy_regret_uses_recalibrate_harm():
    out = policy_regret(
        action="recalibrate",
        harm_no_eraser=0.4,
        harm_eraser=0.3,
        harm_recalibrate=0.1,
        abstain_cost=1.0,
    )
    assert np.isclose(out["policy_harm"], 0.1)
    assert np.isclose(out["oracle_harm"], 0.1)
    assert np.isclose(out["policy_regret"], 0.0)


def test_stratified_label_shift_recalibrator_uses_local_and_global_fallback():
    labels = np.zeros((160, 1), dtype=float)
    labels[:80, 0] = 1.0
    labels[80:88, 0] = 1.0
    strata = np.asarray(["seen"] * 80 + ["small"] * 8 + ["other"] * 72, dtype=object)

    recal = StratifiedLabelShiftRecalibrator.fit_from_labels_and_strata(
        labels,
        strata,
        min_samples=32,
    )
    probs = np.full((3, 1), 0.5, dtype=float)
    eval_strata = np.asarray(["seen", "unseen", "small"], dtype=object)
    adjusted, target_priors_by_stratum, counts_by_stratum = recal.adjust(probs, eval_strata)

    assert adjusted.shape == probs.shape
    assert np.isfinite(adjusted).all()
    assert set(target_priors_by_stratum.keys()) == {"seen", "unseen", "small"}
    assert counts_by_stratum == {"seen": 1, "unseen": 1, "small": 1}
    assert not np.allclose(target_priors_by_stratum["seen"], target_priors_by_stratum["unseen"])
    assert np.allclose(target_priors_by_stratum["small"], target_priors_by_stratum["unseen"])
