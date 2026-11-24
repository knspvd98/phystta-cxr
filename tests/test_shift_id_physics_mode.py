import numpy as np

from phystta.safety.shift_id import ShiftIdentifierV2


def test_shift_id_calibrate_from_scores_supports_physics_labels():
    sid = ShiftIdentifierV2(score_family=["energy", "js"], calibration="none")
    labeled_scores = [
        ("none", {"energy": 0.01, "js": 0.01}),
        ("none", {"energy": 0.02, "js": 0.01}),
        ("heel", {"energy": 0.40, "js": 0.12}),
        ("heel", {"energy": 0.45, "js": 0.10}),
        ("scatter", {"energy": 0.65, "js": 0.18}),
        ("scatter", {"energy": 0.70, "js": 0.20}),
    ]
    sid.calibrate_from_scores(labeled_scores, seed=7)
    assert sid.classes_ is not None
    assert "heel" in sid.classes_
    assert "scatter" in sid.classes_

    probs = sid.predict_proba_from_scores({"energy": 0.42, "js": 0.11})
    assert np.isfinite(probs).all()
    assert np.isclose(probs.sum(), 1.0, atol=1e-6)


def test_shift_id_calibration_report_from_scores_returns_finite_metrics():
    sid = ShiftIdentifierV2(score_family=["energy", "js"], calibration="temperature")
    labeled_scores = [
        ("none", {"energy": 0.01, "js": 0.01}),
        ("none", {"energy": 0.02, "js": 0.01}),
        ("heel", {"energy": 0.40, "js": 0.12}),
        ("heel", {"energy": 0.45, "js": 0.10}),
        ("scatter", {"energy": 0.65, "js": 0.18}),
        ("scatter", {"energy": 0.70, "js": 0.20}),
    ]
    sid.calibrate_from_scores(labeled_scores, seed=7)
    report = sid.calibration_report_from_scores(labeled_scores, n_bins=5)
    assert report["n_scenarios"] == len(labeled_scores)
    assert np.isfinite(report["nll"])
    assert np.isfinite(report["brier"])
    assert np.isfinite(report["ece"])
    assert np.isfinite(report["accuracy"])
    assert np.isclose(report["temperature"], sid.temperature)
    assert report["per_class_support"]["none"] == 2
