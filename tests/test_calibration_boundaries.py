import numpy as np
import pytest

from phystta.metrics.calibration import (
    brier_score, expected_calibration_error, maximum_calibration_error, reliability_curve,
)


def test_empty_cohort_has_undefined_scores():
    for metric in (expected_calibration_error, maximum_calibration_error, brier_score):
        assert np.isnan(metric([], []))


def test_binary_confidence_and_endpoints():
    p = np.array([0, 1, 0.75, 0.25])
    y = np.array([0, 1, 0, 1])
    assert reliability_curve(p, y, 8)["count"].sum() == 4
    assert expected_calibration_error(p, y, 8) == pytest.approx(0.375)
    assert maximum_calibration_error(p, y, 8) == pytest.approx(0.75)
    assert brier_score(p[:, None], y) == pytest.approx(0.28125)
    with pytest.raises(ValueError):
        expected_calibration_error([1, np.nan], [1, 0])


def test_risk_curve_excludes_missing_observations_and_keeps_patient_indices():
    from phystta.metrics.risk import risk_coverage_inputs_multilabel, risk_coverage_curve

    errors, confidence = risk_coverage_inputs_multilabel(
        [[0.8, 0.2], [0.99, 0.99], [0.1, 0.4]], [[1, -1], [-1, -1], [1, 0]]
    )
    np.testing.assert_allclose(errors, [0, np.nan, 0.5], equal_nan=True)
    np.testing.assert_allclose(confidence, [0.8, np.nan, 0.75], equal_nan=True)
    curve = risk_coverage_curve(errors, confidence, points=2)
    np.testing.assert_allclose(curve["risk"], [0.25, 0])
    np.testing.assert_allclose(curve["coverage"], [1, 0.5])
    assert np.isnan(risk_coverage_curve(errors[1:2], confidence[1:2])["risk"]).all()


def test_report_uses_json_null_and_domain_counts_only_observed_labels():
    import json
    import pandas as pd
    from phystta.metrics.aggregation import aggregate_by_domain
    from phystta.utils.io import metric_report_json

    frame = pd.DataFrame({"source": ["a", "a", "b"], "prob": [0.75, 0.4, 0.2], "label": [1, -1, -1]})
    table = aggregate_by_domain(frame)
    assert table["count"].tolist() == [1]
    assert table["ece"].tolist() == [0.25]
    assert aggregate_by_domain(frame.iloc[1:]).columns.tolist() == table.columns.tolist()
    result = metric_report_json({"scores": [expected_calibration_error([], []), table["ece"].iat[0]]})
    assert "NaN" not in result
    assert json.loads(result) == {"scores": [None, 0.25]}


def test_missing_comparison_is_omitted_from_calibration_figure():
    import matplotlib.pyplot as plt
    import pandas as pd
    from phystta.viz.figures import _plot_grouped_bars

    frame = pd.DataFrame({"method": ["no_eraser", "eraser", "eraser"],
                          "scenario_id": ["first", "first", "second"], "ece": [0.0, 0.2, 0.3]})
    fig, ax = plt.subplots()
    try:
        _plot_grouped_bars(ax, frame, "ece", "Calibration")
        baseline = ax.containers[0]
        assert baseline[0].get_height() == 0.0
        assert np.isnan(baseline[1].get_height())
    finally:
        plt.close(fig)
