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
