import numpy as np

from phystta.metrics.operating_points import (
    operating_point_stats,
    threshold_at_sensitivity,
    threshold_at_specificity,
    report_at_operating_points,
)


def test_threshold_at_sensitivity_meets_target():
    y = np.array([1, 1, 1, 0, 0, 0], dtype=float)
    p = np.array([0.95, 0.80, 0.55, 0.60, 0.40, 0.10], dtype=float)
    th = threshold_at_sensitivity(y, p, target_sensitivity=2 / 3)
    stats = operating_point_stats(y, p, threshold=th)
    assert stats["sensitivity"] >= (2 / 3) - 1e-9


def test_threshold_at_specificity_meets_target():
    y = np.array([1, 1, 0, 0, 0, 0], dtype=float)
    p = np.array([0.9, 0.7, 0.8, 0.3, 0.2, 0.1], dtype=float)
    th = threshold_at_specificity(y, p, target_specificity=0.75)
    stats = operating_point_stats(y, p, threshold=th)
    assert stats["specificity"] >= 0.75 - 1e-9


def test_report_at_operating_points_multilabel_shape():
    y = np.array(
        [
            [1, 0],
            [0, 1],
            [1, 1],
            [0, 0],
        ],
        dtype=float,
    )
    p = np.array(
        [
            [0.8, 0.2],
            [0.3, 0.9],
            [0.7, 0.8],
            [0.2, 0.1],
        ],
        dtype=float,
    )
    rows = report_at_operating_points(
        y,
        p,
        label_names=["A", "B"],
        target_sensitivities=[0.5, 0.9],
        target_specificities=[0.5],
    )
    # 2 labels * (2 sens targets + 1 spec target)
    assert len(rows) == 6
    assert {r["label"] for r in rows} == {"A", "B"}
    assert {r["target_kind"] for r in rows} == {"sensitivity", "specificity"}
