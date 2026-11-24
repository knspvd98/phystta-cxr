from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from phystta.utils.report_contract import (
    require_finite_scalar,
    to_scalar_json_value,
    vector_finiteness_status,
)


def test_to_scalar_json_value_accepts_singleton_series_and_array():
    assert to_scalar_json_value(pd.Series([0.95]), "target_value") == 0.95
    assert to_scalar_json_value(np.array([0.95]), "threshold") == 0.95


def test_to_scalar_json_value_rejects_multi_element_series_and_array():
    with pytest.raises(RuntimeError, match="target_value"):
        to_scalar_json_value(pd.Series([0.9, 0.95]), "target_value")
    with pytest.raises(RuntimeError, match="threshold"):
        to_scalar_json_value(np.array([0.9, 0.95]), "threshold")


def test_require_finite_scalar_rejects_nan():
    with pytest.raises(RuntimeError, match="harm"):
        require_finite_scalar(float("nan"), "harm")


def test_vector_finiteness_status_counts_nonfinite():
    status = vector_finiteness_status([0.2, float("nan")], "source_label_priors")
    assert status["field"] == "source_label_priors"
    assert status["present"] is True
    assert status["length"] == 2
    assert status["n_nonfinite"] == 1
    assert status["all_finite"] is False
