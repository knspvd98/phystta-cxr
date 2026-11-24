import numpy as np
import pandas as pd

from phystta.causal import paired_effect_ci, dose_response_slope, build_within_patient_pairs


def test_paired_effect_ci_smoke():
    values = np.array([0.2, 0.3, 0.4, 0.1, 0.5])
    groups = np.array(["p1", "p2", "p3", "p4", "p5"])
    out = paired_effect_ci(values, groups, seed=0, n_boot=100)
    assert np.isfinite(out["effect_mean"])
    assert out["n"] == 5


def test_dose_response_slope_known_linear():
    x = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    y = 2.0 * x + 0.1
    out = dose_response_slope(x, y, groups=None)
    assert abs(out["dose_response_slope"] - 2.0) < 1e-6


def test_build_within_patient_pairs():
    df = pd.DataFrame(
        {
            "patient_id": ["a", "a", "b"],
            "study_datetime": ["2025-01-01 10:00:00", "2025-01-01 12:00:00", "2025-01-02 10:00:00"],
            "source": ["mini", "mini", "mini"],
            "manufacturer": ["m1", "m2", "m1"],
            "station_name": ["s1", "s2", "s1"],
        }
    )
    pairs = build_within_patient_pairs(df, max_hours=24.0, pair_mode="strict_device_change")
    assert len(pairs) == 1
