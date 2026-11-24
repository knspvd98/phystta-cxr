import numpy as np
import pandas as pd

from phystta.shifts.shift_simulators import dirichlet_label_shift_indices, patient_mix_shift_indices


def test_dirichlet_label_shift_no_duplicates():
    rng = np.random.default_rng(0)
    labels = (rng.random((120, 4)) > 0.7).astype(float)
    idx = dirichlet_label_shift_indices(labels, n_samples=80, alpha=0.4, seed=1)
    assert len(idx) == 80
    assert len(np.unique(idx)) == 80
    assert idx.max() < 120


def test_patient_mix_shift_no_duplicates():
    df = pd.DataFrame(
        {
            "view": ["AP"] * 40 + ["PA"] * 40 + ["LATERAL"] * 40,
            "manufacturer": ["m1"] * 60 + ["m2"] * 60,
        }
    )
    idx = patient_mix_shift_indices(df, n_samples=90, strata_cols=["view", "manufacturer"], seed=3)
    assert len(idx) == 90
    assert len(np.unique(idx)) == 90
    assert idx.max() < len(df)
