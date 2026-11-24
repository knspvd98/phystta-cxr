import numpy as np
import pandas as pd
import pytest

from phystta.utils.group import require_group_key
from phystta.intervention.eraser import BiasSubspaceFitter


def test_require_group_key_missing():
    df = pd.DataFrame({"foo": [1, 2, 3]})
    with pytest.raises(RuntimeError):
        require_group_key(df)


def test_probe_fit_requires_group():
    z_clean = np.random.randn(5, 4)
    z_shift = np.random.randn(5, 4)
    intens = np.random.rand(5)
    fitter = BiasSubspaceFitter(seed=1)
    with pytest.raises(RuntimeError):
        fitter.fit("probe_xfit", z_clean, z_shift, intens, None, None, k=2, probe_target="presence")
