from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from .calibration import expected_calibration_error, brier_score
from .risk import aurc
from .clinical import clinical_risk


def aggregate_by_domain(df: pd.DataFrame, label_col: str = "label", prob_col: str = "prob", domain_col: str = "source", cost_fn: float = 10.0, cost_fp: float = 2.0) -> pd.DataFrame:
    rows = []
    for domain, sub in df.groupby(domain_col):
        labels = sub[label_col].values
        probs = sub[prob_col].values
        valid = labels >= 0
        probs = probs[valid]
        labels = labels[valid]
        if len(labels) == 0:
            continue
        errors = (probs >= 0.5).astype(float) != labels
        conf = np.maximum(probs, 1 - probs)
        rows.append({
            "domain": domain,
            "ece": expected_calibration_error(probs, labels),
            "brier": brier_score(probs, labels),
            "aurc": aurc(errors.astype(float), conf),
            "clinical_risk": clinical_risk(probs.reshape(-1, 1), labels.reshape(-1, 1), cost_fn=cost_fn, cost_fp=cost_fp),
            "count": len(sub),
        })
    return pd.DataFrame(rows)
