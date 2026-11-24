from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


def dirichlet_label_shift_indices(
    labels: np.ndarray,
    n_samples: Optional[int] = None,
    alpha: float = 0.3,
    seed: int = 2024,
) -> np.ndarray:
    """
    Creates prevalence shift by reweighting existing samples without duplication.
    labels: [N,C] binary/ignore(-1) multilabel matrix.
    """
    rng = np.random.default_rng(seed)
    N = labels.shape[0]
    n_samples = int(n_samples or N)
    y = labels.copy().astype(float)
    y[y < 0] = 0.0
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    if y.size == 0 or N == 0:
        return np.asarray([], dtype=int)
    base_prev = y.mean(axis=0)
    if y.shape[1] == 1:
        target_prev = np.array([rng.beta(max(alpha, 1e-3), max(alpha, 1e-3))], dtype=float)
    else:
        target_prev = rng.dirichlet(np.maximum(alpha * np.ones_like(base_prev), 1e-3))
    target_prev = np.clip(target_prev, 1e-5, 1.0 - 1e-5)
    # Build per-sample weights aligned to target prevalence.
    logw = np.zeros(N, dtype=float)
    for c in range(y.shape[1]):
        p0 = float(max(base_prev[c], 1e-5))
        p1 = float(target_prev[c])
        # ratio weighting in log-space for stability
        log_ratio_pos = np.log(p1 / p0)
        log_ratio_neg = np.log(max(1.0 - p1, 1e-5) / max(1.0 - p0, 1e-5))
        logw += y[:, c] * log_ratio_pos + (1.0 - y[:, c]) * log_ratio_neg
    logw = logw - np.max(logw)
    w = np.exp(logw)
    if (not np.all(np.isfinite(w))) or float(np.sum(w)) <= 0.0:
        w = np.ones(N, dtype=float)
    w = w / np.sum(w)
    n_samples = min(n_samples, N)
    idx = rng.choice(N, size=n_samples, replace=False, p=w)
    return idx.astype(int)


def patient_mix_shift_indices(
    meta_df: pd.DataFrame,
    n_samples: Optional[int] = None,
    strata_cols: Optional[List[str]] = None,
    seed: int = 2024,
) -> np.ndarray:
    """
    Simulates patient-mix drift by sampling strata at different mixture weights.
    """
    if strata_cols is None:
        strata_cols = [c for c in ["view", "sex", "age_bin", "source", "manufacturer", "station_name"] if c in meta_df.columns]
    if not strata_cols:
        return np.arange(len(meta_df), dtype=int)
    rng = np.random.default_rng(seed)
    df = meta_df.copy()
    for c in strata_cols:
        df[c] = df[c].astype(str).fillna("unknown")
    stratum = df[strata_cols].agg("|".join, axis=1)
    counts = stratum.value_counts()
    keys = counts.index.tolist()
    base = counts.values.astype(float)
    base = base / base.sum()
    # Drift target over strata simplex.
    target = rng.dirichlet(np.maximum(0.5 * np.ones(len(keys)), 1e-3))
    ratio = target / np.maximum(base, 1e-8)
    w = stratum.map(dict(zip(keys, ratio))).values.astype(float)
    w = w / np.sum(w)
    N = len(df)
    n_samples = min(int(n_samples or N), N)
    idx = rng.choice(N, size=n_samples, replace=False, p=w)
    return idx.astype(int)
