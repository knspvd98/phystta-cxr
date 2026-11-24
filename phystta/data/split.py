from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd


def _aggregate_patient_labels(df: pd.DataFrame, label_cols: List[str], group_col: str) -> pd.DataFrame:
    agg = df.groupby(group_col)[label_cols].max().reset_index()
    return agg


def _binarize_labels(y: np.ndarray) -> np.ndarray:
    # treat uncertain/missing (-1) as 0 for stratification
    y = np.where(np.isnan(y), -1, y)
    return (y == 1).astype(int)


def _multilabel_stratified_split(
    X: np.ndarray,
    y: np.ndarray,
    train_size: float,
    seed: int,
):
    n_items = int(len(X))
    if n_items < 2:
        raise RuntimeError("Need at least 2 groups for a stratified split.")
    train_count = int(round(n_items * float(train_size)))
    train_count = max(1, min(train_count, n_items - 1))
    y = np.asarray(y)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    if y.shape[1] <= 1:
        try:
            from sklearn.model_selection import StratifiedShuffleSplit

            splitter = StratifiedShuffleSplit(n_splits=1, train_size=train_count, random_state=seed)
            train_idx, test_idx = next(splitter.split(X, y.reshape(-1)))
            return train_idx, test_idx
        except ValueError:
            rng = np.random.default_rng(seed)
            perm = rng.permutation(n_items)
            return perm[:train_count], perm[train_count:]
        except Exception as e:
            raise RuntimeError("scikit-learn is required for binary stratified splits.") from e
    test_count = n_items - train_count
    try:
        from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
        splitter = MultilabelStratifiedShuffleSplit(
            n_splits=1,
            train_size=train_count,
            test_size=test_count,
            random_state=seed,
        )
        train_idx, test_idx = next(splitter.split(X, y))
        return train_idx, test_idx
    except Exception as e:
        raise RuntimeError(
            f"Multilabel stratified split failed inside iterstrat/sklearn compatibility path: {e}"
        ) from e


def patient_level_split(
    df: pd.DataFrame,
    label_cols: List[str],
    seed: int,
    train_frac: float,
    val_frac: float,
    test_frac: float,
    stratify: bool = True,
    group_col: str = "patient_id",
) -> Dict[str, List[str]]:
    if group_col not in df.columns:
        raise RuntimeError(f"patient_level_split missing group column '{group_col}'. Available: {sorted(df.columns)}")
    patients = df[group_col].astype(str).unique()
    rng = np.random.default_rng(seed)

    if stratify and len(label_cols) > 0:
        patient_labels = _aggregate_patient_labels(df, label_cols, group_col=group_col)
        X = patient_labels[[group_col]].values
        y = _binarize_labels(patient_labels[label_cols].fillna(-1).values)
        train_idx, temp_idx = _multilabel_stratified_split(X, y, train_size=train_frac, seed=seed)
        temp_df = patient_labels.iloc[temp_idx].reset_index(drop=True)
        temp_X = temp_df[[group_col]].values
        temp_y = _binarize_labels(temp_df[label_cols].fillna(-1).values)
        val_size = val_frac / (val_frac + test_frac)
        val_idx, test_idx = _multilabel_stratified_split(temp_X, temp_y, train_size=val_size, seed=seed + 1)
        train_ids = patient_labels.iloc[train_idx][group_col].astype(str).tolist()
        val_ids = temp_df.iloc[val_idx][group_col].astype(str).tolist()
        test_ids = temp_df.iloc[test_idx][group_col].astype(str).tolist()
        return {"train": train_ids, "val": val_ids, "test": test_ids}
    rng.shuffle(patients)
    n = len(patients)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    train_ids = patients[:n_train]
    val_ids = patients[n_train:n_train + n_val]
    test_ids = patients[n_train + n_val:]
    return {"train": list(train_ids), "val": list(val_ids), "test": list(test_ids)}


def patient_level_split_custom(
    df: pd.DataFrame,
    label_cols: List[str],
    seed: int,
    fractions: Dict[str, float],
    stratify: bool = False,
    group_col: str = "patient_id",
) -> Dict[str, List[str]]:
    total = sum(fractions.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Custom split fractions must sum to 1.0, got {total:.3f}")
    if group_col not in df.columns:
        raise RuntimeError(f"patient_level_split_custom missing group column '{group_col}'. Available: {sorted(df.columns)}")
    patients = df[group_col].astype(str).unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(patients)
    splits = {}
    if stratify and len(label_cols) > 0:
        patient_labels = _aggregate_patient_labels(df, label_cols, group_col=group_col)
        X = patient_labels[[group_col]].values
        y = _binarize_labels(patient_labels[label_cols].fillna(-1).values)
        remaining_idx = np.arange(len(patient_labels))
        remaining_X = X
        remaining_y = y
        remaining_names = list(fractions.keys())
        for i, name in enumerate(remaining_names[:-1]):
            frac = fractions[name]
            if len(remaining_idx) == 0:
                splits[name] = []
                continue
            train_size = frac / max(sum(fractions[n] for n in remaining_names[i:]), 1e-8)
            train_idx, rest_idx = _multilabel_stratified_split(remaining_X, remaining_y, train_size=train_size, seed=seed + i)
            splits[name] = patient_labels.iloc[remaining_idx[train_idx]][group_col].astype(str).tolist()
            remaining_idx = remaining_idx[rest_idx]
            remaining_X = X[remaining_idx]
            remaining_y = y[remaining_idx]
        last = remaining_names[-1]
        splits[last] = patient_labels.iloc[remaining_idx][group_col].astype(str).tolist()
        return splits
    # non-stratified fallback
    n = len(patients)
    start = 0
    for name, frac in fractions.items():
        count = int(round(n * frac))
        splits[name] = list(patients[start:start + count])
        start += count
    if start < n:
        last = list(fractions.keys())[-1]
        splits[last].extend(list(patients[start:]))
    return splits
