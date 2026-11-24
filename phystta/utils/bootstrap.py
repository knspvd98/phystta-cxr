from __future__ import annotations

from typing import Any, Callable, Dict

import numpy as np


def cluster_bootstrap_mean(
    values: np.ndarray,
    groups: np.ndarray,
    n_boot: int = 200,
    seed: int = 2024,
    alpha: float = 0.05,
) -> Dict[str, float]:
    x = np.asarray(values, dtype=float).reshape(-1)
    g = np.asarray(groups).reshape(-1)
    if x.shape[0] != g.shape[0]:
        raise ValueError(f"values/groups length mismatch: {x.shape[0]} vs {g.shape[0]}")
    valid = np.isfinite(x)
    x = x[valid]
    g = g[valid]
    if x.size == 0:
        return {
            "mean": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "n": 0.0,
            "n_clusters": 0.0,
            "n_boot": float(n_boot),
        }
    g = g.astype(str)
    uniq = np.unique(g)
    if uniq.size == 0:
        return {
            "mean": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "n": float(x.size),
            "n_clusters": 0.0,
            "n_boot": float(n_boot),
        }
    by_group = {u: x[g == u] for u in uniq}
    rng = np.random.default_rng(seed)
    n_boot = max(int(n_boot), 20)
    boot = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sampled = rng.choice(uniq, size=len(uniq), replace=True)
        vals = np.concatenate([by_group[s] for s in sampled], axis=0)
        boot[i] = float(np.mean(vals))
    lo = float(np.quantile(boot, alpha / 2.0))
    hi = float(np.quantile(boot, 1.0 - alpha / 2.0))
    return {
        "mean": float(np.mean(x)),
        "ci_low": lo,
        "ci_high": hi,
        "n": float(x.size),
        "n_clusters": float(len(uniq)),
        "n_boot": float(n_boot),
    }


def _prepare_cluster_index(groups: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    g = np.asarray(groups).reshape(-1)
    if g.size == 0:
        return np.asarray([], dtype=str), {}
    g = g.astype(str)
    uniq = np.unique(g)
    index_map = {u: np.flatnonzero(g == u) for u in uniq}
    return uniq, index_map


def sample_cluster_bootstrap_indices_from_index(
    uniq: np.ndarray,
    index_map: dict[str, np.ndarray],
    rng: np.random.Generator,
) -> np.ndarray:
    if uniq.size == 0:
        return np.asarray([], dtype=int)
    sampled = rng.choice(uniq, size=len(uniq), replace=True)
    return np.concatenate([index_map[str(g)] for g in sampled], axis=0)


def sample_cluster_bootstrap_indices(groups: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    uniq, index_map = _prepare_cluster_index(groups)
    return sample_cluster_bootstrap_indices_from_index(uniq, index_map, rng)


def cluster_bootstrap_statistic(
    groups: np.ndarray,
    statistic_fn: Callable[[np.ndarray], Any],
    n_boot: int = 200,
    seed: int = 2024,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Cluster bootstrap for scalar or vector-valued statistics.

    `statistic_fn` receives sampled row indices and must return either a scalar
    or a numpy-like vector with fixed shape across calls.
    """
    g = np.asarray(groups).reshape(-1)
    if g.size == 0:
        return {
            "point": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "n_clusters": 0.0,
            "n_boot": float(max(int(n_boot), 20)),
        }
    uniq, index_map = _prepare_cluster_index(g)
    point = statistic_fn(np.arange(len(g), dtype=int))
    point_arr = np.asarray(point, dtype=float)
    rng = np.random.default_rng(seed)
    n_boot = max(int(n_boot), 20)
    boot = []
    for _ in range(n_boot):
        idx = sample_cluster_bootstrap_indices_from_index(uniq, index_map, rng)
        stat = np.asarray(statistic_fn(idx), dtype=float)
        if stat.shape != point_arr.shape:
            raise ValueError(
                f"cluster_bootstrap_statistic shape mismatch: expected {point_arr.shape}, got {stat.shape}"
            )
        boot.append(stat)
    boot_arr = np.stack(boot, axis=0)
    lo = np.quantile(boot_arr, alpha / 2.0, axis=0)
    hi = np.quantile(boot_arr, 1.0 - alpha / 2.0, axis=0)
    if point_arr.ndim == 0:
        point_out: Any = float(point_arr)
        lo_out: Any = float(lo)
        hi_out: Any = float(hi)
    else:
        point_out = point_arr
        lo_out = np.asarray(lo, dtype=float)
        hi_out = np.asarray(hi, dtype=float)
    return {
        "point": point_out,
        "ci_low": lo_out,
        "ci_high": hi_out,
        "n_clusters": float(len(uniq)),
        "n_boot": float(n_boot),
    }
