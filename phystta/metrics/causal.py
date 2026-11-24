from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GroupKFold, KFold


def js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-8) -> float:
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log(p / m), axis=-1)
    kl_qm = np.sum(q * np.log(q / m), axis=-1)
    return float(0.5 * (kl_pm + kl_qm).mean())


def _bernoulli_js(p: np.ndarray, q: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    # p, q in [0,1], compute JS for Bernoulli distributions
    p = np.clip(p, eps, 1.0 - eps)
    q = np.clip(q, eps, 1.0 - eps)
    m = 0.5 * (p + q)
    kl_pm = p * np.log(p / m) + (1 - p) * np.log((1 - p) / (1 - m))
    kl_qm = q * np.log(q / m) + (1 - q) * np.log((1 - q) / (1 - m))
    return 0.5 * (kl_pm + kl_qm)


def invariance_metrics(probs_stack: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """
    probs_stack: [L, N, C] for multi-label (sigmoid) outputs.
    Measures stability per-label, not argmax.
    """
    if probs_stack.ndim != 3:
        raise ValueError("probs_stack must be [L,N,C]")
    L, N, C = probs_stack.shape
    baseline = probs_stack[0]
    deltas = []
    js_vals = []
    flips = []
    for i in range(1, L):
        p = probs_stack[i]
        deltas.append(np.abs(p - baseline))
        js_vals.append(_bernoulli_js(baseline, p))
        # per-label flip across threshold
        flips.append((baseline >= threshold) != (p >= threshold))
    if deltas:
        delta = np.stack(deltas, axis=0)
        mean_abs_delta = float(delta.mean())
        max_abs_delta = float(delta.max())
    else:
        mean_abs_delta = 0.0
        max_abs_delta = 0.0
    if js_vals:
        js = np.stack(js_vals, axis=0)
        js_mean = float(js.mean())
    else:
        js_mean = 0.0
    if flips:
        flip_arr = np.stack(flips, axis=0)
        flip_rate = float(flip_arr.mean())
    else:
        flip_rate = 0.0
    prob_variance = float(np.var(probs_stack, axis=0).mean())
    return {
        "flip_rate": flip_rate,
        "js_divergence": js_mean,
        "prob_variance": prob_variance,
        "mean_abs_delta": mean_abs_delta,
        "max_abs_delta": max_abs_delta,
    }


def aipw_ate(
    X: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    groups: Optional[np.ndarray] = None,
    n_splits: int = 3,
    seed: int = 0,
    eps: float = 1e-3,
) -> Dict[str, float]:
    """
    Cross-fitted AIPW for binary treatment T in {0,1}.
    Returns ate, ate_se, positivity_violation_rate.
    """
    X = np.asarray(X)
    T = np.asarray(T).astype(int)
    Y = np.asarray(Y).astype(float)
    if X.ndim != 2:
        raise ValueError("X must be 2D")
    if len(np.unique(T)) < 2:
        return {"ate": float("nan"), "ate_se": float("nan"), "positivity_violation_rate": 1.0, "n": len(T)}
    splitter = GroupKFold(n_splits=n_splits) if groups is not None else KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    terms = []
    violations = 0
    total = 0
    for train_idx, test_idx in splitter.split(X, T, groups=groups):
        X_tr, X_te = X[train_idx], X[test_idx]
        T_tr, T_te = T[train_idx], T[test_idx]
        Y_tr, Y_te = Y[train_idx], Y[test_idx]
        # propensity
        prop = LogisticRegression(max_iter=500)
        prop.fit(X_tr, T_tr)
        e = prop.predict_proba(X_te)[:, 1]
        e = np.clip(e, eps, 1.0 - eps)
        # outcome models
        mu1 = Ridge(alpha=1.0)
        mu0 = Ridge(alpha=1.0)
        mu1.fit(X_tr[T_tr == 1], Y_tr[T_tr == 1])
        mu0.fit(X_tr[T_tr == 0], Y_tr[T_tr == 0])
        m1 = mu1.predict(X_te)
        m0 = mu0.predict(X_te)
        term = (m1 - m0) + (T_te * (Y_te - m1) / e) - ((1 - T_te) * (Y_te - m0) / (1 - e))
        terms.append(term)
        violations += int(((e <= eps) | (e >= 1.0 - eps)).sum())
        total += len(e)
    all_terms = np.concatenate(terms, axis=0)
    ate = float(np.mean(all_terms))
    ate_se = float(np.std(all_terms, ddof=1) / np.sqrt(len(all_terms))) if len(all_terms) > 1 else float("nan")
    return {"ate": ate, "ate_se": ate_se, "positivity_violation_rate": float(violations / max(total, 1)), "n": int(len(all_terms))}


def aipw_from_potential_outcomes(
    X: np.ndarray,
    y0: np.ndarray,
    y1: np.ndarray,
    groups: Optional[np.ndarray] = None,
    assign_prob: float = 0.5,
    n_splits: int = 3,
    seed: int = 0,
) -> Dict[str, float]:
    """
    Randomized-policy AIPW diagnostic from paired potential outcomes.
    This does not claim observational identification; it verifies DR estimator behavior
    in a randomized do(T) setting derived from paired outcomes.
    """
    X = np.asarray(X)
    y0 = np.asarray(y0, dtype=float).reshape(-1)
    y1 = np.asarray(y1, dtype=float).reshape(-1)
    if X.ndim != 2:
        raise ValueError("X must be 2D")
    if len(y0) != len(X) or len(y1) != len(X):
        raise ValueError("Length mismatch between X and potential outcomes")
    if not (0.0 < assign_prob < 1.0):
        raise ValueError("assign_prob must be in (0,1)")
    rng = np.random.default_rng(seed)
    T = rng.binomial(1, assign_prob, size=len(X)).astype(int)
    Y = np.where(T == 1, y1, y0)
    est = aipw_ate(X, T, Y, groups=groups, n_splits=n_splits, seed=seed)
    est["ate_true"] = float(np.mean(y1 - y0))
    est["assign_prob"] = float(assign_prob)
    return est
