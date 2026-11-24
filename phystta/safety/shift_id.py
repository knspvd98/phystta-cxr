from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from ..shifts.shift_simulators import dirichlet_label_shift_indices, patient_mix_shift_indices


def energy_distance(x: np.ndarray, y: np.ndarray, max_samples: int = 1000) -> float:
    rng = np.random.default_rng(2024)
    if len(x) > max_samples:
        x = x[rng.choice(len(x), max_samples, replace=False)]
    if len(y) > max_samples:
        y = y[rng.choice(len(y), max_samples, replace=False)]
    def _mean_dist(a, b):
        return np.mean(np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1))
    return float(_mean_dist(x, y) - 0.5 * _mean_dist(x, x) - 0.5 * _mean_dist(y, y))


def js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-8) -> float:
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log(p / m))
    kl_qm = np.sum(q * np.log(q / m))
    return float(0.5 * (kl_pm + kl_qm))


def c2st_score(x: np.ndarray, y: np.ndarray) -> float:
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold
    except Exception as e:
        raise RuntimeError("scikit-learn is required for C2ST") from e
    X = np.concatenate([x, y], axis=0)
    ylab = np.concatenate([np.zeros(len(x)), np.ones(len(y))])
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=2024)
    accs = []
    for tr, te in skf.split(X, ylab):
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X[tr], ylab[tr])
        accs.append(clf.score(X[te], ylab[te]))
    return float(np.mean(accs))


def rbf_mmd(x: np.ndarray, y: np.ndarray, gamma: Optional[float] = None, max_samples: int = 1000) -> float:
    rng = np.random.default_rng(2024)
    if len(x) > max_samples:
        x = x[rng.choice(len(x), max_samples, replace=False)]
    if len(y) > max_samples:
        y = y[rng.choice(len(y), max_samples, replace=False)]
    if gamma is None:
        # median heuristic
        z = np.concatenate([x, y], axis=0)
        dists = np.linalg.norm(z[:, None, :] - z[None, :, :], axis=-1)
        med = np.median(dists)
        gamma = 1.0 / max(med ** 2, 1e-8)
    Kxx = np.exp(-gamma * np.linalg.norm(x[:, None, :] - x[None, :, :], axis=-1) ** 2)
    Kyy = np.exp(-gamma * np.linalg.norm(y[:, None, :] - y[None, :, :], axis=-1) ** 2)
    Kxy = np.exp(-gamma * np.linalg.norm(x[:, None, :] - y[None, :, :], axis=-1) ** 2)
    return float(Kxx.mean() + Kyy.mean() - 2 * Kxy.mean())


def ks_statistic(x: np.ndarray, y: np.ndarray) -> float:
    x = np.sort(x)
    y = np.sort(y)
    n = len(x)
    m = len(y)
    i = j = 0
    cdf_x = cdf_y = 0.0
    d = 0.0
    while i < n and j < m:
        if x[i] <= y[j]:
            i += 1
            cdf_x = i / n
        else:
            j += 1
            cdf_y = j / m
        d = max(d, abs(cdf_x - cdf_y))
    return float(d)


def multiclass_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    p = np.asarray(probs, dtype=float)
    y = np.asarray(labels)
    if p.ndim != 2:
        raise ValueError("probs must be [N,C]")
    if y.shape[0] != p.shape[0]:
        raise ValueError(f"labels length mismatch: expected {p.shape[0]}, got {y.shape[0]}")
    if p.shape[0] == 0:
        return float("nan")
    conf = np.max(p, axis=1)
    pred = np.argmax(p, axis=1)
    correct = (pred == y).astype(float)
    bins = np.linspace(0.0, 1.0, int(n_bins) + 1)
    ece = 0.0
    for i in range(len(bins) - 1):
        if i == len(bins) - 2:
            mask = (conf >= bins[i]) & (conf <= bins[i + 1])
        else:
            mask = (conf >= bins[i]) & (conf < bins[i + 1])
        if not np.any(mask):
            continue
        acc = float(correct[mask].mean())
        avg_conf = float(conf[mask].mean())
        ece += float(mask.mean()) * abs(acc - avg_conf)
    return float(ece)


@dataclass
class ShiftIdConfigV2:
    score_family: List[str]
    batch_sizes: List[int]
    cost_matrix: Optional[List[List[float]]] = None


class ShiftIdentifierV2:
    def __init__(self, score_family: List[str] | None = None, cost_matrix: Optional[List[List[float]] | dict] = None, calibration: str = "none"):
        self.score_family = score_family or ["energy", "c2st", "js"]
        if isinstance(cost_matrix, dict):
            self.cost_matrix = cost_matrix
        elif cost_matrix is not None:
            self.cost_matrix = np.array(cost_matrix, dtype=float)
        else:
            self.cost_matrix = None
        self.calibration = calibration
        self.temperature = 1.0
        self.clf = None
        self.classes_ = None

    def compute_scores(self, features_ref: np.ndarray, probs_ref: np.ndarray, features_tgt: np.ndarray, probs_tgt: np.ndarray) -> Dict[str, float]:
        scores = {}
        if "energy" in self.score_family:
            scores["energy"] = energy_distance(features_ref, features_tgt)
        if "c2st" in self.score_family:
            scores["c2st"] = c2st_score(features_ref, features_tgt)
        if "js" in self.score_family:
            scores["js"] = js_divergence(probs_ref.mean(axis=0), probs_tgt.mean(axis=0))
        if "mmd" in self.score_family:
            scores["mmd"] = rbf_mmd(features_ref, features_tgt)
        if "ks" in self.score_family:
            # average KS over labels
            ks_vals = []
            for i in range(probs_ref.shape[1]):
                ks_vals.append(ks_statistic(probs_ref[:, i], probs_tgt[:, i]))
            scores["ks"] = float(np.mean(ks_vals)) if ks_vals else 0.0
        if "cond_mmd" in self.score_family:
            # conditional MMD over predicted label bins
            cond_vals = []
            for i in range(probs_ref.shape[1]):
                mask_r = probs_ref[:, i] >= 0.5
                mask_t = probs_tgt[:, i] >= 0.5
                if mask_r.sum() > 5 and mask_t.sum() > 5:
                    cond_vals.append(rbf_mmd(features_ref[mask_r], features_tgt[mask_t]))
            scores["cond_mmd"] = float(np.mean(cond_vals)) if cond_vals else 0.0
        return scores

    def _scenario_scores(self, features: np.ndarray, probs: np.ndarray, labels: np.ndarray, group: Optional[np.ndarray], rng: np.random.Generator):
        n = len(features)
        idx = rng.permutation(n)
        half = n // 2
        # none
        f_a, f_b = features[idx[:half]], features[idx[half:]]
        p_a, p_b = probs[idx[:half]], probs[idx[half:]]
        yield "none", self.compute_scores(f_a, p_a, f_b, p_b)
        # covariate via technical-group split (stable: frequency-ranked groups)
        if group is not None:
            uniq, counts = np.unique(group, return_counts=True)
            valid = [(u, c) for u, c in zip(uniq.tolist(), counts.tolist()) if u != "unknown" and c > 5]
            valid = sorted(valid, key=lambda x: (-x[1], str(x[0])))
            top_groups = [u for u, _ in valid[:4]]
            if len(top_groups) >= 2:
                for i in range(len(top_groups)):
                    for j in range(i + 1, len(top_groups)):
                        g1, g2 = top_groups[i], top_groups[j]
                        mask_a = group == g1
                        mask_b = group == g2
                        if mask_a.sum() > 5 and mask_b.sum() > 5:
                            yield "covariate", self.compute_scores(features[mask_a], probs[mask_a], features[mask_b], probs[mask_b])
        # prevalence shift via Dirichlet prior shift without duplicates
        if labels is not None and labels.ndim == 2:
            targ = dirichlet_label_shift_indices(labels, n_samples=min(n, len(labels)), alpha=0.3, seed=int(rng.integers(0, 1_000_000)))
            if len(targ) > 10:
                yield "prevalence", self.compute_scores(features, probs, features[targ], probs[targ])
                # mixed: combine patient-mix/covariate shift with prevalence shift target
                if group is not None:
                    meta = pd.DataFrame({"group": group.astype(str)})
                    cov_idx = patient_mix_shift_indices(meta, n_samples=min(len(targ), len(meta)), strata_cols=["group"], seed=int(rng.integers(0, 1_000_000)))
                    if len(cov_idx) > 10:
                        mixed_idx = np.intersect1d(targ, cov_idx, assume_unique=False)
                        if len(mixed_idx) < 10:
                            mixed_idx = cov_idx[: min(len(cov_idx), len(targ))]
                        yield "mixed", self.compute_scores(features[cov_idx], probs[cov_idx], features[mixed_idx], probs[mixed_idx])

    def _fit_classifier(self, X: np.ndarray, y: np.ndarray, seed: int) -> None:
        try:
            from sklearn.dummy import DummyClassifier
            from sklearn.linear_model import LogisticRegression
            from sklearn.model_selection import train_test_split
        except Exception as e:
            raise RuntimeError("scikit-learn is required for shift_id_v2") from e
        unique = np.unique(y)
        if len(unique) < 2:
            clf = DummyClassifier(strategy="constant", constant=unique[0])
            clf.fit(X, y)
            self.clf = clf
            self.classes_ = clf.classes_.tolist()
            self.temperature = 1.0
            return
        try:
            clf = LogisticRegression(max_iter=1000, multi_class="multinomial")
        except TypeError:
            # Newer sklearn may drop/ignore explicit multi_class arg.
            clf = LogisticRegression(max_iter=1000)
        clf.fit(X, y)
        self.clf = clf
        self.classes_ = clf.classes_.tolist()
        # optional temperature scaling on held-out subset
        if self.calibration in {"temperature", "platt"} and X.shape[0] >= 8:
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)
            _ = X_train  # keep API stable for future calibrators
            _ = y_train
            logits = clf.decision_function(X_val)
            if logits.ndim == 1:
                logits = np.stack([-logits, logits], axis=1)
            temps = np.linspace(0.5, 3.0, 11)
            best_t = 1.0
            best_nll = float("inf")
            for t in temps:
                z = logits / t
                z = z - z.max(axis=1, keepdims=True)
                probs_t = np.exp(z)
                probs_t = probs_t / probs_t.sum(axis=1, keepdims=True)
                idx = [list(self.classes_).index(lbl) for lbl in y_val]
                nll = -np.log(probs_t[np.arange(len(idx)), idx] + 1e-8).mean()
                if nll < best_nll:
                    best_nll = nll
                    best_t = float(t)
            self.temperature = best_t

    def calibrate_from_scores(
        self,
        labeled_scores: List[Tuple[str, Dict[str, float]]],
        seed: int = 2024,
    ) -> None:
        if not labeled_scores:
            raise RuntimeError("ShiftIdentifierV2: no labeled score scenarios provided")
        X = np.asarray([[scores.get(k, 0.0) for k in self.score_family] for _, scores in labeled_scores], dtype=float)
        y = np.asarray([label for label, _ in labeled_scores], dtype=object)
        self._fit_classifier(X, y, seed=seed)

    def calibration_report_from_scores(
        self,
        labeled_scores: List[Tuple[str, Dict[str, float]]],
        n_bins: int = 10,
    ) -> Dict[str, object]:
        if self.clf is None or not self.classes_:
            raise RuntimeError("ShiftIdentifierV2 not calibrated")
        if not labeled_scores:
            raise RuntimeError("ShiftIdentifierV2: no labeled score scenarios provided")
        X = np.asarray([[scores.get(k, 0.0) for k in self.score_family] for _, scores in labeled_scores], dtype=float)
        y_labels = np.asarray([label for label, _ in labeled_scores], dtype=object)
        class_to_idx = {label: idx for idx, label in enumerate(self.classes_)}
        y_idx = np.asarray([class_to_idx[label] for label in y_labels], dtype=int)
        probs = np.asarray(self._predict_proba(X), dtype=float)
        probs = np.clip(probs, 1e-8, 1.0)
        probs = probs / probs.sum(axis=1, keepdims=True)
        chosen = probs[np.arange(len(y_idx)), y_idx]
        nll = float(-np.log(chosen).mean()) if len(chosen) else float("nan")
        one_hot = np.zeros_like(probs)
        one_hot[np.arange(len(y_idx)), y_idx] = 1.0
        brier = float(np.mean(np.sum((probs - one_hot) ** 2, axis=1))) if len(y_idx) else float("nan")
        pred_idx = np.argmax(probs, axis=1)
        accuracy = float(np.mean(pred_idx == y_idx)) if len(y_idx) else float("nan")
        per_class_support = {
            str(label): int(np.sum(y_labels == label))
            for label in self.classes_
        }
        return {
            "n_scenarios": int(len(labeled_scores)),
            "classes": [str(label) for label in self.classes_],
            "temperature": float(self.temperature),
            "nll": nll,
            "brier": brier,
            "ece": multiclass_ece(probs, y_idx, n_bins=n_bins),
            "accuracy": accuracy,
            "per_class_support": per_class_support,
        }

    def calibrate(
        self,
        features: np.ndarray,
        probs: np.ndarray,
        labels: np.ndarray,
        group: Optional[np.ndarray],
        seed: int = 2024,
        batch_sizes: Optional[List[int]] = None,
        n_repeats: int = 3,
        n_splits: int = 1,
    ):
        rng = np.random.default_rng(seed)
        X = []
        y = []
        n = len(features)
        sizes = batch_sizes or [n]
        if n_splits > 1 and group is not None:
            try:
                from sklearn.model_selection import GroupKFold
                gkf = GroupKFold(n_splits=min(n_splits, len(np.unique(group))))
                folds = list(gkf.split(features, groups=group))
            except Exception:
                folds = [(np.arange(n), np.arange(n))]
        else:
            folds = [(np.arange(n), np.arange(n))]
        for _, idx_fold in folds:
            f_fold = features[idx_fold]
            p_fold = probs[idx_fold]
            l_fold = labels[idx_fold] if labels is not None else None
            g_fold = group[idx_fold] if group is not None else None
            n_fold = len(f_fold)
            for bs in sizes:
                bs_eff = min(int(bs), n_fold)
                for _ in range(max(1, n_repeats)):
                    idx = rng.choice(n_fold, size=bs_eff, replace=False)
                    f_sub = f_fold[idx]
                    p_sub = p_fold[idx]
                    l_sub = l_fold[idx] if l_fold is not None else None
                    g_sub = g_fold[idx] if g_fold is not None else None
                    for label, scores in self._scenario_scores(f_sub, p_sub, l_sub, g_sub, rng):
                        X.append([scores.get(k, 0.0) for k in self.score_family])
                        y.append(label)
        if not X:
            raise RuntimeError("ShiftIdentifierV2: no calibration scenarios built")
        X = np.asarray(X)
        y = np.asarray(y)
        self._fit_classifier(X, y, seed=seed)

    def _predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.clf is None:
            raise RuntimeError("ShiftIdentifierV2 not calibrated")
        if self.calibration in {"temperature", "platt"} and hasattr(self.clf, "decision_function") and len(self.classes_ or []) > 1:
            logits = self.clf.decision_function(X)
            if logits.ndim == 1:
                logits = np.stack([-logits, logits], axis=1)
            z = logits / max(self.temperature, 1e-6)
            z = z - z.max(axis=1, keepdims=True)
            probs = np.exp(z)
            probs = probs / probs.sum(axis=1, keepdims=True)
            return probs
        return self.clf.predict_proba(X)

    def predict_proba_from_scores(self, scores: Dict[str, float]) -> np.ndarray:
        X = np.asarray([[scores.get(k, 0.0) for k in self.score_family]])
        return self._predict_proba(X)[0]

    def _decision_from_probs(self, probs: np.ndarray) -> str:
        if self.cost_matrix is None:
            return self.classes_[int(np.argmax(probs))]
        if isinstance(self.cost_matrix, dict):
            costs = []
            for pred_label in self.classes_:
                row = self.cost_matrix.get(pred_label, {})
                cost = 0.0
                for true_label, p in zip(self.classes_, probs):
                    cost += float(row.get(true_label, 1.0)) * float(p)
                costs.append(cost)
            return self.classes_[int(np.argmin(costs))]
        C = self.cost_matrix
        if C.shape[0] != len(self.classes_) or C.shape[1] != len(self.classes_):
            return self.classes_[int(np.argmax(probs))]
        expected = C @ probs
        return self.classes_[int(np.argmin(expected))]

    def predict(self, features_ref: np.ndarray, probs_ref: np.ndarray, features_tgt: np.ndarray, probs_tgt: np.ndarray) -> Dict[str, float | str]:
        if self.clf is None:
            raise RuntimeError("ShiftIdentifierV2 not calibrated")
        scores = self.compute_scores(features_ref, probs_ref, features_tgt, probs_tgt)
        X = np.asarray([[scores.get(k, 0.0) for k in self.score_family]])
        probs = self._predict_proba(X)[0]
        pred = self._decision_from_probs(probs)
        out = {"label": pred}
        for k, v in scores.items():
            out[f"score_{k}"] = float(v)
        for cls, p in zip(self.classes_, probs):
            out[f"p_{cls}"] = float(p)
        out["uncertainty"] = float(1.0 - np.max(probs))
        return out

    def save(self, path: str) -> None:
        import pickle
        payload = {
            "score_family": self.score_family,
            "cost_matrix": self.cost_matrix,
            "calibration": self.calibration,
            "temperature": self.temperature,
            "classes": self.classes_,
            "clf": self.clf,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(payload, f)

    @staticmethod
    def load(path: str) -> "ShiftIdentifierV2":
        import pickle
        with open(path, "rb") as f:
            payload = pickle.load(f)
        obj = ShiftIdentifierV2(score_family=payload.get("score_family"), cost_matrix=payload.get("cost_matrix"), calibration=payload.get("calibration", "none"))
        obj.temperature = payload.get("temperature", 1.0)
        obj.classes_ = payload.get("classes")
        obj.clf = payload.get("clf")
        return obj
