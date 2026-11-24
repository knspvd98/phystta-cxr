from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass
class LabelShiftRecalibrator:
    source_priors: np.ndarray
    max_iter: int = 50
    tol: float = 1e-6
    eps: float = 1e-6

    @classmethod
    def fit_from_labels(
        cls,
        labels: np.ndarray,
        max_iter: int = 50,
        tol: float = 1e-6,
        eps: float = 1e-6,
    ) -> "LabelShiftRecalibrator":
        y = np.asarray(labels, dtype=float)
        if y.ndim != 2:
            raise ValueError("labels must be [N,C] for multilabel recalibration")
        priors = []
        for c in range(y.shape[1]):
            yc = y[:, c]
            valid = yc >= 0
            if valid.sum() == 0:
                priors.append(0.5)
            else:
                priors.append(float(np.clip(np.mean(yc[valid] == 1), eps, 1.0 - eps)))
        return cls(source_priors=np.asarray(priors, dtype=float), max_iter=max_iter, tol=tol, eps=eps)

    def adjust(self, probs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        p = np.asarray(probs, dtype=float)
        if p.ndim != 2:
            raise ValueError("probs must be [N,C] for multilabel recalibration")
        c = p.shape[1]
        if self.source_priors.shape[0] != c:
            raise ValueError(
                f"source_priors length mismatch: expected {c}, got {self.source_priors.shape[0]}"
            )
        out = np.zeros_like(p, dtype=float)
        target_priors = np.zeros((c,), dtype=float)
        for j in range(c):
            src = float(np.clip(self.source_priors[j], self.eps, 1.0 - self.eps))
            pj = np.clip(p[:, j], self.eps, 1.0 - self.eps)
            # Saerens-Latinne-Decaestecker EM update for label shift.
            tgt = src
            post = pj
            for _ in range(int(self.max_iter)):
                num = (tgt / src) * pj
                den = num + ((1.0 - tgt) / (1.0 - src)) * (1.0 - pj)
                post = np.clip(num / np.clip(den, self.eps, None), self.eps, 1.0 - self.eps)
                tgt_new = float(np.clip(np.mean(post), self.eps, 1.0 - self.eps))
                if abs(tgt_new - tgt) <= self.tol:
                    tgt = tgt_new
                    break
                tgt = tgt_new
            out[:, j] = post
            target_priors[j] = tgt
        return out, target_priors


@dataclass
class StratifiedLabelShiftRecalibrator:
    global_recalibrator: LabelShiftRecalibrator
    by_stratum: Dict[str, LabelShiftRecalibrator]
    min_samples: int = 64

    @classmethod
    def fit_from_labels_and_strata(
        cls,
        labels: np.ndarray,
        strata: np.ndarray,
        min_samples: int = 64,
        max_iter: int = 50,
        tol: float = 1e-6,
        eps: float = 1e-6,
    ) -> "StratifiedLabelShiftRecalibrator":
        y = np.asarray(labels, dtype=float)
        s = np.asarray(strata, dtype=object).reshape(-1)
        if y.ndim != 2:
            raise ValueError("labels must be [N,C] for multilabel recalibration")
        if s.shape[0] != y.shape[0]:
            raise ValueError(f"strata length mismatch: expected {y.shape[0]}, got {s.shape[0]}")
        global_recalibrator = LabelShiftRecalibrator.fit_from_labels(
            y,
            max_iter=max_iter,
            tol=tol,
            eps=eps,
        )
        by_stratum: Dict[str, LabelShiftRecalibrator] = {}
        uniq, counts = np.unique(s.astype(str), return_counts=True)
        for key, count in zip(uniq.tolist(), counts.tolist()):
            if int(count) < int(min_samples):
                continue
            mask = s.astype(str) == str(key)
            by_stratum[str(key)] = LabelShiftRecalibrator.fit_from_labels(
                y[mask],
                max_iter=max_iter,
                tol=tol,
                eps=eps,
            )
        return cls(
            global_recalibrator=global_recalibrator,
            by_stratum=by_stratum,
            min_samples=int(min_samples),
        )

    def adjust(self, probs: np.ndarray, strata: np.ndarray) -> tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, int]]:
        p = np.asarray(probs, dtype=float)
        s = np.asarray(strata, dtype=object).reshape(-1).astype(str)
        if p.ndim != 2:
            raise ValueError("probs must be [N,C] for multilabel recalibration")
        if s.shape[0] != p.shape[0]:
            raise ValueError(f"strata length mismatch: expected {p.shape[0]}, got {s.shape[0]}")
        out = np.zeros_like(p, dtype=float)
        target_priors_by_stratum: Dict[str, np.ndarray] = {}
        counts_by_stratum: Dict[str, int] = {}
        uniq = np.unique(s)
        for key in uniq.tolist():
            mask = s == str(key)
            counts_by_stratum[str(key)] = int(mask.sum())
            recalibrator = self.by_stratum.get(str(key), self.global_recalibrator)
            adj, priors = recalibrator.adjust(p[mask])
            out[mask] = adj
            target_priors_by_stratum[str(key)] = np.asarray(priors, dtype=float)
        return out, target_priors_by_stratum, counts_by_stratum
