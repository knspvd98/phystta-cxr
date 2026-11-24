from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from ..config.schema import SafetyPolicyConfig
from ..metrics.selective import selective_risk


class SafetyPolicy:
    def __init__(self, cfg: SafetyPolicyConfig):
        self.cfg = cfg
        self.thresholds = {"no_eraser": 0.0, "eraser": 0.0, "apply_eraser": 0.0}
        self.harm = {"no_eraser": float("nan"), "eraser": float("nan"), "apply_eraser": float("nan")}
        self.shift_harm: Dict[str, Dict[str, float]] = {}
        self.thresholds_by_group: Dict[str, Dict[str, float]] = {}
        self.class_thresholds: Optional[np.ndarray] = None

    @staticmethod
    def normalize_action(action: str) -> str:
        a = str(action)
        if a == "apply_eraser":
            return "eraser"
        return a

    @staticmethod
    def _sample_confidence(probs: np.ndarray) -> np.ndarray:
        p = np.asarray(probs, dtype=float)
        if p.ndim == 1:
            return np.maximum(p, 1.0 - p)
        conf = np.maximum(p, 1.0 - p)
        return np.mean(conf, axis=1)

    @staticmethod
    def _resolve_thresholds(probs: np.ndarray, threshold: float | np.ndarray) -> np.ndarray:
        p = np.asarray(probs, dtype=float)
        if np.isscalar(threshold):
            t = np.asarray([float(threshold)], dtype=float)
        else:
            t = np.asarray(threshold, dtype=float).reshape(-1)
        if p.ndim == 1:
            if t.size == 0:
                return np.asarray([0.5], dtype=float)
            return np.asarray([float(t[0])], dtype=float)
        c = p.shape[1]
        if t.size == 1:
            return np.full(c, float(t[0]), dtype=float)
        if t.size != c:
            raise ValueError(f"threshold length mismatch: got {t.size}, expected 1 or {c}")
        return t

    @classmethod
    def _sample_error(cls, probs: np.ndarray, labels: np.ndarray, threshold: float | np.ndarray = 0.5) -> np.ndarray:
        p = np.asarray(probs, dtype=float)
        y = np.asarray(labels, dtype=float)
        if p.shape != y.shape:
            raise ValueError("probs and labels shape mismatch")
        th = cls._resolve_thresholds(p, threshold)
        if p.ndim == 1:
            valid = y >= 0
            pred = (p >= float(th[0])).astype(int)
            err = ((pred != y.astype(int)) & valid).astype(float)
            return err
        pred = (p >= th.reshape(1, -1)).astype(int)
        valid = y >= 0
        err = ((pred != y.astype(int)) & valid).astype(float)
        denom = np.maximum(valid.sum(axis=1), 1)
        return err.sum(axis=1) / denom

    def _expected_harm(
        self,
        probs: np.ndarray,
        labels: np.ndarray,
        critical_idx: Optional[List[int]],
        thresholds: Optional[np.ndarray] = None,
    ) -> float:
        if not critical_idx:
            return float("nan")
        th = self._resolve_thresholds(probs, 0.5 if thresholds is None else thresholds)
        costs = []
        for idx in critical_idx:
            y = labels[:, idx]
            p = probs[:, idx]
            valid = y >= 0
            if valid.sum() == 0:
                continue
            preds = (p >= float(th[idx])).astype(int)
            fn = ((preds == 0) & (y == 1)).astype(float)
            fp = ((preds == 1) & (y == 0)).astype(float)
            cost = self.cfg.cost_fn * fn + self.cfg.cost_fp * fp
            costs.append(cost[valid].mean())
        if not costs:
            return float("nan")
        return float(np.mean(costs))

    def calibrate(
        self,
        probs_no: np.ndarray,
        probs_er: np.ndarray,
        labels: np.ndarray,
        coverage: float,
        critical_idx: Optional[List[int]] = None,
        group_labels: Optional[np.ndarray] = None,
        class_thresholds: Optional[np.ndarray] = None,
    ) -> None:
        # set thresholds on calib split: either coverage budget or risk target
        def _threshold_for_coverage(probs):
            conf = self._sample_confidence(probs)
            return float(np.quantile(conf, 1 - coverage))

        def _threshold_for_risk(probs, labels, risk_target: float):
            errors = self._sample_error(
                probs,
                labels,
                threshold=self.class_thresholds if self.class_thresholds is not None else 0.5,
            )
            conf = self._sample_confidence(probs)
            order = np.argsort(-conf)
            conf_sorted = conf[order]
            err_sorted = errors[order]
            best_th = float(conf_sorted[-1]) if len(conf_sorted) else 0.0
            best_cov = 0.0
            best_risk = 1.0
            for i in range(len(conf_sorted)):
                cov = (i + 1) / len(conf_sorted)
                k = int(err_sorted[: i + 1].sum())
                n = i + 1
                risk = k / max(n, 1)
                if self.cfg.conformal:
                    try:
                        from scipy.stats import beta
                    except Exception as e:
                        raise RuntimeError("scipy is required for conformal risk control") from e
                    upper = float(beta.ppf(1 - self.cfg.conformal_alpha, k + 1, n - k))
                else:
                    upper = risk
                if upper <= risk_target and cov >= best_cov:
                    best_cov = cov
                    best_risk = upper
                    best_th = conf_sorted[i]
            if best_cov == 0.0:
                # fallback: pick min risk
                risks = np.cumsum(err_sorted) / (np.arange(len(err_sorted)) + 1)
                idx = int(np.argmin(risks)) if len(risks) else 0
                best_th = conf_sorted[idx] if len(conf_sorted) else 0.0
            return float(best_th)

        if class_thresholds is not None:
            self.class_thresholds = self._resolve_thresholds(probs_no, class_thresholds)
        else:
            self.class_thresholds = self._resolve_thresholds(probs_no, 0.5)

        if self.cfg.conformal:
            th_no = _threshold_for_risk(probs_no, labels, self.cfg.risk_target)
            th_er = th_no if self.cfg.shared_threshold else _threshold_for_risk(probs_er, labels, self.cfg.risk_target)
        else:
            th_no = _threshold_for_coverage(probs_no)
            th_er = th_no if self.cfg.shared_threshold else _threshold_for_coverage(probs_er)
        self.thresholds["no_eraser"] = th_no
        self.thresholds["eraser"] = th_er
        self.thresholds["apply_eraser"] = th_er
        if self.cfg.stratify_technical and group_labels is not None:
            self.thresholds_by_group = {"no_eraser": {}, "eraser": {}, "apply_eraser": {}}
            for g in np.unique(group_labels):
                mask = group_labels == g
                if mask.sum() < 5:
                    continue
                if self.cfg.conformal:
                    th_g = _threshold_for_risk(probs_no[mask], labels[mask], self.cfg.risk_target)
                    self.thresholds_by_group["no_eraser"][str(g)] = th_g
                    self.thresholds_by_group["eraser"][str(g)] = th_g if self.cfg.shared_threshold else _threshold_for_risk(probs_er[mask], labels[mask], self.cfg.risk_target)
                    self.thresholds_by_group["apply_eraser"][str(g)] = self.thresholds_by_group["eraser"][str(g)]
                else:
                    th_g = _threshold_for_coverage(probs_no[mask])
                    self.thresholds_by_group["no_eraser"][str(g)] = th_g
                    self.thresholds_by_group["eraser"][str(g)] = th_g if self.cfg.shared_threshold else _threshold_for_coverage(probs_er[mask])
                    self.thresholds_by_group["apply_eraser"][str(g)] = self.thresholds_by_group["eraser"][str(g)]
        mask_no = self.apply_threshold(probs_no, "no_eraser")
        mask_er = self.apply_threshold(probs_er, "eraser")
        if mask_no.sum() > 0:
            self.harm["no_eraser"] = self._expected_harm(
                probs_no[mask_no],
                labels[mask_no],
                critical_idx,
                thresholds=self.class_thresholds,
            )
        if mask_er.sum() > 0:
            self.harm["eraser"] = self._expected_harm(
                probs_er[mask_er],
                labels[mask_er],
                critical_idx,
                thresholds=self.class_thresholds,
            )
            self.harm["apply_eraser"] = self.harm["eraser"]

    def decide(self, shift_info: Dict[str, float | str], destructive: bool = False, shift_context: tuple | None = None) -> str:
        if destructive:
            return "abstain"
        label = shift_info.get("label", "unknown")
        if label == "covariate":
            if not self.cfg.apply_on_covariate:
                return "no_eraser"
            # shift_context is intentionally ignored in decision logic to avoid oracle leakage in evaluation.
            harm_no = self.harm.get("no_eraser", float("nan"))
            harm_er = self.harm.get("eraser", float("nan"))
            if np.isnan(harm_no) or np.isnan(harm_er):
                return "no_eraser"
            return "apply_eraser" if harm_er <= harm_no else "no_eraser"
        if label == "prevalence":
            return "no_eraser"
        if label == "mixed":
            return "abstain" if self.cfg.abstain_on_mixed else "no_eraser"
        if label == "none":
            return "no_eraser"
        return "abstain" if self.cfg.abstain_on_unknown else "no_eraser"

    def apply_threshold(self, probs: np.ndarray, action: str, group_labels: Optional[np.ndarray] = None) -> np.ndarray:
        action = self.normalize_action(action)
        if action not in self.thresholds:
            raise ValueError(
                f"Unknown policy action '{action}' for thresholding. "
                f"Expected one of {sorted(self.thresholds.keys())}."
            )
        conf = self._sample_confidence(probs)
        if group_labels is None or not self.thresholds_by_group:
            th = self.thresholds[action]
            return conf >= th
        mask = np.zeros(len(conf), dtype=bool)
        for g in np.unique(group_labels):
            idx = group_labels == g
            th = self.thresholds_by_group[action].get(str(g), self.thresholds[action])
            mask[idx] = conf[idx] >= th
        return mask
