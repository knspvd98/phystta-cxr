from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np


@dataclass
class MonitorEvent:
    step: int
    alert: bool
    score: float
    threshold: float
    reason: str


class ShiftMonitor:
    """Simple post-market style drift monitor over scalar score streams."""

    def __init__(self, alert_quantile: float = 0.99, min_consecutive_alerts: int = 2):
        self.alert_quantile = float(alert_quantile)
        self.min_consecutive_alerts = int(min_consecutive_alerts)
        self.thresholds: Dict[str, float] = {}
        self._consecutive: Dict[str, int] = {}

    def calibrate(self, baseline_scores: Dict[str, List[float]]) -> Dict[str, float]:
        self.thresholds = {}
        self._consecutive = {}
        for key, vals in baseline_scores.items():
            arr = np.asarray(vals, dtype=float)
            arr = arr[np.isfinite(arr)]
            if arr.size < 5:
                raise RuntimeError(f"Not enough baseline scores for monitor key={key}")
            self.thresholds[key] = float(np.quantile(arr, self.alert_quantile))
            self._consecutive[key] = 0
        return self.thresholds

    def update(self, key: str, score: float, step: int) -> MonitorEvent:
        if key not in self.thresholds:
            raise RuntimeError(f"Monitor key not calibrated: {key}")
        th = self.thresholds[key]
        s = float(score)
        if s > th:
            self._consecutive[key] += 1
        else:
            self._consecutive[key] = 0
        alert = self._consecutive[key] >= self.min_consecutive_alerts
        reason = "score_above_threshold" if alert else "ok"
        return MonitorEvent(step=step, alert=bool(alert), score=s, threshold=float(th), reason=reason)


def detection_delay(events: List[MonitorEvent], drift_start_step: int) -> int:
    for ev in events:
        if ev.step >= drift_start_step and ev.alert:
            return int(ev.step - drift_start_step)
    return -1
