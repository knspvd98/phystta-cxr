from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class LabelSchema:
    labels: List[str]
    uncertain_values: List[int] = field(default_factory=lambda: [-1])
    positive_values: List[int] = field(default_factory=lambda: [1])
    negative_values: List[int] = field(default_factory=lambda: [0])
    ignore_value: float = -1.0

    def normalize_label(self, value):
        if value in self.positive_values:
            return 1.0
        if value in self.negative_values:
            return 0.0
        if value in self.uncertain_values:
            return self.ignore_value
        if value is None:
            return self.ignore_value
        try:
            v = float(value)
            if v in self.positive_values:
                return 1.0
            if v in self.negative_values:
                return 0.0
        except Exception:
            pass
        return self.ignore_value


@dataclass
class DatasetSchema:
    name: str
    path_col: str
    patient_id_col: str
    study_id_col: Optional[str]
    label_cols: List[str]
    age_col: Optional[str] = None
    sex_col: Optional[str] = None
    view_col: Optional[str] = None
    domain_label: Optional[str] = None

    def label_schema(self) -> LabelSchema:
        return LabelSchema(labels=self.label_cols)

    def to_dict(self) -> Dict[str, str]:
        return {
            "name": self.name,
            "path_col": self.path_col,
            "patient_id_col": self.patient_id_col,
            "study_id_col": self.study_id_col or "",
            "age_col": self.age_col or "",
            "sex_col": self.sex_col or "",
            "view_col": self.view_col or "",
        }
