from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Tuple

import numpy as np
import pandas as pd

from .templates import CHEXPERT_LABELS, VINDR_LABELS


MIMIC_CHEXPERT_14_LABELS = list(CHEXPERT_LABELS)


_NIH_TARGET_RULES: Dict[str, Dict[str, object]] = {
    "No Finding": {
        "status": "exact",
        "source_columns": ["labels_from_findings_raw"],
        "operator": "contains_no_finding",
        "note": "Derived from the raw NIH Finding Labels string when 'No Finding' is explicitly present.",
    },
    "Enlarged Cardiomediastinum": {
        "status": "unsupported",
        "source_columns": [],
        "note": "NIH public labels do not expose an enlarged cardiomediastinum label.",
    },
    "Cardiomegaly": {
        "status": "exact",
        "source_columns": ["Cardiomegaly"],
        "note": "Source-native NIH label matches the target label.",
    },
    "Lung Opacity": {
        "status": "approximate",
        "source_columns": ["Infiltration"],
        "operator": "max",
        "note": "NIH Infiltration is used as an opacity-like proxy and should not be treated as ontology-equivalent.",
    },
    "Lung Lesion": {
        "status": "approximate",
        "source_columns": ["Mass", "Nodule"],
        "operator": "max",
        "note": "Mass and Nodule are collapsed into an approximate lung-lesion proxy.",
    },
    "Edema": {
        "status": "exact",
        "source_columns": ["Edema"],
        "note": "Source-native NIH label matches the target label.",
    },
    "Consolidation": {
        "status": "exact",
        "source_columns": ["Consolidation"],
        "note": "Source-native NIH label matches the target label.",
    },
    "Pneumonia": {
        "status": "exact",
        "source_columns": ["Pneumonia"],
        "note": "Source-native NIH label matches the target label.",
    },
    "Atelectasis": {
        "status": "exact",
        "source_columns": ["Atelectasis"],
        "note": "Source-native NIH label matches the target label.",
    },
    "Pneumothorax": {
        "status": "exact",
        "source_columns": ["Pneumothorax"],
        "note": "Source-native NIH label matches the target label.",
    },
    "Pleural Effusion": {
        "status": "approximate",
        "source_columns": ["Effusion"],
        "operator": "max",
        "note": "NIH Effusion is treated as an approximate pleural-effusion proxy.",
    },
    "Pleural Other": {
        "status": "approximate",
        "source_columns": ["Pleural_Thickening"],
        "operator": "max",
        "note": "Pleural thickening is mapped into a pleural-other proxy.",
    },
    "Fracture": {
        "status": "unsupported",
        "source_columns": [],
        "note": "NIH public labels do not expose a fracture label in this branch.",
    },
    "Support Devices": {
        "status": "unsupported",
        "source_columns": [],
        "note": "NIH public labels do not expose a support-devices label in this branch.",
    },
}


_VINDR_TARGET_RULES: Dict[str, Dict[str, object]] = {
    "No Finding": {
        "status": "unsupported",
        "source_columns": [],
        "note": (
            "The native VinDr-CXR label set used in this branch does not carry a reliable No Finding flag in the "
            "prepared image-level CSV, so absence of positive labels is not treated as ontology-equivalent."
        ),
    },
    "Enlarged Cardiomediastinum": {
        "status": "unsupported",
        "source_columns": [],
        "note": "VinDr Aortic enlargement is not treated as cardiomediastinal-enlargement equivalence in this branch.",
    },
    "Cardiomegaly": {
        "status": "exact",
        "source_columns": ["Cardiomegaly"],
        "note": "Source-native VinDr label matches the target label.",
    },
    "Lung Opacity": {
        "status": "exact",
        "source_columns": ["Lung Opacity"],
        "note": "Source-native VinDr label matches the target label name.",
    },
    "Lung Lesion": {
        "status": "approximate",
        "source_columns": ["Nodule/Mass", "Other lesion"],
        "operator": "max",
        "note": "VinDr Nodule/Mass and Other lesion are collapsed into an approximate lung-lesion proxy.",
    },
    "Edema": {
        "status": "unsupported",
        "source_columns": [],
        "note": "VinDr public labels do not expose an edema label in this branch.",
    },
    "Consolidation": {
        "status": "exact",
        "source_columns": ["Consolidation"],
        "note": "Source-native VinDr label matches the target label.",
    },
    "Pneumonia": {
        "status": "unsupported",
        "source_columns": [],
        "note": "VinDr public labels do not expose a pneumonia label in this branch.",
    },
    "Atelectasis": {
        "status": "exact",
        "source_columns": ["Atelectasis"],
        "note": "Source-native VinDr label matches the target label.",
    },
    "Pneumothorax": {
        "status": "exact",
        "source_columns": ["Pneumothorax"],
        "note": "Source-native VinDr label matches the target label.",
    },
    "Pleural Effusion": {
        "status": "approximate",
        "source_columns": ["Pleural effusion"],
        "operator": "max",
        "note": "VinDr Pleural effusion is treated as an approximate pleural-effusion proxy.",
    },
    "Pleural Other": {
        "status": "approximate",
        "source_columns": ["Pleural thickening"],
        "operator": "max",
        "note": "VinDr Pleural thickening is projected into a pleural-other proxy.",
    },
    "Fracture": {
        "status": "unsupported",
        "source_columns": [],
        "note": "VinDr public labels do not expose a fracture label in this branch.",
    },
    "Support Devices": {
        "status": "unsupported",
        "source_columns": [],
        "note": "VinDr public labels do not expose support-devices semantics in this branch.",
    },
}


def canonical_external_label_space() -> List[str]:
    return list(MIMIC_CHEXPERT_14_LABELS)


def source_native_label_order(source_name: str, source_cfg) -> List[str]:
    labels = list(getattr(source_cfg, "label_cols", []) or [])
    if str(source_name).lower() in {"nih", "nih_chestxray14", "chestxray14"} and "No Finding" not in labels:
        labels = ["No Finding", *labels]
    if str(source_name).lower() in {"vindr", "vindr-cxr"} and not labels:
        labels = list(VINDR_LABELS)
    return labels


def source_target_mapping(source_name: str, target_labels: Iterable[str]) -> Dict[str, Dict[str, object]]:
    source_key = str(source_name).lower()
    labels = list(target_labels)
    if source_key in {"mimic", "mimic-cxr", "mimic-cxr-jpg", "chexpert"}:
        return {
            label: {
                "status": "exact",
                "source_columns": [label],
                "operator": "identity",
                "note": "Target label is read directly from the source-native CheXpert-style column.",
            }
            for label in labels
        }
    if source_key in {"nih", "nih_chestxray14", "chestxray14"}:
        mapping: Dict[str, Dict[str, object]] = {}
        for label in labels:
            rule = _NIH_TARGET_RULES.get(label)
            if rule is None:
                raise ValueError(
                    f"No NIH harmonization rule defined for target label '{label}'. "
                    "Add an explicit exact/approximate/unsupported mapping before using this label space."
                )
            mapping[label] = dict(rule)
        return mapping
    if source_key in {"vindr", "vindr-cxr"}:
        mapping = {}
        for label in labels:
            rule = _VINDR_TARGET_RULES.get(label)
            if rule is None:
                raise ValueError(
                    f"No VinDr harmonization rule defined for target label '{label}'. "
                    "Add an explicit exact/approximate/unsupported mapping before using this label space."
                )
            mapping[label] = dict(rule)
        return mapping
    raise ValueError(f"External label harmonization is not defined for source '{source_name}'.")


def build_label_projection_manifest(
    source_name: str,
    *,
    target_labels: Iterable[str],
    source_cfg=None,
    ontology_note: str | None = None,
) -> Dict[str, object]:
    target = list(target_labels)
    mapping = source_target_mapping(source_name, target)
    rows = []
    for label in target:
        rule = dict(mapping[label])
        rows.append(
            {
                "target_label": label,
                "status": str(rule.get("status", "unsupported")),
                "source_columns": [str(col) for col in rule.get("source_columns", [])],
                "operator": str(rule.get("operator", "identity")),
                "note": str(rule.get("note", "")),
            }
        )
    return {
        "source": str(source_name),
        "source_native_label_order": source_native_label_order(source_name, source_cfg),
        "target_label_order": target,
        "mapping": rows,
        "exact_labels": [row["target_label"] for row in rows if row["status"] == "exact"],
        "approximate_labels": [row["target_label"] for row in rows if row["status"] == "approximate"],
        "unsupported_labels": [row["target_label"] for row in rows if row["status"] == "unsupported"],
        "ontology_note": ontology_note
        or (
            "Projection is explicit and audited. Approximate labels are utility probes only; unsupported labels "
            "must not be interpreted as negatives."
        ),
    }


def _combine_numeric_columns(df: pd.DataFrame, cols: List[str]) -> np.ndarray:
    out = np.full(len(df), -1.0, dtype=float)
    for col in cols:
        if col not in df.columns:
            continue
        vals = pd.to_numeric(df[col], errors="coerce").fillna(-1.0).to_numpy(dtype=float)
        valid = vals >= 0
        out = np.where(valid & (out < 0), vals, out)
        out = np.where(valid & (out >= 0), np.maximum(out, vals), out)
    return out


def _derive_no_finding_from_raw(df: pd.DataFrame, source_col: str) -> np.ndarray:
    if source_col not in df.columns:
        return np.full(len(df), -1.0, dtype=float)
    out = np.full(len(df), 0.0, dtype=float)
    raw = df[source_col].fillna("").astype(str).str.lower()
    out[raw.str.contains("no finding").to_numpy()] = 1.0
    return out


def harmonize_external_labels(
    index_df: pd.DataFrame,
    *,
    source_map: Mapping[str, object],
    target_labels: Iterable[str],
    eval_sources: Iterable[str] | None = None,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    if "source" not in index_df.columns:
        raise ValueError("index_df must contain a 'source' column for label harmonization.")
    target = list(target_labels)
    eval_source_set = {str(s) for s in (eval_sources or [])}
    if not eval_source_set:
        eval_source_set = {str(s) for s in index_df["source"].astype(str).unique().tolist()}

    harmonized = index_df.copy()
    per_source_manifest: Dict[str, object] = {}
    validation_errors: List[str] = []

    for source in sorted({str(s) for s in harmonized["source"].astype(str).unique().tolist()}):
        df_src = harmonized[harmonized["source"].astype(str) == source]
        if source not in eval_source_set:
            continue
        src_cfg = source_map.get(source)
        native_order = source_native_label_order(source, src_cfg)
        mapping = source_target_mapping(source, target)
        source_errors: List[str] = []
        entries: List[Dict[str, object]] = []

        for label in target:
            rule = dict(mapping[label])
            status = str(rule.get("status", "unsupported"))
            source_columns = [str(col) for col in rule.get("source_columns", [])]
            operator = str(rule.get("operator", "identity"))
            note = str(rule.get("note", ""))

            if status == "exact" and operator == "identity":
                missing = [col for col in source_columns if col not in df_src.columns]
                if missing:
                    source_errors.append(f"{source}: missing source columns {missing} for target '{label}'")
                    values = np.full(len(df_src), -1.0, dtype=float)
                else:
                    values = pd.to_numeric(df_src[source_columns[0]], errors="coerce").fillna(-1.0).to_numpy(dtype=float)
            elif operator == "contains_no_finding":
                values = _derive_no_finding_from_raw(df_src, source_columns[0] if source_columns else "labels_from_findings_raw")
                if values.size and np.all(values < 0):
                    source_errors.append(f"{source}: raw finding labels are unavailable for target '{label}'")
            elif operator == "max":
                missing = [col for col in source_columns if col not in df_src.columns]
                if len(missing) == len(source_columns):
                    source_errors.append(f"{source}: no proxy source columns available for target '{label}'")
                values = _combine_numeric_columns(df_src, source_columns)
            elif status == "unsupported":
                values = np.full(len(df_src), -1.0, dtype=float)
            else:
                source_errors.append(f"{source}: unsupported harmonization operator '{operator}' for target '{label}'")
                values = np.full(len(df_src), -1.0, dtype=float)

            harmonized.loc[df_src.index, label] = values
            entries.append(
                {
                    "target_label": label,
                    "status": status,
                    "source_columns": source_columns,
                    "operator": operator,
                    "note": note,
                }
            )

        per_source_manifest[source] = {
            "source_native_label_order": native_order,
            "target_label_order": target,
            "mapping": entries,
            "exact_labels": [row["target_label"] for row in entries if row["status"] == "exact"],
            "approximate_labels": [row["target_label"] for row in entries if row["status"] == "approximate"],
            "unsupported_labels": [row["target_label"] for row in entries if row["status"] == "unsupported"],
            "validation_ok": len(source_errors) == 0,
            "validation_errors": source_errors,
        }
        validation_errors.extend(source_errors)

    if validation_errors:
        raise ValueError("External label harmonization failed: " + "; ".join(validation_errors))

    # Keep only the active target label space in engine.label_cols, but do not discard auxiliary metadata columns.
    for label in target:
        if label not in harmonized.columns:
            harmonized[label] = -1.0
        harmonized[label] = pd.to_numeric(harmonized[label], errors="coerce").fillna(-1.0).astype(float)

    manifest = {
        "ontology_id": "mimic_chexpert_14_external",
        "ontology_note": (
            "External evaluation is expressed in the MIMIC/CheXpert 14-label head space. "
            "NIH and VinDr labels use explicit exact/approximate/unsupported mappings rather than silent name matching."
        ),
        "target_label_order": target,
        "sources": per_source_manifest,
        "eval_sources": sorted(eval_source_set),
    }
    return harmonized, manifest
