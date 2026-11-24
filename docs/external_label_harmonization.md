# External Label Harmonization

This document defines label-harmonization semantics only.
It does not, by itself, make the external follow-up line an automatic or operator-ready continuation path.

Same-pod external continuation is governed by `docs/canonical_external_followup_runpod_packet.md` and requires frozen artifacts from a successful internal bounded MIMIC run.

The external CheXpert/NIH branch does not assume that source-native labels already match the active head ontology.

Canonical target label space:
- The runtime target ontology is the MIMIC/CheXpert 14-label head space.
- `run_external_policy_eval.py` loads the active frozen head metadata first, uses its `label_cols` order when available, and writes `label_harmonization_manifest.json` beside the outputs.
- For same-pod external continuation, the operator should treat the materialized follow-up config as the runtime contract rather than assuming the raw repo template is the runtime config.

Source-specific rules:
- MIMIC and CheXpert:
  - Direct exact mapping into the 14-label head space.
- NIH ChestX-ray14:
  - Exact mappings: `Atelectasis`, `Cardiomegaly`, `Consolidation`, `Edema`, `Pneumonia`, `Pneumothorax`
  - Exact derivation: `No Finding` from the raw NIH `Finding Labels` string when it explicitly contains `No Finding`
  - Approximate mappings:
    - `Effusion` -> `Pleural Effusion`
    - `Infiltration` -> `Lung Opacity`
    - `Mass` or `Nodule` -> `Lung Lesion`
    - `Pleural_Thickening` -> `Pleural Other`
  - Unsupported in this public branch:
    - `Enlarged Cardiomediastinum`
    - `Fracture`
    - `Support Devices`
- VinDr-CXR:
  - This branch keeps VinDr native labels as its primary ontology for localization/classification sanity checks.
  - If VinDr is projected into the MIMIC/CheXpert 14-label head space, that projection must stay explicit and partial:
    - Exact examples: `Cardiomegaly`, `Lung Opacity`, `Consolidation`, `Atelectasis`, `Pneumothorax`
    - Approximate examples: `Nodule/Mass` or `Other lesion` -> `Lung Lesion`, `Pleural effusion` -> `Pleural Effusion`, `Pleural thickening` -> `Pleural Other`
    - Unsupported examples: `No Finding`, `Edema`, `Pneumonia`, `Support Devices`, `Fracture`
  - Treat this projection as an audit artifact for sanity checks, not as ontology equivalence.

Interpretation rules:
- Approximate mappings are explicitly marked as approximate in the manifest. They are utility probes, not ontology-equivalent labels.
- Unsupported labels are materialized as `-1` and therefore excluded from metric computation rather than silently forced to `0`.
- If a target label has no explicit harmonization rule for a source, runtime validation must fail. Silent label-order drift is not allowed.

Audit artifacts:
- `label_harmonization_manifest.json`
  - `target_label_order`
  - `sources.<name>.source_native_label_order`
  - `sources.<name>.mapping`
  - `sources.<name>.exact_labels`
  - `sources.<name>.approximate_labels`
  - `sources.<name>.unsupported_labels`
  - `sources.<name>.validation_ok`
  - `sources.<name>.validation_errors`
- `train.label_harmonization_manifest.json` for prepared VinDr CSVs
  - `native_label_order`
  - `projection.target_label_order`
  - `projection.exact_labels`
  - `projection.approximate_labels`
  - `projection.unsupported_labels`
