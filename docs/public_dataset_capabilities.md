# Public Dataset Capabilities

`phystta` mainline now treats public CXR datasets as capability-scoped sources, not as interchangeable metadata carriers.

Capability presence in the repository is not the same thing as operator-ready same-pod continuation.

For post-MIMIC continuation, the operator-facing truth is defined by `docs/canonical_external_followup_runpod_packet.md`.
Only the selected external follow-up branches covered by that packet should be treated as eligible same-pod continuation.

MIMIC-CXR-JPG public branch:
- Supported: temporal pairing, `ViewPosition`, procedure/context descriptors, orientation descriptors, `Rows/Columns`, derived `view_family`, `procedure_context`, `orientation_context`, `resolution_bucket`.
- Not supported by default: `manufacturer`, `station_name`, fairness claims from age/sex/race on the public JPG metadata branch.
- Mainline claim: acquisition-context robustness under public metadata.
- Strict device-change path is forbidden unless a source really materializes `manufacturer` and `station_name`; public MIMIC-JPG should use `temporal_same_view` or `temporal_acquisition_context`.

Mixed stress on public MIMIC:
- Current role: separate legacy CNN / 6-class mixed-stress contour, not the canonical continuation of the bounded CLIP six-pack.
- Operator-facing truth is defined by `docs/canonical_mixed_stress_runpod_packet.md`.
- Current operator readiness includes honest RunPod/offline/frozen config semantics, mixed-stress-specific preflight/legacy-head validation, and a dedicated staged runner via `scripts/run_mixed_stress_suite.py`.
- The contour currently depends on the legacy public-atomic head lineage:
  - `/workspace/outputs/heads/mimic_public_atomic_head.pt`
  - `/workspace/outputs/heads/mimic_public_atomic_head.meta.json`
- Capability presence in repo does not mean the mixed-stress contour is runtime-proven or equivalent to the hardened bounded MIMIC/external lines.
- Mixed stress is not part of the canonical same-pod external continuation packet.

CheXpert:
- Supported: sex, age, AP/PA, frontal/lateral, 14 observation labels.
- Capability: external stratified evaluation on sex/age/view-family.
- Same-pod operator-ready continuation exists only through the canonical external follow-up packet, materialized runtime configs, and frozen artifacts from a successful bounded MIMIC internal branch.
- External evaluation is expressed in the MIMIC/CheXpert 14-label head space and should emit `label_harmonization_manifest.json`.
- Not a public temporal-pair or real-pair realism source in this pipeline.

NIH ChestX-ray14:
- Supported: patient age, patient gender, `View Position`, `Finding Labels`.
- Capability: external stratified evaluation and label-space sanity checks.
- Same-pod continuation is only valid inside the canonical external follow-up packet; NIH should not be treated as an automatic next wave after MIMIC.
- NIH enters the external branch through an explicit harmonization layer with exact, approximate, and unsupported target-label mappings. It is not ontology-identical to CheXpert-derived labels.
- Not a temporal or device-realism source here.

VinDr-CXR:
- Supported: image-level labels and lesion-level localization/boxes.
- Not supported: temporal-pair semantics, demographic fairness semantics, device robustness semantics.
- Current role: external classification/localization sanity branch only.
- Same-pod continuation is limited to the canonical external follow-up packet and should be treated as a scoped sanity branch, not as a full public-mainline contour.
- If VinDr labels are projected into the MIMIC/CheXpert head space, the projection must stay explicit and partial via a manifest; unsupported labels are not negatives.

Allowed claim surface:
- Public MIMIC mainline: the current operator-ready bounded CLIP cycle is heel/scatter-only and MIMIC-only.
- CheXpert/NIH branch: external stratified synthetic-generalization follow-up on sex/age/view after a successful internal bounded run.
- VinDr branch: external localization/classification sanity follow-up after a successful internal bounded run.

Shift-family caveat:
- The operator-ready bounded public mainline is still partial. The current canonical first cycle is heel/scatter-only.
- Broader shift-family preparation is now governed separately by `docs/canonical_broader_shift_family_runpod_packet.md`.
- The current broader public atomic contour is limited to:
  - `configs/mimic_public_acquisition_atomic.yaml`
  - `configs/mimic_public_acquisition_atomic_headsubspace.yaml`
- That contour remains:
  - legacy CNN
  - `atomic_calibrated_eval`
  - realism enabled
  - `heel`, `scatter`, `exposure_poisson`, `motion_blur`
- Capability presence in the repository is still not equivalent to operator-ready execution.
- Current broader-family readiness now includes honest `/workspace/...` config truth, contour-specific preflight/runtime validation, and a surfaced execution path via `scripts/run_public_mainline.py`.
- That path currently reuses generic public-mainline reporting/manifests rather than a broader-family-owned runner report.
- This still is not runtime proof and not a broader-family scientific summary contract.
- Broader shift-family expansion is not part of the canonical same-pod external continuation surface.
- Mixed stress exists as a separate contour with a truthful staged runner and preflight/rehearsal surface, but it still has a weaker evidence surface than the hardened CLIP bounded core and the hardened external continuation line.

VLM representation-surgery line:
- Current operator-facing truth is governed by `docs/canonical_vlm_runpod_packet.md`.
- The current canonical RunPod config is `configs/representation_surgery_2026_vlm_runpod.yaml`.
- Current VLM readiness includes dedicated config/runtime truth and contour-specific `vlm_contract` validation inside `runpod_preflight.json`.
- Current truthful higher-order execution now uses `scripts/run_vlm_suite.py`, which delegates the stage execution to `scripts/run_vlm_erasure_eval.py`.
- The contour now has branch-owned VLM plan/report/manifest artifacts, but this is still not runtime proof or scientific proof.
