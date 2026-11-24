# Reprovenance Contract

Every key training or evaluation script must save:
- `resolved_runtime_config.yaml`
- `artifact_manifest.json`
- `code_fingerprint.json`
- `input_artifact_hashes.json`

Minimum manifest fields:
- `config_path`
- `resolved_config_dump`
- `git_sha_or_repo_fingerprint`
- `model_revision`
- `index_sha256`
- `source_capabilities`
- `encoder_fingerprint`
- `preprocess_signature`
- `pair_mode`
- `pair_filter_stats`
- `shift_families`
- `mixed_profiles`
- `probe_sha256`
- `head_sha256`
- `realism_sha256`
- `figure_outputs`

Artifact-specific additions:
- Probes: `n_samples`, `lengths_by_field`, `consistency_ok`, `feature_dim`, `source_split`, `shift_family`
- Erasers: `supported_shift_families`, `fit_pair_mode`, `pair_quality_summary`, `probe_basis_id`, `code_revision`
- Realism calibrators: `validity_flags`, `is_graded`, `level_separation_margin`, `n_pairs_total`, `n_pairs_accepted`, `realism_validity_summary`
- Realism calibrators: `pair_quality_audit.json`, `pair_quality_pairs.csv`, `physics_semantics`, `physical_dependence_warnings`
- Safety eval: scenario-level trace CSV with decision posterior columns and accepted/abstained counts
- External eval: `label_harmonization_manifest.json` plus a manifest hash reference
- VinDr prep/localization sanity branch: explicit native-to-head-space projection manifest when any 14-label projection is reported

Artifact-loading contract:
- Trusted local project artifacts (`.pt`, `.pth`) must be loaded through the centralized helper `load_torch_artifact(...)`, which uses an explicit `weights_only=False` policy for project-owned checkpoints.
- Pure weight/state-dict checkpoints must be loaded through `load_torch_state_dict(...)` so the code path is explicit about expecting a mapping instead of an arbitrary serialized module.
- Fingerprint checks must read `preprocess_signature` from a top-level manifest/meta field first, while still accepting older artifacts that stored it inside `encoder_fingerprint`.

Interpretation rule:
- If the sidecars are missing, the artifact is not considered scientifically attributable.
- If external labels were harmonized or figures were generated, the corresponding manifest/figure sidecars are part of the attributable artifact surface.
