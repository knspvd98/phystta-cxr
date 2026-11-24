# Dataflow Map (2026-2027 Spine)

## Core inference path
1. Dataset index and splits: `phystta/data/engine.py`
2. Sample loading and DICOM normalization: `phystta/data/datasets.py`
3. Encoder preprocessing and feature extraction: `phystta/models/encoder_factory.py`
4. Representation surgery:
   - no eraser: `z = encoder.encode(x)`
   - with eraser: `z' = eraser(z)` or token wrapper path
5. Prediction head: `head(z)`
6. Metrics and reports: `phystta/metrics/*`, scripts under `scripts/`

## Intervention path
1. Real image batch `x`
2. Controlled shift generation: `phystta/shifts/manager.py`
3. Shifted batch `x_shift`
4. Feature extraction on clean/shifted
5. Probe collection and eraser fitting: `scripts/collect_probes.py`, `scripts/fit_eraser.py`

## Causal path
1. Structural target and estimands: `phystta/causal/objectives.py`
2. Paired quasi-experimental effect: same-patient short-window technical pairs
3. Pair quality gate (technical-dominant vs ambiguous): `phystta/causal/pair_quality.py`
4. Dose-response sensitivity: `dose_response_slope`

## Safety path
1. Shift scores/type estimation: `phystta/safety/shift_id.py`
2. Decision policy:
   - legacy threshold policy: `phystta/safety/policy.py`
   - decision-theoretic policy: `phystta/safety/decision.py`
3. Coverage/risk application and reporting: `scripts/run_safety_policy_eval.py`
4. Harm/regret decomposition: `phystta/safety/policy_harm.py`

## Erasure certificates
1. Adaptive post-erasure adversaries (linear/kernel/MLP): `phystta/intervention/adversary_eval.py`
2. Unified detectability+distortion certificate: `phystta/intervention/erasure_certificates.py`

## Monitoring path
1. Baseline calibration window
2. Streaming score updates
3. Alert logic and event logs
4. Module: `phystta/monitoring/shift_monitor.py`
5. Script: `scripts/run_shift_monitor.py`
