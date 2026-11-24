# Shift Physics Semantics

The current atomic shift families are physics-inspired acquisition-context stressors, not a complete simulator of all clinically relevant technical variability.

Current atomic families:
- `heel`
  - Models: a heel-effect-like fluence gradient across the field.
  - Does not model: true device provenance, detector gain maps, or full beam-geometry changes.
  - Approximation: deterministic multiplicative field on normalized intensity.
- `scatter`
  - Models: low-frequency veiling/scatter contamination.
  - Does not model: anti-scatter grid behavior or full Monte Carlo scatter transport.
  - Approximation: attenuation-informed low-pass additive proxy.
- `exposure`
  - Models: effective dose reduction with AEC-like renormalization.
  - Does not model: exact tube settings, beam hardening, detector saturation, or full acquisition physics.
  - Approximation: dose scaling plus AEC normalization.
- `poisson`
  - Models: quantum-limited shot noise plus a small electronic term.
  - Does not model: independent detector-processing noise or a fully separate non-dose noise process.
  - Approximation: Poisson sampling under the same effective dose axis used by `exposure`.
- `exposure_poisson`
  - Models: a coupled dose-reduction plus quantum-noise stressor where lower effective exposure directly raises noise burden.
  - Does not model: true tube-current trajectories, beam hardening, detector saturation, or detector-specific post-processing.
  - Approximation: one coupled dose-noise path with AEC-like renormalization. This is the cleaner canonical atomic family for dose/noise realism.
- `motion_blur`
  - Models: directional motion blur as a proxy for respiratory, patient, or bedside motion during acquisition.
  - Does not model: full rigid-body repositioning, multi-directional motion trajectories, or detector readout smear.
  - Approximation: fixed-axis line-spread blur in image space.

Critical dependence note:
- `exposure` and `poisson` are not orthogonal mechanisms in the current implementation.
- Both families operate through an effective dose/noise approximation.
- Outputs and manifests now surface this as a `physical_dependence_warning` instead of implying independent ground-truth mechanisms.
- `exposure` and `poisson` remain available as legacy separable stressors, but canonical public-mainline atomic configs should prefer `exposure_poisson`.

Coverage limits:
- The current realism family is partial, not exhaustive.
- Important technical variability still outside the implemented family:
  - stronger blur / MTF degradation beyond the current proxies
  - complex motion beyond the current directional blur proxy
  - rotation / positioning
  - collimation / crop
  - grid / detector processing effects

Interpretation rule:
- Acquisition-context robustness under these atomic shifts must not be overclaimed as full technical robustness or true device provenance robustness.
