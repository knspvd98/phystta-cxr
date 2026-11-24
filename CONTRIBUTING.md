# Contributing

PhysTTA is a research project. Small, reviewable changes with a clear scientific or engineering rationale are welcome.

1. Create a focused branch.
2. Install the project with `pip install -e ".[test]"`.
3. Add or update tests for changed behavior.
4. Run `pytest` before opening a pull request.
5. Describe any effect on dataset assumptions, artifact compatibility, determinism, or reported metrics.

Do not commit clinical images, protected metadata, model weights without redistribution rights, access credentials, or generated experiment outputs. New dataset adapters must document their license, patient-level grouping key, label mapping, and missing-value policy.
