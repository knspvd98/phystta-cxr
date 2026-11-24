from pathlib import Path


def test_calibration_entrypoint_is_unique():
    scripts = sorted(Path("scripts").glob("calibrate_physics_realism*.py"))
    assert len(scripts) == 1, f"Expected one calibrate_physics_realism entrypoint, found: {[s.name for s in scripts]}"


def test_real_anchor_entrypoint_is_unique():
    scripts = sorted(Path("scripts").glob("run_real_anchor_eval*.py"))
    assert len(scripts) == 1, f"Expected one run_real_anchor_eval entrypoint, found: {[s.name for s in scripts]}"


def test_calibration_script_uses_group_key_filtering():
    text = Path("scripts/calibrate_physics_realism.py").read_text(encoding="utf-8")
    assert "pairs_df[group_key]" in text
    assert "groups=pairs_td[group_key]" in text
