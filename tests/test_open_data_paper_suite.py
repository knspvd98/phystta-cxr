import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

from phystta.data.templates import OPEN_CXR_CANONICAL_LABELS


def test_open_data_mode_bypasses_real_pair_realism_requirement(tmp_path: Path):
    repo = Path(__file__).resolve().parents[1]
    out_dir = tmp_path / "suite_out"
    labels = list(OPEN_CXR_CANONICAL_LABELS)
    for name in ("padchest", "chexpert", "nih_chestxray14"):
        (tmp_path / name).mkdir(parents=True)

    cfg = {
        "data": {
            "label_cols": labels,
            "split_group_key": "global_patient_id",
            "sources": [
                {"name": "padchest", "root": str(tmp_path / "padchest"), "label_cols": labels},
                {"name": "chexpert", "root": str(tmp_path / "chexpert"), "label_cols": labels},
                {"name": "nih_chestxray14", "root": str(tmp_path / "nih_chestxray14"), "label_cols": labels},
            ],
        },
        "model": {
            "num_classes": len(labels),
            "encoder": {
                "encoder_source": "baseline_cnn",
                "architecture": "densenet121",
                "pretrained": False,
            },
        },
        "intervention": {
            "use_realism": True,
            "group_key": "global_patient_id",
            "intensity_grid": [0.0, 0.2],
        },
        "erasure_eval": {
            "modes": ["clinical"],
            "intensity_grid": [0.0, 0.2],
        },
        "runtime": {
            "allow_download": True,
            "device": "cpu",
        },
        "experiment": {
            "tier": "dev",
            "open_data_fallback": True,
        },
    }
    cfg_path = tmp_path / "open_cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    cmd = [
        sys.executable,
        str(repo / "scripts" / "run_paper_suite.py"),
        "--config",
        str(cfg_path),
        "--eraser",
        str(tmp_path / "eraser.pt"),
        "--out_dir",
        str(out_dir),
        "--dry_run",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    subprocess.run(cmd, check=True, cwd=str(repo), env=env)

    plan = out_dir / "paper_suite_plan.json"
    payload = json.loads(plan.read_text(encoding="utf-8"))

    assert payload["run_mode"]["open_data_fallback"] is True
    assert payload["run_mode"]["full_paper_tier"] is False
    assert payload["run_mode"]["real_pair_realism_enabled"] is False
    assert payload["run_mode"]["real_anchor_enabled"] is False
    notes = "\n".join(payload.get("preflight", {}).get("notes", []))
    assert "Open-data fallback run enabled" in notes
    assert "Sources without real-pair metadata" in notes
    assert "real-pair realism is disabled" in notes.lower()
    assert "synthetic intervention grid only" in notes.lower()
    assert "real-anchor evaluation is skipped" in notes.lower()

    joined = "\n".join(" ".join(cmd) for cmd in payload["commands"])
    assert "scripts/run_erasure_eval.py" in joined
    assert "scripts/run_safety_policy_eval.py" in joined
    assert "scripts/calibrate_physics_realism.py" not in joined
    assert "scripts/run_real_anchor_eval.py" not in joined


def test_open_data_fallback_is_not_automatic(tmp_path: Path):
    repo = Path(__file__).resolve().parents[1]
    out_dir = tmp_path / "suite_out"
    labels = list(OPEN_CXR_CANONICAL_LABELS)
    for name in ("padchest", "chexpert", "nih_chestxray14"):
        (tmp_path / name).mkdir(parents=True)

    cfg = {
        "data": {
            "label_cols": labels,
            "split_group_key": "global_patient_id",
            "sources": [
                {"name": "padchest", "root": str(tmp_path / "padchest"), "label_cols": labels},
                {"name": "chexpert", "root": str(tmp_path / "chexpert"), "label_cols": labels},
                {"name": "nih_chestxray14", "root": str(tmp_path / "nih_chestxray14"), "label_cols": labels},
            ],
        },
        "model": {
            "num_classes": len(labels),
            "encoder": {
                "encoder_source": "baseline_cnn",
                "architecture": "densenet121",
                "pretrained": False,
            },
        },
        "intervention": {
            "use_realism": True,
            "group_key": "global_patient_id",
            "intensity_grid": [0.0, 0.2],
        },
        "erasure_eval": {
            "modes": ["clinical"],
            "intensity_grid": [0.0, 0.2],
        },
        "runtime": {
            "allow_download": True,
            "device": "cpu",
        },
        "experiment": {
            "tier": "dev",
            "open_data_fallback": False,
        },
    }
    cfg_path = tmp_path / "strict_cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    cmd = [
        sys.executable,
        str(repo / "scripts" / "run_paper_suite.py"),
        "--config",
        str(cfg_path),
        "--eraser",
        str(tmp_path / "eraser.pt"),
        "--out_dir",
        str(out_dir),
        "--dry_run",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    proc = subprocess.run(cmd, cwd=str(repo), env=env, capture_output=True, text=True)

    assert proc.returncode != 0
    assert "run_paper_suite requires experiment.tier=paper" in (proc.stderr + proc.stdout)
