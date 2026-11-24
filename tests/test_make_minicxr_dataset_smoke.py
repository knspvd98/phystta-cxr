import json
import os
import subprocess
import sys
from pathlib import Path


def test_make_minicxr_dataset_smoke(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    out_dir = tmp_path / "mini"
    cmd = [
        sys.executable,
        str(repo / "scripts" / "make_minicxr_dataset.py"),
        "--out_dir",
        str(out_dir),
        "--n_patients",
        "8",
        "--write_config_patch",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    subprocess.run(cmd, check=True, cwd=str(repo), env=env)
    report = json.loads((out_dir / "minicxr_report.json").read_text(encoding="utf-8"))
    assert report["n_patients"] == 8
    assert (out_dir / "metadata.csv").exists()
    assert (out_dir / "minicxr_config_patch.json").exists()
