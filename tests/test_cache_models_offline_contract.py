import os
import subprocess
import sys
from pathlib import Path

import pytest


def test_cache_models_rejects_download_when_runtime_disallows(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    report = tmp_path / "cache_report.json"
    cmd = [
        sys.executable,
        str(repo / "scripts" / "cache_models.py"),
        "--config",
        str(repo / "configs" / "representation_surgery_2026.yaml"),
        "--profiles",
        "study",
        "--out_report",
        str(report),
        "--download",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    with pytest.raises(subprocess.CalledProcessError):
        subprocess.run(cmd, check=True, cwd=str(repo), env=env)
