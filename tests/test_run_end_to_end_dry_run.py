import json
import os
import subprocess
import sys
from pathlib import Path


def test_run_end_to_end_dry_run_writes_manifest(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    out_dir = tmp_path / "e2e_out"
    cmd = [
        sys.executable,
        str(repo / "scripts" / "run_end_to_end.py"),
        "--config",
        str(repo / "configs" / "representation_surgery_2026.yaml"),
        "--out_dir",
        str(out_dir),
        "--run_id",
        "smoke",
        "--dry_run",
        "--skip_realism_calibration",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    subprocess.run(cmd, check=True, cwd=str(repo), env=env)

    manifest = out_dir / "smoke" / "run_manifest.json"
    assert manifest.exists()
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    joined = "\n".join(" ".join(c) for c in payload["commands"])
    assert "train_linear_head.py" in joined
    assert "collect_probes.py" in joined
    assert "fit_eraser.py" in joined
    assert "run_paper_suite.py" in joined
