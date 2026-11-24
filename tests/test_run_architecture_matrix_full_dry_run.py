import json
import os
import subprocess
import sys
from pathlib import Path


def test_run_architecture_matrix_full_dry_run_writes_manifest(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    out_dir = tmp_path / "arch_full"
    cmd = [
        sys.executable,
        str(repo / "scripts" / "run_architecture_matrix_full.py"),
        "--config",
        str(repo / "configs" / "representation_surgery_2026.yaml"),
        "--out_dir",
        str(out_dir),
        "--profiles",
        "hf_vit",
        "--dry_run",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    subprocess.run(cmd, check=True, cwd=str(repo), env=env)

    manifest = out_dir / "architecture_matrix_full_manifest.json"
    assert manifest.exists()
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert "hf_vit" in payload["profiles"]
    joined = "\n".join(" ".join(c) for c in payload["commands"]["hf_vit"])
    assert "train_linear_head.py" in joined
    assert "collect_probes.py" in joined
    assert "fit_eraser.py" in joined
    assert "run_erasure_eval.py" in joined
    assert "run_safety_policy_eval.py" in joined
