import json
import os
import subprocess
import sys
from pathlib import Path


def test_architecture_manifest_includes_pinned_profile_specs(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    out_dir = tmp_path / "arch"
    cmd = [
        sys.executable,
        str(repo / "scripts" / "run_architecture_matrix.py"),
        "--config",
        str(repo / "configs" / "representation_surgery_2026.yaml"),
        "--out_dir",
        str(out_dir),
        "--aggregate_only",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    subprocess.run(cmd, check=True, cwd=str(repo), env=env)

    manifest_path = out_dir / "architecture_protocol_manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text())

    assert manifest["experiment_tier"] == "paper"
    assert manifest["runtime"]["allow_download"] is False
    assert manifest["runtime"]["freeze_artifacts"] is True

    specs = manifest["profile_specs"]
    assert isinstance(specs, list) and len(specs) > 0
    mutable = {"main", "master", "head", "latest"}
    for spec in specs:
        src = str(spec.get("encoder_source", ""))
        rev = spec.get("revision")
        if src.startswith("hf_") and spec.get("model_id"):
            assert isinstance(rev, str) and rev.strip()
            assert rev.lower() not in mutable

    code_hashes = manifest.get("code_sha256")
    assert isinstance(code_hashes, dict) and len(code_hashes) > 0
    for rel, sha in code_hashes.items():
        assert isinstance(rel, str) and rel
        assert isinstance(sha, str) and len(sha) == 64
