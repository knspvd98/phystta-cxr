import json
import os
import subprocess
import sys
from pathlib import Path


def test_run_paper_suite_dry_run(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    out_dir = tmp_path / "suite_out"
    cmd = [
        sys.executable,
        str(repo / "scripts" / "run_paper_suite.py"),
        "--config",
        str(repo / "configs" / "representation_surgery_2026.yaml"),
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
    assert plan.exists()
    payload = json.loads(plan.read_text(encoding="utf-8"))
    assert "commands" in payload
    assert len(payload["commands"]) >= 4
    assert "code_sha256" in payload and isinstance(payload["code_sha256"], str)
    assert len(payload["code_sha256"]) == 64
    assert "code_sha256_files" in payload and isinstance(payload["code_sha256_files"], dict)
    assert "scripts/run_paper_suite.py" in payload["code_sha256_files"]
    preflight = payload.get("preflight", {})
    errors = preflight.get("errors", [])
    assert any("Eraser artifact not found" in str(e) for e in errors)
    joined = "\n".join(" ".join(c) for c in payload["commands"])
    assert "build_paper_report.py" in joined
    assert "build_dissertation_artifacts.py" in joined


def test_run_paper_suite_dry_run_with_external_eval(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    out_dir = tmp_path / "suite_out_ext"
    cmd = [
        sys.executable,
        str(repo / "scripts" / "run_paper_suite.py"),
        "--config",
        str(repo / "configs" / "representation_surgery_2026.yaml"),
        "--eraser",
        str(tmp_path / "eraser.pt"),
        "--out_dir",
        str(out_dir),
        "--external_eval_sources",
        "externalA,externalB",
        "--dry_run",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    subprocess.run(cmd, check=True, cwd=str(repo), env=env)

    plan = out_dir / "paper_suite_plan.json"
    payload = json.loads(plan.read_text(encoding="utf-8"))
    joined = "\n".join(" ".join(c) for c in payload["commands"])
    assert "run_external_policy_eval.py" in joined
