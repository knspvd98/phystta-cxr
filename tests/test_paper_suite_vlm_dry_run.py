import json
import os
import subprocess
import sys
from pathlib import Path

import yaml


def test_run_paper_suite_dry_run_includes_vlm_stage_when_enabled(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    cfg_src = repo / "configs" / "representation_surgery_2026.yaml"
    cfg_dst = tmp_path / "cfg_vlm.yaml"
    cfg = yaml.safe_load(cfg_src.read_text(encoding="utf-8"))

    # Minimal patch: enable VLM and point to an existing local model path directory.
    local_vlm = tmp_path / "local_vlm"
    local_vlm.mkdir(parents=True, exist_ok=True)
    cfg.setdefault("vlm", {})
    cfg["vlm"]["enabled"] = True
    cfg["vlm"]["model_id"] = None
    cfg["vlm"]["model_path"] = str(local_vlm)
    cfg_dst.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    out_dir = tmp_path / "suite_out_vlm"
    cmd = [
        sys.executable,
        str(repo / "scripts" / "run_paper_suite.py"),
        "--config",
        str(cfg_dst),
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
    joined = "\n".join(" ".join(c) for c in payload["commands"])
    assert "run_vlm_erasure_eval.py" in joined
