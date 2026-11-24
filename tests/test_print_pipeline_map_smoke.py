import json
import os
import subprocess
import sys
from pathlib import Path


def test_print_pipeline_map_from_plan(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    plan = tmp_path / "paper_suite_plan.json"
    plan.write_text(
        json.dumps(
            {
                "config": "configs/representation_surgery_2026.yaml",
                "eraser": "outputs/representation_surgery/eraser.pt",
                "preflight": {"errors": [], "notes": ["ok"]},
                "commands": [
                    [sys.executable, "scripts/run_erasure_eval.py", "--config", "cfg.yaml"],
                    [sys.executable, "scripts/run_safety_policy_eval.py", "--config", "cfg.yaml"],
                ],
                "artifact_dirs": {"erasure": "out/erasure", "safety": "out/safety"},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    out_json = tmp_path / "pipeline_map.json"
    cmd = [
        sys.executable,
        str(repo / "scripts" / "print_pipeline_map.py"),
        "--plan",
        str(plan),
        "--save_json",
        str(out_json),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    subprocess.run(cmd, check=True, cwd=str(repo), env=env)
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert len(payload["entrypoints"]) == 2
    assert payload["entrypoints"][0]["script"] == "scripts/run_erasure_eval.py"
