from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


SCRIPT_MAP = {
    "public-mainline": "scripts/run_public_mainline.py",
    "paper-suite": "scripts/run_paper_suite.py",
    "end-to-end": "scripts/run_end_to_end.py",
    "architecture-matrix": "scripts/run_architecture_matrix.py",
    "safety-eval": "scripts/run_safety_policy_eval.py",
    "erasure-eval": "scripts/run_erasure_eval.py",
    "confounding-audit": "scripts/run_confounding_audit.py",
    "real-anchor": "scripts/run_real_anchor_eval.py",
    "validate-env": "scripts/validate_env.py",
}

PIPELINE_NOTES = {
    "public-mainline": "recommended bounded public acquisition-context launcher",
    "paper-suite": "historical umbrella alias; prefer public-mainline",
    "end-to-end": "historical convenience wrapper",
    "safety-eval": "canonical safety-policy evaluation entrypoint",
    "erasure-eval": "canonical erasure evaluation entrypoint",
    "confounding-audit": "confounding audit for acquisition-context residual shortcut risk",
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _run_script(script: str, script_args: list[str]) -> int:
    root = _repo_root()
    if not (root / script).is_file():
        root = Path.cwd()
    script_path = root / script
    if not script_path.is_file():
        print("Pipeline scripts require a checkout of the PhysTTA repository.", file=sys.stderr)
        return 2
    cmd = [sys.executable, str(script_path), *script_args]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    proc = subprocess.run(cmd, cwd=str(root), env=env)
    return int(proc.returncode)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PhysTTA CLI (public acquisition-context mainline)")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run a pipeline/script entrypoint")
    run.add_argument(
        "pipeline",
        choices=sorted(SCRIPT_MAP.keys()),
        help="Named pipeline entrypoint.",
    )
    run.add_argument("args", nargs=argparse.REMAINDER, help="Arguments passed to selected pipeline script.")

    list_cmd = sub.add_parser("list", help="List available pipeline entrypoints and canonical/historical notes")
    list_cmd.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "list":
        if args.json:
            import json

            print(json.dumps(SCRIPT_MAP, indent=2))
        else:
            for name in sorted(SCRIPT_MAP.keys()):
                note = PIPELINE_NOTES.get(name, "")
                suffix = f" [{note}]" if note else ""
                print(f"{name:18s} -> {SCRIPT_MAP[name]}{suffix}")
        return

    script = SCRIPT_MAP[args.pipeline]
    script_args = list(args.args or [])
    if script_args and script_args[0] == "--":
        script_args = script_args[1:]
    code = _run_script(script, script_args)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
