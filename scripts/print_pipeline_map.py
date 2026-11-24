#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _entrypoint(cmd: list[str]) -> str:
    for token in cmd:
        if token.endswith(".py"):
            return token
    return cmd[0] if cmd else ""


def main() -> None:
    ap = argparse.ArgumentParser(description="Print a compact map of paper-suite entrypoints and artifacts.")
    ap.add_argument("--plan", default=None, help="Path to paper_suite_plan.json.")
    ap.add_argument("--config", default=None, help="If --plan is omitted, create dry-run plan from this config.")
    ap.add_argument("--eraser", default=None, help="Eraser path (required with --config).")
    ap.add_argument("--out_dir", default=None, help="Output dir for dry-run plan (required with --config).")
    ap.add_argument("--encoder_profile", default=None)
    ap.add_argument("--shift_type", default=None)
    ap.add_argument("--save_json", default=None, help="Optional path to write normalized pipeline map JSON.")
    args = ap.parse_args()

    if args.plan is None:
        if not (args.config and args.eraser and args.out_dir):
            raise SystemExit("Provide --plan, or provide --config --eraser --out_dir.")
        out_dir = Path(args.out_dir).expanduser()
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            "scripts/run_paper_suite.py",
            "--config",
            str(args.config),
            "--eraser",
            str(args.eraser),
            "--out_dir",
            str(out_dir),
            "--dry_run",
        ]
        if args.encoder_profile:
            cmd += ["--encoder_profile", args.encoder_profile]
        if args.shift_type:
            cmd += ["--shift_type", args.shift_type]
        subprocess.run(cmd, check=True)
        plan_path = out_dir / "paper_suite_plan.json"
    else:
        plan_path = Path(args.plan).expanduser()

    if not plan_path.exists():
        raise SystemExit(f"Plan file not found: {plan_path}")

    payload = _load_json(plan_path)
    commands = payload.get("commands", [])
    preflight = payload.get("preflight", {})
    artifacts = payload.get("artifact_dirs", {})

    mapped = {
        "plan_path": str(plan_path),
        "config": payload.get("config"),
        "eraser": payload.get("eraser"),
        "entrypoints": [
            {"index": i + 1, "script": _entrypoint(cmd), "cmd": cmd}
            for i, cmd in enumerate(commands)
        ],
        "preflight_errors": preflight.get("errors", []),
        "preflight_notes": preflight.get("notes", []),
        "artifact_dirs": artifacts,
    }

    print(f"plan: {mapped['plan_path']}")
    print(f"config: {mapped.get('config')}")
    print(f"eraser: {mapped.get('eraser')}")
    if mapped["preflight_errors"]:
        print("preflight_errors:")
        for e in mapped["preflight_errors"]:
            print(f"  - {e}")
    else:
        print("preflight_errors: none")
    print("entrypoints:")
    for row in mapped["entrypoints"]:
        print(f"  {row['index']:02d}. {row['script']}")
    print("artifact_dirs:")
    for k, v in mapped["artifact_dirs"].items():
        print(f"  - {k}: {v}")

    if args.save_json:
        out = Path(args.save_json).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(mapped, indent=2), encoding="utf-8")
        print(f"saved: {out}")


if __name__ == "__main__":
    main()
