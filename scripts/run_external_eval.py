#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make wrapper resilient when called from different CWD/module contexts.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from scripts.run_external_policy_eval import main as external_policy_main
except Exception:
    from run_external_policy_eval import main as external_policy_main


def main():
    parser = argparse.ArgumentParser(description="External frozen eval wrapper")
    parser.add_argument("--config", required=True)
    parser.add_argument("--eraser", required=True)
    parser.add_argument("--shift_id_model", required=True)
    parser.add_argument("--policy_json", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--eval_sources", required=True)
    parser.add_argument("--shift_type", default=None)
    parser.add_argument("--intensity_grid", default=None)
    parser.add_argument("--encoder_profile", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--bootstrap_samples", type=int, default=None)
    parser.add_argument("--target_sensitivities", default=None)
    parser.add_argument("--target_specificities", default=None)
    args = parser.parse_args()
    print(
        f"[run_external_eval] stage_start config={args.config} out_dir={args.out_dir} shift_type={args.shift_type or 'auto'}",
        flush=True,
    )
    # delegate to policy eval
    import sys
    argv = ["run_external_policy_eval.py"]
    for k, v in vars(args).items():
        if v is None:
            continue
        flag = f"--{k.replace('_','-')}"
        argv.extend([flag, str(v)])
    sys.argv = argv
    print(f"[run_external_eval] delegated_argv={' '.join(argv[1:])}", flush=True)
    try:
        external_policy_main()
    except Exception as exc:
        print(
            f"[run_external_eval] failure_context out_dir={args.out_dir} policy_json={args.policy_json} error={type(exc).__name__}:{exc}",
            flush=True,
        )
        raise
    print(f"[run_external_eval] output_materialized_hint out_dir={args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
