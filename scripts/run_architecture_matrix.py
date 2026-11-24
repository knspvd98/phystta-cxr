#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

from phystta.config.loader import load_config
from phystta.studies import (
    infer_paradigm,
    infer_pretraining,
    build_interaction_table,
    build_robustness_table,
    summarize_architecture_matrix,
)

def _profile_names(cfg) -> List[str]:
    if cfg.model.encoder_profiles:
        return [p.name for p in cfg.model.encoder_profiles]
    return ["__default__"]


def _profile_meta(cfg, profile: str, tags: Dict[str, Dict[str, str]]) -> Dict[str, str]:
    if profile in tags:
        return {
            "paradigm": tags[profile].get("paradigm", "Other"),
            "pretraining": tags[profile].get("pretraining", "Unknown"),
            "training_regime": tags[profile].get("training_regime", tags[profile].get("training", "unspecified")),
            "capacity_m": tags[profile].get("capacity_m", ""),
        }
    enc = cfg.model.encoder
    if profile != "__default__" and cfg.model.encoder_profiles:
        matches = [p for p in cfg.model.encoder_profiles if p.name == profile]
        if matches:
            enc = matches[0].encoder
    return {
        "paradigm": infer_paradigm(enc.encoder_source),
        "pretraining": infer_pretraining(enc.encoder_source, enc.model_id),
        "training_regime": "fine_tune",
        "capacity_m": "",
    }


def _run(cmd: List[str], log):
    log("RUN: " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def _sha256_if_exists(path: Path) -> str | None:
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run architecture matrix study (profiles x shifts x interventions)")
    parser.add_argument("--config", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--eraser_template", default=None, help="Path template with {profile}, e.g. outputs/rs/{profile}/eraser.pt")
    parser.add_argument("--shift_type", default=None)
    parser.add_argument("--intensity_grid", default=None)
    parser.add_argument("--modes", default=None, help="comma-separated modes for run_erasure_eval")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--aggregate_only", action="store_true")
    args = parser.parse_args()

    cfg_path = Path(args.config).expanduser()
    cfg = load_config(str(cfg_path))
    study_cfg = cfg.architecture_study
    tags = study_cfg.profile_tags or {}

    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "log.txt"

    def log(msg: str):
        print(msg)
        log_path.open("a", encoding="utf-8").write(msg + "\n")

    profile_names = list(study_cfg.encoder_profiles or _profile_names(cfg))
    interventions = list(study_cfg.interventions or ["erasure", "policy"])
    eraser_paths_cfg: Dict[str, str] = dict(study_cfg.eraser_paths or {})
    clean_intensity = float(study_cfg.reporting.clean_intensity)
    interaction_pairs = list(study_cfg.interaction_pairs or [["no_eraser", "eraser"]])
    modes = args.modes or ",".join(study_cfg.modes or (cfg.erasure_eval.modes or [cfg.erasure_eval.mode]))

    all_metrics = []
    all_fairness = []
    all_causal = []
    all_safety = []
    all_policy_decisions = []

    profile_map = {p.name: p.encoder for p in (cfg.model.encoder_profiles or [])}

    for profile in profile_names:
        profile_name = str(profile)
        meta = _profile_meta(cfg, profile_name, tags)
        profile_dir = out_dir / profile_name
        eval_dir = profile_dir / "eval"
        safety_dir = profile_dir / "safety"
        profile_dir.mkdir(parents=True, exist_ok=True)

        eraser_path = eraser_paths_cfg.get(profile_name)
        if eraser_path is None and args.eraser_template:
            eraser_path = args.eraser_template.format(profile=profile_name)
        if eraser_path is None:
            # fallback convention
            candidate = Path(f"outputs/representation_surgery/{profile_name}/eraser.pt")
            if candidate.exists():
                eraser_path = str(candidate)

        if not args.aggregate_only:
            if "erasure" in interventions:
                if eraser_path is None:
                    raise RuntimeError(f"Eraser path missing for profile={profile_name}. Provide architecture_study.eraser_paths or --eraser_template")
                cmd = [
                    sys.executable,
                    "scripts/run_erasure_eval.py",
                    "--config",
                    str(cfg_path),
                    "--eraser",
                    str(eraser_path),
                    "--out_dir",
                    str(eval_dir),
                    "--modes",
                    modes,
                ]
                if profile_name != "__default__":
                    cmd += ["--encoder_profile", profile_name]
                if args.shift_type:
                    cmd += ["--shift_type", args.shift_type]
                if args.intensity_grid:
                    cmd += ["--intensity_grid", args.intensity_grid]
                if args.seed is not None:
                    cmd += ["--seed", str(args.seed)]
                if args.device:
                    cmd += ["--device", args.device]
                _run(cmd, log)

            if "policy" in interventions:
                if eraser_path is None:
                    raise RuntimeError(f"Eraser path missing for profile={profile_name}. Required for policy evaluation.")
                shift_id_path = safety_dir / "shift_id.pkl"
                policy_json_path = safety_dir / "policy.json"
                cmd = [
                    sys.executable,
                    "scripts/run_safety_policy_eval.py",
                    "--config",
                    str(cfg_path),
                    "--eraser",
                    str(eraser_path),
                    "--out_dir",
                    str(safety_dir),
                    "--save_shift_id",
                    str(shift_id_path),
                    "--save_policy",
                    str(policy_json_path),
                ]
                if profile_name != "__default__":
                    cmd += ["--encoder_profile", profile_name]
                if args.shift_type:
                    cmd += ["--shift_type", args.shift_type]
                if args.intensity_grid:
                    cmd += ["--intensity_grid", args.intensity_grid]
                if args.seed is not None:
                    cmd += ["--seed", str(args.seed)]
                if args.device:
                    cmd += ["--device", args.device]
                _run(cmd, log)

        # aggregate outputs if present
        m_path = eval_dir / "metrics.csv"
        if m_path.exists():
            df = pd.read_csv(m_path)
            df["profile"] = profile_name
            df["paradigm"] = meta["paradigm"]
            df["pretraining"] = meta["pretraining"]
            df["training_regime"] = meta["training_regime"]
            df["capacity_m"] = meta["capacity_m"]
            all_metrics.append(df)
        f_path = eval_dir / "fairness.csv"
        if f_path.exists():
            df = pd.read_csv(f_path)
            df["profile"] = profile_name
            df["paradigm"] = meta["paradigm"]
            df["pretraining"] = meta["pretraining"]
            df["training_regime"] = meta["training_regime"]
            df["capacity_m"] = meta["capacity_m"]
            all_fairness.append(df)
        c_path = eval_dir / "causal_invariance.csv"
        if c_path.exists():
            df = pd.read_csv(c_path)
            df["profile"] = profile_name
            df["paradigm"] = meta["paradigm"]
            df["pretraining"] = meta["pretraining"]
            df["training_regime"] = meta["training_regime"]
            df["capacity_m"] = meta["capacity_m"]
            all_causal.append(df)
        s_path = safety_dir / "safety.csv"
        if s_path.exists():
            df = pd.read_csv(s_path)
            df["profile"] = profile_name
            df["paradigm"] = meta["paradigm"]
            df["pretraining"] = meta["pretraining"]
            df["training_regime"] = meta["training_regime"]
            df["capacity_m"] = meta["capacity_m"]
            all_safety.append(df)
        d_path = safety_dir / "policy_decisions.csv"
        if d_path.exists():
            df = pd.read_csv(d_path)
            df["profile"] = profile_name
            df["paradigm"] = meta["paradigm"]
            df["pretraining"] = meta["pretraining"]
            df["training_regime"] = meta["training_regime"]
            df["capacity_m"] = meta["capacity_m"]
            all_policy_decisions.append(df)

    # save consolidated artifacts
    metrics_df = pd.concat(all_metrics, ignore_index=True) if all_metrics else pd.DataFrame()
    fairness_df = pd.concat(all_fairness, ignore_index=True) if all_fairness else pd.DataFrame()
    causal_df = pd.concat(all_causal, ignore_index=True) if all_causal else pd.DataFrame()
    safety_df = pd.concat(all_safety, ignore_index=True) if all_safety else pd.DataFrame()
    policy_df = pd.concat(all_policy_decisions, ignore_index=True) if all_policy_decisions else pd.DataFrame()

    if not metrics_df.empty:
        metrics_df.to_csv(out_dir / "architecture_matrix_metrics.csv", index=False)
        interaction_frames = []
        for pair in interaction_pairs:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                raise RuntimeError(f"architecture_study.interaction_pairs entries must be [baseline, compare], got: {pair}")
            base, cmp = str(pair[0]), str(pair[1])
            inter = build_interaction_table(metrics_df, baseline_method=base, compare_method=cmp)
            if not inter.empty:
                interaction_frames.append(inter)
        interactions = pd.concat(interaction_frames, ignore_index=True) if interaction_frames else pd.DataFrame()
        interactions.to_csv(out_dir / "architecture_interactions.csv", index=False)
        robustness = build_robustness_table(metrics_df, clean_intensity=clean_intensity)
        robustness.to_csv(out_dir / "architecture_robustness.csv", index=False)
        if not robustness.empty:
            agg_cols = ["degradation"]
            if "relative_degradation" in robustness.columns:
                agg_cols.append("relative_degradation")
            robustness_summary = (
                robustness.groupby(["paradigm", "method", "metric"], dropna=False)[agg_cols]
                .agg(["mean", "std", "count"])
                .reset_index()
            )
            robustness_summary.columns = [
                c if isinstance(c, str) else (c[0] if c[1] == "" else f"{c[0]}_{c[1]}")
                for c in robustness_summary.columns
            ]
            robustness_summary.to_csv(out_dir / "architecture_robustness_summary.csv", index=False)
        else:
            robustness_summary = pd.DataFrame()
        summary = summarize_architecture_matrix(metrics_df)
        summary.to_csv(out_dir / "architecture_summary.csv", index=False)
    else:
        interactions = pd.DataFrame()
        robustness = pd.DataFrame()
        robustness_summary = pd.DataFrame()
        summary = pd.DataFrame()

    if not fairness_df.empty:
        fairness_df.to_csv(out_dir / "architecture_fairness.csv", index=False)
    if not causal_df.empty:
        causal_df.to_csv(out_dir / "architecture_causal.csv", index=False)
    if not safety_df.empty:
        safety_df.to_csv(out_dir / "architecture_safety.csv", index=False)
    if not policy_df.empty:
        policy_df.to_csv(out_dir / "architecture_policy_decisions.csv", index=False)

    report = {
        "profiles": profile_names,
        "interventions": interventions,
        "rows_metrics": int(len(metrics_df)),
        "rows_interactions": int(len(interactions)),
        "rows_robustness": int(len(robustness)),
        "rows_robustness_summary": int(len(robustness_summary)),
        "rows_summary": int(len(summary)),
        "rows_fairness": int(len(fairness_df)),
        "rows_causal": int(len(causal_df)),
        "rows_safety": int(len(safety_df)),
        "rows_policy_decisions": int(len(policy_df)),
    }
    key_code_files = [
        "phystta/config/schema.py",
        "phystta/config/loader.py",
        "phystta/models/encoder_factory.py",
        "phystta/shifts/realism.py",
        "scripts/run_erasure_eval.py",
        "scripts/run_safety_policy_eval.py",
        "scripts/run_architecture_matrix.py",
    ]
    code_hashes = {
        rel: _sha256_if_exists(Path(rel)) for rel in key_code_files
    }
    protocol_manifest = {
        "config_path": str(cfg_path),
        "config_sha256": hashlib.sha256(cfg_path.read_bytes()).hexdigest(),
        "experiment_tier": cfg.experiment.tier,
        "runtime": {
            "allow_download": cfg.runtime.allow_download,
            "freeze_artifacts": cfg.runtime.freeze_artifacts,
            "seed": cfg.runtime.seed,
        },
        "profiles": profile_names,
        "profile_specs": [
            {
                "name": p,
                "encoder_source": (
                    profile_map[p].encoder_source
                    if p in profile_map
                    else cfg.model.encoder.encoder_source
                ),
                "model_id": (
                    profile_map[p].model_id
                    if p in profile_map
                    else cfg.model.encoder.model_id
                ),
                "revision": (
                    profile_map[p].revision
                    if p in profile_map
                    else cfg.model.encoder.revision
                ),
                "encoder_path": str(profile_map[p].encoder_path) if p in profile_map and profile_map[p].encoder_path is not None else None,
                "head_path": str(profile_map[p].head_path) if p in profile_map and profile_map[p].head_path is not None else None,
                "head_meta_path": str(profile_map[p].head_meta_path) if p in profile_map and getattr(profile_map[p], "head_meta_path", None) is not None else None,
            }
            for p in profile_names
        ],
        "interventions": interventions,
        "shift_types": args.shift_type.split(",") if args.shift_type and "," in args.shift_type else ([args.shift_type] if args.shift_type else cfg.erasure_eval.shift_types),
        "intensity_grid": args.intensity_grid.split(",") if args.intensity_grid else cfg.erasure_eval.intensity_grid,
        "modes": [m.strip() for m in modes.split(",") if m.strip()],
        "interaction_pairs": interaction_pairs,
        "clean_intensity": clean_intensity,
        "code_sha256": code_hashes,
    }
    (out_dir / "architecture_matrix_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out_dir / "architecture_protocol_manifest.json").write_text(json.dumps(protocol_manifest, indent=2), encoding="utf-8")
    log(f"Saved architecture matrix report to {out_dir}")


if __name__ == "__main__":
    main()
