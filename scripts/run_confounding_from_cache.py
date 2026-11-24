from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd

from phystta.causal import PairQualityConfig, run_confounding_audit
from phystta.config.loader import load_config
from phystta.data import DataEngine

CACHE_DIR = Path("/workspace/outputs/model_family_runs/clip_base_patch32_mainline/confounding/cache")
OUT_DIR = Path("/workspace/outputs/model_family_runs/clip_base_patch32_mainline/confounding_from_cache")
CFG_PATH = "/workspace/outputs/model_family_runs/clip_base_patch32_mainline/public_mainline.effective.confounding_opt.yaml"

OUT_DIR.mkdir(parents=True, exist_ok=True)

print("[cache-run] loading cache", flush=True)
features = np.load(CACHE_DIR / "features.npy")
probs = np.load(CACHE_DIR / "probs.npy")
labels = np.load(CACHE_DIR / "labels.npy")
metadata = pd.read_csv(CACHE_DIR / "metadata.csv.gz")
pairs = pd.read_csv(CACHE_DIR / "pairs.csv.gz") if (CACHE_DIR / "pairs.csv.gz").exists() else None

print(f"[cache-run] features={features.shape} probs={probs.shape} labels={labels.shape} metadata={metadata.shape}", flush=True)
if pairs is not None:
    print(f"[cache-run] pairs={pairs.shape}", flush=True)
else:
    print("[cache-run] pairs=None", flush=True)

cfg = load_config(CFG_PATH)
seed = int(cfg.runtime.seed)

data_engine = DataEngine(cfg.data).attach_runtime_contract(cfg)
data_engine.build_index()

pair_quality_cfg = PairQualityConfig(**cfg.intervention.pair_quality.__dict__)
shift_id_group_key = str(cfg.shift_id.group_key) if getattr(cfg, "shift_id", None) is not None else None
shift_id_score_family = list(cfg.shift_id.score_family) if getattr(cfg, "shift_id", None) is not None else None

print(f"[cache-run] label_cols={len(data_engine.label_cols)}", flush=True)
print(f"[cache-run] shift_id_group_key={shift_id_group_key}", flush=True)
print(f"[cache-run] shift_id_score_family={shift_id_score_family}", flush=True)

audit = run_confounding_audit(
    features=features,
    probs=probs,
    labels=labels,
    metadata_df=metadata,
    label_cols=data_engine.label_cols,
    index_df=data_engine.index,
    pairs_df=pairs,
    pair_quality_cfg=pair_quality_cfg,
    shift_id_group_key=shift_id_group_key,
    shift_id_score_family=shift_id_score_family,
    bootstrap_samples=25,
    seed=seed,
)

print("[cache-run] audit returned", flush=True)

predictability_df = audit["predictability_df"]
shift_id_df = audit["shift_id_proxy_df"]
pair_delta_df = audit["technical_pair_delta_df"]

predictability_df.to_csv(OUT_DIR / "confounding_predictability.csv", index=False)
shift_id_df.to_csv(OUT_DIR / "shift_id_proxy_audit.csv", index=False)
pair_delta_df.to_csv(OUT_DIR / "technical_pair_pathology_deltas.csv", index=False)

summary = {
    **audit["summary"],
    "shift_id_summary": audit["shift_id_summary"],
    "pair_quality_summary": audit["pair_quality_summary"],
    "config_path": str(CFG_PATH),
    "bootstrap_samples": 25,
}
(OUT_DIR / "confounding_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

print(f"[cache-run] saved outputs to {OUT_DIR}", flush=True)
