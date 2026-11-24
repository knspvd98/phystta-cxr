from __future__ import annotations
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from phystta.config.loader import load_config
from phystta.data import DataEngine
from phystta.causal.confounding import build_predictability_audit, available_confounding_context_keys

RUN_ROOT = Path("/workspace/outputs/model_family_runs/clip_base_patch32_mainline")
CACHE_DIR = RUN_ROOT / "confounding" / "cache"
SHIFT_DIR = RUN_ROOT / "shift_id_from_cache"
PAIR_DIR = RUN_ROOT / "pair_delta_from_cache"
OUT_DIR = RUN_ROOT / "confounding_pairs"
CFG_PATH = RUN_ROOT / "public_mainline.effective.confounding_opt.yaml"

OUT_DIR.mkdir(parents=True, exist_ok=True)

print("[materialize] loading cache", flush=True)
features = np.load(CACHE_DIR / "features.npy")
probs = np.load(CACHE_DIR / "probs.npy")
labels = np.load(CACHE_DIR / "labels.npy")
metadata = pd.read_csv(CACHE_DIR / "metadata.csv.gz")

cfg = load_config(str(CFG_PATH))
seed = int(cfg.runtime.seed)

data_engine = DataEngine(cfg.data).attach_runtime_contract(cfg)
data_engine.build_index()
label_cols = list(data_engine.label_cols)

print(f"[materialize] features={features.shape} probs={probs.shape} labels={labels.shape} metadata={metadata.shape}", flush=True)
print(f"[materialize] label_cols={len(label_cols)}", flush=True)

print("[materialize] build_predictability_audit start", flush=True)
predictability_df = build_predictability_audit(
    features=features,
    probs=probs,
    labels=labels,
    metadata_df=metadata,
    label_cols=label_cols,
    bootstrap_samples=25,
    seed=seed,
)
print(f"[materialize] build_predictability_audit done shape={predictability_df.shape}", flush=True)

shift_df = pd.read_csv(SHIFT_DIR / "shift_id_proxy_audit.csv")
shift_summary = json.loads((SHIFT_DIR / "shift_id_summary.json").read_text())

pair_df = pd.read_csv(PAIR_DIR / "technical_pair_pathology_deltas.csv")
pair_summary = json.loads((PAIR_DIR / "pair_delta_summary.json").read_text())

predictability_df.to_csv(OUT_DIR / "confounding_predictability.csv", index=False)
shift_df.to_csv(OUT_DIR / "shift_id_proxy_audit.csv", index=False)
pair_df.to_csv(OUT_DIR / "technical_pair_pathology_deltas.csv", index=False)

if (PAIR_DIR / "pairs_annotated.csv.gz").exists():
    shutil.copy2(PAIR_DIR / "pairs_annotated.csv.gz", OUT_DIR / "pairs_annotated.csv.gz")

summary = {
    "n_samples": int(len(metadata)),
    "context_keys": available_confounding_context_keys(metadata),
    "predictability_tests": int(len(predictability_df)),
    "shift_id_proxy_tests": int(len(shift_df)),
    "active_shift_id_group_key": shift_summary.get("active_group_key"),
    "n_technical_pairs_audited": int(pair_summary.get("n_pairs_technical_dominant", 0)),
    "shift_id_summary": shift_summary,
    "pair_quality_summary": pair_summary,
    "cache_dir": str(CACHE_DIR),
    "bootstrap_samples": 25,
}

(OUT_DIR / "confounding_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(f"[materialize] saved outputs to {OUT_DIR}", flush=True)
