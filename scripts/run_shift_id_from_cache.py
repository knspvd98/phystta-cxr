from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd

from phystta.config.loader import load_config
from phystta.causal.confounding import build_shift_id_proxy_audit

CACHE_DIR = Path("/workspace/outputs/model_family_runs/clip_base_patch32_mainline/confounding/cache")
OUT_DIR = Path("/workspace/outputs/model_family_runs/clip_base_patch32_mainline/shift_id_from_cache")
CFG_PATH = "/workspace/outputs/model_family_runs/clip_base_patch32_mainline/public_mainline.effective.confounding_opt.yaml"

OUT_DIR.mkdir(parents=True, exist_ok=True)

print("[shift-id-cache] loading cache", flush=True)
features = np.load(CACHE_DIR / "features.npy")
probs = np.load(CACHE_DIR / "probs.npy")
labels = np.load(CACHE_DIR / "labels.npy")
metadata = pd.read_csv(CACHE_DIR / "metadata.csv.gz")

cfg = load_config(CFG_PATH)
preferred_group_key = str(cfg.shift_id.group_key) if getattr(cfg, "shift_id", None) is not None else None
score_family = list(cfg.shift_id.score_family) if getattr(cfg, "shift_id", None) is not None else None
seed = int(cfg.runtime.seed)

print(f"[shift-id-cache] features={features.shape} probs={probs.shape} labels={labels.shape} metadata={metadata.shape}", flush=True)
print(f"[shift-id-cache] preferred_group_key={preferred_group_key}", flush=True)
print(f"[shift-id-cache] score_family={score_family}", flush=True)

df, summary = build_shift_id_proxy_audit(
    features=features,
    probs=probs,
    labels=labels,
    metadata_df=metadata,
    preferred_group_key=preferred_group_key,
    score_family=score_family,
    seed=seed,
)

print(f"[shift-id-cache] returned df_shape={df.shape}", flush=True)
df.to_csv(OUT_DIR / "shift_id_proxy_audit.csv", index=False)
(OUT_DIR / "shift_id_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(f"[shift-id-cache] saved outputs to {OUT_DIR}", flush=True)
