from __future__ import annotations
import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from phystta.config.loader import load_config
from phystta.data import DataEngine
from phystta.causal import PairQualityConfig
from phystta.causal.confounding import cluster_bootstrap_statistic
from phystta.causal.pair_quality import annotate_pair_quality

CACHE_DIR = Path("/workspace/outputs/model_family_runs/clip_base_patch32_mainline/confounding/cache")
OUT_DIR = Path("/workspace/outputs/model_family_runs/clip_base_patch32_mainline/pair_delta_from_cache")
CFG_PATH = "/workspace/outputs/model_family_runs/clip_base_patch32_mainline/public_mainline.effective.confounding_opt.yaml"

BOOTSTRAP_SAMPLES = 25
N_WORKERS = 24
CHUNK_SIZE = 1500

OUT_DIR.mkdir(parents=True, exist_ok=True)

_INDEX_DF = None
_LABEL_COLS = None
_PAIR_QUALITY_CFG = None

def _init_worker(index_df, label_cols, pair_quality_cfg):
    global _INDEX_DF, _LABEL_COLS, _PAIR_QUALITY_CFG
    _INDEX_DF = index_df
    _LABEL_COLS = label_cols
    _PAIR_QUALITY_CFG = pair_quality_cfg

def _annotate_chunk(args):
    chunk_id, chunk_df = args
    out = annotate_pair_quality(
        chunk_df,
        index_df=_INDEX_DF,
        label_cols=_LABEL_COLS,
        cfg=_PAIR_QUALITY_CFG,
    )
    return chunk_id, out

def main():
    print("[pair-delta-cache] loading cache", flush=True)
    pairs_df = pd.read_csv(CACHE_DIR / "pairs.csv.gz")
    cfg = load_config(CFG_PATH)
    seed = int(cfg.runtime.seed)

    data_engine = DataEngine(cfg.data).attach_runtime_contract(cfg)
    data_engine.build_index()

    index_df = data_engine.index
    label_cols = list(data_engine.label_cols)

    print(f"[pair-delta-cache] index_df={index_df.shape} pairs_df={pairs_df.shape} label_cols={len(label_cols)}", flush=True)

    pair_quality_cfg = PairQualityConfig(**cfg.intervention.pair_quality.__dict__)
    pair_work = pairs_df

    if "pair_quality_bucket" not in pair_work.columns:
        print("[pair-delta-cache] annotate_pair_quality parallel start", flush=True)
        chunks = []
        for i in range(0, len(pair_work), CHUNK_SIZE):
            chunk_id = len(chunks)
            chunks.append((chunk_id, pair_work.iloc[i:i + CHUNK_SIZE].copy()))
        print(f"[pair-delta-cache] n_chunks={len(chunks)} chunk_size={CHUNK_SIZE} workers={N_WORKERS}", flush=True)

        ctx = mp.get_context("fork")
        outs = [None] * len(chunks)
        with ProcessPoolExecutor(
            max_workers=N_WORKERS,
            mp_context=ctx,
            initializer=_init_worker,
            initargs=(index_df, label_cols, pair_quality_cfg),
        ) as ex:
            futs = {ex.submit(_annotate_chunk, item): item[0] for item in chunks}
            done = 0
            for fut in as_completed(futs):
                chunk_id, out_df = fut.result()
                outs[chunk_id] = out_df
                done += 1
                print(f"[pair-delta-cache] annotate chunk done {done}/{len(chunks)} chunk_id={chunk_id} rows={len(out_df)}", flush=True)

        pair_work = pd.concat(outs, axis=0, ignore_index=True)
        pair_work.to_csv(OUT_DIR / "pairs_annotated.csv.gz", index=False)
        print("[pair-delta-cache] annotate_pair_quality parallel done", flush=True)
    else:
        print("[pair-delta-cache] pair_quality already present in input", flush=True)

    technical_pairs = pair_work[pair_work.get("pair_quality_bucket", "ambiguous").astype(str) == "technical_dominant"].copy()
    print(f"[pair-delta-cache] technical_pairs={technical_pairs.shape}", flush=True)

    if technical_pairs.empty:
        summary = {
            "n_pairs_total": int(len(pair_work)),
            "n_pairs_technical_dominant": 0,
            "rows_emitted": 0,
        }
        pd.DataFrame().to_csv(OUT_DIR / "technical_pair_pathology_deltas.csv", index=False)
        (OUT_DIR / "pair_delta_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"[pair-delta-cache] saved outputs to {OUT_DIR}", flush=True)
        return

    idx = index_df.set_index("image_path")
    fallback_groups = pd.Series(range(len(technical_pairs)), index=technical_pairs.index)
    pair_groups = technical_pairs.get("global_patient_id", technical_pairs.get("patient_id", fallback_groups)).astype(str).to_numpy()

    rows = []
    for i, label_name in enumerate(label_cols, start=1):
        print(f"[pair-delta-cache] label {i}/{len(label_cols)} start name={label_name}", flush=True)

        if label_name not in idx.columns:
            print(f"[pair-delta-cache] label {i}/{len(label_cols)} skip missing_in_index", flush=True)
            continue

        vals_a = pd.to_numeric(
            idx.reindex(technical_pairs["image_path_a"])[label_name],
            errors="coerce",
        ).to_numpy(dtype=float)
        vals_b = pd.to_numeric(
            idx.reindex(technical_pairs["image_path_b"])[label_name],
            errors="coerce",
        ).to_numpy(dtype=float)

        valid = pd.notna(vals_a) & pd.notna(vals_b) & (vals_a >= 0) & (vals_b >= 0)
        n_valid = int(valid.sum())
        if n_valid < 4:
            print(f"[pair-delta-cache] label {i}/{len(label_cols)} skip n_valid={n_valid}", flush=True)
            continue

        abs_delta = abs(vals_b[valid] - vals_a[valid])
        signed_delta = vals_b[valid] - vals_a[valid]
        groups = pair_groups[valid]

        if len(set(groups)) > 1:
            print(f"[pair-delta-cache] label {i}/{len(label_cols)} bootstrap start n_valid={n_valid}", flush=True)
            boot_abs = cluster_bootstrap_statistic(
                groups,
                lambda idx_sel: float(abs_delta[idx_sel].mean()),
                n_boot=BOOTSTRAP_SAMPLES,
                seed=seed,
            )
            boot_signed = cluster_bootstrap_statistic(
                groups,
                lambda idx_sel: float(signed_delta[idx_sel].mean()),
                n_boot=BOOTSTRAP_SAMPLES,
                seed=seed + 17,
            )
            print(f"[pair-delta-cache] label {i}/{len(label_cols)} bootstrap done", flush=True)
        else:
            boot_abs = {"ci_low": float("nan"), "ci_high": float("nan")}
            boot_signed = {"ci_low": float("nan"), "ci_high": float("nan")}
            print(f"[pair-delta-cache] label {i}/{len(label_cols)} single_group_no_bootstrap", flush=True)

        rows.append(
            {
                "label": str(label_name),
                "mean_abs_delta": float(abs_delta.mean()),
                "mean_abs_delta_ci_low": float(boot_abs.get("ci_low", float("nan"))),
                "mean_abs_delta_ci_high": float(boot_abs.get("ci_high", float("nan"))),
                "mean_signed_delta": float(signed_delta.mean()),
                "mean_signed_delta_ci_low": float(boot_signed.get("ci_low", float("nan"))),
                "mean_signed_delta_ci_high": float(boot_signed.get("ci_high", float("nan"))),
                "prevalence_a": float((vals_a[valid] > 0).mean()),
                "prevalence_b": float((vals_b[valid] > 0).mean()),
                "n_pairs_valid": n_valid,
            }
        )
        print(f"[pair-delta-cache] label {i}/{len(label_cols)} row_saved", flush=True)

    out_df = pd.DataFrame(rows)
    summary = {
        "n_pairs_total": int(len(pair_work)),
        "n_pairs_technical_dominant": int(len(technical_pairs)),
        "rows_emitted": int(len(out_df)),
    }

    out_df.to_csv(OUT_DIR / "technical_pair_pathology_deltas.csv", index=False)
    (OUT_DIR / "pair_delta_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[pair-delta-cache] saved outputs to {OUT_DIR}", flush=True)

if __name__ == "__main__":
    main()
