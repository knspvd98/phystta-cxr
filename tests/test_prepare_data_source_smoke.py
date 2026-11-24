import json
import os
import subprocess
import sys
from pathlib import Path

import yaml


def test_prepare_data_source_builds_manifest_and_index(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    mini_dir = tmp_path / "mini"
    # Build tiny deterministic dataset first.
    cmd_make = [
        sys.executable,
        str(repo / "scripts" / "make_minicxr_dataset.py"),
        "--out_dir",
        str(mini_dir),
        "--n_patients",
        "10",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    subprocess.run(cmd_make, check=True, cwd=str(repo), env=env)

    cfg = {
        "data": {
            "require_real_data": False,
            "split_group_key": "global_patient_id",
            "split_fractions": {"train_head": 0.5, "train_eraser": 0.2, "calib": 0.1, "eval": 0.2},
            "head_split": "train_head",
            "eraser_split": "train_eraser",
            "calib_split": "calib",
            "eval_split": "eval",
            "sources": [
                {
                    "name": "mini",
                    "root": str(tmp_path / "unset"),
                    "label_cols": ["No Finding"],
                    "path_col": "path",
                    "patient_id_col": "patient_id",
                    "study_id_col": "study_id",
                    "view_col": "view",
                    "manufacturer_col": "manufacturer",
                    "station_name_col": "station_name",
                    "study_date_col": "study_date",
                    "study_time_col": "study_time",
                }
            ],
        },
        "runtime": {"allow_download": False},
    }
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    index_out = tmp_path / "index.csv"
    manifest_out = tmp_path / "data_manifest.json"
    cmd_prepare = [
        sys.executable,
        str(repo / "scripts" / "prepare_data_source.py"),
        "--config",
        str(cfg_path),
        "--source_name",
        "mini",
        "--dataset_root",
        str(mini_dir),
        "--metadata_csv",
        str(mini_dir / "metadata.csv"),
        "--image_root",
        str(mini_dir / "images"),
        "--index_out",
        str(index_out),
        "--manifest_out",
        str(manifest_out),
        "--set_config",
    ]
    subprocess.run(cmd_prepare, check=True, cwd=str(repo), env=env)

    payload = json.loads(manifest_out.read_text(encoding="utf-8"))
    assert payload["index_sha256"]
    assert payload["n_rows"] > 0
    cfg_after = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    assert cfg_after["data"]["prebuilt_index_csv"] == str(index_out)
    assert cfg_after["data"]["prebuilt_index_sha256"] == payload["index_sha256"]
