from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

from ..config import KaggleConfig, KaggleDatasetConfig


def _ensure_kaggle_api():
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except Exception as e:
        raise RuntimeError("kaggle package is required. Install with `pip install kaggle`.") from e
    _ensure_kaggle_credentials()
    api = KaggleApi()
    api.authenticate()
    return api


def _ensure_kaggle_credentials() -> None:
    config_dir = os.environ.get("KAGGLE_CONFIG_DIR")
    if config_dir:
        return
    default_path = Path.home() / ".kaggle" / "kaggle.json"
    if default_path.exists():
        return
    username = os.environ.get("KAGGLE_USERNAME")
    key = os.environ.get("KAGGLE_KEY")
    if not username or not key:
        raise RuntimeError(
            "Kaggle credentials not found. Place kaggle.json in ~/.kaggle or set "
            "KAGGLE_USERNAME and KAGGLE_KEY in the environment."
        )
    tmp_dir = Path(os.environ.get("PHYSTTA_KAGGLE_DIR", ".kaggle_tmp")).expanduser()
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_file = tmp_dir / "kaggle.json"
    tmp_file.write_text(json.dumps({"username": username, "key": key}))
    os.chmod(tmp_file, 0o600)
    os.environ["KAGGLE_CONFIG_DIR"] = str(tmp_dir)


def download_dataset(spec: KaggleDatasetConfig, logger=None) -> Path:
    api = _ensure_kaggle_api()
    dest = Path(spec.dest).expanduser()
    dest.mkdir(parents=True, exist_ok=True)
    if spec.competition:
        if spec.files:
            for f in spec.files:
                if logger:
                    logger.info(f"Kaggle competition download: {spec.dataset}/{f}")
                api.competition_download_file(spec.dataset, f, path=str(dest))
        else:
            if logger:
                logger.info(f"Kaggle competition download: {spec.dataset}")
            api.competition_download_files(spec.dataset, path=str(dest), quiet=False)
    else:
        if spec.files:
            for f in spec.files:
                if logger:
                    logger.info(f"Kaggle dataset download: {spec.dataset}/{f}")
                api.dataset_download_file(spec.dataset, f, path=str(dest), quiet=False)
        else:
            if logger:
                logger.info(f"Kaggle dataset download: {spec.dataset}")
            api.dataset_download_files(spec.dataset, path=str(dest), quiet=False)
    if spec.unzip:
        for zip_file in dest.glob("*.zip"):
            if logger:
                logger.info(f"Unzipping {zip_file}")
            import zipfile
            with zipfile.ZipFile(zip_file, "r") as zf:
                zf.extractall(dest)
    return dest


def download_all(cfg: KaggleConfig, logger=None) -> None:
    for ds in cfg.datasets:
        download_dataset(ds, logger=logger)
