import json
import math
import hashlib
import subprocess
from collections.abc import Mapping as MappingABC
from pathlib import Path
from typing import Any, Dict, Mapping, Optional


def atomic_write(path: Path, data: str) -> None:
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(data)
    tmp.replace(path)


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text())


def metric_report_json(data: Any) -> str:
    """Encode unavailable numerical results as JSON null in published reports."""
    def normalize(value):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        if isinstance(value, dict):
            return {key: normalize(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [normalize(item) for item in value]
        return value

    return json.dumps(normalize(data), indent=2, allow_nan=False)


def write_json(path: Path, data: Dict[str, Any], indent: int = 2) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    atomic_write(Path(path), json.dumps(data, indent=indent))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_torch_artifact(path, map_location: str | None = "cpu"):
    import torch

    return torch.load(path, map_location=map_location, weights_only=False)


def load_torch_state_dict(path, map_location: str | None = "cpu", key: str | None = None):
    payload = load_torch_artifact(path, map_location=map_location)
    if key is not None and isinstance(payload, MappingABC) and key in payload:
        payload = payload[key]
    if not isinstance(payload, MappingABC):
        raise RuntimeError(f"Expected a state_dict-like mapping in {path}, got {type(payload).__name__}.")
    return payload


def repo_fingerprint(root: Optional[Path] = None) -> Dict[str, Any]:
    root = Path(root or Path(__file__).resolve().parents[2])
    git_sha = None
    try:
        git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root).decode().strip()
    except Exception:
        git_sha = None
    from .hash import hash_code
    from .versions import get_versions

    return {
        "repo_root": str(root),
        "git_sha_or_repo_fingerprint": git_sha or hash_code(root),
        "code_sha256": hash_code(root),
        "versions": get_versions(["torch", "torchvision", "numpy", "pandas", "scikit-learn", "PyYAML"]),
    }


def write_yaml(path: Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import yaml
    except Exception:
        atomic_write(path, json.dumps(data, indent=2))
        return
    atomic_write(path, yaml.safe_dump(data, sort_keys=False))


def write_reprovenance_bundle(
    out_dir: Path,
    *,
    cfg=None,
    config_path: Optional[str | Path] = None,
    artifact_manifest: Optional[Mapping[str, Any]] = None,
    input_artifact_paths: Optional[Mapping[str, str | Path | None]] = None,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if cfg is not None:
        from phystta.config.loader import save_config

        save_config(str(out_dir / "resolved_runtime_config.yaml"), cfg)
    repo_meta = repo_fingerprint()
    input_hashes: Dict[str, Any] = {}
    for name, path in (input_artifact_paths or {}).items():
        if path is None:
            continue
        p = Path(path)
        input_hashes[str(name)] = {
            "path": str(p),
            "sha256": sha256_file(p) if p.exists() else None,
            "exists": bool(p.exists()),
        }
    write_json(out_dir / "input_artifact_hashes.json", input_hashes)
    write_json(out_dir / "code_fingerprint.json", repo_meta)
    manifest = dict(artifact_manifest or {})
    if config_path is not None:
        manifest["config_path"] = str(config_path)
    if cfg is not None:
        manifest["resolved_config_dump"] = cfg.as_dict()
    encoder_fp = manifest.get("encoder_fingerprint")
    if isinstance(encoder_fp, MappingABC):
        manifest.setdefault("model_revision", encoder_fp.get("model_revision", encoder_fp.get("revision")))
        if "preprocess_signature" not in manifest and encoder_fp.get("preprocess_signature") is not None:
            manifest["preprocess_signature"] = encoder_fp.get("preprocess_signature")
    manifest.setdefault("git_sha_or_repo_fingerprint", repo_meta["git_sha_or_repo_fingerprint"])
    write_json(out_dir / "artifact_manifest.json", manifest)
