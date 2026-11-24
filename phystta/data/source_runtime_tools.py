from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
from typing import Any
from urllib.request import urlretrieve


TOOL_OVERRIDE_ENVS = {
    "aws": "PHYSTTA_AWS_BIN",
    "azcopy": "PHYSTTA_AZCOPY_BIN",
    "kaggle": "PHYSTTA_KAGGLE_BIN",
}

TOOL_MODULES = {
    "aws": "awscli",
    "kaggle": "kaggle",
}

DEFAULT_AZCOPY_DOWNLOAD_URL = "https://aka.ms/downloadazcopy-v10-linux"


def _module_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def _tool_install_root(bundle_root: str | Path, tool_install_root: str | Path | None = None) -> Path:
    if tool_install_root is not None:
        return Path(tool_install_root).expanduser().resolve()
    return Path(bundle_root).expanduser().resolve() / "tools" / "source_runtime"


def resolve_tool_command(
    tool: str,
    *,
    bundle_root: str | Path,
    tool_install_root: str | Path | None = None,
) -> list[str] | None:
    override_env = TOOL_OVERRIDE_ENVS.get(tool)
    override = os.environ.get(override_env or "")
    if override:
        candidate = Path(override).expanduser().resolve()
        if candidate.exists():
            return [str(candidate)]
    install_root = _tool_install_root(bundle_root, tool_install_root)
    if tool == "azcopy":
        installed = install_root / "azcopy" / "azcopy"
        if installed.exists():
            return [str(installed)]
    which = shutil.which(tool)
    if which:
        return [which]
    module_name = TOOL_MODULES.get(tool)
    if module_name and _module_available(module_name):
        return [sys.executable, "-m", module_name]
    return None


def bootstrap_runtime_tool(
    tool: str,
    *,
    bundle_root: str | Path,
    tool_install_root: str | Path | None = None,
) -> dict[str, Any]:
    install_root = _tool_install_root(bundle_root, tool_install_root)
    install_root.mkdir(parents=True, exist_ok=True)
    if tool in {"aws", "kaggle"}:
        package_name = "awscli" if tool == "aws" else "kaggle"
        command = [sys.executable, "-m", "pip", "install", package_name]
        subprocess.run(command, check=True)
        resolved = resolve_tool_command(tool, bundle_root=bundle_root, tool_install_root=install_root)
        return {
            "tool": tool,
            "bootstrap_method": "pip_install",
            "command": command,
            "resolved_command": resolved,
        }
    if tool == "azcopy":
        download_url = os.environ.get("PHYSTTA_AZCOPY_DOWNLOAD_URL", DEFAULT_AZCOPY_DOWNLOAD_URL)
        with tempfile.TemporaryDirectory(prefix="phystta-azcopy-") as tmp_dir:
            archive_path = Path(tmp_dir) / "azcopy.tar.gz"
            urlretrieve(download_url, archive_path)
            with tarfile.open(archive_path, "r:gz") as tar:
                tar.extractall(path=tmp_dir)
            extracted = next(Path(tmp_dir).glob("azcopy_linux_amd64_*/azcopy"), None)
            if extracted is None:
                raise RuntimeError("azcopy_archive_missing_binary")
            target_dir = install_root / "azcopy"
            target_dir.mkdir(parents=True, exist_ok=True)
            target = target_dir / "azcopy"
            shutil.copy2(extracted, target)
            target.chmod(0o755)
        return {
            "tool": tool,
            "bootstrap_method": "binary_download",
            "download_url": download_url,
            "resolved_command": [str((install_root / 'azcopy' / 'azcopy').resolve())],
        }
    raise RuntimeError(f"unsupported_tool_bootstrap:{tool}")
