from __future__ import annotations

from typing import Dict, List


def get_versions(packages: List[str]) -> Dict[str, str]:
    versions = {}
    try:
        from importlib import metadata
    except Exception:
        return versions
    for pkg in packages:
        try:
            versions[pkg] = metadata.version(pkg)
        except Exception:
            versions[pkg] = "not_installed"
    return versions
