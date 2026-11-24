from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable


def hash_files(paths: Iterable[Path]) -> str:
    h = hashlib.sha256()
    for p in sorted(paths):
        h.update(p.read_bytes())
    return h.hexdigest()


def hash_code(root: Path, patterns=("*.py",)) -> str:
    files = []
    for pattern in patterns:
        files.extend(root.rglob(pattern))
    return hash_files(files)
