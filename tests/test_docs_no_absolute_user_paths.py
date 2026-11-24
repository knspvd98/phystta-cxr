import re
from pathlib import Path


def test_readme_and_docs_do_not_contain_local_users_paths():
    repo = Path(__file__).resolve().parents[1]
    targets = [repo / "README.md"] + sorted((repo / "docs").glob("*.md"))
    pat = re.compile(r"(^|\\s)/Users/", re.MULTILINE)
    offenders = []
    for path in targets:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        if pat.search(text):
            offenders.append(str(path.relative_to(repo)))
    assert not offenders, f"Remove absolute local paths from docs: {offenders}"
