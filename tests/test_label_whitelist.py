import textwrap
import pytest

from phystta.config.loader import load_config
from phystta.data import DataEngine


def _write(tmp_path, content: str) -> str:
    path = tmp_path / "cfg.yaml"
    path.write_text(textwrap.dedent(content))
    return str(path)


def test_label_cols_required(tmp_path):
    cfg_path = _write(
        tmp_path,
        f"""
        data:
          require_labels: true
          sources:
            - name: mimic
              root: {tmp_path}
              metadata_csv: {tmp_path}/meta.csv
        """,
    )
    cfg = load_config(cfg_path)
    engine = DataEngine(cfg.data)
    with pytest.raises(RuntimeError, match="label_cols"):
        engine.build_index()
