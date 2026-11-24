import textwrap

import pytest

from phystta.config.loader import load_config
from phystta.data import DataEngine


def _write(tmp_path, content: str) -> str:
    path = tmp_path / "cfg.yaml"
    path.write_text(textwrap.dedent(content))
    return str(path)


def test_lidc_drr_blocked(tmp_path):
    cfg_path = _write(
        tmp_path,
        f"""
        data:
          require_real_data: true
          sources:
            - name: lidc_drr
              root: {tmp_path}
              source_type: lidc_drr
        """,
    )
    with pytest.raises(ValueError, match="Synthetic DRR"):
        load_config(cfg_path)
