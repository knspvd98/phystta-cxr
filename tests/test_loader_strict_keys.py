import textwrap

import pytest

from phystta.config.loader import load_config


def _write(tmp_path, content: str) -> str:
    path = tmp_path / "cfg.yaml"
    path.write_text(textwrap.dedent(content))
    return str(path)


def test_max_samples_rejected(tmp_path):
    cfg_path = _write(
        tmp_path,
        """
        data:
          require_real_data: false
          sources:
            - name: dummy
              root: /tmp
              source_type: chexpert
        intervention:
          max_samples: 10
        """,
    )
    with pytest.raises(ValueError, match="max_samples"):
        load_config(cfg_path)


def test_max_images_accepted(tmp_path):
    cfg_path = _write(
        tmp_path,
        """
        data:
          require_real_data: false
          sources:
            - name: dummy
              root: /tmp
              source_type: chexpert
        intervention:
          max_images: 10
        """,
    )
    cfg = load_config(cfg_path)
    assert cfg.intervention.max_images == 10
