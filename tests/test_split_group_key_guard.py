import textwrap

import pytest

from phystta.config.loader import load_config


def test_multisource_requires_global_split_group_key(tmp_path):
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        textwrap.dedent(
            """
            data:
              require_real_data: false
              sources:
                - name: s1
                  root: /tmp
                - name: s2
                  root: /tmp
            """
        ).strip(),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="split_group_key=global_patient_id"):
        load_config(str(cfg_path))


def test_multisource_patient_id_allowed_only_with_explicit_override(tmp_path):
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        textwrap.dedent(
            """
            data:
              require_real_data: false
              split_group_key: patient_id
              single_source_mode: true
              sources:
                - name: s1
                  root: /tmp
                - name: s2
                  root: /tmp
            """
        ).strip(),
        encoding="utf-8",
    )
    cfg = load_config(str(cfg_path))
    assert cfg.data.single_source_mode is True
