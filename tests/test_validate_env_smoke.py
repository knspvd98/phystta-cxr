import os
import subprocess
import sys
from pathlib import Path


def test_validate_env_smoke(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    cfg_path = tmp_path / "cfg.yaml"
    out_path = tmp_path / "env.json"
    cfg_path.write_text(
        """
data:
  require_real_data: false
  sources:
    - name: dummy
      root: /tmp
      source_type: chexpert
model:
  encoder:
    encoder_source: baseline_cnn
runtime:
  allow_download: false
experiment:
  tier: dev
""".strip(),
        encoding="utf-8",
    )

    cmd = [
        sys.executable,
        str(repo / "scripts" / "validate_env.py"),
        "--config",
        str(cfg_path),
        "--out",
        str(out_path),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    subprocess.run(cmd, check=True, cwd=str(repo), env=env)
    assert out_path.exists()
