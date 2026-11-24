import json
import subprocess
import sys


def test_cli_list_outputs_json():
    cmd = [sys.executable, "-m", "phystta.cli.main", "list", "--json"]
    out = subprocess.check_output(cmd, text=True)
    payload = json.loads(out)
    assert "paper-suite" in payload
    assert "end-to-end" in payload
