import json
import os
import subprocess
import sys
from pathlib import Path


def test_cache_models_dry_run_writes_report(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    report = tmp_path / "cache_report.json"
    cmd = [
        sys.executable,
        str(repo / "scripts" / "cache_models.py"),
        "--config",
        str(repo / "configs" / "representation_surgery_2026.yaml"),
        "--profiles",
        "study",
        "--out_report",
        str(report),
        "--dry_run",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    subprocess.run(cmd, check=True, cwd=str(repo), env=env)
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["dry_run"] is True
    assert payload["download"] is False
    assert isinstance(payload.get("items"), list) and len(payload["items"]) > 0
    hf_rows = [r for r in payload["items"] if r.get("encoder_source") == "hf_clip_or_vit" and r.get("model_id")]
    assert len(hf_rows) > 0
    assert all(r.get("status") in {"planned_verify", "verified", "missing", "error"} for r in hf_rows)
    cnn_rows = [r for r in payload["items"] if r.get("encoder_source") == "baseline_cnn"]
    assert len(cnn_rows) > 0
    assert all(r.get("status") in {"planned_verify", "planned_export", "planned_missing"} for r in cnn_rows)
    assert all("local_path" in r for r in cnn_rows)
    assert all("use_medical_pretrain" in r for r in cnn_rows)
    assert all(r.get("use_medical_pretrain") is False for r in cnn_rows)


def test_cache_models_dry_run_download_marks_medical_pretrain_flag(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    cfg_src = repo / "configs" / "representation_surgery_2026.yaml"
    cfg_dst = tmp_path / "cfg_download.yaml"
    cfg_text = cfg_src.read_text(encoding="utf-8")
    cfg_text = cfg_text.replace("tier: paper", "tier: dev")
    cfg_text = cfg_text.replace("allow_download: false", "allow_download: true")
    cfg_dst.write_text(cfg_text, encoding="utf-8")
    report = tmp_path / "cache_report_download.json"
    cmd = [
        sys.executable,
        str(repo / "scripts" / "cache_models.py"),
        "--config",
        str(cfg_dst),
        "--profiles",
        "study",
        "--out_report",
        str(report),
        "--download",
        "--dry_run",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    subprocess.run(cmd, check=True, cwd=str(repo), env=env)
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["download"] is True
    cnn_rows = [r for r in payload["items"] if r.get("encoder_source") == "baseline_cnn"]
    assert len(cnn_rows) > 0
    assert all(r.get("use_medical_pretrain") is True for r in cnn_rows)
