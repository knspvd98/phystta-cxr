from pathlib import Path


def test_no_base_model_eval_path():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_erasure_eval.py"
    text = path.read_text()
    assert "base_model(" not in text, "base_model(x) path found in eval script"
