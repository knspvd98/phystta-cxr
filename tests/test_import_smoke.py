import importlib


def test_import_metrics_module():
    importlib.import_module("phystta.metrics")


def test_import_encoder_factory_module():
    importlib.import_module("phystta.models.encoder_factory")


def test_build_encoder_symbol_exists():
    mod = importlib.import_module("phystta.models.encoder_factory")
    assert hasattr(mod, "build_encoder")
