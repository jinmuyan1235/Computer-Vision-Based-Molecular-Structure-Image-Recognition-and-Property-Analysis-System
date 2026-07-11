"""Compatibility tests for optional backend result normalization."""

from pathlib import Path

from src.ocsr.decimer_adapter import DECIMERAdapter
from src.ocsr.demo_adapter import DemoOCSRAdapter
from src.ocsr.molscribe_adapter import MolScribeAdapter


def test_demo_backend_reports_available() -> None:
    status = DemoOCSRAdapter().status()
    assert status["backend"] == "demo"
    assert status["available"] is True


def test_decimer_tuple_result_with_confidence(tmp_path: Path) -> None:
    image_path = tmp_path / "molecule.png"
    image_path.write_bytes(b"fake-image")
    adapter = DECIMERAdapter()
    adapter.predictor = lambda _path, confidence=True: ("CCO", 0.88)
    result = adapter.recognize(image_path)
    assert result.status == "success"
    assert result.smiles == "CCO"
    assert result.confidence == 0.88


def test_molscribe_dict_result_with_confidence(tmp_path: Path) -> None:
    class FakeModel:
        def predict_image_file(self, _path: str, return_confidence: bool = True) -> dict:
            return {"smiles": "CCO", "confidence": 0.91}

    image_path = tmp_path / "molecule.png"
    image_path.write_bytes(b"fake-image")
    adapter = MolScribeAdapter()
    adapter.model = FakeModel()
    adapter.import_error = None
    result = adapter.recognize(image_path)
    assert result.status == "success"
    assert result.confidence == 0.91
