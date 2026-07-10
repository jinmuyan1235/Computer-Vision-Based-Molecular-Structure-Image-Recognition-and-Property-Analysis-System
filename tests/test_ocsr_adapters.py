"""Compatibility tests for optional backend result normalization."""

from src.ocsr.decimer_adapter import DECIMERAdapter
from src.ocsr.demo_adapter import DemoOCSRAdapter
from src.ocsr.molscribe_adapter import MolScribeAdapter


def test_demo_backend_reports_available() -> None:
    status = DemoOCSRAdapter().status()
    assert status["backend"] == "demo"
    assert status["available"] is True


def test_decimer_tuple_result_with_confidence() -> None:
    adapter = DECIMERAdapter()
    adapter.predictor = lambda _path, confidence=True: ("CCO", 0.88)
    result = adapter.recognize("molecule.png")
    assert result.status == "success"
    assert result.smiles == "CCO"
    assert result.confidence == 0.88


def test_molscribe_dict_result_with_confidence() -> None:
    class FakeModel:
        def predict_image_file(self, _path: str, return_confidence: bool = True) -> dict:
            return {"smiles": "CCO", "confidence": 0.91}

    adapter = MolScribeAdapter()
    adapter.model = FakeModel()
    adapter.import_error = None
    result = adapter.recognize("molecule.png")
    assert result.status == "success"
    assert result.confidence == 0.91
