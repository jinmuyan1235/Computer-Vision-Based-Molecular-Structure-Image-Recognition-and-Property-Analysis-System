"""Tests for the optional, failure-isolated ADMET baseline."""

from pathlib import Path

import pytest

from src.ml.admet_baseline import ADMETBaseline, ConfiguredADMETPredictor, smiles_to_fingerprint


def test_morgan_fingerprint_shape_and_invalid_input() -> None:
    fingerprint = smiles_to_fingerprint("CCO", n_bits=256)
    assert fingerprint.shape == (256,)
    assert set(fingerprint.tolist()) <= {0, 1}
    with pytest.raises(ValueError, match="SMILES"):
        smiles_to_fingerprint("not_a_smiles")


def test_configured_predictor_is_disabled_by_default(tmp_path: Path) -> None:
    result = ConfiguredADMETPredictor(enabled=False, model_path=tmp_path / "missing.joblib").predict("CCO")
    assert result["status"] == "disabled"
    assert "RDKit" in result["message"]


def test_train_save_load_and_predict_classification(tmp_path: Path) -> None:
    smiles = ["CCO", "CCN", "CCC", "c1ccccc1", "CC(=O)O", "CCCl"]
    labels = [0, 0, 0, 1, 1, 1]
    model = ADMETBaseline.train(smiles, labels, target_name="ames", random_state=7)
    model_path = model.save(tmp_path / "ames.joblib")
    loaded = ADMETBaseline.load(model_path)
    result = loaded.predict("CCO")
    assert result["status"] == "success"
    assert result["target"] == "ames"
    assert result["prediction"] in {0, 1}
    assert 0 <= result["probability"] <= 1


def test_enabled_predictor_reports_missing_model(tmp_path: Path) -> None:
    result = ConfiguredADMETPredictor(enabled=True, model_path=tmp_path / "missing.joblib").predict("CCO")
    assert result["status"] == "unavailable"
    assert "不存在" in result["message"]
