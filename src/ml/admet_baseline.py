"""Optional Morgan-fingerprint Random Forest baseline for one ADMET endpoint."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence

import joblib
import numpy as np
from pandas import isna
from rdkit import DataStructs
from rdkit.Chem import rdFingerprintGenerator
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

import config
from src.chem.smiles_validator import smiles_to_mol


TaskType = Literal["classification", "regression"]
DISCLAIMER = "该预测仅为教学 baseline，不能替代实验、毒理研究或专业决策。"


def smiles_to_fingerprint(smiles: str, radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    """Convert valid SMILES to a fixed-length Morgan bit vector."""
    molecule = smiles_to_mol(smiles)
    if molecule is None:
        raise ValueError("无法生成指纹：输入的 SMILES 无效。")
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    fingerprint = generator.GetFingerprint(molecule)
    array = np.zeros((n_bits,), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fingerprint, array)
    return array


@dataclass
class ADMETBaseline:
    """Serializable single-endpoint Random Forest model."""

    estimator: Any
    target_name: str
    task_type: TaskType
    radius: int = 2
    n_bits: int = 2048
    training_samples: int = 0

    @classmethod
    def train(
        cls,
        smiles_values: Sequence[str] | Iterable[str],
        labels: Sequence[Any] | Iterable[Any],
        target_name: str,
        task_type: TaskType = "classification",
        random_state: int = 42,
    ) -> "ADMETBaseline":
        """Train a baseline after filtering rows with invalid SMILES or missing labels."""
        if task_type not in {"classification", "regression"}:
            raise ValueError("task_type 必须是 classification 或 regression。")
        if not target_name.strip():
            raise ValueError("target_name 不能为空。")
        smiles_list = list(smiles_values)
        label_list = list(labels)
        if len(smiles_list) != len(label_list):
            raise ValueError("SMILES 与标签数量必须一致。")
        fingerprints: list[np.ndarray] = []
        clean_labels: list[Any] = []
        for smiles, label in zip(smiles_list, label_list):
            if label is None or bool(isna(label)):
                continue
            try:
                fingerprints.append(smiles_to_fingerprint(str(smiles)))
                clean_labels.append(label)
            except ValueError:
                continue
        if len(fingerprints) < 4:
            raise ValueError("至少需要 4 条具有有效 SMILES 和标签的数据才能训练 baseline。")
        features = np.vstack(fingerprints)
        targets = np.asarray(clean_labels)
        if task_type == "classification":
            if np.unique(targets).size < 2:
                raise ValueError("分类任务至少需要两个不同类别。")
            estimator = RandomForestClassifier(
                n_estimators=300,
                class_weight="balanced",
                random_state=random_state,
                n_jobs=-1,
            )
        else:
            estimator = RandomForestRegressor(
                n_estimators=300,
                random_state=random_state,
                n_jobs=-1,
            )
        estimator.fit(features, targets)
        return cls(
            estimator=estimator,
            target_name=target_name.strip(),
            task_type=task_type,
            training_samples=len(fingerprints),
        )

    def predict(self, smiles: str) -> dict[str, Any]:
        """Predict one endpoint and return a JSON-friendly result."""
        features = smiles_to_fingerprint(smiles, radius=self.radius, n_bits=self.n_bits).reshape(1, -1)
        prediction = self.estimator.predict(features)[0]
        probability: float | None = None
        if self.task_type == "classification" and hasattr(self.estimator, "predict_proba"):
            probabilities = self.estimator.predict_proba(features)[0]
            probability = round(float(np.max(probabilities)), 4)
        if hasattr(prediction, "item"):
            prediction = prediction.item()
        return {
            "status": "success",
            "target": self.target_name,
            "task_type": self.task_type,
            "prediction": prediction,
            "probability": probability,
            "model": type(self.estimator).__name__,
            "training_samples": self.training_samples,
            "message": "ADMET baseline 预测完成。",
            "disclaimer": DISCLAIMER,
        }

    def save(self, output_path: str | Path) -> str:
        """Save the model artifact and return its absolute path."""
        destination = Path(output_path).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        artifact = {
            "format_version": 1,
            "estimator": self.estimator,
            "target_name": self.target_name,
            "task_type": self.task_type,
            "radius": self.radius,
            "n_bits": self.n_bits,
            "training_samples": self.training_samples,
        }
        joblib.dump(artifact, destination)
        return str(destination)

    @classmethod
    def load(cls, model_path: str | Path) -> "ADMETBaseline":
        """Load a trusted model artifact created by :meth:`save`."""
        source = Path(model_path).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"ADMET 模型文件不存在：{source}")
        artifact = joblib.load(source)
        required = {"estimator", "target_name", "task_type", "radius", "n_bits"}
        if not isinstance(artifact, dict) or not required <= artifact.keys():
            raise ValueError("ADMET 模型文件格式无效或版本不受支持。")
        return cls(
            estimator=artifact["estimator"],
            target_name=str(artifact["target_name"]),
            task_type=artifact["task_type"],
            radius=int(artifact["radius"]),
            n_bits=int(artifact["n_bits"]),
            training_samples=int(artifact.get("training_samples", 0)),
        )


class ConfiguredADMETPredictor:
    """Lazy, failure-isolated access to the configured optional baseline."""

    def __init__(
        self,
        enabled: bool = config.ENABLE_ADMET_MODEL,
        model_path: str | Path = config.ADMET_MODEL_PATH,
    ) -> None:
        self.enabled = enabled
        self.model_path = Path(model_path).expanduser()
        self._model: ADMETBaseline | None = None
        self._load_error: str | None = None

    def _load(self) -> ADMETBaseline | None:
        if self._model is None and self._load_error is None:
            try:
                self._model = ADMETBaseline.load(self.model_path)
            except Exception as exc:
                self._load_error = str(exc)
        return self._model

    def predict(self, smiles: str) -> dict[str, Any]:
        """Return disabled/unavailable/success without breaking the chemistry workflow."""
        if not self.enabled:
            return {
                "status": "disabled",
                "message": "未启用可选 ADMET baseline，当前仅执行 RDKit 规则分析。",
                "disclaimer": DISCLAIMER,
            }
        model = self._load()
        if model is None:
            return {
                "status": "unavailable",
                "message": self._load_error or "ADMET 模型不可用。",
                "model_path": str(self.model_path),
                "disclaimer": DISCLAIMER,
            }
        try:
            return model.predict(smiles)
        except Exception as exc:
            return {
                "status": "failed",
                "message": f"ADMET baseline 预测失败：{exc}",
                "disclaimer": DISCLAIMER,
            }
