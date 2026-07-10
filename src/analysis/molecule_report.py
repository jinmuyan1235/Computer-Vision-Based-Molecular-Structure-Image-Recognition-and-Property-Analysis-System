"""End-to-end molecular image/SMILES analysis workflow."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import OUTPUT_DIR
from src.chem.descriptors import calculate_descriptors
from src.chem.lipinski import evaluate_lipinski
from src.chem.mol_drawer import draw_molecule
from src.chem.smiles_validator import validate_smiles
from src.ocsr.base import OCSRResult
from src.ocsr.recognizer import MoleculeRecognizer
from src.preprocess.image_preprocessor import ImagePreprocessor
from src.preprocess.visualization import save_preprocessing_stages
from src.utils.file_utils import ensure_directory, safe_stem


class MoleculeReportGenerator:
    """Coordinate preprocessing, OCSR, validation, descriptors and rendering."""

    def __init__(self, backend: str | None = None, output_dir: str | Path = OUTPUT_DIR) -> None:
        self.output_dir = ensure_directory(output_dir)
        self.recognizer = MoleculeRecognizer(backend)
        self.preprocessor = ImagePreprocessor()

    @staticmethod
    def _base_report(input_data: dict[str, Any]) -> dict[str, Any]:
        return {
            "status": "failed",
            "message": "分析尚未完成。",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "input": input_data,
            "ocsr": None,
            "validation": {"valid": False, "canonical_smiles": None, "error": None},
            "descriptors": None,
            "lipinski": None,
            "images": {"preprocessing": {}, "preprocessed": None, "redrawn_molecule": None},
        }

    def generate(self, image_path: str | Path | None = None, smiles: str | None = None) -> dict[str, Any]:
        """Generate a complete report from exactly one image or SMILES input."""
        if (image_path is None) == (smiles is None):
            raise ValueError("请且仅提供 image_path 或 smiles 其中一项。")
        return self._from_image(image_path) if image_path is not None else self._from_smiles(smiles or "")

    def _from_image(self, image_path: str | Path) -> dict[str, Any]:
        path = Path(image_path).expanduser().resolve()
        prefix = safe_stem(path.stem)
        report = self._base_report({"type": "image", "filename": path.name, "path": str(path)})
        try:
            stages = self.preprocessor.preprocess_pipeline(path)
            stage_paths = save_preprocessing_stages(stages, self.output_dir / "preprocessed", prefix)
            report["images"]["preprocessing"] = stage_paths
            report["images"]["preprocessed"] = stage_paths["normalized"]
        except Exception as exc:
            report["message"] = f"图像预处理失败：{exc}"
            return report

        recognition_target = path if self.recognizer.is_demo else report["images"]["preprocessed"]
        result = self.recognizer.recognize(recognition_target)
        report["ocsr"] = result.to_dict()
        if result.status != "success" or not result.smiles:
            report["message"] = result.message
            report["validation"]["error"] = "未获得可校验的 SMILES。"
            return report
        return self._complete_chemistry(report, result.smiles, prefix)

    def _from_smiles(self, smiles: str) -> dict[str, Any]:
        prefix = f"manual_{safe_stem(smiles, 'smiles')[:50]}"
        report = self._base_report({"type": "smiles", "smiles": smiles})
        report["ocsr"] = OCSRResult(
            smiles=smiles,
            confidence=None,
            backend="manual",
            status="success",
            message="使用手动输入的 SMILES。",
        ).to_dict()
        return self._complete_chemistry(report, smiles, prefix)

    def _complete_chemistry(self, report: dict[str, Any], smiles: str, prefix: str) -> dict[str, Any]:
        validation = validate_smiles(smiles)
        report["validation"] = validation
        if not validation["valid"]:
            report["message"] = validation["error"]
            return report
        canonical = validation["canonical_smiles"]
        try:
            descriptors = calculate_descriptors(canonical)
            lipinski = evaluate_lipinski(descriptors)
            drawing_path = self.output_dir / "redrawn" / f"{prefix}_structure.png"
            report["descriptors"] = descriptors
            report["lipinski"] = lipinski
            report["images"]["redrawn_molecule"] = draw_molecule(canonical, drawing_path)
            report["status"] = "success"
            report["message"] = "分子识别与性质分析完成。"
        except Exception as exc:
            report["message"] = f"分子性质分析失败：{exc}"
        return report
