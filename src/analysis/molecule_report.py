"""End-to-end molecular image/SMILES analysis workflow."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from config import OUTPUT_DIR
from src.chem.descriptors import calculate_descriptors
from src.chem.lipinski import evaluate_lipinski
from src.chem.mol_drawer import draw_molecule
from src.chem.standardization import standardize_smiles
from src.analysis.correction import (
    default_correction_state,
    default_final_state,
    normalize_ocsr_block,
    sha256_file,
)
from src.ocsr.base import OCSRResult
from src.ocsr.recognizer import MoleculeRecognizer
from src.ml.admet_baseline import ConfiguredADMETPredictor
from src.preprocess.image_preprocessor import ImagePreprocessor
from src.preprocess.visualization import save_preprocessing_stages
from src.utils.file_utils import ensure_directory, safe_stem


class MoleculeReportGenerator:
    """Coordinate preprocessing, OCSR, validation, descriptors and rendering."""

    def __init__(
        self,
        backend: str | None = None,
        output_dir: str | Path = OUTPUT_DIR,
        runtime_config: dict[str, Any] | None = None,
    ) -> None:
        self.output_dir = ensure_directory(output_dir)
        self.recognizer = MoleculeRecognizer(backend, runtime_config=runtime_config)
        self.preprocessor = ImagePreprocessor()
        self.admet_predictor = ConfiguredADMETPredictor()

    @staticmethod
    def _base_report(input_data: dict[str, Any]) -> dict[str, Any]:
        return {
            "analysis_id": uuid4().hex,
            "status": "failed",
            "message": "分析尚未完成。",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "input": input_data,
            "ocsr": None,
            "correction": default_correction_state(),
            "final": default_final_state(),
            "validation": {"valid": False, "canonical_smiles": None, "standardized_smiles": None, "error": None},
            "chemical_identity": None,
            "standardization": {"profile": None, "changed": False, "steps": [], "warnings": []},
            "structure_warnings": [],
            "descriptors": None,
            "lipinski": None,
            "admet": None,
            "images": {
                "preprocessing": {},
                "preprocessed": None,
                "redrawn_molecule": None,
                "predicted_molecule": None,
                "corrected_molecule": None,
            },
        }

    def generate(self, image_path: str | Path | None = None, smiles: str | None = None) -> dict[str, Any]:
        """Generate a complete report from exactly one image or SMILES input."""
        if (image_path is None) == (smiles is None):
            raise ValueError("请且仅提供 image_path 或 smiles 其中一项。")
        return self._from_image(image_path) if image_path is not None else self._from_smiles(smiles or "")

    def _from_image(self, image_path: str | Path) -> dict[str, Any]:
        path = Path(image_path).expanduser().resolve()
        report = self._base_report({
            "type": "image",
            "filename": path.name,
            "path": str(path),
            "image_sha256": sha256_file(path),
        })
        prefix = f"{safe_stem(path.stem)}_{report['analysis_id'][:8]}"
        try:
            stages = self.preprocessor.preprocess_pipeline(path)
            stage_paths = save_preprocessing_stages(stages, self.output_dir / "preprocessed", prefix)
            report["images"]["preprocessing"] = stage_paths
            report["images"]["preprocessed"] = stage_paths["normalized"]
        except Exception as exc:
            report["message"] = f"图像预处理失败：{exc}"
            return report

        recognition_target = path if self.recognizer.preferred_image_stage == "original" else report["images"]["preprocessed"]
        result = self.recognizer.recognize(recognition_target)
        report["ocsr"] = normalize_ocsr_block(result.to_dict())
        if result.status != "success" or not result.smiles:
            report["message"] = result.message
            report["validation"]["error"] = "未获得可校验的 SMILES。"
            return report
        final_source = "ensemble_recommendation" if result.backend == "ensemble" else "ocsr"
        return self._complete_chemistry(report, result.smiles, prefix, final_source=final_source)

    def _from_smiles(self, smiles: str) -> dict[str, Any]:
        report = self._base_report({"type": "smiles", "smiles": smiles})
        prefix = f"manual_{safe_stem(smiles, 'smiles')[:40]}_{report['analysis_id'][:8]}"
        report["ocsr"] = OCSRResult(
            smiles=smiles,
            confidence=None,
            backend="manual",
            status="success",
            message="使用手动输入的 SMILES。",
            inference_time_ms=0.0,
            model_name="manual",
            model_version="built-in",
            device="cpu",
        ).to_dict()
        report["ocsr"] = normalize_ocsr_block(report["ocsr"])
        return self._complete_chemistry(report, smiles, prefix, final_source="manual")

    def _complete_chemistry(
        self,
        report: dict[str, Any],
        smiles: str,
        prefix: str,
        final_source: str,
    ) -> dict[str, Any]:
        standardization_result = standardize_smiles(smiles)
        identity = standardization_result["chemical_identity"]
        validation = {
            "valid": standardization_result["valid"],
            "canonical_smiles": identity.get("canonical_smiles"),
            "standardized_smiles": identity.get("standardized_smiles"),
            "error": standardization_result["error"],
        }
        report["validation"] = validation
        report["chemical_identity"] = identity
        report["standardization"] = standardization_result["standardization"]
        report["structure_warnings"] = standardization_result["structure_warnings"]
        if not validation["valid"]:
            report["message"] = validation["error"]
            return report
        canonical = str(validation["canonical_smiles"])
        analysis_smiles = str(validation["standardized_smiles"] or canonical)
        try:
            descriptors = calculate_descriptors(analysis_smiles)
            lipinski = evaluate_lipinski(descriptors)
            drawing_path = self.output_dir / "redrawn" / f"{prefix}_structure.png"
            report["descriptors"] = descriptors
            report["lipinski"] = lipinski
            report["admet"] = self.admet_predictor.predict(analysis_smiles)
            report["images"]["redrawn_molecule"] = draw_molecule(analysis_smiles, drawing_path)
            if final_source in {"ocsr", "ensemble_recommendation"}:
                report["images"]["predicted_molecule"] = report["images"]["redrawn_molecule"]
            report["final"] = {
                "smiles": analysis_smiles,
                "raw_smiles": smiles,
                "canonical_smiles": canonical,
                "standardized_smiles": analysis_smiles,
                "source": final_source,
            }
            report["status"] = "success"
            report["message"] = "分子识别与性质分析完成。"
        except Exception as exc:
            report["message"] = f"分子性质分析失败：{exc}"
        return report
