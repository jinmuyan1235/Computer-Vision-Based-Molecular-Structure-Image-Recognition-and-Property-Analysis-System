"""Regression tests for shared visual candidate screening and evaluation."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image, ImageDraw

import src.datasets.pipeline as dataset_pipeline_module
import src.documents.candidate_screening as screening_module
import src.documents.detectors as detectors_module
import src.documents.processor as processor_module
from scripts.collect_ocsr_dataset import _pipeline, build_parser
from src.documents.candidate_screening import (
    get_crop_screening_config, get_proposal_config, get_screening_config,
    screen_region_candidate,
)
from src.documents.detectors import HeuristicMoleculeRegionDetector
from src.evaluation.visual_detector import evaluate_visual_detector
from src.evaluation.visual_detector_compare import compare_visual_detector_runs


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FROZEN_CHECKSUM_FILE_SHA256 = {
    "visual-dev-v0.1": "74e0999340ccb99e50ed0970957d6cd6a532131c3dd28c7e164f851ac88ef9ab",
    "visual-holdout-v0.1": "ec144520fbc6b8df682f63659ad8e3d04c30406266eaf4a0588bd7932fed7cdf",
    "visual-page-holdout-v0.1": "8f16dda80677e6cfbd243a34b77862b8ad80162a2ff409cee28442d87724aa5f",
}


def _screen(image: Image.Image, initial: str = "molecule") -> str:
    array = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    return screen_region_candidate(
        array, (0, 0, image.width, image.height), initial, 0.9, config="candidate",
    ).recommended_region_type


def _arrow(size: tuple[int, int] = (500, 180)) -> Image.Image:
    image = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(image)
    draw.line((60, size[1] // 2, size[0] - 70, size[1] // 2), fill="black", width=4)
    x, y = size[0] - 70, size[1] // 2
    draw.polygon([(x, y), (x - 30, y - 15), (x - 30, y + 15)], fill="black")
    return image


def _molecule() -> Image.Image:
    return Image.open(PROJECT_ROOT / "data" / "samples" / "caffeine.png").convert("RGB")


def test_reaction_arrow_is_not_a_molecule() -> None:
    assert _screen(_arrow()) == "reaction"


def test_molecule_plus_arrow_routes_to_reaction_or_uncertain() -> None:
    image = Image.new("RGB", (700, 220), "white")
    image.paste(_molecule().resize((250, 180)), (20, 20))
    draw = ImageDraw.Draw(image)
    draw.line((330, 110, 650, 110), fill="black", width=4)
    draw.polygon([(650, 110), (620, 95), (620, 125)], fill="black")
    assert _screen(image) in {"reaction", "uncertain"}


def test_plain_text_is_classified_as_text() -> None:
    image = Image.new("RGB", (700, 220), "white")
    draw = ImageDraw.Draw(image)
    for index in range(5):
        draw.text((20, 20 + index * 35), "This is ordinary article text and not a molecule.", fill="black")
    assert _screen(image) == "text"


def test_grid_is_classified_as_table() -> None:
    image = Image.new("RGB", (500, 300), "white")
    draw = ImageDraw.Draw(image)
    for y in range(20, 281, 65):
        draw.line((20, y, 480, y), fill="black", width=3)
    for x in range(20, 481, 115):
        draw.line((x, 20, x, 280), fill="black", width=3)
    assert _screen(image) == "table"


def test_single_clear_molecule_passes_candidate_gate() -> None:
    assert _screen(_molecule()) == "molecule"


def test_two_separate_molecules_do_not_pass_single_molecule_gate() -> None:
    molecule = _molecule().resize((240, 180))
    image = Image.new("RGB", (600, 220), "white")
    image.paste(molecule, (20, 20))
    image.paste(molecule, (340, 20))
    assert _screen(image) == "multiple_molecules"


def test_interactive_and_collection_flows_import_the_same_screening_function() -> None:
    assert processor_module.screen_region_candidate is screening_module.screen_region_candidate
    assert dataset_pipeline_module.screen_region_candidate is screening_module.screen_region_candidate
    assert detectors_module.screen_region_candidate is screening_module.screen_region_candidate


def test_baseline_and_candidate_profiles_are_distinct_and_selectable() -> None:
    baseline = get_screening_config("baseline")
    candidate = get_screening_config("candidate")
    assert baseline.name == "baseline"
    assert candidate.name == "candidate"
    assert baseline.dilation_kernel != candidate.dilation_kernel
    assert baseline.merge_overlap_ratio != candidate.merge_overlap_ratio
    with pytest.raises(ValueError, match="Unknown candidate-screening config"):
        get_screening_config("missing")  # type: ignore[arg-type]


def test_proposal_and_crop_profiles_are_independent_and_default_is_safe() -> None:
    detector = HeuristicMoleculeRegionDetector()
    assert detector.proposal_config is get_proposal_config("baseline")
    assert detector.crop_screening_config is get_crop_screening_config("candidate")
    assert not hasattr(get_proposal_config("candidate"), "molecule_score_threshold")
    assert not hasattr(get_crop_screening_config("candidate"), "dilation_kernel")


def test_legacy_screening_config_warns_and_maps_both_axes() -> None:
    with pytest.warns(DeprecationWarning, match="deprecated"):
        detector = HeuristicMoleculeRegionDetector(screening_config="candidate")
    assert detector.proposal_config.name == "candidate"
    assert detector.crop_screening_config.name == "candidate"


def test_legacy_cli_screening_flag_emits_deprecation_warning(tmp_path: Path) -> None:
    args = build_parser().parse_args([
        "--dataset-root", str(tmp_path), "--dry-run", "--screening-config", "candidate",
        "pmc", "--pmcid", "PMC1234567",
    ])
    with pytest.warns(FutureWarning, match="deprecated"):
        pipeline = _pipeline(args)
    assert pipeline.proposal_config.name == "candidate"
    assert pipeline.crop_screening_config.name == "candidate"


def test_three_way_crop_decisions_are_explicit() -> None:
    molecule = cv2.cvtColor(np.asarray(_molecule()), cv2.COLOR_RGB2BGR)
    accepted = screen_region_candidate(molecule, (0, 0, molecule.shape[1], molecule.shape[0]), "molecule", 0.9)
    rejected_image = cv2.cvtColor(np.asarray(_arrow()), cv2.COLOR_RGB2BGR)
    rejected = screen_region_candidate(rejected_image, (0, 0, rejected_image.shape[1], rejected_image.shape[0]), "molecule", 0.9)
    multiple_image = Image.new("RGB", (600, 220), "white")
    crop = _molecule().resize((240, 180)); multiple_image.paste(crop, (20, 20)); multiple_image.paste(crop, (340, 20))
    multiple_array = cv2.cvtColor(np.asarray(multiple_image), cv2.COLOR_RGB2BGR)
    review = screen_region_candidate(multiple_array, (0, 0, 600, 220), "molecule", 0.9)
    assert accepted.decision == "accept_molecule"
    assert rejected.decision == "reject_negative"
    assert review.decision == "review_needed"
    assert accepted.config_version == "crop-screening-candidate-v1"


def _evaluation_manifest(tmp_path: Path) -> Path:
    molecule_path = tmp_path / "molecule.png"
    text_path = tmp_path / "text.png"
    _molecule().save(molecule_path)
    text = Image.new("RGB", (500, 180), "white")
    draw = ImageDraw.Draw(text)
    for index in range(4):
        draw.text((20, 20 + index * 34), "ordinary article text", fill="black")
    text.save(text_path)
    manifest = tmp_path / "detector_training_manifest.csv"
    fields = ["sample_id", "image_path", "visual_review_status", "source_document", "source_page_path"]
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([
            {
                "sample_id": "pmc_demo_molecule_abcdef", "image_path": str(molecule_path),
                "visual_review_status": "valid_single_molecule_crop", "source_document": "doc-a",
                "source_page_path": "page-1.png",
            },
            {
                "sample_id": "pmc_demo_text_123abc", "image_path": str(text_path),
                "visual_review_status": "text", "source_document": "doc-a",
                "source_page_path": "page-1.png",
            },
        ])
    return manifest


def test_evaluation_and_comparison_write_required_fields(tmp_path: Path) -> None:
    manifest = _evaluation_manifest(tmp_path)
    baseline_dir = tmp_path / "baseline"
    candidate_dir = tmp_path / "candidate"
    baseline = evaluate_visual_detector(manifest, baseline_dir, config_name="baseline")
    candidate = evaluate_visual_detector(manifest, candidate_dir, config_name="candidate")
    required = {
        "metrics.json", "predictions.csv", "confusion_matrix.csv", "per_class_metrics.csv",
        "per_document_metrics.csv", "errors.csv", "report.md",
    }
    assert required == {path.name for path in baseline_dir.iterdir()}
    assert baseline["molecule_vs_non_molecule"]["precision"] == 1.0
    assert "macro_f1" in candidate["multiclass"]
    assert "cannot measure complete-page molecule detection recall" in candidate["scope_limitation"]
    comparison = compare_visual_detector_runs(baseline_dir, candidate_dir, tmp_path / "comparison")
    assert "molecule_precision" in comparison["metrics"]
    assert "development_set_improved" in comparison
    assert (tmp_path / "comparison" / "comparison.json").is_file()
    assert (tmp_path / "comparison" / "comparison.md").is_file()


def test_dataset_role_is_explicit_then_summary_then_development_default(tmp_path: Path) -> None:
    manifest = _evaluation_manifest(tmp_path)
    default = evaluate_visual_detector(manifest, tmp_path / "default", config_name="baseline")
    assert default["dataset_role"] == "development"
    assert default["dataset_role_source"] == "default"

    (manifest.parent / "dataset_summary.json").write_text(
        json.dumps({"dataset_role": "holdout"}), encoding="utf-8",
    )
    automatic = evaluate_visual_detector(manifest, tmp_path / "automatic", config_name="baseline")
    assert automatic["dataset_role"] == "holdout"
    assert automatic["dataset_role_source"] == "dataset_summary"
    explicit = evaluate_visual_detector(
        manifest, tmp_path / "explicit", config_name="baseline", dataset_role="development",
    )
    assert explicit["dataset_role"] == "development"
    assert explicit["dataset_role_source"] == "command_line"


@pytest.mark.parametrize("version", sorted(FROZEN_CHECKSUM_FILE_SHA256))
def test_frozen_visual_snapshots_are_unchanged_when_present(version: str) -> None:
    checksum_file = PROJECT_ROOT / "data" / "datasets" / version / "checksums.sha256"
    if not checksum_file.is_file():
        pytest.skip("Local frozen dataset is intentionally not committed.")
    assert hashlib.sha256(checksum_file.read_bytes()).hexdigest() == FROZEN_CHECKSUM_FILE_SHA256[version]
    root = checksum_file.parent
    for line in checksum_file.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        assert hashlib.sha256((root / relative).read_bytes()).hexdigest() == expected
