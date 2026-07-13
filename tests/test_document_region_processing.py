from __future__ import annotations

import sys
import types
import zipfile
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image, ImageDraw
from reportlab.pdfgen import canvas

import config
from src.documents.detectors import HeuristicMoleculeRegionDetector
from src.documents.input_loader import DocumentInputError, DocumentInputLoader, PDFRenderer, check_file_size
from src.documents.models import DocumentPage, DocumentRegion
from src.documents.processor import DocumentOCSRProcessor


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _make_page(path: Path, molecule_names: list[str] | None = None, text: str = "") -> Path:
    page = Image.new("RGB", (900, 600), "white")
    draw = ImageDraw.Draw(page)
    if text:
        draw.text((60, 30), text, fill="black")
    molecule_names = molecule_names or []
    for index, name in enumerate(molecule_names):
        source = PROJECT_ROOT / "data" / "samples" / f"{name}.png"
        image = Image.open(source).convert("RGB").resize((260, 220))
        page.paste(image, (80 + index * 420, 130))
    path.parent.mkdir(parents=True, exist_ok=True)
    page.save(path)
    return path


def _make_pdf(path: Path, pages: int = 1) -> Path:
    pdf = canvas.Canvas(str(path))
    for index in range(pages):
        pdf.drawString(72, 720, f"Generated fixture page {index + 1}")
        pdf.showPage()
    pdf.save()
    return path


class FakePDFRenderer(PDFRenderer):
    def __init__(self, fixture_pages: list[Path]) -> None:
        super().__init__()
        self.fixture_pages = fixture_pages

    def render(self, pdf_path: str | Path, output_dir: str | Path, document_id: str) -> list[DocumentPage]:
        pages: list[DocumentPage] = []
        page_dir = Path(output_dir) / "pages"
        page_dir.mkdir(parents=True, exist_ok=True)
        for index, source in enumerate(self.fixture_pages, start=1):
            destination = page_dir / f"{document_id}_p{index:03d}.png"
            destination.write_bytes(source.read_bytes())
            with Image.open(destination) as image:
                width, height = image.size
            pages.append(DocumentPage(
                document_id=document_id,
                page_number=index,
                image_path=str(destination),
                width=width,
                height=height,
                source_path=str(pdf_path),
                source_type="pdf",
                render_dpi=200,
            ))
        return pages


class FakeDetector:
    name = "fake-detector"

    def __init__(self, regions: list[DocumentRegion]) -> None:
        self.regions = regions

    def detect(self, page: DocumentPage) -> list[DocumentRegion]:
        return [
            DocumentRegion(
                document_id=page.document_id,
                page_number=page.page_number,
                region_id=region.region_id,
                bbox=region.bbox,
                region_type=region.region_type,
                detection_confidence=region.detection_confidence,
                detector_name=self.name,
            )
            for region in self.regions
        ]


class FakeReportGenerator:
    def __init__(self) -> None:
        self.calls: list[Path] = []
        self.recognizer = types.SimpleNamespace(backend="demo")

    def generate(self, image_path: str | Path) -> dict:
        self.calls.append(Path(image_path))
        return {
            "status": "success",
            "message": "fake ok",
            "input": {},
            "ocsr": {"backend": "demo", "status": "success", "smiles": "CCO"},
            "final": {"smiles": "CCO", "canonical_smiles": "CCO"},
        }


def test_single_page_pdf_processes_with_fake_renderer(tmp_path: Path) -> None:
    page = _make_page(tmp_path / "aspirin_page.png", ["aspirin"], text="single molecule")
    pdf = _make_pdf(tmp_path / "aspirin_doc.pdf")
    loader = DocumentInputLoader(tmp_path / "out", renderer=FakePDFRenderer([page]))
    processor = DocumentOCSRProcessor("demo", tmp_path / "out", loader=loader)
    result = processor.process(pdf)
    assert result["summary"]["page_count"] == 1
    assert result["summary"]["molecule_region_count"] >= 1
    assert result["summary"]["recognized_region_count"] >= 1
    first = next(region for region in result["regions"] if region["region_type"] == "molecule")
    assert first["page_number"] == 1
    assert len(first["bbox"]) == 4
    assert Path(first["crop_path"]).is_file()
    assert Path(result["exports"]["json"]).is_file()
    assert Path(result["exports"]["regions_csv"]).is_file()
    assert Path(result["exports"]["zip"]).is_file()


def test_real_pymupdf_pdf_rendering_when_available(tmp_path: Path) -> None:
    pytest.importorskip("fitz")
    pdf_path = tmp_path / "aspirin_real_pdf.pdf"
    pdf = canvas.Canvas(str(pdf_path))
    pdf.drawString(72, 720, "Aspirin PDF fixture")
    pdf.drawImage(
        str(PROJECT_ROOT / "data" / "samples" / "aspirin.png"),
        90,
        420,
        width=260,
        height=195,
        preserveAspectRatio=True,
        mask="auto",
    )
    pdf.showPage()
    pdf.save()

    result = DocumentOCSRProcessor("demo", tmp_path / "out").process(pdf_path)

    assert result["summary"]["page_count"] == 1
    assert result["summary"]["molecule_region_count"] >= 1
    assert result["summary"]["recognized_region_count"] >= 1
    molecule_region = next(region for region in result["regions"] if region["region_type"] == "molecule")
    assert molecule_region["page_number"] == 1
    assert molecule_region["bbox"][2] > molecule_region["bbox"][0]
    assert Path(result["exports"]["json"]).is_file()


def test_multi_page_pdf_and_blank_page(tmp_path: Path) -> None:
    molecule_page = _make_page(tmp_path / "aspirin_page.png", ["aspirin"])
    blank_page = _make_page(tmp_path / "blank.png", [])
    pdf = _make_pdf(tmp_path / "aspirin_multi.pdf", pages=2)
    loader = DocumentInputLoader(tmp_path / "out", renderer=FakePDFRenderer([molecule_page, blank_page]))
    result = DocumentOCSRProcessor("demo", tmp_path / "out", loader=loader).process(pdf, run_ocsr=False)
    assert result["summary"]["page_count"] == 2
    assert any(page["quality"].get("blank") for page in result["pages"])
    assert result["summary"]["molecule_region_count"] >= 1


def test_detect_only_mode_does_not_call_ocsr(tmp_path: Path) -> None:
    page_path = _make_page(tmp_path / "aspirin_page.png", ["aspirin"])
    detector = FakeDetector([
        DocumentRegion("doc", 1, "p001_r001", (70, 120, 360, 380), "molecule", 0.9),
    ])
    processor = DocumentOCSRProcessor("demo", tmp_path / "out", detector=detector)
    fake_generator = FakeReportGenerator()
    processor.report_generator = fake_generator

    result = processor.process(page_path, run_ocsr=False)

    assert result["processing"]["mode"] == "detect_only"
    assert fake_generator.calls == []
    assert result["regions"][0]["status"] == "detected"


def test_full_document_mode_only_recognizes_screened_molecule_regions(tmp_path: Path) -> None:
    page_path = _make_page(tmp_path / "aspirin_page.png", ["aspirin"], text="Aspirin")
    detector = FakeDetector([
        DocumentRegion("doc", 1, "p001_r001", (70, 120, 360, 380), "molecule", 0.9),
        DocumentRegion("doc", 1, "p001_r002", (45, 20, 340, 70), "molecule", 0.8),
        DocumentRegion("doc", 1, "p001_r003", (30, 480, 400, 540), "text", 0.8),
    ])
    processor = DocumentOCSRProcessor("demo", tmp_path / "out", detector=detector)
    fake_generator = FakeReportGenerator()
    processor.report_generator = fake_generator

    result = processor.process(page_path, run_ocsr=True)

    assert result["processing"]["mode"] == "detect_and_recognize"
    assert len(fake_generator.calls) == 1
    statuses = {region["region_id"]: region for region in result["regions"]}
    assert statuses["p001_r001"]["status"] == "recognized"
    assert statuses["p001_r002"]["status"] == "skipped"
    assert "文字" in statuses["p001_r002"]["message"] or "过小" in statuses["p001_r002"]["message"]
    assert statuses["p001_r003"]["status"] == "skipped"


def test_invalid_bbox_is_not_sent_to_ocsr(tmp_path: Path) -> None:
    page_path = _make_page(tmp_path / "aspirin_page.png", ["aspirin"])
    page = DocumentPage("doc", 1, str(page_path), 900, 600)
    region = DocumentRegion("doc", 1, "p001_r001", (100, 100, 100, 120), "molecule", 0.9)
    processor = DocumentOCSRProcessor("demo", tmp_path / "out")
    fake_generator = FakeReportGenerator()
    processor.report_generator = fake_generator

    processor.recognize_region(region, [page], tmp_path / "out")

    assert region.status == "skipped"
    assert fake_generator.calls == []
    assert region.screening["passed"] is False


def test_damaged_pdf_reports_readable_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    broken_pdf = tmp_path / "broken.pdf"
    broken_pdf.write_bytes(b"%PDF broken")

    fake_fitz = types.SimpleNamespace(open=lambda _path: (_ for _ in ()).throw(RuntimeError("cannot open xref")))
    monkeypatch.setitem(sys.modules, "fitz", fake_fitz)
    with pytest.raises(DocumentInputError, match="damaged|encrypted|unsupported"):
        PDFRenderer().render(broken_pdf, tmp_path / "out", "broken")


def test_multi_molecule_page_and_text_not_molecule(tmp_path: Path) -> None:
    page_path = _make_page(tmp_path / "aspirin_benzene_page.png", ["aspirin", "benzene"], text="Two molecule page")
    page = DocumentPage("doc", 1, str(page_path), 900, 600)
    regions = HeuristicMoleculeRegionDetector().detect(page)
    molecule_regions = [region for region in regions if region.region_type == "molecule"]
    text_regions = [region for region in regions if region.region_type == "text"]
    assert len(molecule_regions) >= 2
    assert text_regions


def test_complex_linework_is_not_filtered_as_text() -> None:
    assert HeuristicMoleculeRegionDetector._looks_like_text(
        width=480,
        height=300,
        aspect=1.6,
        ink_ratio=0.055,
        significant_components=[20] * 22,
        text_line_count=5,
        page_area_ratio=0.08,
        small_component_ratio=0.06,
        skeletal_linework=True,
    ) is False
    assert HeuristicMoleculeRegionDetector._looks_like_text(
        width=480,
        height=300,
        aspect=1.6,
        ink_ratio=0.055,
        significant_components=[20] * 22,
        text_line_count=5,
        page_area_ratio=0.08,
        small_component_ratio=0.06,
        skeletal_linework=False,
    ) is True


def test_complex_molecule_page_keeps_text_and_structure_separate(tmp_path: Path) -> None:
    page_image = Image.new("RGB", (1000, 850), "white")
    draw = ImageDraw.Draw(page_image)
    lines = [
        "Rivaroxaban prevents thromboembolism during surgical operations.",
        "The methods are simple and accurate and do not require equipment.",
    ]
    for index, line in enumerate(lines * 6):
        draw.text((40, 20 + index * 28), line, fill="black")
    caffeine = Image.open(PROJECT_ROOT / "data" / "samples" / "caffeine.png").convert("RGB").resize((420, 300))
    page_image.paste(caffeine, (460, 430))
    page_path = tmp_path / "complex_molecule_with_text.png"
    page_image.save(page_path)

    regions = HeuristicMoleculeRegionDetector().detect(DocumentPage("doc", 1, str(page_path), 1000, 850))

    assert any(region.region_type == "text" for region in regions)
    assert any(region.region_type == "molecule" and region.bbox[0] > 400 for region in regions)


def test_no_molecule_page_does_not_emit_molecule_regions(tmp_path: Path) -> None:
    page_path = _make_page(tmp_path / "text_only.png", [], text="This page contains text but no structure")
    page = DocumentPage("doc", 1, str(page_path), 900, 600)
    regions = HeuristicMoleculeRegionDetector().detect(page)
    assert not [region for region in regions if region.region_type == "molecule"]


def test_multiline_text_block_is_filtered_before_ocsr(tmp_path: Path) -> None:
    page_image = Image.new("RGB", (900, 600), "white")
    draw = ImageDraw.Draw(page_image)
    lines = [
        "This paragraph describes aspirin and rivaroxaban in an article.",
        "It contains many compact character components but no molecule.",
        "The detector should avoid sending this text block to OCSR.",
        "A single bad region must not waste DECIMER inference time.",
        "Only real chemical structure crops should be recognized.",
        "Figure captions and labels are not molecular structures.",
    ]
    for index, line in enumerate(lines):
        draw.text((80, 80 + index * 32), line, fill="black")
    page_path = tmp_path / "text_block.png"
    page_image.save(page_path)

    page = DocumentPage("doc", 1, str(page_path), 900, 600)
    region = DocumentRegion("doc", 1, "p001_r001", (50, 50, 820, 310), "molecule", 0.9)
    processor = DocumentOCSRProcessor("demo", tmp_path / "out")
    fake_generator = FakeReportGenerator()
    processor.report_generator = fake_generator

    processor.recognize_region(region, [page], tmp_path / "out")

    assert region.status == "skipped"
    assert fake_generator.calls == []
    assert region.screening["passed"] is False
    assert region.screening["text_line_count"] >= 4


def test_reaction_like_region_is_not_ocrd_as_single_molecule(tmp_path: Path) -> None:
    page = Image.new("RGB", (700, 260), "white")
    draw = ImageDraw.Draw(page)
    draw.line((80, 130, 620, 130), fill="black", width=4)
    draw.polygon([(620, 130), (590, 115), (590, 145)], fill="black")
    page_path = tmp_path / "reaction_like.png"
    page.save(page_path)
    result = DocumentOCSRProcessor("demo", tmp_path / "out").process(page_path)
    assert any(region["region_type"] == "reaction_like" for region in result["regions"])
    assert all(region["status"] != "recognized" for region in result["regions"] if region["region_type"] == "reaction_like")


def test_zip_image_collection_is_supported(tmp_path: Path) -> None:
    image_a = _make_page(tmp_path / "aspirin_page.png", ["aspirin"])
    image_b = _make_page(tmp_path / "benzene_page.png", ["benzene"])
    archive = tmp_path / "pages.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.write(image_a, "a/aspirin_page.png")
        zf.write(image_b, "b/benzene_page.png")
    result = DocumentOCSRProcessor("demo", tmp_path / "out").process(archive, run_ocsr=False)
    assert result["summary"]["page_count"] == 2


def test_region_editing_records_audit_and_reruns(tmp_path: Path) -> None:
    page_path = _make_page(tmp_path / "aspirin_page.png", ["aspirin"])
    processor = DocumentOCSRProcessor("demo", tmp_path / "out")
    result = processor.process(page_path, run_ocsr=False)
    page = result["pages"][0]
    edited = processor.apply_edits(
        result,
        [{
            "action": "add",
            "page_number": page["page_number"],
            "bbox": [70, 120, 360, 380],
            "region_type": "molecule",
        }],
        rerun_ocsr=True,
    )
    added = next(region for region in edited["regions"] if region["source"] == "user")
    assert added["audit"]
    assert added["status"] == "recognized"
    assert added["report"]["input"]["bbox"] == added["bbox"]


def test_single_region_failure_does_not_stop_other_regions(tmp_path: Path) -> None:
    page_path = _make_page(tmp_path / "unknown_page.png", ["aspirin", "benzene"])
    result = DocumentOCSRProcessor("demo", tmp_path / "out").process(page_path)
    molecule_regions = [region for region in result["regions"] if region["region_type"] == "molecule"]
    assert len(molecule_regions) >= 2
    assert all(region["status"] == "failed" for region in molecule_regions)
    assert Path(result["exports"]["failure_cases_csv"]).is_file()


def test_file_size_limit_is_reported(tmp_path: Path) -> None:
    big_file = tmp_path / "large.png"
    big_file.write_bytes(b"0" * 100)
    with pytest.raises(DocumentInputError, match="safety limit"):
        check_file_size(big_file, max_size_mb=0.00001)


def test_region_csv_contains_coordinates_and_final_result(tmp_path: Path) -> None:
    page_path = _make_page(tmp_path / "aspirin_page.png", ["aspirin"])
    result = DocumentOCSRProcessor("demo", tmp_path / "out").process(page_path)
    frame = pd.read_csv(result["exports"]["regions_csv"])
    assert {"document_id", "page_number", "region_id", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"}.issubset(frame.columns)
    molecule_rows = frame[frame["region_type"] == "molecule"]
    assert not molecule_rows.empty
    assert "final_smiles" in frame.columns
