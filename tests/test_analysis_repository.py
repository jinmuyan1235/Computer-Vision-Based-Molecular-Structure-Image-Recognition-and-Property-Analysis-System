from __future__ import annotations

import json
from pathlib import Path

from src.analysis.molecule_report import MoleculeReportGenerator
from src.storage.analysis_repository import AnalysisRepository, record_result_payload


def _report(tmp_path: Path, smiles: str, analysis_id: str, filename: str) -> dict:
    report = MoleculeReportGenerator("manual", tmp_path).generate(smiles=smiles, analysis_id=analysis_id)
    report["input"].update({"type": "image", "filename": filename, "path": str(tmp_path / filename), "image_sha256": f"sha-{analysis_id}"})
    report["ocsr"]["backend"] = "demo"
    report["recognition_decision"] = {"decision": "accepted", "manual_review_recommended": False}
    return report


def test_repository_saves_searches_filters_and_favorites(tmp_path: Path) -> None:
    db_path = tmp_path / "app.db"
    repository = AnalysisRepository(db_path)
    aspirin = _report(tmp_path, "CC(=O)Oc1ccccc1C(=O)O", "aspirin001", "aspirin.png")
    review = _report(tmp_path, "CCO", "ethanol001", "ethanol.png")
    review["recognition_decision"] = {"decision": "review_needed", "manual_review_recommended": True}
    repository.save_analysis(aspirin, tmp_path / "aspirin_report.json")
    repository.save_analysis(review, tmp_path / "review_report.json")

    assert [row["analysis_id"] for row in repository.list_analyses(query="aspirin")] == ["aspirin001"]
    assert [row["analysis_id"] for row in repository.list_analyses(query=aspirin["chemical_identity"]["inchikey"])] == ["aspirin001"]
    assert [row["analysis_id"] for row in repository.list_analyses(status_filter="review_needed")] == ["ethanol001"]

    repository.set_favorite("aspirin001", True)
    favorite = repository.get_analysis("aspirin001")
    assert favorite is not None
    assert favorite["is_favorite"] is True
    assert repository.list_analyses(favorites_only=True)[0]["analysis_id"] == "aspirin001"

    csv_text = repository.export_rows_csv(repository.list_analyses())
    assert "analysis_id,created_at" in csv_text
    assert "aspirin001" in csv_text

    repository.delete_analysis("aspirin001")
    assert repository.get_analysis("aspirin001") is None


def test_repository_loads_report_from_single_and_batch_payload(tmp_path: Path) -> None:
    db_path = tmp_path / "app.db"
    repository = AnalysisRepository(db_path)
    report = _report(tmp_path, "CCO", "ethanol001", "ethanol.png")
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
    repository.save_analysis(report, report_path)
    assert repository.load_report("ethanol001")["analysis_id"] == "ethanol001"

    batch_report = _report(tmp_path, "c1ccccc1", "benzene001", "benzene.png")
    batch_path = tmp_path / "batch.json"
    batch_path.write_text(json.dumps({"reports": [batch_report]}, ensure_ascii=False), encoding="utf-8")
    repository.save_analysis(batch_report, batch_path)
    assert repository.load_report("benzene001")["analysis_id"] == "benzene001"


def test_repository_records_corrections_jobs_and_result_payloads(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "app.db"
    repository = AnalysisRepository(db_path)
    report = _report(tmp_path, "CCO", "ethanol001", "ethanol.png")
    repository.save_analysis(report, tmp_path / "report.json")

    correction = repository.record_correction("ethanol001", "CO", "CCO", "user", "fixed missing carbon")
    assert correction["analysis_id"] == "ethanol001"

    job = repository.save_job({
        "job_id": "batch001",
        "status": "running",
        "total": 10,
        "completed": 4,
        "current_file": "compound_004.png",
        "created_at": "2026-01-01T00:00:00+00:00",
        "updated_at": "2026-01-01T00:01:00+00:00",
        "result_path": str(tmp_path / "batch.json"),
    })
    assert job["job_id"] == "batch001"

    batch_report = _report(tmp_path, "CCN", "amine001", "amine.png")
    payload_path = tmp_path / "payload.json"
    payload_path.write_text(json.dumps({"reports": [batch_report]}, ensure_ascii=False), encoding="utf-8")

    import src.storage.analysis_repository as module

    monkeypatch.setattr(module, "AnalysisRepository", lambda: AnalysisRepository(db_path))
    assert record_result_payload({"reports": [batch_report]}, payload_path) == 1
    assert repository.get_analysis("amine001") is not None
