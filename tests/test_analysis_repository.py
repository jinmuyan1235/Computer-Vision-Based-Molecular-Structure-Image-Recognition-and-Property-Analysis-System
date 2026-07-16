from __future__ import annotations

import json
from pathlib import Path

from src.analysis.molecule_report import MoleculeReportGenerator
from src.runtime.run_store import cleanup_runs, create_image_run_from_bytes, save_run_report, write_runtime_metadata
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
    aspirin_path = tmp_path / "aspirin_report.json"
    review_path = tmp_path / "review_report.json"
    aspirin_path.write_text(json.dumps(aspirin, ensure_ascii=False), encoding="utf-8")
    review_path.write_text(json.dumps(review, ensure_ascii=False), encoding="utf-8")
    repository.save_analysis(aspirin, aspirin_path)
    repository.save_analysis(review, review_path)

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
    assert aspirin_path.exists()


def test_repository_delete_analysis_and_files_removes_owned_run_dir(tmp_path: Path) -> None:
    repository = AnalysisRepository(tmp_path / "app.db")
    image_run = create_image_run_from_bytes(
        b"owned-image",
        "owned.png",
        runs_root=tmp_path / "runs",
        analysis_id="owned001",
    )
    report = MoleculeReportGenerator("manual", image_run.run_dir).generate(
        smiles="CCO",
        analysis_id=image_run.analysis_id,
    )
    report_path = save_run_report(report, image_run)
    repository.save_analysis(report, report_path)

    result = repository.delete_analysis_and_files(image_run.analysis_id)

    assert repository.get_analysis(image_run.analysis_id) is None
    assert not image_run.run_dir.exists()
    assert str(image_run.run_dir.resolve()) in result["deleted_paths"]
    assert result["errors"] == []


def test_repository_delete_analysis_and_files_removes_direct_report_file(tmp_path: Path) -> None:
    repository = AnalysisRepository(tmp_path / "app.db")
    report = _report(tmp_path, "CCO", "ethanol001", "ethanol.png")
    report_path = tmp_path / "ethanol_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
    repository.save_analysis(report, report_path)

    result = repository.delete_analysis_and_files("ethanol001")

    assert repository.get_analysis("ethanol001") is None
    assert not report_path.exists()
    assert str(report_path.resolve()) in result["deleted_paths"]


def test_repository_delete_analysis_and_files_keeps_shared_batch_payload(tmp_path: Path) -> None:
    repository = AnalysisRepository(tmp_path / "app.db")
    first = _report(tmp_path, "CCO", "ethanol001", "ethanol.png")
    second = _report(tmp_path, "c1ccccc1", "benzene001", "benzene.png")
    payload_path = tmp_path / "batch_payload.json"
    payload_path.write_text(json.dumps({"reports": [first, second]}, ensure_ascii=False), encoding="utf-8")
    repository.save_analysis(first, payload_path)
    repository.save_analysis(second, payload_path)

    result = repository.delete_analysis_and_files("ethanol001")

    assert repository.get_analysis("ethanol001") is None
    assert repository.get_analysis("benzene001") is not None
    assert payload_path.exists()
    assert result["deleted_paths"] == []


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


def test_repository_favorites_protect_persistent_image_runs(tmp_path: Path) -> None:
    repository = AnalysisRepository(tmp_path / "app.db")
    image_run = create_image_run_from_bytes(
        b"favorite-image",
        "favorite.png",
        runs_root=tmp_path / "runs",
        analysis_id="favorite001",
    )
    report = MoleculeReportGenerator("manual", image_run.run_dir).generate(
        smiles="CCO",
        analysis_id=image_run.analysis_id,
    )
    report_path = save_run_report(report, image_run)
    repository.save_analysis(report, report_path)

    repository.set_favorite(image_run.analysis_id, True)
    runtime = json.loads(image_run.runtime_path.read_text(encoding="utf-8"))
    assert runtime["protected"] is True
    assert "favorite" in runtime["protected_reasons"]

    write_runtime_metadata(image_run, {"created_at": "2020-01-01T00:00:00+00:00"})
    protected_cleanup = cleanup_runs(tmp_path / "runs", retention_days=1, max_storage_gb=10)
    assert protected_cleanup["deleted_count"] == 0
    assert image_run.run_dir.exists()

    repository.set_favorite(image_run.analysis_id, False)
    runtime = json.loads(image_run.runtime_path.read_text(encoding="utf-8"))
    assert runtime["protected"] is False

    unprotected_cleanup = cleanup_runs(tmp_path / "runs", retention_days=1, max_storage_gb=10)
    assert unprotected_cleanup["deleted_count"] == 1
    assert not image_run.run_dir.exists()
