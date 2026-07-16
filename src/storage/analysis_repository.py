"""Repository for searchable local analysis history."""

from __future__ import annotations

from contextlib import closing
from datetime import datetime, timezone
import csv
import io
import json
from pathlib import Path
from typing import Any, Iterable, Mapping
from uuid import uuid4

from src.storage.database import connect


STATUS_FILTERS = {
    "all": "",
    "success": "status = 'success'",
    "review_needed": "decision = 'review_needed'",
    "rejected": "decision = 'rejected'",
    "failed": "status != 'success'",
}


class AnalysisRepository:
    """Read/write the local SQLite history index."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        self.db_path = Path(db_path).expanduser().resolve() if db_path is not None else None

    def save_analysis(self, report: Mapping[str, Any], report_path: str | Path | None = None) -> dict[str, Any]:
        """Upsert one analysis report into the history index."""
        record = analysis_record_from_report(report, report_path)
        with closing(connect(self.db_path)) as connection:
            existing = connection.execute(
                "SELECT is_favorite FROM analyses WHERE analysis_id = ?",
                (record["analysis_id"],),
            ).fetchone()
            if existing is not None:
                record["is_favorite"] = int(existing["is_favorite"])
            connection.execute(
                """
                INSERT INTO analyses (
                    analysis_id, created_at, updated_at, input_type, filename, input_path,
                    image_sha256, backend, decision, status, final_smiles, inchikey,
                    report_path, is_favorite
                )
                VALUES (
                    :analysis_id, :created_at, :updated_at, :input_type, :filename, :input_path,
                    :image_sha256, :backend, :decision, :status, :final_smiles, :inchikey,
                    :report_path, :is_favorite
                )
                ON CONFLICT(analysis_id) DO UPDATE SET
                    updated_at = excluded.updated_at,
                    input_type = excluded.input_type,
                    filename = excluded.filename,
                    input_path = excluded.input_path,
                    image_sha256 = excluded.image_sha256,
                    backend = excluded.backend,
                    decision = excluded.decision,
                    status = excluded.status,
                    final_smiles = excluded.final_smiles,
                    inchikey = excluded.inchikey,
                    report_path = excluded.report_path,
                    is_favorite = analyses.is_favorite
                """,
                record,
            )
            connection.commit()
        return record

    def save_many(self, reports: Iterable[Mapping[str, Any]], report_path: str | Path | None = None) -> int:
        """Upsert many reports and return the saved count."""
        count = 0
        for report in reports:
            self.save_analysis(report, report_path=report_path)
            count += 1
        return count

    def record_correction(
        self,
        analysis_id: str,
        previous_smiles: str | None,
        new_smiles: str | None,
        source: str,
        notes: str = "",
    ) -> dict[str, Any]:
        """Persist one correction event."""
        record = {
            "correction_id": uuid4().hex,
            "analysis_id": analysis_id,
            "previous_smiles": previous_smiles,
            "new_smiles": new_smiles,
            "source": source,
            "created_at": utc_now(),
            "notes": notes,
        }
        with closing(connect(self.db_path)) as connection:
            connection.execute(
                """
                INSERT INTO corrections (
                    correction_id, analysis_id, previous_smiles, new_smiles, source, created_at, notes
                )
                VALUES (
                    :correction_id, :analysis_id, :previous_smiles, :new_smiles, :source, :created_at, :notes
                )
                """,
                record,
            )
            connection.commit()
        return record

    def save_job(self, job: Mapping[str, Any], job_type: str = "batch") -> dict[str, Any]:
        """Upsert a background job row."""
        now = utc_now()
        progress = {
            "total": job.get("total"),
            "completed": job.get("completed"),
            "current_file": job.get("current_file"),
            "accepted": job.get("accepted"),
            "review_needed": job.get("review_needed"),
            "failed": job.get("failed"),
        }
        record = {
            "job_id": str(job.get("job_id") or uuid4().hex),
            "job_type": job_type,
            "status": str(job.get("status") or "unknown"),
            "progress": json.dumps(progress, ensure_ascii=False),
            "created_at": str(job.get("created_at") or now),
            "updated_at": str(job.get("updated_at") or now),
            "result_path": str(job.get("result_path") or ""),
        }
        with closing(connect(self.db_path)) as connection:
            connection.execute(
                """
                INSERT INTO jobs (job_id, job_type, status, progress, created_at, updated_at, result_path)
                VALUES (:job_id, :job_type, :status, :progress, :created_at, :updated_at, :result_path)
                ON CONFLICT(job_id) DO UPDATE SET
                    job_type = excluded.job_type,
                    status = excluded.status,
                    progress = excluded.progress,
                    updated_at = excluded.updated_at,
                    result_path = excluded.result_path
                """,
                record,
            )
            connection.commit()
        return record

    def list_analyses(
        self,
        query: str = "",
        status_filter: str = "all",
        favorites_only: bool = False,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Search analyses by filename, SMILES, InChIKey, or id."""
        clauses: list[str] = []
        params: list[Any] = []
        query = query.strip()
        if query:
            like = f"%{query}%"
            clauses.append(
                "(filename LIKE ? OR final_smiles LIKE ? OR inchikey LIKE ? OR analysis_id LIKE ? OR image_sha256 LIKE ?)"
            )
            params.extend([like, like, like, like, like])
        filter_clause = STATUS_FILTERS.get(status_filter, "")
        if filter_clause:
            clauses.append(filter_clause)
        if favorites_only:
            clauses.append("is_favorite = 1")
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        params.extend([int(limit), int(offset)])
        with closing(connect(self.db_path)) as connection:
            rows = connection.execute(
                f"""
                SELECT analysis_id, created_at, updated_at, input_type, filename, input_path,
                       image_sha256, backend, decision, status, final_smiles, inchikey,
                       report_path, is_favorite
                FROM analyses
                {where}
                ORDER BY is_favorite DESC, created_at DESC
                LIMIT ? OFFSET ?
                """,
                params,
            ).fetchall()
        return [_row_dict(row) for row in rows]

    def get_analysis(self, analysis_id: str) -> dict[str, Any] | None:
        """Return one analysis row by id."""
        with closing(connect(self.db_path)) as connection:
            row = connection.execute("SELECT * FROM analyses WHERE analysis_id = ?", (analysis_id,)).fetchone()
        return _row_dict(row) if row is not None else None

    def load_report(self, analysis_id: str) -> dict[str, Any] | None:
        """Load the original report JSON for an indexed analysis."""
        row = self.get_analysis(analysis_id)
        if not row:
            return None
        path = Path(str(row.get("report_path") or ""))
        if not path.is_file():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("analysis_id") == analysis_id:
            return payload
        for key in ("results", "reports"):
            for report in payload.get(key) or []:
                if isinstance(report, Mapping) and report.get("analysis_id") == analysis_id:
                    return dict(report)
        for region in payload.get("regions") or []:
            report = region.get("report") if isinstance(region, Mapping) else None
            if isinstance(report, Mapping) and report.get("analysis_id") == analysis_id:
                return dict(report)
        return None

    def set_favorite(self, analysis_id: str, favorite: bool) -> None:
        with closing(connect(self.db_path)) as connection:
            connection.execute(
                "UPDATE analyses SET is_favorite = ?, updated_at = ? WHERE analysis_id = ?",
                (1 if favorite else 0, utc_now(), analysis_id),
            )
            connection.commit()

    def delete_analysis(self, analysis_id: str) -> None:
        """Delete an analysis index row and correction rows; report files are left untouched."""
        with closing(connect(self.db_path)) as connection:
            connection.execute("DELETE FROM analyses WHERE analysis_id = ?", (analysis_id,))
            connection.commit()

    def export_rows_csv(self, rows: Iterable[Mapping[str, Any]]) -> str:
        """Return selected history rows as UTF-8 CSV text."""
        rows = list(rows)
        fields = [
            "analysis_id",
            "created_at",
            "input_type",
            "filename",
            "image_sha256",
            "backend",
            "decision",
            "status",
            "final_smiles",
            "inchikey",
            "report_path",
            "is_favorite",
        ]
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        return buffer.getvalue()


def record_report(report: Mapping[str, Any], report_path: str | Path | None = None) -> dict[str, Any]:
    """Convenience wrapper for UI code."""
    return AnalysisRepository().save_analysis(report, report_path=report_path)


def record_reports(reports: Iterable[Mapping[str, Any]], report_path: str | Path | None = None) -> int:
    """Convenience wrapper for indexing many reports."""
    return AnalysisRepository().save_many(reports, report_path=report_path)


def record_result_payload(result: Mapping[str, Any], report_path: str | Path | None = None) -> int:
    """Index all reports inside a batch or document result payload."""
    reports = list(result.get("reports") or [])
    if not reports and result.get("regions"):
        reports = [
            region.get("report")
            for region in result.get("regions") or []
            if isinstance(region, Mapping) and isinstance(region.get("report"), Mapping)
        ]
    return record_reports([report for report in reports if isinstance(report, Mapping)], report_path=report_path)


def analysis_record_from_report(report: Mapping[str, Any], report_path: str | Path | None = None) -> dict[str, Any]:
    input_data = _block(report, "input")
    ocsr = _block(report, "ocsr")
    final = _block(report, "final")
    decision = _block(report, "recognition_decision")
    identity = _block(report, "chemical_identity")
    validation = _block(report, "validation")
    analysis_id = str(report.get("analysis_id") or uuid4().hex)
    created = str(report.get("created_at") or utc_now())
    path = report_path or _report_path_from_report(report)
    return {
        "analysis_id": analysis_id,
        "created_at": created,
        "updated_at": utc_now(),
        "input_type": _text(input_data.get("type")),
        "filename": _text(input_data.get("filename") or Path(str(input_data.get("path") or "")).name),
        "input_path": _text(input_data.get("path")),
        "image_sha256": _text(input_data.get("image_sha256")),
        "backend": _text(ocsr.get("backend")),
        "decision": _text(decision.get("decision") or ocsr.get("decision")),
        "status": _text(report.get("status")),
        "final_smiles": _text(final.get("smiles") or validation.get("canonical_smiles") or ocsr.get("smiles")),
        "inchikey": _text(identity.get("inchikey")),
        "report_path": str(Path(path).expanduser().resolve()) if path else "",
        "is_favorite": 0,
    }


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _report_path_from_report(report: Mapping[str, Any]) -> str | None:
    run = _block(report, "run")
    if run.get("report_path"):
        return str(run["report_path"])
    return None


def _block(value: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    nested = value.get(key)
    return nested if isinstance(nested, Mapping) else {}


def _text(value: Any) -> str:
    return "" if value is None else str(value)


def _row_dict(row: Any) -> dict[str, Any]:
    data = dict(row)
    if "is_favorite" in data:
        data["is_favorite"] = bool(data["is_favorite"])
    return data
