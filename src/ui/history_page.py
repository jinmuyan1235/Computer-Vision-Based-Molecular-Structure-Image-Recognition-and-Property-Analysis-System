"""Searchable local analysis history page."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
import zipfile

import streamlit as st

from config import OUTPUT_DIR
from src.analysis.molecule_report import MoleculeReportGenerator
from src.export.json_exporter import to_json_text
from src.export.structure_exporter import export_structure_files
from src.runtime.run_store import create_image_run_from_file, save_run_report
from src.storage.analysis_repository import AnalysisRepository, record_report
from src.ui.report_view import show_report
from src.ui.state import current_runtime_key, runtime_config_from_key
from src.ui.styles import page_intro


STATUS_OPTIONS = {
    "全部": "all",
    "成功": "success",
    "待审核": "review_needed",
    "拒绝": "rejected",
    "失败": "failed",
}


def render_history_page(backend: str, show_preprocessing: bool, export_pdf: bool) -> None:
    """Render local analysis history search and actions."""
    page_intro("分析历史", "按文件名、SMILES、InChIKey 或分析 ID 搜索本地历史记录。")
    repository = AnalysisRepository()
    controls = st.columns([0.46, 0.24, 0.15, 0.15])
    query = controls[0].text_input("搜索", value="", placeholder="文件名 / SMILES / InChIKey / analysis_id")
    status_label = controls[1].selectbox("状态", list(STATUS_OPTIONS), index=0)
    favorites_only = controls[2].checkbox("只看收藏", value=False)
    limit = controls[3].number_input("数量", min_value=10, max_value=500, value=100, step=10)
    rows = repository.list_analyses(
        query=query,
        status_filter=STATUS_OPTIONS[status_label],
        favorites_only=favorites_only,
        limit=int(limit),
    )
    st.caption(f"匹配记录：{len(rows)}")
    _render_batch_exports(repository, rows)
    if not rows:
        st.info("暂无匹配记录。新的图片、SMILES、文档区域和批量结果会自动进入这里。")
        return

    for row in rows:
        _render_history_row(repository, row, backend)

    report = st.session_state.get("history_report")
    if report:
        st.divider()
        st.subheader("历史报告预览")
        show_report(report, show_preprocessing, export_pdf, f"history_{str(report.get('analysis_id', 'report'))[:8]}")


def _render_batch_exports(repository: AnalysisRepository, rows: list[dict]) -> None:
    left, right = st.columns(2)
    left.download_button(
        "批量导出历史 CSV",
        repository.export_rows_csv(rows).encode("utf-8-sig"),
        "analysis_history.csv",
        "text/csv",
        key="history_export_csv",
        disabled=not rows,
    )
    zip_bytes = _reports_zip(repository, rows)
    right.download_button(
        "批量导出报告 ZIP",
        zip_bytes,
        "analysis_reports.zip",
        "application/zip",
        key="history_export_reports_zip",
        disabled=not zip_bytes,
    )


def _render_history_row(repository: AnalysisRepository, row: dict, backend: str) -> None:
    analysis_id = str(row["analysis_id"])
    title = f"{'★ ' if row.get('is_favorite') else ''}{row.get('filename') or analysis_id} | {row.get('status') or '-'} | {row.get('decision') or '-'}"
    with st.expander(title, expanded=False):
        fields = {
            "analysis_id": analysis_id,
            "created_at": row.get("created_at"),
            "backend": row.get("backend"),
            "final_smiles": row.get("final_smiles"),
            "inchikey": row.get("inchikey"),
            "report_path": row.get("report_path"),
        }
        st.json(fields)
        actions = st.columns(6)
        if actions[0].button("打开旧报告", key=f"history_open_{analysis_id}"):
            report = repository.load_report(analysis_id)
            if report:
                st.session_state["history_report"] = report
                st.rerun()
            else:
                st.warning("报告文件不存在或无法定位该 analysis_id。")
        if actions[1].button("重新识别", key=f"history_rerun_{analysis_id}", disabled=not Path(str(row.get("input_path") or "")).is_file()):
            _rerun_analysis(repository, row, backend)
        if actions[2].button("重新导出", key=f"history_reexport_{analysis_id}"):
            _reexport_analysis(repository, analysis_id)
        favorite_label = "取消收藏" if row.get("is_favorite") else "标记收藏"
        if actions[3].button(favorite_label, key=f"history_fav_{analysis_id}"):
            repository.set_favorite(analysis_id, not bool(row.get("is_favorite")))
            st.rerun()
        if actions[4].button("删除记录", key=f"history_delete_{analysis_id}"):
            repository.delete_analysis(analysis_id)
            if (st.session_state.get("history_report") or {}).get("analysis_id") == analysis_id:
                st.session_state.pop("history_report", None)
            st.rerun()
        if actions[5].button("复制 ID", key=f"history_copy_{analysis_id}"):
            st.code(analysis_id, language=None)


def _rerun_analysis(repository: AnalysisRepository, row: dict, backend: str) -> None:
    input_path = Path(str(row.get("input_path") or "")).expanduser().resolve()
    try:
        run = create_image_run_from_file(input_path, original_filename=row.get("filename") or input_path.name)
        generator = MoleculeReportGenerator(backend, run.run_dir, runtime_config=runtime_config_from_key(current_runtime_key()))
        report = generator.generate(image_path=run.input_path, analysis_id=run.analysis_id)
        report_path = save_run_report(report, run)
        repository.save_analysis(report, report_path)
        st.session_state["history_report"] = report
        st.success("已完成重新识别。")
        st.rerun()
    except Exception as exc:
        st.warning(f"重新识别失败：{exc}")


def _reexport_analysis(repository: AnalysisRepository, analysis_id: str) -> None:
    report = repository.load_report(analysis_id)
    if not report:
        st.warning("报告文件不存在，无法重新导出。")
        return
    export_dir = OUTPUT_DIR / "history_exports" / analysis_id
    try:
        exports = export_structure_files(report, export_dir, prefix=f"history_{analysis_id[:8]}")
        record_report(report, repository.get_analysis(analysis_id).get("report_path") if repository.get_analysis(analysis_id) else None)
        st.success(f"已重新导出：{exports['zip']}")
    except Exception as exc:
        st.warning(f"重新导出失败：{exc}")


def _reports_zip(repository: AnalysisRepository, rows: list[dict]) -> bytes:
    buffer = BytesIO()
    count = 0
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for row in rows:
            analysis_id = str(row.get("analysis_id") or "")
            report = repository.load_report(analysis_id)
            if not report:
                continue
            archive.writestr(f"{analysis_id}.json", to_json_text(report))
            count += 1
    return buffer.getvalue() if count else b""
