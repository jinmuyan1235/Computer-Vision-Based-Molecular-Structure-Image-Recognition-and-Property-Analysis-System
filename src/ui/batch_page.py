"""Batch image processing page."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.export.json_exporter import to_json_text
from src.runtime.batch_job_store import BatchJobStore
from src.runtime.job_registry import (
    cancel_batch_job,
    clear_batch_job,
    load_batch_job_result,
    refresh_batch_job,
    request_skip_current,
    start_batch_job,
    start_batch_job_from_uploads,
    start_batch_retry_job,
)
from src.storage.analysis_repository import AnalysisRepository, record_result_payload
from src.ui.labels import BATCH_COLUMN_LABELS, localize_batch_rows
from src.ui.records import render_records
from src.ui.state import (
    current_runtime_key,
    remember_backend_status,
    runtime_config_from_key,
)
from src.ui.styles import page_intro

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_COLUMNS = [
    "filename",
    "status",
    "backend",
    "final_smiles",
    "valid",
    "confidence",
    "inference_time_ms",
    "message",
]

RUNNING_BATCH_STATUSES = {"queued", "running", "cancelling"}


def render_batch_page(backend: str) -> None:
    page_intro("批量处理", "批量任务在后台运行；页面刷新后可以恢复进度和下载结果。")
    store = BatchJobStore()
    folder_path = st.text_input("输入文件夹路径（可选）", value="")
    uploaded_files = st.file_uploader(
        "批量上传图片",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True,
        key="batch_upload",
    )

    active_job = _active_job(store)
    running = bool(active_job and _is_running_batch_status(active_job.get("status")))
    if st.button("开始后台批量任务", type="primary", key="analyze_batch", disabled=running):
        try:
            runtime = runtime_config_from_key(current_runtime_key())
            if uploaded_files:
                uploads = [(item.name, item.getvalue()) for item in uploaded_files]
                active_job = start_batch_job_from_uploads(uploads, backend, runtime, store=store)
            elif folder_path.strip():
                active_job = start_batch_job(folder_path.strip(), backend, runtime, store=store, source="folder")
            else:
                st.warning("请上传至少一张图片或填写输入文件夹路径。")
                active_job = None
            if active_job:
                st.session_state["batch_job_id"] = active_job["job_id"]
                st.session_state.pop("batch_result", None)
                remember_backend_status(backend)
                st.rerun()
        except Exception as exc:
            st.error(f"启动批量任务失败：{exc}")

    active_job = _active_job(store)
    if active_job:
        _index_job(active_job)
        if _is_running_batch_status(active_job.get("status")):
            _render_live_job_status(str(active_job["job_id"]), backend)
        else:
            _render_job_status(active_job, store, backend)
        batch_result = load_batch_job_result(active_job["job_id"], store)
        if batch_result:
            record_result_payload(batch_result, active_job.get("result_path"))
            st.session_state["batch_result"] = batch_result
    else:
        _render_restore_jobs(store)

    if "batch_result" not in st.session_state:
        return
    _render_batch_result(st.session_state["batch_result"])


def _active_job(store: BatchJobStore) -> dict | None:
    job_id = st.session_state.get("batch_job_id")
    if not job_id or not store.exists(str(job_id)):
        return None
    try:
        return refresh_batch_job(str(job_id), store)
    except Exception as exc:
        st.warning(f"恢复批量任务失败：{exc}")
        st.session_state.pop("batch_job_id", None)
        return None


def _render_restore_jobs(store: BatchJobStore) -> None:
    jobs = store.list_jobs(limit=8)
    if not jobs:
        return
    with st.expander("恢复批量任务", expanded=False):
        labels = [_job_label(job) for job in jobs]
        selected = st.selectbox("选择历史任务", list(range(len(jobs))), format_func=lambda index: labels[index], key="restore_batch_job")
        if st.button("恢复所选任务", key="restore_batch_job_button"):
            st.session_state["batch_job_id"] = jobs[int(selected)]["job_id"]
            result = load_batch_job_result(jobs[int(selected)]["job_id"], store)
            if result:
                record_result_payload(result, jobs[int(selected)].get("result_path"))
                st.session_state["batch_result"] = result
            st.rerun()


def _render_job_status(job: dict, store: BatchJobStore, backend: str) -> None:
    status = str(job.get("status") or "unknown")
    total = int(job.get("total") or 0)
    completed = int(job.get("completed") or 0)
    st.subheader("后台任务状态")
    st.progress((completed / total) if total else 0.0)
    metrics = st.columns(2)
    metrics[0].metric("总数", total)
    metrics[1].metric("已完成", completed)
    _render_batch_status_metrics(job)
    st.caption(
        f"状态：{_status_label(status)}；"
        f"当前文件：{job.get('current_file') or '-'}；"
        f"任务 ID：{job.get('job_id')}"
    )
    col_refresh, col_cancel, col_skip, col_retry_failed, col_retry_review, col_clear = st.columns(6)
    if col_refresh.button("刷新状态", key="refresh_batch_job"):
        st.rerun()
    if col_cancel.button("取消任务", key="cancel_batch_job", disabled=not _is_running_batch_status(status)):
        cancel_batch_job(job["job_id"], store)
        st.rerun()
    if col_skip.button(
        "跳过下一张未开始文件",
        key="skip_batch_next_unstarted",
        disabled=status != "running",
        help="不会中断正在推理的图片；请求会在下一张图片开始前生效。",
    ):
        request_skip_current(job["job_id"], store)
        st.rerun()
    result = load_batch_job_result(job["job_id"], store)
    if col_retry_failed.button("重试失败项", key="retry_failed_batch", disabled=not result):
        _start_retry_job(result, backend, "failed", store, job["job_id"])
    if col_retry_review.button("只重试待审核项", key="retry_review_batch", disabled=not result):
        _start_retry_job(result, backend, "review", store, job["job_id"])
    if col_clear.button("清除任务", key="clear_batch_job", disabled=_is_running_batch_status(status)):
        clear_batch_job(job["job_id"], store)
        st.session_state.pop("batch_job_id", None)
        st.session_state.pop("batch_result", None)
        st.rerun()


@st.fragment(run_every="3s")
def _render_live_job_status(job_id: str, backend: str) -> None:
    store = BatchJobStore()
    try:
        job = refresh_batch_job(job_id, store)
    except Exception as exc:
        st.warning(f"恢复批量任务失败：{exc}")
        if st.session_state.get("batch_job_id") == job_id:
            st.session_state.pop("batch_job_id", None)
        st.rerun()
    _index_job(job)
    if _is_running_batch_status(job.get("status")):
        _render_job_status(job, store, backend)
        st.caption("任务运行中，状态区域每 3 秒自动刷新。")
        return
    result = load_batch_job_result(job_id, store)
    if result:
        record_result_payload(result, job.get("result_path"))
        st.session_state["batch_result"] = result
    st.rerun()


def _is_running_batch_status(status: object) -> bool:
    return str(status or "") in RUNNING_BATCH_STATUSES


def _start_retry_job(result: dict, backend: str, mode: str, store: BatchJobStore, parent_job_id: str) -> None:
    try:
        retry = start_batch_retry_job(
            result,
            backend,
            mode,
            runtime_config_from_key(current_runtime_key()),
            store=store,
            parent_job_id=parent_job_id,
        )
        st.session_state["batch_job_id"] = retry["job_id"]
        st.session_state.pop("batch_result", None)
        st.rerun()
    except Exception as exc:
        st.warning(str(exc))


def _index_job(job: dict) -> None:
    try:
        AnalysisRepository().save_job(job, job_type="batch")
    except Exception:
        return


def _render_batch_result(batch_result: dict) -> None:
    summary = batch_result["summary"]
    metrics = st.columns(4)
    metrics[0].metric("总图片", summary["total"])
    metrics[1].metric("已处理", summary.get("completed", summary["total"]))
    metrics[2].metric("有效 SMILES", summary["valid_smiles"])
    metrics[3].metric("成功率", f"{summary['success_rate']:.1%}")
    _render_batch_status_metrics(summary)

    rows = batch_result["rows"]
    default_rows = [{key: row.get(key) for key in DEFAULT_COLUMNS} for row in rows]
    render_records(
        localize_batch_rows(default_rows),
        title_keys=("文件名",),
        summary_keys=("状态", "识别后端", "最终 SMILES", "是否有效", "推理耗时(ms)"),
        max_records=50,
    )
    if st.checkbox("查看完整字段", value=False, key="show_batch_full_fields"):
        render_records(
            localize_batch_rows(rows),
            title_keys=("文件名",),
            summary_keys=("状态", "识别后端", "最终 SMILES", "失败原因"),
            max_records=100,
        )

    chart = batch_result["exports"]["summary_chart"]
    if Path(chart).is_file():
        st.image(chart, caption="批量结果统计", width=640)

    with st.expander("结果下载", expanded=True):
        csv_bytes = Path(batch_result["exports"]["csv"]).read_bytes()
        st.download_button("下载批量结果表 CSV", csv_bytes, "batch_results.csv", "text/csv", key="batch_csv")
        st.download_button(
            "下载完整 JSON",
            to_json_text({"summary": summary, "results": batch_result["reports"]}),
            "batch_results.json",
            "application/json",
            key="batch_json",
        )
        _download_export_if_present(batch_result["exports"], "merged_sdf", "下载合并 SDF", "chemical/x-mdl-sdfile", "batch_merged_sdf")
        _download_export_if_present(batch_result["exports"], "successful_zip", "下载成功结果 ZIP", "application/zip", "batch_success_zip")
        _download_export_if_present(batch_result["exports"], "failed_csv", "下载失败清单 CSV", "text/csv", "batch_failed_csv")
        _download_export_if_present(batch_result["exports"], "review_csv", "下载待审核清单 CSV", "text/csv", "batch_review_csv")


def _download_export_if_present(exports: dict, field: str, label: str, mime: str, key: str) -> None:
    path_value = exports.get(field)
    if not path_value:
        return
    path = Path(path_value)
    if not path.is_file():
        return
    st.download_button(label, path.read_bytes(), path.name, mime, key=key)


def _render_batch_status_metrics(summary: dict) -> None:
    counts = _batch_status_counts(summary)
    metrics = st.columns(7)
    metrics[0].metric("自动接受", counts["accepted"])
    metrics[1].metric("警告待确认", counts["accepted_with_warning"])
    metrics[2].metric("明确需要审核", counts["review_needed"])
    metrics[3].metric("拒绝", counts["rejected"])
    metrics[4].metric("失败", counts["failed"])
    metrics[5].metric("跳过", counts["skipped"])
    metrics[6].metric("人工审核总数", counts["manual_review_total"])


def _batch_status_counts(summary: dict) -> dict[str, int]:
    accepted = int(summary.get("accepted") or 0)
    accepted_with_warning = int(summary.get("accepted_with_warning") or 0)
    review_needed = int(summary.get("review_needed") or 0)
    rejected = int(summary.get("rejected") or 0)
    failed = int(summary.get("failed") or 0)
    skipped = int(summary.get("skipped") or 0)
    schema_version = int(summary.get("summary_schema_version") or 0)
    if "manual_review_total" in summary or schema_version >= 2:
        manual_review_total = int(summary.get("manual_review_total") or (accepted_with_warning + review_needed))
    else:
        manual_review_total = review_needed
        review_needed = max(0, review_needed - accepted_with_warning)
    return {
        "accepted": accepted,
        "accepted_with_warning": accepted_with_warning,
        "review_needed": review_needed,
        "manual_review_total": manual_review_total,
        "rejected": rejected,
        "failed": failed,
        "skipped": skipped,
    }


def default_batch_columns_chinese() -> list[str]:
    return [BATCH_COLUMN_LABELS[column] for column in DEFAULT_COLUMNS]


def _job_label(job: dict) -> str:
    return (
        f"{_status_label(str(job.get('status')))} | "
        f"{job.get('completed', 0)}/{job.get('total', 0)} | "
        f"{job.get('backend')} | {job.get('updated_at') or job.get('created_at')}"
    )


def _status_label(status: str) -> str:
    return {
        "queued": "排队中",
        "running": "运行中",
        "cancelling": "取消中",
        "cancelled": "已取消",
        "completed": "已完成",
        "failed": "失败",
    }.get(status, status)
