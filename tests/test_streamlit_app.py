"""Streamlit smoke tests that execute UI code without a browser."""

import json
from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_streamlit_app_starts_without_exception() -> None:
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(str(app_path), default_timeout=30).run()
    assert not app.exception


def test_dataset_batch_classification_workspace_renders() -> None:
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(str(app_path), default_timeout=30).run()

    app.segmented_control[0].set_value("Batch classify").run(timeout=30)

    assert not app.exception
    assert any(item.value == "Batch visual classification" for item in app.subheader)
    checkbox_labels = [item.label for item in app.checkbox]
    info_messages = [item.value for item in app.info]
    assert "Select image" in checkbox_labels or any(
        message.startswith("No unreviewed samples in ") for message in info_messages
    )
    assert app.title[0].value == "分子结构识别与性质分析"
    assert len(app.tabs) == 0
    source = app_path.read_text(encoding="utf-8")
    review_source = (app_path.parent / "src" / "ui" / "dataset_review_page.py").read_text(encoding="utf-8")
    assert 'st.checkbox("Select image"' in review_source
    assert '"Machine rejected"' in review_source
    assert "submit_recheck_batch" in review_source
    assert "PAGE_LABELS" in source
    assert "active_page" in source
    assert "st.tabs" not in source
    assert "st.navigation" not in source


def test_smiles_result_page_renders_without_exception(tmp_path: Path) -> None:
    """Execute result widgets directly so image/dataframe API errors cannot hide."""
    from app import show_report
    from src.analysis.molecule_report import MoleculeReportGenerator

    report = MoleculeReportGenerator("demo", tmp_path).generate(smiles="CCO")
    assert report["status"] == "success"
    show_report(report, show_preprocessing=False, export_pdf=False, key_prefix="test_smiles")


def test_app_avoids_newer_only_stretch_width_api() -> None:
    """Keep the UI compatible with the project's existing Streamlit 1.41 environment."""
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    source = app_path.read_text(encoding="utf-8")
    assert 'width="stretch"' not in source


def test_streamlit_correction_widgets_are_present() -> None:
    source = (Path(__file__).resolve().parents[1] / "src" / "ui" / "report_view.py").read_text(encoding="utf-8")
    assert "校验并应用修正" in source
    assert "恢复模型原始结果" in source
    assert "确认与修正" in source
    assert "纠错反馈与数据回流" not in source
    assert "保存为待审核" not in source
    assert "确认进入训练集" not in source
    assert "多后端候选与一致性" in source
    assert "确认结构正确" in source
    assert "修改 SMILES" in source
    assert "无法确认" in source
    assert "撤销确认" in source
    assert "重新修改结构" in source


def test_candidate_and_strategy_updates_rerun_immediately() -> None:
    source = (Path(__file__).resolve().parents[1] / "src" / "ui" / "report_view.py").read_text(encoding="utf-8")
    assert "def _apply_report_update_and_rerun" in source
    assert "st.rerun()" in source
    assert "user_selected_{backend}_candidate" in source
    assert "strategy_selection" in source


def test_batch_skip_button_matches_worker_behavior_and_autorefresh() -> None:
    source = (Path(__file__).resolve().parents[1] / "src" / "ui" / "batch_page.py").read_text(encoding="utf-8")
    assert "跳过下一张未开始文件" in source
    assert "跳过当前文件" not in source
    assert "time.sleep" not in source
    assert '@st.fragment(run_every="3s")' in source
    assert "def _render_live_job_status" in source
    assert "RUNNING_BATCH_STATUSES" in source
    fragment_body = source.split("def _render_live_job_status", 1)[1].split("def _is_running_batch_status", 1)[0]
    assert "start_batch_job" not in fragment_body
    assert "uploaded_files" not in fragment_body
    assert "明确需要审核" in source
    assert "人工审核总数" in source
    assert "def _batch_status_counts" in source


def test_history_delete_actions_distinguish_index_and_files() -> None:
    source = (Path(__file__).resolve().parents[1] / "src" / "ui" / "history_page.py").read_text(encoding="utf-8")
    assert "从历史中移除" in source
    assert "删除记录及本地文件" in source
    assert "确认删除本地文件" in source
    assert "重试删除残留文件" in source
    assert "报告文件和运行目录已保留" in source
    assert "artifact_status" in source
    assert "disabled=not report_available" in source
    assert "报告文件已过期" in source


def test_review_queue_supports_return_revision_loop() -> None:
    source = (Path(__file__).resolve().parents[1] / "src" / "ui" / "review_queue_page.py").read_text(encoding="utf-8")
    assert "审核人" in source
    assert "打开原报告" in source
    assert "修订 SMILES" in source
    assert "重新提交审核" in source
    assert "disabled=reviewer_missing" in source
    assert "revised_by" in source
    assert "revise_and_resubmit" in source


def test_image_editor_supports_visual_two_point_crop() -> None:
    source = (Path(__file__).resolve().parents[1] / "src" / "ui" / "image_editor.py").read_text(encoding="utf-8")
    assert "components.html" in source
    assert "crop_editor_key" in source
    assert "点击两个角点" in source
    assert "crop_bbox_from_points" in source


def test_document_page_uses_chinese_mode_labels() -> None:
    source = (Path(__file__).resolve().parents[1] / "src" / "ui" / "document_page.py").read_text(encoding="utf-8")
    assert "全文检测与审核识别" in source
    assert "开始{DOCUMENT_WORKFLOW_LABEL}" in source
    assert "仅检测分子区域（速度快，不执行结构识别）" not in source
    assert "检测并识别已确认分子区域（新检测结果需先人工确认）" not in source
    assert 'st.radio("处理模式"' not in source
    assert "论文页码（全文共" in source
    assert "所有页面均已保留在当前任务中" in source
    assert "Update bbox and rerun" not in source
    assert "Delete region" not in source
    assert "确认并识别" not in source
    assert "仅保存框选" in source
    assert "保存并识别" in source
    assert "空白处拖动画新框" in source
    assert "框选已自动保存" in source
    assert "裁剪预览" in source
    assert '@st.fragment(run_every="2s")' in source
    assert "document_region_job" in source
    assert "步骤 1/2：已保存并锁定" in source
    assert "步骤 2/2：正在加载 OCSR 并识别结构" in source
    assert "识别本页已确认区域" in source
    assert "识别全文全部已确认区域" in source
    assert "未确认框不会进入 OCSR" in source
    render_page_body = source.split("def render_document_page", 1)[1].split("def _run_demo_document", 1)[0]
    inspector_body = source.split("def _render_region_inspector", 1)[1].split("def _region_candidate_smiles", 1)[0]
    assert "_render_region_ocsr_job_status()" not in render_page_body
    assert "_render_region_ocsr_job_status(selected_id)" in inspector_body
    assert "合并两个或多个区域" in source
    assert "拆分当前区域" in source
    assert "下载检测训练标注" in source


def test_document_subprocess_json_parser_ignores_trailing_logs() -> None:
    from src.ui.document_page import _extract_document_progress, _extract_json_object

    parsed = _extract_json_object('native log\n{"status": "success", "result_path": "x.json"}\n[09:46] warning')
    assert parsed == {"status": "success", "result_path": "x.json"}
    progress = _extract_document_progress(
        'native log\nDOCUMENT_PROGRESS_JSON={"stage":"detecting","current":12,"total":30,"detail":"第12页"}\n'
    )
    assert progress == {"stage": "detecting", "current": 12, "total": 30, "detail": "第12页"}

    mixed = (
        'DOCUMENT_PROGRESS_JSON={"stage":"detecting","current":30,"total":30}\n'
        '{"status":"success","result_path":"final.json","summary":{"page_count":30}}\n'
    )
    assert _extract_json_object(mixed)["result_path"] == "final.json"
    marked = (
        'DOCUMENT_PROGRESS_JSON={"stage":"detecting","current":1,"total":2}\n'
        'DOCUMENT_RESULT_JSON={"status":"success","result_path":"marked.json"}\n'
    )
    assert _extract_json_object(marked)["result_path"] == "marked.json"


def test_document_result_recovery_uses_completed_worker_log(tmp_path: Path) -> None:
    from src.ui.document_page import _latest_recoverable_document_result

    result_path = tmp_path / "document_result.json"
    result_path.write_text('{"document_id":"doc"}', encoding="utf-8")
    job_dir = tmp_path / "ui_jobs" / "job-1"
    job_dir.mkdir(parents=True)
    (job_dir / "stdout.log").write_text(
        'DOCUMENT_PROGRESS_JSON={"stage":"detecting","current":15,"total":15}\n'
        + 'DOCUMENT_RESULT_JSON={"status":"success","result_path":'
        + json.dumps(str(result_path))
        + ',"summary":{"page_count":15,"region_count":377}}\n',
        encoding="utf-8",
    )

    recovered = _latest_recoverable_document_result(tmp_path / "ui_jobs")

    assert recovered is not None
    payload, recovered_path = recovered
    assert payload["summary"] == {"page_count": 15, "region_count": 377}
    assert recovered_path == result_path
