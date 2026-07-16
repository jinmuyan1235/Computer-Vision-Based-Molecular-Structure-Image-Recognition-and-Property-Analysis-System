"""Streamlit smoke tests that execute UI code without a browser."""

from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_streamlit_app_starts_without_exception() -> None:
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(str(app_path), default_timeout=30).run()
    assert not app.exception
    assert app.title[0].value == "分子结构识别与性质分析"
    assert len(app.tabs) == 8


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
    assert "保存为待审核" in source
    assert "不保存" in source
    assert "确认进入训练集" not in source
    assert "多后端候选与一致性" in source


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
    assert "仅检测分子区域（速度快，不执行结构识别）" in source
    assert "检测并识别分子结构（调用 OCSR，耗时较长）" in source
    assert "Update bbox and rerun" not in source
    assert "Delete region" not in source


def test_document_subprocess_json_parser_ignores_trailing_logs() -> None:
    from src.ui.document_page import _extract_json_object

    parsed = _extract_json_object('native log\n{"status": "success", "result_path": "x.json"}\n[09:46] warning')
    assert parsed == {"status": "success", "result_path": "x.json"}
