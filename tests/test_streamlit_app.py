"""Streamlit smoke tests that execute UI code without a browser."""

from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_streamlit_app_starts_without_exception() -> None:
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(str(app_path), default_timeout=30).run()
    assert not app.exception
    assert app.title[0].value == "分子结构识别与性质分析"
    assert len(app.tabs) == 5


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
    assert "仅保存纠错" in source
    assert "确认进入训练集" in source
    assert "多后端候选与一致性" in source


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
