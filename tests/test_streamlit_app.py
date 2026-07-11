"""Streamlit smoke test that executes the app script without a browser."""

from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_streamlit_app_starts_without_exception() -> None:
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(str(app_path), default_timeout=30).run()
    assert not app.exception
    assert app.title[0].value == "基于计算机视觉的分子结构图像识别与性质分析系统"
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


def test_streamlit_molscribe_unavailable_selection_does_not_crash() -> None:
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(str(app_path), default_timeout=30).run()
    app.selectbox[0].set_value("molscribe").run()
    assert not app.exception


def test_streamlit_decimer_unavailable_selection_does_not_crash() -> None:
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(str(app_path), default_timeout=30).run()
    app.selectbox[0].set_value("decimer").run()
    assert not app.exception


def test_streamlit_ensemble_selection_does_not_crash() -> None:
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(str(app_path), default_timeout=30).run()
    app.selectbox[0].set_value("ensemble").run()
    assert not app.exception


def test_streamlit_correction_widgets_are_present() -> None:
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    source = app_path.read_text(encoding="utf-8")
    assert "校验并应用修正" in source
    assert "恢复模型原始结果" in source
    assert "保存为纠错反馈样本" in source
    assert "多后端候选与共识" in source
    assert "PDF/多分子文档" in source
    assert "Update bbox and rerun" in source
