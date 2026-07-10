"""Streamlit smoke test that executes the app script without a browser."""

from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_streamlit_app_starts_without_exception() -> None:
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(str(app_path), default_timeout=30).run()
    assert not app.exception
    assert app.title[0].value == "基于计算机视觉的分子结构图像识别与性质分析系统"
    assert len(app.tabs) == 4
