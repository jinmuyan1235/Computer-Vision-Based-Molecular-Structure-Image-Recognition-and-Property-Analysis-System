"""Cached Streamlit state and service factories."""

from __future__ import annotations

from typing import Any

import streamlit as st

from config import OUTPUT_DIR
from src.analysis.batch_analyzer import BatchAnalyzer
from src.analysis.molecule_report import MoleculeReportGenerator
from src.documents.processor import DocumentOCSRProcessor


@st.cache_resource(show_spinner=False)
def get_report_generator(backend: str) -> MoleculeReportGenerator:
    return MoleculeReportGenerator(backend, OUTPUT_DIR)


@st.cache_resource(show_spinner=False)
def get_batch_analyzer(backend: str) -> BatchAnalyzer:
    return BatchAnalyzer(backend, OUTPUT_DIR)


@st.cache_resource(show_spinner=False)
def get_document_processor(backend: str) -> DocumentOCSRProcessor:
    return DocumentOCSRProcessor(backend=backend)


@st.cache_data(show_spinner=False, ttl=10)
def get_backend_statuses() -> dict[str, dict[str, Any]]:
    statuses: dict[str, dict[str, Any]] = {}
    for backend in ("demo", "molscribe", "decimer", "ensemble"):
        try:
            statuses[backend] = get_report_generator(backend).recognizer.status()
        except Exception as exc:
            statuses[backend] = {
                "backend": backend,
                "available": False,
                "message": str(exc),
            }
    return statuses


def remember_backend_status(backend: str) -> None:
    try:
        st.session_state["backend_last_status"] = get_report_generator(backend).recognizer.status()
    except Exception:
        return


def merged_backend_status(backend: str) -> dict[str, Any]:
    status = dict(get_backend_statuses().get(backend, {"backend": backend, "available": False}))
    latest = st.session_state.get("backend_last_status") or {}
    if latest.get("backend") == backend:
        status.update({key: value for key, value in latest.items() if value is not None})
    return status
