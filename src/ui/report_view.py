"""Shared report and correction rendering."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from config import OUTPUT_DIR
from src.analysis.correction import (
    apply_smiles_correction,
    restore_original_prediction,
    save_correction_feedback,
    structure_similarity,
)
from src.export.json_exporter import to_json_text
from src.export.pdf_exporter import save_pdf
from src.ui.image_viewer import show_preprocess_thumbnail, show_structure
from src.ui.labels import backend_label, status_label
from src.ui.streamlit_compat import dataframe_stretch


def show_ensemble_details(ocsr: dict[str, Any]) -> None:
    candidates = ocsr.get("candidates") or []
    consensus = ocsr.get("consensus") or {}
    if not candidates and not consensus:
        return
    with st.expander("多后端候选与一致性", expanded=False):
        status = consensus.get("status") or "unknown"
        reason = consensus.get("reason") or ""
        if status == "agreement":
            st.success(f"候选一致：{reason}")
        elif status == "disagreement":
            st.warning(consensus.get("warning") or reason)
        else:
            st.info(reason or "暂无一致性结论。")
        if candidates:
            rows = []
            for candidate in candidates:
                rows.append({
                    "后端": backend_label(candidate.get("backend"), short=True),
                    "状态": status_label(candidate.get("status")),
                    "原始 SMILES": candidate.get("raw_smiles"),
                    "Canonical SMILES": candidate.get("canonical_smiles"),
                    "有效": status_label(candidate.get("valid")),
                    "耗时(ms)": candidate.get("inference_time_ms"),
                    "错误": candidate.get("error"),
                })
            dataframe_stretch(pd.DataFrame(rows), hide_index=True)


def show_chemical_identity(report: dict[str, Any]) -> None:
    identity = report.get("chemical_identity") or {}
    standardization = report.get("standardization") or {}
    warnings = report.get("structure_warnings") or []
    if not identity:
        return
    with st.expander("化学标准化与身份信息", expanded=False):
        metrics = st.columns(3)
        metrics[0].metric("片段数", identity.get("fragment_count") if identity.get("fragment_count") is not None else "-")
        metrics[1].metric("形式电荷", identity.get("formal_charge") if identity.get("formal_charge") is not None else "-")
        metrics[2].metric("结构提示", len(warnings))
        st.write(f"**Standardized SMILES：** `{identity.get('standardized_smiles') or '-'}`")
        st.write(f"**InChIKey：** `{identity.get('inchikey') or '当前不可用'}`")
        st.caption(
            f"标准化 profile：{standardization.get('profile') or '-'}；"
            f"是否改变：{'是' if standardization.get('changed') else '否'}。"
        )
        if warnings:
            dataframe_stretch(pd.DataFrame(warnings), hide_index=True)


def show_report(report: dict[str, Any], show_preprocessing: bool, export_pdf: bool, key_prefix: str) -> None:
    """Render a molecule analysis report in Streamlit."""
    if report.get("status") != "success":
        st.error(report.get("message", "分析失败。"))
        ocsr = report.get("ocsr") or {}
        if ocsr:
            st.caption(f"后端：{backend_label(ocsr.get('backend'), short=True)}；状态：{status_label(ocsr.get('status'))}")
            show_ensemble_details(ocsr)
        return

    ocsr = report.get("ocsr") or {}
    correction = report.get("correction") or {}
    final = report.get("final") or {}
    validation = report.get("validation") or {}
    st.success(report.get("message", "分析完成。"))

    left, right = st.columns([1.1, 0.9])
    with left:
        st.subheader("核心识别结果")
        st.code(final.get("smiles") or ocsr.get("smiles") or "", language=None)
        st.write(f"**Canonical SMILES：** `{final.get('canonical_smiles') or validation.get('canonical_smiles') or '-'}`")
        if final.get("standardized_smiles") or validation.get("standardized_smiles"):
            st.write(f"**Standardized SMILES：** `{final.get('standardized_smiles') or validation.get('standardized_smiles')}`")
        confidence = ocsr.get("confidence")
        st.write(f"**识别后端：** {backend_label(ocsr.get('backend'), short=True)}")
        st.write(f"**置信度：** {confidence if confidence is not None else '模型未提供'}")
        st.write(f"**当前结果来源：** {final.get('source') or '未知'}")
        st.write(f"**RDKit 校验：** {'有效' if validation.get('valid') else '无效'}")
        if correction.get("applied"):
            st.info(f"已应用人工修正：`{correction.get('corrected_canonical_smiles')}`")
    with right:
        st.subheader("分子结构重绘")
        show_structure((report.get("images") or {}).get("redrawn_molecule"), "最终分析结构")

    descriptors = report.get("descriptors") or {}
    if descriptors:
        st.subheader("分子性质")
        labels = {
            "formula": "分子式",
            "molecular_weight": "分子量",
            "logp": "LogP",
            "tpsa": "TPSA",
            "hbd": "HBD",
            "hba": "HBA",
            "rotatable_bonds": "可旋转键",
            "heavy_atom_count": "重原子数",
        }
        compact = st.columns(min(4, max(1, len(descriptors))))
        for index, (key, value) in enumerate(descriptors.items()):
            compact[index % len(compact)].metric(labels.get(key, key), str(value))

    lipinski = report.get("lipinski") or {}
    if lipinski:
        if lipinski.get("passed"):
            st.info(lipinski.get("summary", "符合规则。"))
        else:
            st.warning(lipinski.get("summary", "存在规则提示。"))

    show_ensemble_details(ocsr)
    show_chemical_identity(report)

    if show_preprocessing and report.get("input", {}).get("type") == "image":
        with st.expander("OpenCV 预处理过程", expanded=False):
            stage_paths = (report.get("images") or {}).get("preprocessing") or {}
            titles = {
                "original": "原图",
                "gray": "灰度",
                "denoised": "去噪",
                "binary": "二值化",
                "cropped": "裁剪",
                "deskewed": "旋转校正",
                "normalized": "归一化",
            }
            columns = st.columns(3)
            for index, name in enumerate(["original", "gray", "denoised", "binary", "cropped", "deskewed", "normalized"]):
                if name in stage_paths:
                    with columns[index % 3]:
                        show_preprocess_thumbnail(stage_paths[name], titles[name])

    with st.expander("结果导出", expanded=True):
        st.download_button(
            "下载 JSON 报告",
            to_json_text(report),
            file_name=f"{key_prefix}_report.json",
            mime="application/json",
            key=f"json_{key_prefix}",
        )
        if export_pdf:
            pdf_result = save_pdf(report, OUTPUT_DIR / f"{key_prefix}_report.pdf")
            if pdf_result["success"]:
                st.download_button(
                    "下载 PDF 报告",
                    Path(pdf_result["path"]).read_bytes(),
                    file_name=f"{key_prefix}_report.pdf",
                    mime="application/pdf",
                    key=f"pdf_{key_prefix}",
                )
            else:
                st.caption(pdf_result["message"])

    with st.expander("技术诊断", expanded=False):
        st.json({
            "backend": ocsr.get("backend"),
            "device": ocsr.get("device"),
            "model_name": ocsr.get("model_name"),
            "model_version": ocsr.get("model_version"),
            "package_version": ocsr.get("package_version"),
            "inference_time_ms": ocsr.get("inference_time_ms"),
        })


def show_correction_panel(report: dict[str, Any]) -> dict[str, Any]:
    """Render human correction controls for image reports."""
    if (report.get("input") or {}).get("type") != "image":
        return report
    analysis_id = report.get("analysis_id") or "image"
    ocsr = report.get("ocsr") or {}
    correction = report.get("correction") or {}
    final = report.get("final") or {}
    predicted = ocsr.get("predicted_smiles") or ocsr.get("smiles") or ""
    default_smiles = correction.get("corrected_smiles") or final.get("smiles") or predicted

    st.subheader("人工纠错")
    st.caption(
        f"纠错状态：{'已人工修正' if correction.get('applied') else '未人工修正'}；"
        f"当前结果来源：{final.get('source') or '暂无有效结果'}"
    )
    st.text_input("模型原始预测", value=predicted, disabled=True, key=f"predicted_{analysis_id}")
    corrected_input = st.text_input(
        "修正 SMILES",
        value=default_smiles or "",
        key=f"corrected_smiles_{analysis_id}",
        placeholder="OCSR 失败时也可以手动输入 SMILES",
    )
    apply_col, restore_col, feedback_col = st.columns(3)
    current_report = report
    if apply_col.button("校验并应用修正", type="primary", key=f"apply_correction_{analysis_id}"):
        candidate = apply_smiles_correction(report, corrected_input, OUTPUT_DIR)
        error = (candidate.get("correction") or {}).get("last_error")
        if error:
            st.error(error)
        else:
            current_report = candidate
            st.session_state["image_report"] = current_report
            st.success("人工修正已应用，性质和结构图已重新生成。")
    if restore_col.button("恢复模型原始结果", key=f"restore_prediction_{analysis_id}"):
        candidate = restore_original_prediction(report, OUTPUT_DIR)
        error = (candidate.get("correction") or {}).get("last_error")
        if error:
            st.warning(error)
        else:
            current_report = candidate
            st.session_state["image_report"] = current_report
            st.success("已恢复为模型原始预测。")

    notes = st.text_area("反馈备注（可选）", value="", key=f"feedback_notes_{analysis_id}", height=70)
    if feedback_col.button(
        "保存为纠错反馈样本",
        key=f"save_feedback_{analysis_id}",
        disabled=not bool((current_report.get("correction") or {}).get("applied")),
    ):
        feedback_path = save_correction_feedback(current_report, OUTPUT_DIR, notes)
        st.session_state[f"feedback_path_{analysis_id}"] = feedback_path
        st.success(f"反馈样本已保存：{feedback_path}")

    images = current_report.get("images") or {}
    predicted_image = images.get("predicted_molecule")
    corrected_image = images.get("corrected_molecule")
    if predicted_image or corrected_image:
        st.subheader("结构对比")
        columns = st.columns(2) if predicted_image and corrected_image else st.columns(1)
        if predicted_image:
            with columns[0]:
                show_structure(predicted_image, "模型预测结构")
        if corrected_image:
            target = columns[1] if predicted_image and corrected_image else columns[0]
            with target:
                show_structure(corrected_image, "人工修正结构")
        if predicted and (current_report.get("correction") or {}).get("corrected_smiles"):
            similarity = structure_similarity(predicted, (current_report.get("correction") or {}).get("corrected_smiles"))
            if similarity is not None:
                st.caption(f"Morgan Tanimoto 相似度：{similarity:.3f}。该值只比较两个分子结果，不代表与原图一致。")
    return current_report
