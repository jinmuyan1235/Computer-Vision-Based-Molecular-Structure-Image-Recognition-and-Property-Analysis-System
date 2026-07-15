"""Shared report and correction rendering."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st

from config import DATA_DIR, OUTPUT_DIR
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
from src.ui.records import render_records


def show_ensemble_details(ocsr: dict[str, Any]) -> None:
    candidates = ocsr.get("candidates") or []
    consensus = ocsr.get("consensus") or {}
    if not candidates and not consensus:
        return
    with st.expander("多后端候选与一致性", expanded=False):
        status = consensus.get("status") or "unknown"
        decision = consensus.get("decision") or "unknown"
        reason = consensus.get("reason") or ""
        if decision == "accepted" and status == "agreement":
            st.success(f"自动接受：{reason}")
        elif decision == "accepted":
            st.warning(consensus.get("warning") or reason)
        elif decision == "review_needed":
            st.warning(consensus.get("warning") or reason or "需要人工确认。")
        elif decision == "rejected":
            st.error(reason or "无可靠候选。")
        else:
            st.info(reason or "暂无一致性结论。")
        st.caption(f"ensemble 决策：{decision}；候选状态：{status}")
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
            render_records(rows, title_keys=("后端",))


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
            normalized_warnings = [item if isinstance(item, dict) else {"提示": item} for item in warnings]
            render_records(normalized_warnings, title_keys=("type", "提示"))


def show_report(report: dict[str, Any], show_preprocessing: bool, export_pdf: bool, key_prefix: str) -> None:
    """Render a molecule analysis report in Streamlit."""
    if report.get("status") != "success":
        ocsr = report.get("ocsr") or {}
        consensus = ocsr.get("consensus") or {}
        if consensus.get("decision") == "review_needed":
            st.warning(report.get("message", "多个后端结果不一致，需要人工确认。"))
        else:
            st.error(report.get("message", "分析失败。"))
        if ocsr:
            st.caption(f"后端：{backend_label(ocsr.get('backend'), short=True)}；状态：{status_label(ocsr.get('status'))}")
            if ocsr.get("backend") == "demo":
                st.warning("这是演示后端，不会识别任意图片；请使用真实后端或手动输入 SMILES。")
            show_ensemble_details(ocsr)
        return

    ocsr = report.get("ocsr") or {}
    correction = report.get("correction") or {}
    final = report.get("final") or {}
    validation = report.get("validation") or {}
    st.success(report.get("message", "分析完成。"))
    if ocsr.get("backend") == "demo":
        st.warning("当前是演示结果：系统按内置样例文件名返回固定 SMILES，并没有进行真实图片识别。")
    elif ocsr.get("result_origin") in {"real_model", "real_model_ensemble"}:
        st.caption("当前结果来自真实 OCSR 模型推理。")

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
        runtime = report.get("runtime") or {}
        st.json({
            "backend": ocsr.get("backend"),
            "device": ocsr.get("device"),
            "model_name": ocsr.get("model_name"),
            "model_version": ocsr.get("model_version"),
            "model_sha256": ocsr.get("model_sha256"),
            "package_version": ocsr.get("package_version"),
            "git_commit": ocsr.get("git_commit") or runtime.get("git_commit"),
            "app_mode": runtime.get("app_mode"),
            "dependency_versions": ocsr.get("dependency_versions") or runtime.get("dependency_versions"),
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
    apply_col, restore_col = st.columns(2)
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

    with st.expander("纠错反馈与数据回流", expanded=False):
        correction_type = st.selectbox(
            "纠错类型",
            ["atom", "bond", "charge", "stereo", "missing_fragment", "other"],
            format_func={
                "atom": "原子错误",
                "bond": "键类型/连接错误",
                "charge": "电荷错误",
                "stereo": "立体化学错误",
                "missing_fragment": "缺失片段",
                "other": "其他",
            }.get,
            key=f"feedback_type_{analysis_id}",
        )
        review_status = st.selectbox(
            "二次审核状态",
            ["pending", "verified", "rejected"],
            format_func={"pending": "待审核", "verified": "已核验", "rejected": "已拒绝"}.get,
            key=f"feedback_review_{analysis_id}",
        )
        source_col, license_col = st.columns(2)
        source_reference = source_col.text_input(
            "来源或引用（可选）",
            value="",
            key=f"feedback_source_{analysis_id}",
            placeholder="DOI / 专利号 / 内部数据批次 / URL",
        )
        source_license = license_col.text_input(
            "许可说明（可选）",
            value="",
            key=f"feedback_license_{analysis_id}",
            placeholder="CC-BY-4.0 / internal / unknown",
        )
        privacy_notes = st.text_input(
            "隐私/脱敏说明（可选）",
            value="",
            key=f"feedback_privacy_{analysis_id}",
            placeholder="例如：已裁去个人信息；仅保留分子区域",
        )
        notes = st.text_area("反馈备注（可选）", value="", key=f"feedback_notes_{analysis_id}", height=70)
        can_save_feedback = bool((current_report.get("correction") or {}).get("applied"))
        save_col, accept_col = st.columns(2)
        if save_col.button("仅保存纠错", key=f"save_feedback_{analysis_id}", disabled=not can_save_feedback):
            result = save_correction_feedback(
                current_report,
                DATA_DIR,
                notes=notes,
                correction_type=correction_type,
                review_status=review_status,
                feedback_action="correction_only",
                include_in_training=False,
                source_reference=source_reference,
                source_license=source_license,
                privacy_notes=privacy_notes,
            )
            st.session_state[f"feedback_result_{analysis_id}"] = result
            if result.get("duplicate_image"):
                st.warning(f"已保存纠错；检测到重复图片，首次记录：{result.get('duplicate_of')}")
            else:
                st.success(f"纠错已保存：{result.get('annotation_path')}")
        if accept_col.button("确认进入训练集", key=f"accept_feedback_{analysis_id}", disabled=not can_save_feedback):
            result = save_correction_feedback(
                current_report,
                DATA_DIR,
                notes=notes,
                correction_type=correction_type,
                review_status="verified",
                feedback_action="accepted_for_dataset",
                include_in_training=True,
                source_reference=source_reference,
                source_license=source_license,
                privacy_notes=privacy_notes,
            )
            st.session_state[f"feedback_result_{analysis_id}"] = result
            if result.get("duplicate_image"):
                st.warning(f"已确认入库；检测到重复图片，首次记录：{result.get('duplicate_of')}")
            else:
                st.success(f"已确认进入训练集：{result.get('manifest_path')}")

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
