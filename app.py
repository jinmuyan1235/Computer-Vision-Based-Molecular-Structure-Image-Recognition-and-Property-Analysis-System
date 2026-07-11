"""Streamlit demonstration UI for Molecule Vision OCSR."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from config import OCSR_BACKEND, OUTPUT_DIR
from src.analysis.batch_analyzer import BatchAnalyzer
from src.analysis.correction import (
    apply_smiles_correction,
    restore_original_prediction,
    save_correction_feedback,
    structure_similarity,
)
from src.analysis.molecule_report import MoleculeReportGenerator
from src.documents.input_loader import DocumentInputError, OptionalDependencyError
from src.documents.processor import DocumentOCSRProcessor
from src.export.json_exporter import to_json_text
from src.export.pdf_exporter import save_pdf


st.set_page_config(page_title="分子结构图像识别与性质分析", page_icon="🧪", layout="wide")


@st.cache_resource(show_spinner=False)
def get_report_generator(backend: str) -> MoleculeReportGenerator:
    """Cache expensive optional OCSR model initialization between reruns."""
    return MoleculeReportGenerator(backend, OUTPUT_DIR)


@st.cache_resource(show_spinner=False)
def get_batch_analyzer(backend: str) -> BatchAnalyzer:
    """Reuse a batch analyzer and its selected backend between reruns."""
    return BatchAnalyzer(backend, OUTPUT_DIR)


@st.cache_resource(show_spinner=False)
def get_document_processor(backend: str) -> DocumentOCSRProcessor:
    """Reuse document detector and selected OCSR backend between reruns."""
    return DocumentOCSRProcessor(backend=backend)


def get_backend_status(backend: str) -> dict:
    """Return current backend status, including recent inference details."""
    status = get_report_generator(backend).recognizer.status()
    latest = st.session_state.get("backend_last_status") or {}
    if latest.get("backend") == backend:
        status.update({key: value for key, value in latest.items() if value is not None})
    return status


def remember_backend_status(backend: str) -> None:
    """Store backend diagnostics after an inference or batch run."""
    st.session_state["backend_last_status"] = get_report_generator(backend).recognizer.status()


def show_ensemble_details(ocsr: dict) -> None:
    """Render ensemble candidates and disagreement diagnostics."""
    candidates = ocsr.get("candidates") or []
    consensus = ocsr.get("consensus") or {}
    if not candidates and not consensus:
        return
    st.subheader("多后端候选与共识")
    status = consensus.get("status") or "unknown"
    reason = consensus.get("reason") or ""
    if status == "agreement":
        st.success(f"共识：{reason}")
    elif status == "disagreement":
        st.warning(consensus.get("warning") or reason)
        st.caption(reason)
    elif status in {"single_valid", "invalid_candidates", "all_failed"}:
        st.info(reason)
    if consensus.get("recommended_smiles"):
        st.write(f"**推荐结果：** `{consensus.get('recommended_smiles')}`")
        st.write(f"**推荐来源：** {consensus.get('recommended_backend')}")
    if consensus.get("confidence_policy"):
        st.caption(consensus.get("confidence_policy"))
    if candidates:
        rows = [
            {
                "backend": candidate.get("backend"),
                "status": candidate.get("status"),
                "raw_smiles": candidate.get("raw_smiles"),
                "canonical_smiles": candidate.get("canonical_smiles"),
                "valid": candidate.get("valid"),
                "confidence": candidate.get("confidence"),
                "time_ms": candidate.get("inference_time_ms"),
                "model": candidate.get("model_name"),
                "device": candidate.get("device"),
                "error": candidate.get("error"),
            }
            for candidate in candidates
        ]
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    similarity = ocsr.get("similarity_analysis") or []
    if similarity:
        st.caption("候选差异指标只解释模型输出之间的差异，不代表与原图一致。")
        st.dataframe(pd.DataFrame(similarity), hide_index=True, use_container_width=True)


def show_chemical_identity(report: dict) -> None:
    """Render chemical identity and standardization audit summary."""
    identity = report.get("chemical_identity") or {}
    standardization = report.get("standardization") or {}
    warnings = report.get("structure_warnings") or []
    if not identity:
        return
    st.subheader("化学身份与标准化")
    columns = st.columns(3)
    columns[0].metric("片段数", identity.get("fragment_count") if identity.get("fragment_count") is not None else "-")
    columns[1].metric("形式电荷", identity.get("formal_charge") if identity.get("formal_charge") is not None else "-")
    columns[2].metric("结构提示", len(warnings))
    st.write(f"**Standardized SMILES：** `{identity.get('standardized_smiles') or '-'}`")
    st.write(f"**InChIKey：** `{identity.get('inchikey') or '当前 RDKit/InChI 不可用或未生成'}`")
    st.caption(
        f"标准化 profile：{standardization.get('profile') or '-'} · "
        f"是否改变：{'是' if standardization.get('changed') else '否'}。"
        "结构提示只反映表示质量，不是毒理或药理结论。"
    )
    if warnings:
        with st.expander("查看结构提示与标准化审计"):
            st.dataframe(pd.DataFrame(warnings), hide_index=True, use_container_width=True)
            steps = standardization.get("steps") or []
            if steps:
                st.dataframe(pd.DataFrame(steps), hide_index=True, use_container_width=True)


def show_report(report: dict, show_preprocessing: bool, export_pdf: bool, key_prefix: str) -> None:
    """Render a molecule analysis report in Streamlit."""
    if report.get("status") != "success":
        st.error(report.get("message", "分析失败。"))
        ocsr = report.get("ocsr") or {}
        if ocsr:
            st.caption(f"后端：{ocsr.get('backend')} · 状态：{ocsr.get('status')}")
            show_ensemble_details(ocsr)
        return

    ocsr = report.get("ocsr") or {}
    correction = report.get("correction") or {}
    final = report.get("final") or {}
    validation = report.get("validation") or {}
    st.success(report.get("message", "分析完成。"))
    left, right = st.columns([1.1, 1])
    with left:
        st.subheader("结构识别")
        st.code(final.get("smiles") or ocsr.get("smiles") or "", language=None)
        st.write(f"**Canonical SMILES：** `{final.get('canonical_smiles') or validation.get('canonical_smiles')}`")
        if final.get("standardized_smiles") or validation.get("standardized_smiles"):
            st.write(f"**Standardized SMILES：** `{final.get('standardized_smiles') or validation.get('standardized_smiles')}`")
        confidence = ocsr.get("confidence")
        st.write(f"**识别后端：** {ocsr.get('backend')}　 **置信度：** {confidence if confidence is not None else '模型未提供'}")
        st.write(f"**当前结果来源：** {final.get('source') or 'unknown'}")
        if correction.get("applied"):
            st.info(f"已应用人工修正：`{correction.get('corrected_canonical_smiles')}`")
        diagnostic_line = " · ".join(
            item
            for item in [
                f"设备：{ocsr.get('device')}" if ocsr.get("device") else None,
                f"模型：{ocsr.get('model_name')}" if ocsr.get("model_name") else None,
                f"模型版本：{ocsr.get('model_version')}" if ocsr.get("model_version") else None,
                f"耗时：{ocsr.get('inference_time_ms')} ms" if ocsr.get("inference_time_ms") is not None else None,
            ]
            if item
        )
        if diagnostic_line:
            st.caption(diagnostic_line)
        st.write(f"**RDKit 校验：** {'有效' if validation.get('valid') else '无效'}")
        show_ensemble_details(ocsr)
        show_chemical_identity(report)
    with right:
        st.subheader("标准化结构重绘")
        drawing = (report.get("images") or {}).get("redrawn_molecule")
        if drawing:
            st.image(drawing, use_container_width=True)

    descriptors = report.get("descriptors") or {}
    st.subheader("分子基本性质")
    display_names = {
        "formula": "分子式", "molecular_weight": "MW", "logp": "LogP",
        "tpsa": "TPSA", "hbd": "HBD", "hba": "HBA",
        "rotatable_bonds": "可旋转键", "heavy_atom_count": "重原子数",
    }
    property_frame = pd.DataFrame(
        [{"性质": display_names.get(key, key), "数值": str(value)} for key, value in descriptors.items()]
    )
    st.dataframe(property_frame, hide_index=True, use_container_width=True)

    lipinski = report.get("lipinski") or {}
    if lipinski.get("passed"):
        st.info("✅ " + lipinski.get("summary", "符合规则。"))
    else:
        violations = "、".join(lipinski.get("violations") or [])
        st.warning(f"⚠️ {lipinski.get('summary', '')} 超限项：{violations}")

    admet = report.get("admet") or {}
    if admet.get("status") == "success":
        st.subheader("可选 ADMET baseline")
        admet_columns = st.columns(3)
        admet_columns[0].metric("预测终点", str(admet.get("target", "-")))
        admet_columns[1].metric("预测值", str(admet.get("prediction", "-")))
        probability = admet.get("probability")
        admet_columns[2].metric("模型置信度", f"{probability:.1%}" if probability is not None else "未提供")
        st.caption(admet.get("disclaimer", ""))
    elif admet.get("status") in {"unavailable", "failed"}:
        st.warning(admet.get("message", "ADMET baseline 不可用。"))

    if show_preprocessing and report.get("input", {}).get("type") == "image":
        st.subheader("OpenCV 图像预处理过程")
        stage_paths = (report.get("images") or {}).get("preprocessing") or {}
        preferred = ["original", "gray", "denoised", "binary", "cropped", "deskewed", "normalized"]
        titles = {"original": "原图", "gray": "灰度", "denoised": "去噪", "binary": "二值化", "cropped": "裁剪", "deskewed": "旋转校正", "normalized": "归一化"}
        columns = st.columns(4)
        for index, name in enumerate(preferred):
            if name in stage_paths:
                columns[index % 4].image(
                    stage_paths[name], caption=titles[name], use_container_width=True
                )

    json_text = to_json_text(report)
    st.download_button(
        "下载 JSON 报告", json_text, file_name=f"{key_prefix}_report.json",
        mime="application/json", key=f"json_{key_prefix}",
    )
    if export_pdf:
        pdf_result = save_pdf(report, OUTPUT_DIR / f"{key_prefix}_report.pdf")
        if pdf_result["success"]:
            st.download_button(
                "下载 PDF 报告", Path(pdf_result["path"]).read_bytes(),
                file_name=f"{key_prefix}_report.pdf", mime="application/pdf", key=f"pdf_{key_prefix}",
            )
        else:
            st.caption(pdf_result["message"])


def show_correction_panel(report: dict) -> dict:
    """Render human correction controls for an image report and return the current report."""
    if (report.get("input") or {}).get("type") != "image":
        return report
    analysis_id = report.get("analysis_id") or "image"
    ocsr = report.get("ocsr") or {}
    correction = report.get("correction") or {}
    final = report.get("final") or {}
    predicted = ocsr.get("predicted_smiles") or ocsr.get("smiles") or ""
    default_smiles = correction.get("corrected_smiles") or final.get("smiles") or predicted

    st.subheader("人工纠错")
    status_label = "已人工修正" if correction.get("applied") else "未人工修正"
    st.caption(f"纠错状态：{status_label} · 当前结果来源：{final.get('source') or '暂无有效结果'}")
    st.text_input("模型原始预测", value=predicted, disabled=True, key=f"predicted_{analysis_id}")
    corrected_input = st.text_input(
        "修正 SMILES",
        value=default_smiles or "",
        key=f"corrected_smiles_{analysis_id}",
        placeholder="OCSR 失败时也可以在这里手动输入 SMILES",
    )
    actions = st.columns([1, 1, 1])
    current_report = report
    if actions[0].button("校验并应用修正", type="primary", key=f"apply_correction_{analysis_id}"):
        candidate = apply_smiles_correction(report, corrected_input, OUTPUT_DIR)
        error = (candidate.get("correction") or {}).get("last_error")
        if error:
            st.error(error)
        else:
            current_report = candidate
            st.session_state["image_report"] = current_report
            st.success("人工修正已应用，性质和结构图已重新生成。")
    if actions[1].button("恢复模型原始结果", key=f"restore_prediction_{analysis_id}"):
        candidate = restore_original_prediction(report, OUTPUT_DIR)
        error = (candidate.get("correction") or {}).get("last_error")
        if error:
            st.warning(error)
        else:
            current_report = candidate
            st.session_state["image_report"] = current_report
            st.success("已恢复为模型原始预测。")

    updated_correction = current_report.get("correction") or {}
    feedback_notes = st.text_area("反馈备注（可选）", value="", key=f"feedback_notes_{analysis_id}", height=80)
    if actions[2].button(
        "保存为纠错反馈样本",
        key=f"save_feedback_{analysis_id}",
        disabled=not bool(updated_correction.get("applied")),
    ):
        feedback_path = save_correction_feedback(current_report, OUTPUT_DIR, feedback_notes)
        st.session_state[f"feedback_path_{analysis_id}"] = feedback_path
        st.success(f"反馈样本已保存：{feedback_path}")
    if st.session_state.get(f"feedback_path_{analysis_id}"):
        st.caption(f"最近保存的反馈样本：{st.session_state[f'feedback_path_{analysis_id}']}")

    images = current_report.get("images") or {}
    predicted_image = images.get("predicted_molecule")
    corrected_image = images.get("corrected_molecule")
    if predicted_image or corrected_image:
        st.subheader("结构对比")
        columns = st.columns(2) if predicted_image and corrected_image else st.columns(1)
        if predicted_image:
            columns[0].image(predicted_image, caption="模型预测结构", use_container_width=True)
        elif predicted:
            st.caption("模型原始预测不能被 RDKit 解析，无法绘制预测结构。")
        if corrected_image:
            target_column = columns[1] if predicted_image and corrected_image else columns[0]
            target_column.image(corrected_image, caption="人工修正结构", use_container_width=True)
        if predicted and updated_correction.get("corrected_smiles"):
            similarity = structure_similarity(predicted, updated_correction.get("corrected_smiles"))
            if similarity is not None:
                st.caption(f"Morgan Tanimoto 相似度：{similarity:.3f}。该值只比较两个分子结果，不代表与原图一致。")
    elif predicted:
        st.caption("模型原始预测不能被 RDKit 解析，无法绘制预测结构。")
    return current_report


def show_document_result(document_result: dict, backend: str) -> dict:
    """Render document pages, detected regions, exports, and bbox edit controls."""
    summary = document_result.get("summary") or {}
    st.subheader("Document regions")
    metrics = st.columns(4)
    metrics[0].metric("Pages", summary.get("page_count", 0))
    metrics[1].metric("Regions", summary.get("region_count", 0))
    metrics[2].metric("Molecule regions", summary.get("molecule_region_count", 0))
    metrics[3].metric("Recognized", summary.get("recognized_region_count", 0))
    if document_result.get("detection_errors"):
        st.warning(to_json_text(document_result.get("detection_errors"), indent=2))

    annotated = [item for item in (document_result.get("exports", {}).get("annotated_pages") or "").split(",") if item]
    if annotated:
        st.subheader("Annotated pages")
        columns = st.columns(min(3, max(1, len(annotated))))
        for index, page_path in enumerate(annotated[:6]):
            columns[index % len(columns)].image(page_path, caption=Path(page_path).name, use_container_width=True)

    rows = DocumentOCSRProcessor.region_rows(document_result)
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    else:
        st.info("No molecule-like regions detected. You can add a region manually below.")

    active_regions = [region for region in document_result.get("regions", []) if region.get("status") != "deleted"]
    if active_regions:
        region_ids = [region["region_id"] for region in active_regions]
        selected_id = st.selectbox("Edit region", region_ids, key="document_region_select")
        selected = next(region for region in active_regions if region["region_id"] == selected_id)
        bbox = selected.get("bbox") or [0, 0, 1, 1]
        edit_columns = st.columns(5)
        x1 = edit_columns[0].number_input("x1", min_value=0, value=int(bbox[0]), key=f"edit_x1_{selected_id}")
        y1 = edit_columns[1].number_input("y1", min_value=0, value=int(bbox[1]), key=f"edit_y1_{selected_id}")
        x2 = edit_columns[2].number_input("x2", min_value=1, value=int(bbox[2]), key=f"edit_x2_{selected_id}")
        y2 = edit_columns[3].number_input("y2", min_value=1, value=int(bbox[3]), key=f"edit_y2_{selected_id}")
        allowed_types = ["molecule", "text", "table", "reaction_like", "unknown", "non_molecule"]
        current_type = selected.get("region_type") if selected.get("region_type") in allowed_types else "unknown"
        region_type = edit_columns[4].selectbox(
            "type",
            allowed_types,
            index=allowed_types.index(current_type),
            key=f"edit_type_{selected_id}",
        )
        actions = st.columns(3)
        processor = get_document_processor(backend)
        if actions[0].button("Update bbox and rerun", key=f"update_region_{selected_id}"):
            document_result = processor.apply_edits(
                document_result,
                [{"action": "update", "region_id": selected_id, "bbox": [x1, y1, x2, y2], "region_type": region_type}],
                rerun_ocsr=True,
            )
            st.session_state["document_result"] = document_result
            st.success("Region updated and reprocessed.")
        if actions[1].button("Delete region", key=f"delete_region_{selected_id}"):
            document_result = processor.apply_edits(
                document_result,
                [{"action": "delete", "region_id": selected_id, "note": "Deleted in Streamlit UI."}],
                rerun_ocsr=False,
            )
            st.session_state["document_result"] = document_result
            st.success("Region marked as deleted.")
        if actions[2].button("Mark non-molecule", key=f"mark_region_{selected_id}"):
            document_result = processor.apply_edits(
                document_result,
                [{"action": "mark", "region_id": selected_id, "region_type": "non_molecule"}],
                rerun_ocsr=False,
            )
            st.session_state["document_result"] = document_result
            st.success("Region marked as non-molecule.")

    st.subheader("Add missed region")
    page_numbers = [int(page["page_number"]) for page in document_result.get("pages", [])]
    if page_numbers:
        add_columns = st.columns(6)
        page_number = add_columns[0].selectbox("page", page_numbers, key="add_region_page")
        add_x1 = add_columns[1].number_input("new x1", min_value=0, value=0, key="add_x1")
        add_y1 = add_columns[2].number_input("new y1", min_value=0, value=0, key="add_y1")
        add_x2 = add_columns[3].number_input("new x2", min_value=1, value=200, key="add_x2")
        add_y2 = add_columns[4].number_input("new y2", min_value=1, value=200, key="add_y2")
        add_type = add_columns[5].selectbox("new type", ["molecule", "text", "table", "reaction_like", "unknown"], key="add_type")
        if st.button("Add region and rerun", key="add_region"):
            processor = get_document_processor(backend)
            document_result = processor.apply_edits(
                document_result,
                [{"action": "add", "page_number": page_number, "bbox": [add_x1, add_y1, add_x2, add_y2], "region_type": add_type}],
                rerun_ocsr=True,
            )
            st.session_state["document_result"] = document_result
            st.success("Region added.")

    exports = document_result.get("exports") or {}
    if exports.get("json") and Path(exports["json"]).is_file():
        st.download_button("Download document_result.json", Path(exports["json"]).read_bytes(), "document_result.json", "application/json")
    if exports.get("regions_csv") and Path(exports["regions_csv"]).is_file():
        st.download_button("Download regions.csv", Path(exports["regions_csv"]).read_bytes(), "regions.csv", "text/csv")
    if exports.get("zip") and Path(exports["zip"]).is_file():
        st.download_button("Download ZIP result package", Path(exports["zip"]).read_bytes(), "document_results.zip", "application/zip")
    return document_result


st.title("基于计算机视觉的分子结构图像识别与性质分析系统")
st.caption("图片 → OpenCV 预处理 → OCSR → SMILES → RDKit 校验 → 性质分析 → 报告")

with st.sidebar:
    st.header("运行设置")
    backend_options = ["demo", "molscribe", "decimer", "ensemble"]
    backend_index = backend_options.index(OCSR_BACKEND) if OCSR_BACKEND in backend_options else 0
    backend = st.selectbox("OCSR 后端", backend_options, index=backend_index)
    show_preprocessing = st.checkbox("显示预处理过程", value=True)
    export_pdf = st.checkbox("启用 PDF 报告", value=False)
    if backend == "demo":
        st.info(
            "当前主动选择的是 demo 演示后端；这与 RDKit/OpenCV 是否安装无关。"
            "如已安装并配置 MolScribe/DECIMER，请在上方切换对应后端。"
        )
    backend_status = get_backend_status(backend)
    if backend_status["available"]:
        st.success(backend_status["message"])
    else:
        st.error(backend_status["message"])
        if backend == "molscribe":
            st.warning("MolScribe 当前不可用。请配置模型权重，或切换 demo 后端，也可以使用手动 SMILES 分析。")
        if backend == "decimer":
            st.warning("DECIMER 当前不可用。请安装兼容 decimer 包并确认 TensorFlow/设备环境，或切换 demo 后端。")
        if backend == "ensemble":
            st.warning("ensemble 当前没有可用真实子后端时不会伪装成功；可继续使用人工 SMILES 分析。")
    st.write(f"**当前后端：** {backend_status.get('backend', backend)}")
    st.write(f"**是否可用：** {'是' if backend_status.get('available') else '否'}")
    st.write(f"**模型：** {backend_status.get('model_name') or backend_status.get('model_path') or '无'}")
    st.write(f"**设备：** {backend_status.get('device') or '未指定'}")
    st.write(f"**包版本：** {backend_status.get('package_version') or '未安装/未提供'}")
    if backend_status.get("image_strategy"):
        st.write(f"**输入策略：** {backend_status.get('image_strategy')}")
    if backend_status.get("enabled_backends"):
        st.write(f"**子后端：** {', '.join(backend_status.get('enabled_backends') or [])}")
        st.write(f"**执行模式：** {'并行' if backend_status.get('parallel') else '串行安全'}")
    for child in backend_status.get("child_statuses") or []:
        st.caption(
            f"{child.get('backend')}: {'可用' if child.get('available') else '不可用'} · "
            f"{child.get('device') or 'device n/a'} · {child.get('message') or ''}"
        )
    last_time = backend_status.get("last_inference_time_ms")
    st.write(f"**最近推理耗时：** {last_time} ms" if last_time is not None else "**最近推理耗时：** 暂无")
    st.caption("默认串行安全运行；MolScribe 可用 PyTorch CUDA，DECIMER 可用 TensorFlow GPU/auto。")

image_tab, document_tab, smiles_tab, batch_tab, about_tab = st.tabs(["图片识别", "PDF/多分子文档", "SMILES 分析", "批量处理", "项目说明"])

with image_tab:
    uploaded = st.file_uploader("上传 PNG/JPG/JPEG 分子结构图", type=["png", "jpg", "jpeg"], key="single_upload")
    if uploaded is not None:
        st.image(uploaded, caption=f"上传原图：{uploaded.name}", width=500)
        if st.button("开始识别与分析", type="primary", key="analyze_image"):
            with st.spinner("正在执行图像预处理、OCSR 与 RDKit 分析……"):
                suffix = Path(uploaded.name).suffix.lower()
                prefix = Path(uploaded.name).stem + "_"
                with tempfile.NamedTemporaryFile(prefix=prefix, suffix=suffix, delete=False) as temporary:
                    temporary.write(uploaded.getvalue())
                    temporary_path = Path(temporary.name)
                try:
                    st.session_state["image_report"] = get_report_generator(backend).generate(image_path=temporary_path)
                    st.session_state["image_report"]["input"]["filename"] = uploaded.name
                    remember_backend_status(backend)
                finally:
                    temporary_path.unlink(missing_ok=True)
        if "image_report" in st.session_state:
            active_report = show_correction_panel(st.session_state["image_report"])
            show_report(active_report, show_preprocessing, export_pdf, f"image_{active_report.get('analysis_id', 'report')[:8]}")

with document_tab:
    st.write("Upload a PDF, a full-page PNG/JPG image, or a ZIP of page images. Regions keep page numbers and bbox coordinates.")
    document_upload = st.file_uploader(
        "Upload PDF / page image / ZIP",
        type=["pdf", "png", "jpg", "jpeg", "zip"],
        key="document_upload",
    )
    detect_only = st.checkbox("Detect only, do not run OCSR", value=False, key="document_detect_only")
    if document_upload is not None and st.button("Process document", type="primary", key="process_document"):
        suffix = Path(document_upload.name).suffix.lower()
        prefix = Path(document_upload.name).stem + "_"
        with tempfile.NamedTemporaryFile(prefix=prefix, suffix=suffix, delete=False) as temporary:
            temporary.write(document_upload.getvalue())
            temporary_path = Path(temporary.name)
        try:
            with st.spinner("Rendering pages, detecting molecule regions, and processing region OCSR..."):
                st.session_state["document_result"] = get_document_processor(backend).process(
                    temporary_path,
                    run_ocsr=not detect_only,
                )
                remember_backend_status(backend)
        except OptionalDependencyError as exc:
            st.error(str(exc))
        except (DocumentInputError, FileNotFoundError, ValueError) as exc:
            st.error(str(exc))
        finally:
            temporary_path.unlink(missing_ok=True)
    if "document_result" in st.session_state:
        st.session_state["document_result"] = show_document_result(st.session_state["document_result"], backend)

with smiles_tab:
    smiles_input = st.text_input("输入 SMILES", value="CCO", placeholder="例如：CCO")
    if st.button("分析 SMILES", type="primary", key="analyze_smiles"):
        with st.spinner("正在进行 RDKit 校验与性质计算……"):
            st.session_state["smiles_report"] = get_report_generator(backend).generate(smiles=smiles_input)
            remember_backend_status(backend)
    if "smiles_report" in st.session_state:
        show_report(st.session_state["smiles_report"], False, export_pdf, "smiles")

with batch_tab:
    st.write("可输入服务器上的文件夹路径，或一次上传多张图片。")
    folder_path = st.text_input("输入文件夹路径（可选）", value="")
    uploaded_files = st.file_uploader(
        "批量上传图片", type=["png", "jpg", "jpeg"], accept_multiple_files=True, key="batch_upload"
    )
    if st.button("开始批量处理", type="primary", key="analyze_batch"):
        with st.spinner("正在逐张处理并生成汇总……"):
            try:
                if uploaded_files:
                    with tempfile.TemporaryDirectory() as temp_dir:
                        for item in uploaded_files:
                            (Path(temp_dir) / Path(item.name).name).write_bytes(item.getvalue())
                        st.session_state["batch_result"] = get_batch_analyzer(backend).analyze_folder(temp_dir)
                        remember_backend_status(backend)
                elif folder_path.strip():
                    st.session_state["batch_result"] = get_batch_analyzer(backend).analyze_folder(folder_path.strip())
                    remember_backend_status(backend)
                else:
                    st.warning("请上传至少一张图片或填写输入文件夹路径。")
            except Exception as exc:
                st.error(f"批量处理失败：{exc}")
    if "batch_result" in st.session_state:
        batch_result = st.session_state["batch_result"]
        summary = batch_result["summary"]
        metrics = st.columns(4)
        metrics[0].metric("总图片", summary["total"])
        metrics[1].metric("成功", summary["successful"])
        metrics[2].metric("有效 SMILES", summary["valid_smiles"])
        metrics[3].metric("成功率", f"{summary['success_rate']:.1%}")
        st.dataframe(batch_result["dataframe"], use_container_width=True, hide_index=True)
        chart = batch_result["exports"]["summary_chart"]
        if Path(chart).is_file():
            st.image(chart, caption="批量结果统计", width=700)
        csv_bytes = Path(batch_result["exports"]["csv"]).read_bytes()
        st.download_button("下载 batch_results.csv", csv_bytes, "batch_results.csv", "text/csv", key="batch_csv")
        st.download_button(
            "下载 batch_results.json", to_json_text({"summary": summary, "results": batch_result["reports"]}),
            "batch_results.json", "application/json", key="batch_json",
        )

with about_tab:
    st.markdown("""
### 项目背景

医药研发、化学文献与专利中存在大量无法直接检索和计算的分子结构图片。本系统将二维结构图转换为 SMILES，并完成校验、重绘和基础性质分析。

### 技术路线

1. Pillow/OpenCV 读取图片并执行灰度化、去噪、二值化、白边裁剪、旋转校正和尺寸归一化；
2. 通过可替换的 MolScribe、DECIMER、ensemble 或 demo 适配器识别 SMILES；
3. 使用 RDKit 校验、标准化、绘图并计算 MW、LogP、TPSA、HBD、HBA 等描述符；
4. 通过 Lipinski 与扩展规则给出教学性质的风险提示；
5. 导出 JSON、CSV 和可选 PDF 报告。

### 局限性

- 主要支持清晰的二维分子结构图片；复杂背景、手绘结构和低分辨率图片可能失败；
- demo 模式只按样例文件名匹配，不是真实 OCSR；
- 真实识别需要单独安装和配置 MolScribe 或 DECIMER；ensemble 只解释模型一致/分歧，不会证明哪个候选一定符合原图；
- 性质与规则分析仅供教学演示，不能替代药物实验或专业决策。
""")
