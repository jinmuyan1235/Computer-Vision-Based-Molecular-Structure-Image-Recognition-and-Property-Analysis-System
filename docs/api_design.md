# API 设计

## `ImagePreprocessor`

- `load_image(source) -> np.ndarray`
- `to_grayscale(image) -> np.ndarray`
- `denoise(image) -> np.ndarray`
- `binarize(image) -> np.ndarray`
- `crop_whitespace(image) -> np.ndarray`
- `deskew(image) -> np.ndarray`
- `resize_normalize(image, size) -> np.ndarray`
- `preprocess_pipeline(source) -> dict[str, np.ndarray]`

## OCSR

`BaseOCSRAdapter.recognize(image_path_or_array) -> OCSRResult`

`OCSRResult` 包含 `smiles`、`confidence`、`backend`、`status` 和 `message`。可用后端为 `demo`、`molscribe`、`decimer`。

## 化学模块

- `validate_smiles(smiles) -> dict`
- `canonicalize_smiles(smiles) -> str | None`
- `calculate_descriptors(smiles) -> dict`
- `evaluate_lipinski(descriptors) -> dict`
- `draw_molecule(smiles, output_path) -> str`

## 分析服务

### `MoleculeReportGenerator.generate`

只接受 `image_path` 或 `smiles` 之一，返回统一报告字典。`status=success` 表示校验、描述符和结构重绘全部成功。

### `BatchAnalyzer.analyze_folder`

输入文件夹，返回 `summary`、`rows`、`dataframe`、`reports` 和 `exports`。输出目录包含 CSV、JSON、统计图、预处理图和结构重绘图。

## 导出

- `save_json(data, output_path) -> str`
- `save_csv(rows, output_path) -> str`
- `save_pdf(report, output_path) -> {success, path, message}`
