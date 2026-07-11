# 基于计算机视觉的分子结构图像识别与性质分析系统

本项目面向医药研发、化学文献数字化和实验室数据归档场景，基于 Python、OpenCV、OCSR 与 RDKit，实现二维分子结构图片到 SMILES 的自动识别、结构校验、分子重绘和基础性质分析，形成一个可演示、可批量处理、可导出报告的计算机视觉应用原型。

## 项目简介与企业需求

论文、专利、实验记录和药物资料中的分子结构常以图片存在，无法直接用于数据库检索或计算模型。系统提供以下数据入口：

- 将 PNG/JPG/JPEG 分子结构图转换为 SMILES；
- 使用 RDKit 校验和标准化结果，减少人工录入错误；
- 自动重绘结构，计算分子式、MW、LogP、TPSA、HBD、HBA 等；
- 批量处理文献或实验记录中的结构图，导出 CSV/JSON；
- 通过统一适配器接入 MolScribe、DECIMER 或其他 OCSR 模型；
- 在没有真实模型时使用稳定的 demo 流程完成教学演示。

项目重点是 OCSR 工程闭环，不是普通图像分类，也不以从零训练大型模型为目标。

## 功能列表

- 单图识别：格式检查、可视化预处理、OCSR、校验、重绘和报告；
- OpenCV 流程：灰度化、去噪、Otsu 二值化、白边裁剪、旋转校正、等比例归一化；
- 手动 SMILES：作为识别失败后的可靠补充入口；
- 描述符：分子式、MW、LogP、TPSA、HBD、HBA、可旋转键和重原子数；
- 规则判断：Lipinski 五规则指标及扩展可旋转键规则；
- 批量处理：逐图容错、结果表格、成功率/有效率和失败原因汇总；
- 导出：单图 JSON、批量 CSV/JSON、统计图和可选 PDF；
- 可选 ADMET baseline：用户提供带标签 CSV 后，可训练 Morgan Fingerprint + Random Forest 单终点模型；
- 工程稳定性：后端可用性诊断、模型实例缓存、分析 ID 隔离输出和端到端测试；
- Web 演示：Streamlit 四个页签，一条命令启动；
- 自动化测试：SMILES、描述符、规则、预处理和 demo 识别。

## 技术路线图

```mermaid
flowchart LR
    A["PNG/JPG 分子结构图"] --> B["OpenCV 预处理"]
    B --> C["OCSR 适配器"]
    C --> D["候选 SMILES"]
    D --> E["RDKit 校验与标准化"]
    E --> F["结构重绘"]
    E --> G["描述符与规则分析"]
    E --> K["可选 ADMET baseline"]
    F --> H["Streamlit 展示"]
    G --> H
    K --> H
    H --> I["JSON / CSV / PDF"]
    J["手动 SMILES"] --> E
```

## 环境要求

- Python 3.10 或 3.11；
- CPU 即可运行 demo、OpenCV 和 RDKit 主流程；
- GPU 仅作为真实 OCSR 后端的可选加速设备。

### Conda 安装（推荐）

完整环境可直接创建：

```bash
conda env create -f environment.yml
conda activate molecule-vision
```

也可以手动安装：

```bash
conda create -n molecule-vision python=3.10
conda activate molecule-vision
conda install -c conda-forge rdkit
pip install -r requirements.txt
```

> RDKit 优先推荐通过 conda-forge 安装。如果已由 conda 安装，`pip install -r requirements.txt` 会检测已有版本；遇到平台相关问题时可从 requirements 中临时注释 `rdkit` 一行。

### pip 安装

在支持 RDKit wheel 的 Python 3.10/3.11 平台上：

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 快速启动

先生成清晰的演示结构图：

```bash
python scripts/make_demo_samples.py
```

启动 Web：

```bash
python -m streamlit run app.py
```

浏览器中可上传 `data/samples/aspirin.png`，查看原图、所有预处理阶段、识别 SMILES、RDKit 重绘和性质结果。

### PyCharm 运行注意事项

项目解释器应明确指向：

```text
C:\Users\17679\.conda\envs\molecule-vision-310\python.exe
```

推荐在 Run Configuration 中以模块方式启动，Module name 填 `streamlit`，Parameters 填 `run app.py`。在终端中也优先使用 `python -m streamlit run app.py`，避免调用到其他环境中的 `streamlit.exe`。

可用以下命令核对当前解释器和核心依赖：

```powershell
python -c "import sys; print(sys.executable)"
python -c "import cv2, rdkit, streamlit; print(cv2.__version__, rdkit.__version__, streamlit.__version__)"
```

若终端启动时把整段 PATH 当作 PowerShell 命令执行，应关闭从异常终端继承环境的 PyCharm 实例，并从 Windows 开始菜单重新启动 PyCharm；也可将 PyCharm Terminal 的 Shell path 临时设为 `powershell.exe -NoProfile`。

## 命令行使用

分析手动 SMILES：

```bash
python scripts/analyze_smiles.py --smiles "CCO"
```

批量处理：

```bash
python scripts/run_batch.py --input data/batch_input --output data/outputs
```

也可直接批量处理生成的样例：

```bash
python scripts/run_batch.py --input data/samples --output data/outputs --backend demo
```

## OCSR 后端说明

后端可在 Streamlit 侧边栏选择，也可设置环境变量：

```bash
# Windows PowerShell
$env:OCSR_BACKEND="demo"

# macOS/Linux
export OCSR_BACKEND=demo
```

### demo

默认模式不加载机器学习模型，而是根据文件名匹配四个内置样例：

| 文件名关键词 | SMILES |
|---|---|
| `aspirin` | `CC(=O)OC1=CC=CC=C1C(=O)O` |
| `caffeine` | `Cn1cnc2c1c(=O)n(C)c(=O)n2C` |
| `benzene` | `c1ccccc1` |
| `ethanol` | `CCO` |

界面会明确提示当前主动选择的是 demo 后端。RDKit 和 OpenCV 不是 OCSR 模型；只有额外安装并配置 MolScribe/DECIMER 后，才能切换到真实图像识别后端。

### MolScribe

MolScribe 是可选依赖，不会在项目启动时被强制导入。按照其对应版本文档安装后，设置模型文件与设备：

```bash
$env:MOLSCRIBE_MODEL_PATH="C:\path\to\checkpoint.pth"
$env:OCSR_DEVICE="cpu"  # 或 cuda
python -m streamlit run app.py
```

不同 MolScribe 发行版本的模型构造与推理 API 可能变化；适配点集中在 `src/ocsr/molscribe_adapter.py`，不影响其他模块。

### DECIMER

DECIMER 同样是可选依赖。当前适配器支持暴露 `predict_SMILES` 的发行形式。若已安装版本的模块路径不同，只需调整 `src/ocsr/decimer_adapter.py` 中标注的适配位置。未安装或初始化失败时系统返回可读错误，不会崩溃。

适配器会优先请求后端返回置信度，并兼容不支持置信度参数的旧版本。Web 侧边栏会显示所选后端是否已成功加载。

## 可选 ADMET baseline

项目不会附带或伪造 ADMET 数据。准备一个至少包含 `smiles` 和目标标签列的可信 CSV 后，可训练单个分类或回归终点：

```bash
python scripts/train_admet.py \
  --input data/admet.csv \
  --smiles-column smiles \
  --target-column ames \
  --task classification \
  --output models/admet_baseline.joblib
```

训练完成后启用模型：

```bash
# Windows PowerShell
$env:ENABLE_ADMET_MODEL="true"
$env:ADMET_MODEL_PATH="models/admet_baseline.joblib"
python -m streamlit run app.py
```

模型文件使用 joblib 序列化，只应加载自己训练或可信来源的本地文件。未启用、文件缺失或预测失败时，系统会继续完成 RDKit 描述符与 Lipinski 规则分析。ADMET 输出仅是教学 baseline，不替代实验或专业结论。

## 测试

```bash
pytest -q
```

测试覆盖合法/非法 SMILES、canonical SMILES、描述符字段、规则超限、图像预处理、OCSR 兼容层、单图/手动 SMILES 端到端流程、批处理导出、PDF 报告以及可选 ADMET baseline。

## 项目目录

```text
molecule-vision-ocsr/
├── README.md
├── requirements.txt
├── environment.yml
├── config.py
├── app.py
├── data/
│   ├── samples/
│   ├── batch_input/
│   └── outputs/
├── models/                # 可选本地模型；模型文件不提交到 Git
├── src/
│   ├── preprocess/        # 图片读取、OpenCV 处理、过程可视化
│   ├── ocsr/              # 统一接口与 demo/MolScribe/DECIMER 适配器
│   ├── chem/              # RDKit 校验、描述符、规则、绘图
│   ├── analysis/          # 单分子报告与批处理编排
│   ├── export/            # JSON、CSV、可选 PDF
│   ├── ml/                # 可选 Morgan + Random Forest ADMET baseline
│   └── utils/             # 文件与日志工具
├── scripts/               # 批处理、SMILES 分析、样例生成、ADMET 训练
├── tests/                 # pytest 测试
└── docs/                  # 说明书、API 与报告模板
```

所有运行路径由 `config.py` 统一管理。默认输出写入 `data/outputs`，包括预处理图、重绘结构、批量表格和统计图。

## 答辩演示建议

1. 说明医药研发和专利中结构图片难以直接检索的企业需求；
2. 展示“图片 → OpenCV → OCSR → SMILES → RDKit → 报告”的路线；
3. 上传 `aspirin.png`，逐步展示 CV 中间结果；
4. 展示 canonical SMILES、结构重绘、描述符和 Lipinski 判断；
5. 输入 `CCO` 展示手动分析的稳定备用流程；
6. 批量处理 `data/samples` 并下载 CSV；
7. 说明 demo 与真实 OCSR 的边界和后续升级方向。

## 局限性与免责声明

- 当前系统主要支持清晰的二维分子结构图片；
- 复杂图片、手绘结构、低分辨率图片可能识别失败；
- demo 模式不是真实分子识别，只用于系统演示；
- 真实 OCSR 需要安装 MolScribe 或 DECIMER；
- MolScribe/DECIMER 的安装方式、模型权重和推理 API 可能随版本变化；
- Lipinski 结果只反映简单规则，不代表吸收、毒性、疗效或可开发性；
- 性质分析为教学演示，不能替代真实药物实验或专业判断。

## 生产 MolScribe 后端配置

本项目默认仍使用 `demo` 后端，demo 只用于教学演示：它会按内置样例文件名返回固定 SMILES，不是真实图片识别。真实 OCSR 需要单独安装 MolScribe、下载模型权重，并把 `OCSR_BACKEND` 切换为 `molscribe`。如果 MolScribe 未安装、模型文件缺失或加载失败，Streamlit、手动 SMILES、RDKit 性质分析、demo 后端和 DECIMER 后端仍会继续工作，并显示可读错误。

本适配器按 MolScribe 官方仓库公开接口进行兼容：构造模型时优先使用 `MolScribe(model_path, device=...)`，推理时优先使用 `predict_image_file(path, return_confidence=True)`；不同发行版本若返回 `dict`、字符串、元组或列表，适配器会归一化为统一结果字段。已对公开仓库接口形状做验证，未声称支持未测试的私有改版 API。

### 安装步骤

```bash
conda create -n molecule-vision-310 python=3.10
conda activate molecule-vision-310
conda install -c conda-forge rdkit
python -m pip install --upgrade pip
pip install -r requirements.txt
```

按 MolScribe 官方说明安装可选依赖。安装方式可能随 MolScribe 版本变化，请以其官方仓库为准：[thomas0809/MolScribe](https://github.com/thomas0809/MolScribe)。

```bash
pip install MolScribe
```

模型权重请从可信来源下载到本机，例如：

```text
models/molscribe_model.pth
```

不要把模型权重、大型数据集、虚拟环境或缓存文件提交到 Git。本仓库 `.gitignore` 已忽略 `models/*`、`*.pt`、`*.pth` 和 `*.onnx`。

### 环境变量

Windows PowerShell：

```powershell
$env:OCSR_BACKEND="molscribe"
$env:MOLSCRIBE_MODEL_PATH="C:\path\to\molscribe_model.pth"
$env:OCSR_DEVICE="cpu"
$env:MOLSCRIBE_IMAGE_STRATEGY="original"
python -m streamlit run app.py
```

Linux/macOS：

```bash
export OCSR_BACKEND=molscribe
export MOLSCRIBE_MODEL_PATH=/path/to/molscribe_model.pth
export OCSR_DEVICE=cpu
export MOLSCRIBE_IMAGE_STRATEGY=original
python -m streamlit run app.py
```

CUDA 示例：

```powershell
$env:OCSR_BACKEND="molscribe"
$env:MOLSCRIBE_MODEL_PATH="C:\path\to\molscribe_model.pth"
$env:OCSR_DEVICE="cuda"
python -m streamlit run app.py
```

常用配置：

| 变量 | 默认值 | 说明 |
|---|---|---|
| `MOLSCRIBE_MODEL_PATH` | `models/molscribe_model.pth` | MolScribe 权重文件路径，支持相对路径和绝对路径 |
| `MOLSCRIBE_MODEL_NAME` | 权重文件名 | 侧边栏和结果中显示的模型名 |
| `MOLSCRIBE_MODEL_VERSION` | 空 | 可选模型版本或标识 |
| `OCSR_DEVICE` | `cpu` | `cpu`、`cuda` 或 `cuda:0` |
| `OCSR_TIMEOUT_SECONDS` | `120` | 单次推理超时时间 |
| `OCSR_STRICT_MODE` | `false` | 为 `true` 时 CUDA 不可用会直接报错；默认可回退 CPU |
| `OCSR_USE_PREPROCESSED_IMAGE` | `false` | 兼容旧流程；默认 MolScribe 使用原图 |
| `MOLSCRIBE_IMAGE_STRATEGY` | `original` | `original`、`grayscale`、`normalized`、`binary` |

默认 `original` 更贴近 MolScribe 官方模型输入预期；不要默认假设二值化图片一定更好。只有在实验需要时再切换 `grayscale`、`normalized` 或 `binary`。

### 诊断命令

```bash
python scripts/check_ocsr_backend.py --backend demo
python scripts/check_ocsr_backend.py --backend molscribe
```

诊断输出包含 Python 版本、后端名称、包是否安装、包版本、模型路径、模型文件是否存在、设备、CUDA 是否可用、模型是否成功加载和可读错误。MolScribe 未安装时不会打印 Python 堆栈并崩溃。

### 常见错误

- `未安装 MolScribe`：先确认当前 Python 环境是否正确，再按官方说明安装 MolScribe。
- `模型文件不存在`：检查 `MOLSCRIBE_MODEL_PATH`，Windows 路径建议使用 PowerShell 字符串。
- `请求 CUDA 设备，但 torch.cuda.is_available() 为 False`：检查 NVIDIA 驱动、CUDA 版 PyTorch 和 `OCSR_DEVICE`；也可先用 `OCSR_DEVICE=cpu` 验证流程。
- `模型加载失败`：通常是权重文件与 MolScribe 代码版本不匹配，或模型文件损坏。请重新下载与当前 MolScribe 版本匹配的权重。
- `MolScribe 未返回 SMILES`：图片可能不符合模型输入分布，或该版本返回格式发生变化。可运行诊断脚本并保留错误信息用于适配。
