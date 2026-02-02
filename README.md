# MHier-Agent（MMRAG-DocQA + MDocAgent 融合版）

本项目将 **MMRAG-DocQA（索引与检索）** 和 **MDocAgent（多智能体生成）** 融合，形成三层架构：

1. **数据层（索引）**：Docling 解析 PDF，多模态结构化；构建 Flat Index（Faiss）与 RAPTOR Tree。
2. **检索层（上下文组装）**：并行检索 Flat/Tree，LLM Rerank，输出“摘要 + 细节”上下文。
3. **生成层（多智能体推理）**：General/Critical/Text/Image/Sum 多智能体协作。

---

## 目录结构

```
MHier-Agent/
├── MMRAG-DocQA/                 # MMRAG 核心
├── MDocAgent/                   # 多智能体核心
├── data/
│   ├── raw_pdfs/                # 原始 PDF
│   ├── processed_json/          # Docling 解析 + 合并 + 切分
│   ├── vector_db/               # Faiss 索引
│   ├── raptor_tree/             # RAPTOR Tree
│   └── page_images/             # PDF 页面截图
├── model_cache/                 # Docling / HF 模型缓存
├── prepare_data_mhier.py        # 一键数据准备脚本
├── requirements.txt             # 统一依赖
└── README.md
```

---

## 环境与依赖

- 建议 Python >= 3.11（已验证 3.12 / 3.11）
- 使用 `uv` 管理依赖：

```
uv pip install -r requirements.txt
```

---

## 配置（必须）

根目录 `.env` 需要配置 DashScope 兼容 OpenAI 的 Key（每种功能独立）：

```
DASHSCOPE_API_KEY_LLM=...
DASHSCOPE_API_KEY_VLM=...
DASHSCOPE_API_KEY_EMBEDDING=...
DASHSCOPE_API_KEY_RERANK=...
DASHSCOPE_API_KEY_SUMMARY=...
DASHSCOPE_API_KEY_QA=...
```

可选项（默认已内置）：

```
# DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
# DASHSCOPE_MODEL_LLM=qwen3-vl-plus
# DASHSCOPE_MODEL_VLM=qwen3-vl-plus
# DASHSCOPE_MODEL_EMBEDDING=text-embedding-v4
# DASHSCOPE_MODEL_RERANK=qwen3-vl-plus
# DASHSCOPE_MODEL_SUMMARY=qwen3-vl-plus
# DASHSCOPE_MODEL_QA=qwen3-vl-plus
# DASHSCOPE_EMBEDDING_DIMENSION=1024
# DASHSCOPE_EMBEDDING_MAX_BATCH_SIZE=10

# Docling 模型下载路径（默认到 model_cache/docling_models）
# DOCLING_ARTIFACTS_PATH=.\model_cache\docling_models
# DOCLING_FORCE_DOWNLOAD=1
```

---

## 快速开始（单个 PDF）

### 1) 放 PDF

把 PDF 放在：
```
data/raw_pdfs/你的文件名.pdf
```

### 2) 跑数据管线（解析 + 索引）

```
uv run python prepare_data_mhier.py
```

输出目录：
- `data/processed_json/01_parsed_reports`
- `data/processed_json/02_merged_reports`
- `data/processed_json/03_chunked_reports`
- `data/vector_db`
- `data/raptor_tree`
- `data/page_images`

### 3) 准备 Query（最小样例）

在 `data/mhier/samples.json`：

```json
[
  {
    "doc_id": "你的文件名.pdf",
    "question": "这里填写你的问题"
  }
]
```

注意：`doc_id` 必须和 `data/raw_pdfs` 下的文件名一致（含 `.pdf`）。

### 4) 检索（生成 texts/images）

```
uv run python MDocAgent/scripts/retrieve.py retrieval=mhier dataset.name=mhier dataset.data_dir=./data/mhier
```

输出：
```
data/mhier/sample-with-retrieval-results.json
```

### 5) 推理（多智能体回答）

```
uv run python MDocAgent/scripts/predict.py retrieval=mhier dataset.name=mhier dataset.data_dir=./data/mhier
```

输出：
```
results/mhier/...
```

---

## 常见问题

### 1) Docling 报模型缺失 / 权重损坏

如果看到 `model.safetensors` 相关错误：

```
DOCLING_FORCE_DOWNLOAD=1
rmdir /s /q model_cache\docling_models
uv run python prepare_data_mhier.py
```

### 2) Hydra 报 dataset.name 不存在

如果提示 `Key 'name' is not in struct`，用 `+` 强制新增：
```
... +dataset.name=mhier +dataset.data_dir=./data/mhier
```

或在 `MDocAgent/config/base.yaml` 中补 `dataset.name` 字段。

### 3) NumPy/Transformers 版本冲突

若 NumPy > 2.3 触发 numba 报错，降级：
```
uv pip install "numpy<2.0"
```

---

## 关键模块说明

- **MMRAGRetrieval**：`MDocAgent/retrieval/mhier_retrieval.py`
  - 使用 MMRAG HybridRetriever
  - 输出 `texts = [RAPTOR Summary, Parent Page Texts...]`
  - 输出 `images = 对应 page 的截图`

- **数据管线**：`prepare_data_mhier.py`
  - Docling 解析 → 页面图片 → 合并 → 切分 → RAPTOR Tree → Faiss

---

## 版本说明

本仓库集成了 MMRAG 与 MDocAgent 的多模态 RAG 方案，支持 DashScope 兼容 OpenAI API，适合长文档多模态问答。
