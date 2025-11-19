# RAGDemo 项目文档

## 项目概述
RAGDemo 是一个轻量级的 Retrieval-Augmented Generation (RAG) 演示项目，展示如何：

- 使用本地 ONNX 嵌入模型生成文本向量
- 构建和管理向量索引（本地内存/文件持久化）
- 基于检索的段落进行问答或生成任务

该项目以 .NET (C#) 编写，采用 ONNX 模型来离线生成嵌入（Embedding），并将向量存储在项目内的向量存储实现中。

## 快速开始

前提：已安装 .NET SDK（建议 .NET 8/10 兼容）以及 Git。将仓库克隆到本地并进入项目根目录。

1. 克隆与构建：

```pwsh
git clone <your-repo-url>
cd RAGDemo
dotnet build RAGDemo.slnx
```

2. 运行：

```pwsh
dotnet run --project RAGDemo/RAGDemo.csproj
```

3. 常见运行选项：
- 如果需要重新生成索引或使用不同的数据集，请先查看 `RAGDemo/Config.cs` 中的配置并更新路径。

## 目录结构（摘录）

- `RAGDemo/` - 主程序与核心实现
  - `Config.cs` - 配置项（模型目录、索引文件路径等）
  - `Program.cs` - 程序入口
  - `Retriever.cs` - 检索器实现（检索逻辑与相似度度量）
  - `Embeddings/OnnxEmbeddings.cs` - ONNX 嵌入模型调用逻辑
  - `Embeddings/TokenizerBridge.cs` - 分词/tokenizer 桥接实现
  - `IO/DocumentLoader.cs` - 文档加载与分段逻辑
- `VectorStore/` - 向量存储实现（索引、持久化）
- `models/all-MiniLM-L6-v2/` - 预置 ONNX 模型与 tokenizer 资源
- `docs/` - 本地文档目录（本文件）
- `test_data/` - 示例数据

## 关键组件说明

### 嵌入 (Embeddings)
实现位于 `RAGDemo/Embeddings/OnnxEmbeddings.cs`，它负责：
- 调用 ONNX 运行时对文本进行前向推理
- 返回向量表示（通常做 L2/normalize 处理）

`TokenizerBridge.cs` 用于把文本转换为模型需要的 token id/attention masks，桥接 C# 与 tokenizer 资源或 Python 脚本（`scripts/tokenize.py`）的差异。

### 向量存储 (VectorStore)
`VectorStore/` 目录包含向量索引的接口 `IVectorStore.cs` 与内存实现 `InMemoryVectorStore.cs`。功能包括：
- 插入向量与元数据
- 按相似度检索 top-k 向量
- 可序列化到文件以便持久化（项目中有示例实现）

### 检索器 (Retriever)
位于 `RAGDemo/Retriever.cs`，检索器主要职责：
- 接受查询文本
- 通过嵌入模块将查询编码为向量
- 在向量存储中检索相似段落
- 返回文本片段及其匹配得分，供下游生成/问答使用

## 运行示例

示例：使用自带的 `test_data/chinese_notes.txt` 构建索引并进行一次查询。

1. 确保模型文件存在于 `models/all-MiniLM-L6-v2/`。
2. 修改 `RAGDemo/Config.cs` 中的输入文件与模型路径（如需要）。
3. 运行程序：

```pwsh
dotnet run --project RAGDemo/RAGDemo.csproj
```

程序运行后会输出索引构建进度及一次示例查询的检索结果。

## 开发指南

- 在修改嵌入或 Tokenizer 逻辑时，最好先在 `scripts/tokenize.py` 里复现 tokenizer 的行为，确保 token 与 ONNX 模型期待的一致。
- 如果要替换模型：
  - 将新的 ONNX 模型与相应的 tokenizer 文件放到 `models/<your-model>/` 下
  - 更新 `RAGDemo/Config.cs` 中的 `ModelPath` 等字段
  - 运行并验证嵌入输出维度是否匹配向量存储（常见问题：维度不匹配）

## 测试与调试

- 当前仓库没有使用专门的测试框架（如 xUnit）的示例测试，您可以：
  - 将关键函数封装并添加单元测试
  - 或者编写一个小的 `Console` 测试程序，加载模型并对已知文本进行嵌入，验算相似度

## 常见问题 (FAQ)

Q: 运行时报错找不到 ONNX 模型或 tokenizer
A: 请检查 `models/all-MiniLM-L6-v2/` 是否完整（`model.onnx`, `tokenizer.json`, `vocab.txt` 等）。也确认 `RAGDemo/Config.cs` 中的路径指向正确位置。

Q: 嵌入维度或输出不匹配
A: 检查您使用的 ONNX 模型是否与分词器（tokenizer）匹配；模型量化或不同导出的模型可能改变输出格式/维度。

## 贡献与许可

欢迎提交 issue 或 PR 改进示例、添加单元测试或完善向量存储后端。

---
文档由自动化工具生成，若需补充示例或 API 细节，请告知具体需求。