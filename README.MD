# RAGDemo

轻量级的 Retrieval-Augmented Generation (RAG) 演示项目，用于展示如何：
- 加载文档并构建向量索引
- 基于本地 ONNX 嵌入模型生成向量
- 在向量库中检索相关段落并用于下游问答/生成

## 快速开始

1. 构建方案：
```sh
dotnet build RAGDemo.slnx
```

2. 运行项目：
```sh
dotnet run --project RAGDemo/RAGDemo.csproj
```

也可以直接打开解决方案文件 [RAGDemo.slnx](RAGDemo.slnx)。

## 主要组件（与源码对应）

- 配置：[`RAGDemo.Config`](RAGDemo/Config.cs) — [RAGDemo/Config.cs](RAGDemo/Config.cs)  
- 程序入口：[`RAGDemo.Program`](RAGDemo/Program.cs) — [RAGDemo/Program.cs](RAGDemo/Program.cs)  
- 检索器实现：[`RAGDemo.Retriever`](RAGDemo/Retriever.cs) — [RAGDemo/Retriever.cs](RAGDemo/Retriever.cs)  
- 文档加载：[`RAGDemo.IO.DocumentLoader`](RAGDemo/IO/DocumentLoader.cs) — [RAGDemo/IO/DocumentLoader.cs](RAGDemo/IO/DocumentLoader.cs)  
- 嵌入实现（ONNX）：[`RAGDemo.Embeddings.OnnxEmbeddings`](RAGDemo/Embeddings/OnnxEmbeddings.cs) — [RAGDemo/Embeddings/OnnxEmbeddings.cs](RAGDemo/Embeddings/OnnxEmbeddings.cs)  
- 分词桥接：[`RAGDemo.Embeddings.TokenizerBridge`](RAGDemo/Embeddings/TokenizerBridge.cs) — [RAGDemo/Embeddings/TokenizerBridge.cs](RAGDemo/Embeddings/TokenizerBridge.cs)  
- 向量存储：项目内 [VectorStore/](VectorStore/)（向量索引、持久化逻辑）  
- 预置模型： [models/all-MiniLM-L6-v2/](models/all-MiniLM-L6-v2/)  
- 索引文件： [RAGDemo/index.json](RAGDemo/index.json)  
- 辅助脚本： [scripts/tokenize.py](scripts/tokenize.py) (参考) 
- 示例数据： [test_data/chinese_notes.txt](test_data/chinese_notes.txt)  
- 测试引导： [TESTING_GUIDE.md](TESTING_GUIDE.md) 与 [sample_test.txt](sample_test.txt)

更多详细文档请参阅 `docs/PROJECT_DOCUMENTATION.md`，包含快速开始、目录说明、运行与开发指南等完整说明。
## 常见工作流

1. 准备模型和词表：将 ONNX 模型放到 [models/all-MiniLM-L6-v2/](models/all-MiniLM-L6-v2/)。  
2. 载入文档并分段：使用 [`RAGDemo.IO.DocumentLoader`](RAGDemo/IO/DocumentLoader.cs)。  
3. 生成嵌入：通过 [`RAGDemo.Embeddings.OnnxEmbeddings`](RAGDemo/Embeddings/OnnxEmbeddings.cs) 与 [`RAGDemo.Embeddings.TokenizerBridge`](RAGDemo/Embeddings/TokenizerBridge.cs)。  
4. 建索引并保存：向量写入 [VectorStore/](VectorStore/)。  
5. 检索相关段落：使用 [`RAGDemo.Retriever`](RAGDemo/Retriever.cs)。  
6. 问答或生成：基于检索结果进行下游任务。