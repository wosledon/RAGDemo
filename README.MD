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

## 运行
```bash
RAGDemo — 本地向量化与检索演示
[TokenizerBridge] Loaded tokenizer from: ./models/all-MiniLM-L6-v2\tokenizer.json
ONNX model loaded. Input metadata:
 - input_ids: System.Int64 -1x-1
 - attention_mask: System.Int64 -1x-1
 - token_type_ids: System.Int64 -1x-1
Using model path: ./models/all-MiniLM-L6-v2\model.onnx
输入 'help' 查看可用命令。

rag> help
可用命令:
  import <dir>         — 从目录导入并索引所有支持的文档（txt, md, html, pdf, docx, xlsx）。
  index <file>         — 对单个文件解析并索引（支持同上扩展名）。
  query <text>         — 对查询文本向量化并检索 top-K 文档。返回 id, score, metadata。
  save <path>          — 将当前索引保存到磁盘（默认 ./index.json）。
  load <path>          — 从磁盘加载索引（默认 ./index.json）。
  help                 — 显示此帮助。
  exit | quit          — 退出程序。

索引说明:
  文档会被分块（chunk），默认 chunk size= 800, overlap= 200。
  每个 chunk 会单独向量化并存为 id: <filename>#chunk-i，metadata 为原始文件路径。
rag> load
Index loaded from ./index.json
rag> query 劳斯莱斯
[TokenizerBridge] Loaded tokenizer from: ./models/all-MiniLM-L6-v2\tokenizer.json
Top results:
Result #1 — id=新建 文本文档 (2).txt#chunk-0 score=0.8838
  source: C:\Users\BSI\Desktop\doc\新建 文本文档 (2).txt
  chunk: 0
  snippet:
劳斯莱斯是英国的豪华汽车品牌，经常被用来比喻极致的品质。
Result #2 — id=新建 文本文档 (3).txt#chunk-0 score=0.8770
  source: C:\Users\BSI\Desktop\doc\新建 文本文档 (3).txt
  chunk: 0
  snippet:
科目, 科目余额, 科目是什么, 啊啊啊啊科目
Result #3 — id=chinese_notes.txt#chunk-0 score=0.7866
  source: C:\Users\BSI\Desktop\doc\chinese_notes.txt
  chunk: 0
  snippet:
记账 报表 这是一些测试文本，用于验证记账查询是否能够匹配。 账目分类：资产、负债、收入、费用。 记账原则：借贷平衡，分录清晰。
rag> query 记账
[TokenizerBridge] Loaded tokenizer from: ./models/all-MiniLM-L6-v2\tokenizer.json
Top results:
Result #1 — id=chinese_notes.txt#chunk-0 score=0.7685
  source: C:\Users\BSI\Desktop\doc\chinese_notes.txt
  chunk: 0
  snippet:
记账 报表 这是一些测试文本，用于验证记账查询是否能够匹配。 账目分类：资产、负债、收入、费用。 记账原则：借贷平衡，分录清晰。
Result #2 — id=新建 文本文档 (2).txt#chunk-0 score=0.8682
  source: C:\Users\BSI\Desktop\doc\新建 文本文档 (2).txt
  chunk: 0
  snippet:
劳斯莱斯是英国的豪华汽车品牌，经常被用来比喻极致的品质。
Result #3 — id=新建 文本文档 (3).txt#chunk-0 score=0.8618
  source: C:\Users\BSI\Desktop\doc\新建 文本文档 (3).txt
  chunk: 0
  snippet:
科目, 科目余额, 科目是什么, 啊啊啊啊科目
rag>
```