# 测试指南

## 快速测试步骤

### 1. 编译项目
```bash
cd d:\repos\github\RAGDemo\RAGDemo
dotnet build -c Release
```

### 2. 运行程序
```bash
dotnet run
```

### 3. 测试命令

#### 导入测试数据
```
rag> import d:\your\data\directory
```

程序会自动：
- 扫描目录中的所有文档（txt, md, html, pdf, docx 等）
- 使用 Tokenizers.DotNet 进行分词
- 使用 ONNX 模型生成嵌入（自动归一化）
- 存储到内存向量库

#### 查询测试
```
rag> query 用户表结构
rag> query user table schema
rag> query 如何创建索引
```

### 4. 观察日志输出

查询时会输出详细日志：
```
[Retriever] Query: "用户表结构"
[Retriever] Tokens: 用, 户, 表, 结, 构 (count: 5)
[Retriever] Candidate: doc1#chunk-0, VectorScore: 0.856, Boost: 0.720, Combined: 1.576, TokenMatch: True, FullMatch: False
[Retriever] Candidate: doc2#chunk-1, VectorScore: 0.734, Boost: 0.000, Combined: 0.434, TokenMatch: False, FullMatch: False
...
[Retriever] Returning 5 results
```

关键指标：
- **VectorScore**: 语义相似度（0-1）
- **Boost**: 词法加权分数
- **Combined**: 最终综合分数
- **TokenMatch**: 是否有分词匹配
- **FullMatch**: 是否有完整查询匹配

### 5. 保存和加载索引

```
rag> save index.json
rag> load index.json
```

## 验证优化效果

### 检查点 1: Tokenizer 加载
启动时应看到：
```
[TokenizerBridge] Loaded tokenizer from: ./models/all-MiniLM-L6-v2/tokenizer.json
```

### 检查点 2: ONNX 模型加载
```
ONNX model loaded. Input metadata:
 - input_ids: Int64 1x...
 - attention_mask: Int64 1x...
 - token_type_ids: Int64 1x...
```

### 检查点 3: 嵌入生成
如果看到大量 "using hash fallback"，说明 tokenizer 未正常工作。
应该很少或没有 fallback 信息。

### 检查点 4: 检索结果
- Combined Score 应该在 0.3-2.0 之间
- 相关文档的 TokenMatch 或 FullMatch 应该为 True
- Boost 值应该合理（0-1.0）

## 性能对比

### 优化前（使用 Python）
- 每次查询需要启动 Python 进程
- 分词依赖 jieba 库
- 可能使用 hash embedding fallback

### 优化后（纯 C#）
- 零 Python 依赖
- 原生 Tokenizers.DotNet 分词
- 正确的 ONNX 嵌入 + 归一化
- 混合检索策略

## 常见问题

### Q: Tokenizer 未加载怎么办？
A: 确保 `models/all-MiniLM-L6-v2/tokenizer.json` 文件存在。

### Q: 所有查询都使用 hash embedding？
A: 检查：
1. tokenizer.json 是否存在
2. Tokenizers.DotNet NuGet 包是否正确安装
3. 运行时是否有错误日志

### Q: 检索结果不准确？
A: 检查：
1. MinScore 阈值（Config.cs 中，默认 0.3）
2. Boost 参数（Retriever.cs 中）
3. 是否有足够的词法匹配（查看 TokenMatch）

### Q: 如何调整参数？
A: 修改 `Config.cs`:
- `MinScore`: 最小分数阈值（降低=更多结果，提高=更精确）
- `LowPriorityPenalty`: 低优先级文件惩罚值
- `TopK`: 返回结果数量
