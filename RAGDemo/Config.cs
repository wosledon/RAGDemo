using System;

namespace RAGDemo
{
    public class Config
    {
        public string ModelsDir { get; set; } = "./models/all-MiniLM-L6-v2";
        public string IndexPath { get; set; } = "./index.json";
        public int TopK { get; set; } = 5;
        public int EmbeddingDimension { get; set; } = 384;
        public string IndexType { get; set; } = "inmemory";
        // Document chunking defaults
        public int ChunkSize { get; set; } = 800;
        public int ChunkOverlap { get; set; } = 200;
        // Maximum concurrent embedding tasks (for parallel indexing). Default to number of processors.
        public int MaxConcurrency { get; set; } = Math.Max(1, Environment.ProcessorCount);
        // Batch size for ONNX batch inference (number of chunks per forward pass)
        public int BatchSize { get; set; } = 16;
        // Minimum combined score (vector + lexical boost) to include in final results.
        // 调低阈值到 0.3，避免过度过滤相关结果
        public float MinScore { get; set; } = 0.3f;
        // 允许被加载并索引的文件扩展名白名单（小写，包含点号），用于过滤目录中文件类型。
        public string[] AllowedExtensions { get; set; } = new[]
        {
            ".txt", ".md", ".markdown", ".html", ".htm", ".pdf", ".docx"
        };
        // 需要降权的扩展名（例如代码、日志、配置、模板类文件），会在排序时按惩罚值降低得分。
        // 注意：这里不包含 .sql，避免统一压低 DDL/DML 脚本，方便在字段查询场景下召回表结构。
        public string[] LowPriorityExtensions { get; set; } = new[]
        {
            ".proto", ".json", ".log", ".http", ".xml"
        };
        // Penalty to subtract from combined score for low-priority extensions (0~1，值越大惩罚越重)
        // 降低惩罚值到 0.4，避免过度压低相关代码文件
        public float LowPriorityPenalty { get; set; } = 0.4f;
    }
}
