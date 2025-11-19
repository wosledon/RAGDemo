using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using DocumentFormat.OpenXml.Packaging;
using ExcelDataReader;
using UglyToad.PdfPig;
using HtmlAgilityPack;

namespace RAGDemo.IO
{
    public record DocChunk(string Id, string Text, string SourcePath, int ChunkIndex, int CharStart, int CharEnd);

    public class DocumentLoader
    {
        public int ChunkSize { get; }
        public int ChunkOverlap { get; }

        private readonly Config _config = new();

        public DocumentLoader(int chunkSize = 800, int chunkOverlap = 200)
        {
            ChunkSize = chunkSize;
            ChunkOverlap = chunkOverlap;
            // Required for ExcelDataReader on .NET Core to support System.Text.Encoding code pages
            System.Text.Encoding.RegisterProvider(System.Text.CodePagesEncodingProvider.Instance);
        }

        public IEnumerable<DocChunk> LoadDirectory(string dir)
        {
            if (!Directory.Exists(dir)) yield break;
            var files = Directory.GetFiles(dir, "*.*", SearchOption.TopDirectoryOnly);
            foreach (var f in files)
            {
                foreach (var c in LoadFileChunks(f)) yield return c;
            }
        }

        public IEnumerable<DocChunk> LoadFileChunks(string path)
        {
            if (!File.Exists(path)) yield break;
            var ext = Path.GetExtension(path).ToLowerInvariant();
            // 使用 Config 中的白名单限制可读文件类型，避免将脚本/二进制等噪声文件读入索引
            if (_config.AllowedExtensions != null && !_config.AllowedExtensions.Contains(ext))
                yield break;
            string text;
            try
            {
                text = ext switch
                {
                    ".txt" => File.ReadAllText(path),
                    ".md" or ".markdown" => File.ReadAllText(path),
                    ".html" or ".htm" => ExtractTextFromHtml(path),
                    ".pdf" => ExtractTextFromPdf(path),
                    ".docx" => ExtractTextFromDocx(path),
                    _ => File.ReadAllText(path)
                };
            }
            catch
            {
                text = string.Empty;
            }

            if (string.IsNullOrWhiteSpace(text)) yield break;
            foreach (var chunk in ChunkText(text, path)) yield return chunk;
        }

        private IEnumerable<DocChunk> ChunkText(string text, string path)
        {
            int pos = 0;
            int idx = 0;
            while (pos < text.Length)
            {
                var end = Math.Min(text.Length, pos + ChunkSize);
                var snippet = text[pos..end];
                yield return new DocChunk(Path.GetFileName(path) + $"#chunk-{idx}", snippet, path, idx, pos, end);
                idx++;
                if (end == text.Length) break;
                pos = Math.Max(0, end - ChunkOverlap);
            }
        }

        private static string ExtractTextFromPdf(string path)
        {
            try
            {
                var sb = new StringBuilder();
                using var doc = PdfDocument.Open(path);
                foreach (var page in doc.GetPages())
                {
                    var t = page.Text;
                    if (!string.IsNullOrWhiteSpace(t)) sb.AppendLine(t);
                }
                return sb.ToString();
            }
            catch
            {
                return string.Empty;
            }
        }

        private static string ExtractTextFromDocx(string path)
        {
            try
            {
                using var doc = WordprocessingDocument.Open(path, false);
                var body = doc.MainDocumentPart?.Document?.Body;
                return body?.InnerText ?? string.Empty;
            }
            catch
            {
                return string.Empty;
            }
        }

        private static string ExtractTextFromExcel(string path)
        {
            try
            {
                using var stream = File.Open(path, FileMode.Open, FileAccess.Read, FileShare.ReadWrite);
                using var reader = ExcelReaderFactory.CreateReader(stream);
                var sb = new StringBuilder();
                do
                {
                    while (reader.Read())
                    {
                        for (int i = 0; i < reader.FieldCount; i++)
                        {
                            var v = reader.GetValue(i);
                            if (v != null) sb.Append(v.ToString());
                            sb.Append('\t');
                        }
                        sb.AppendLine();
                    }
                } while (reader.NextResult());
                return sb.ToString();
            }
            catch
            {
                return string.Empty;
            }
        }

        private static string ExtractTextFromHtml(string path)
        {
            try
            {
                var doc = new HtmlDocument();
                doc.Load(path);
                return doc.DocumentNode.InnerText;
            }
            catch
            {
                return string.Empty;
            }
        }
    }
}
