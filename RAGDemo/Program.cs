// Minimal local RAG vector demo (embeddings + in-memory vector store)
using System;
using RAGDemo;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using RAGDemo.Embeddings;
using RAGDemo.VectorStore;
using RAGDemo.IO;
using System.Text.Json;
using System.Text.RegularExpressions;
using System.Threading;
using System.Threading.Tasks;

Console.WriteLine("RAGDemo — 本地向量化与检索演示");

var cfg = new Config();
var modelPath = Path.Combine(cfg.ModelsDir, "model.onnx");
var tokenizer = new TokenizerBridge(cfg.ModelsDir);
using var emb = new OnnxEmbeddings(modelPath, tokenizer, cfg.EmbeddingDimension);
IVectorStore store = new InMemoryVectorStore();
var retriever = new Retriever(emb, store);

Console.WriteLine($"Using model path: {modelPath}");
Console.WriteLine("输入 'help' 查看可用命令。\n");

// Helper to clean/normalize snippet previews: remove control chars, collapse whitespace and redact long blobs
static string CleanSnippet(string s)
{
	if (string.IsNullOrEmpty(s)) return s;
	// remove control characters
	s = Regex.Replace(s, "\\p{C}+", " ");
	// replace very long non-whitespace sequences (likely base64/jwt/blob) with placeholder
	s = Regex.Replace(s, "\\S{100,}", "[BLOB]");
	// collapse whitespace
	s = Regex.Replace(s, "\\s+", " ").Trim();
	return s;
}

while (true)
{
	Console.Write("rag> ");
	var line = Console.ReadLine();
	if (string.IsNullOrWhiteSpace(line)) continue;
	var parts = line.Split(' ', 2, StringSplitOptions.RemoveEmptyEntries);
	var cmd = parts[0].ToLowerInvariant();
	var arg = parts.Length > 1 ? parts[1] : null;
	try
	{
		if (cmd == "exit" || cmd == "quit") break;
		if (cmd == "help")
		{
			Console.WriteLine("可用命令:");
			Console.WriteLine("  import <dir>         — 从目录导入并索引所有支持的文档（txt, md, html, pdf, docx, xlsx）。");
			Console.WriteLine("  index <file>         — 对单个文件解析并索引（支持同上扩展名）。");
			Console.WriteLine("  query <text>         — 对查询文本向量化并检索 top-K 文档。返回 id, score, metadata。");
			Console.WriteLine("  save <path>          — 将当前索引保存到磁盘（默认 ./index.json）。");
			Console.WriteLine("  load <path>          — 从磁盘加载索引（默认 ./index.json）。");
			Console.WriteLine("  help                 — 显示此帮助。");
			Console.WriteLine("  exit | quit          — 退出程序。");
			Console.WriteLine();
			Console.WriteLine("索引说明:");
			Console.WriteLine("  文档会被分块（chunk），默认 chunk size= {0}, overlap= {1}。", cfg.ChunkSize, cfg.ChunkOverlap);
			Console.WriteLine("  每个 chunk 会单独向量化并存为 id: <filename>#chunk-i，metadata 为原始文件路径。");
			continue;
		}
		if (cmd == "import")
		{
			if (string.IsNullOrEmpty(arg)) { Console.WriteLine("请提供目录路径。"); continue; }
			var dir = arg;
			if (!Directory.Exists(dir)) { Console.WriteLine("目录不存在: " + dir); continue; }
			var loader = new DocumentLoader(cfg.ChunkSize, cfg.ChunkOverlap);
			var chunks = loader.LoadDirectory(dir).ToList();
			int count = 0;
			int batchSize = Math.Max(1, cfg.BatchSize);
			var batchIndices = Enumerable.Range(0, (chunks.Count + batchSize - 1) / batchSize);
			var po = new ParallelOptions { MaxDegreeOfParallelism = cfg.MaxConcurrency };
			await Parallel.ForEachAsync(batchIndices, po, async (b, ct) =>
			{
				var items = chunks.Skip(b * batchSize).Take(batchSize).ToArray();
				var texts = items.Select(x => x.Text).ToArray();
				var vecs = await emb.EmbedBatchAsync(texts);
				for (int i = 0; i < items.Length; i++)
				{
					var chunk = items[i];
					var vec = vecs[i];
					var preview = chunk.Text.Length > 1000 ? chunk.Text.Substring(0, 1000) : chunk.Text;
					preview = CleanSnippet(preview);
					var metaObj = new { source = chunk.SourcePath, chunk = chunk.ChunkIndex, text = preview };
					var metaJson = JsonSerializer.Serialize(metaObj);
					store.Upsert(chunk.Id, vec, metaJson);
					Interlocked.Increment(ref count);
					Console.WriteLine($"Indexed {chunk.Id} ({chunk.SourcePath})");
				}
			});
			Console.WriteLine($"导入完成，已索引 {count} chunks。");
			continue;
		}
		if (cmd == "index")
		{
			if (string.IsNullOrEmpty(arg)) { Console.WriteLine("请提供文件路径。"); continue; }
			var f = arg;
			if (!File.Exists(f)) { Console.WriteLine("文件不存在: " + f); continue; }
			var loader = new DocumentLoader(cfg.ChunkSize, cfg.ChunkOverlap);
			var chunks = loader.LoadFileChunks(f).ToList();
			int count = 0;
			int batchSize = Math.Max(1, cfg.BatchSize);
			var batchIndices = Enumerable.Range(0, (chunks.Count + batchSize - 1) / batchSize);
			var po2 = new ParallelOptions { MaxDegreeOfParallelism = cfg.MaxConcurrency };
			await Parallel.ForEachAsync(batchIndices, po2, async (b, ct) =>
			{
				var items = chunks.Skip(b * batchSize).Take(batchSize).ToArray();
				var texts = items.Select(x => x.Text).ToArray();
				var vecs = await emb.EmbedBatchAsync(texts);
				for (int i = 0; i < items.Length; i++)
				{
					var chunk = items[i];
					var vec = vecs[i];
					var preview = chunk.Text.Length > 1000 ? chunk.Text.Substring(0, 1000) : chunk.Text;
					preview = CleanSnippet(preview);
					var metaObj = new { source = chunk.SourcePath, chunk = chunk.ChunkIndex, text = preview };
					var metaJson = JsonSerializer.Serialize(metaObj);
					store.Upsert(chunk.Id, vec, metaJson);
					Interlocked.Increment(ref count);
					Console.WriteLine($"Indexed {chunk.Id} ({chunk.SourcePath})");
				}
			});
			Console.WriteLine($"索引完成，已添加 {count} chunks。");
			continue;
		}
		if (cmd == "query")
		{
			if (string.IsNullOrEmpty(arg)) { Console.WriteLine("请提供查询文本。"); continue; }
			var q = arg;
			var results = await retriever.RetrieveAsync(q, cfg.TopK);
			Console.WriteLine("Top results:");
			int i = 1;
			foreach (var r in results)
			{
				Console.WriteLine($"Result #{i} — id={r.Id} score={r.Score:F4}");
				if (!string.IsNullOrEmpty(r.Metadata))
				{
					try
					{
						using var doc = JsonDocument.Parse(r.Metadata);
						var root = doc.RootElement;
						if (root.TryGetProperty("source", out var sourceEl)) Console.WriteLine($"  source: {sourceEl.GetString()}");
						if (root.TryGetProperty("chunk", out var chunkEl)) Console.WriteLine($"  chunk: {chunkEl.GetInt32()}");
						if (root.TryGetProperty("text", out var textEl))
						{
							var txt = textEl.GetString() ?? string.Empty;
							var preview = txt.Length > 500 ? txt.Substring(0, 500) + "..." : txt;
							preview = CleanSnippet(preview);
							Console.WriteLine("  snippet:");
							Console.WriteLine(preview.Replace("\n", " ").Trim());
						}
					}
					catch
					{
						Console.WriteLine("  meta: " + r.Metadata);
					}
				}
				i++;
			}
			continue;
		}
		if (cmd == "save")
		{
			var path = string.IsNullOrEmpty(arg) ? cfg.IndexPath : arg;
			store.Save(path);
			Console.WriteLine("Index saved to " + path);
			continue;
		}
		if (cmd == "load")
		{
			var path = string.IsNullOrEmpty(arg) ? cfg.IndexPath : arg;
			store.Load(path);
			Console.WriteLine("Index loaded from " + path);
			continue;
		}

		Console.WriteLine("未知命令：" + cmd + "。输入 help 查看可用命令。");
	}
	catch (Exception ex)
	{
		Console.WriteLine("错误: " + ex.Message);
	}
}

Console.WriteLine("退出。");