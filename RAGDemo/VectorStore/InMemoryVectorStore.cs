using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;

namespace RAGDemo.VectorStore
{
    public class InMemoryVectorStore : IVectorStore
    {
        private class Item { public string Id { get; set; } = null!; public float[] Vector { get; set; } = null!; public string? Metadata { get; set; } }
        private readonly List<Item> _items = new();
        private readonly object _lock = new();

        public void Upsert(string id, float[] vector, string? metadata = null)
        {
            lock (_lock)
            {
                var existing = _items.FirstOrDefault(x => x.Id == id);
                if (existing != null)
                {
                    existing.Vector = vector;
                    existing.Metadata = metadata;
                }
                else
                {
                    _items.Add(new Item { Id = id, Vector = vector, Metadata = metadata });
                }
            }
        }

        public IEnumerable<VectorResult> Query(float[] vector, int topK = 5)
        {
            if (vector == null) yield break;
            List<Item> snapshot;
            lock (_lock)
            {
                snapshot = new List<Item>(_items);
            }
            var scored = new List<VectorResult>();
            foreach (var it in snapshot)
            {
                var score = CosineSimilarity(vector, it.Vector);
                scored.Add(new VectorResult(it.Id, score, it.Metadata));
            }
            foreach (var v in scored.OrderByDescending(s => s.Score).Take(topK)) yield return v;
        }

        private static float CosineSimilarity(float[] a, float[] b)
        {
            if (a == null || b == null) return 0f;
            int n = Math.Min(a.Length, b.Length);
            double dot = 0, na = 0, nb = 0;
            for (int i = 0; i < n; i++)
            {
                dot += a[i] * b[i];
                na += a[i] * a[i];
                nb += b[i] * b[i];
            }
            if (na == 0 || nb == 0) return 0f;
            return (float)(dot / (Math.Sqrt(na) * Math.Sqrt(nb)));
        }

        public void Save(string path)
        {
            List<Item> snapshot;
            lock (_lock)
            {
                snapshot = new List<Item>(_items);
            }
            var arr = snapshot.Select(it => new { it.Id, it.Vector, it.Metadata }).ToArray();
            var opts = new JsonSerializerOptions { WriteIndented = true };
            File.WriteAllText(path, JsonSerializer.Serialize(arr, opts));
        }

        public void Load(string path)
        {
            lock (_lock)
            {
                _items.Clear();
            }
            if (!File.Exists(path)) return;
            var text = File.ReadAllText(path);
            try
            {
                var doc = JsonSerializer.Deserialize<JsonElement>(text);
                if (doc.ValueKind == JsonValueKind.Array)
                {
                    foreach (var el in doc.EnumerateArray())
                    {
                        var id = el.GetProperty("Id").GetString() ?? Guid.NewGuid().ToString();
                        var metadata = el.TryGetProperty("Metadata", out var m) && m.ValueKind != JsonValueKind.Null ? m.GetString() : null;
                        var vecEl = el.GetProperty("Vector");
                        var list = new List<float>();
                        foreach (var v in vecEl.EnumerateArray()) list.Add(v.GetSingle());
                        lock (_lock)
                        {
                            _items.Add(new Item { Id = id, Vector = list.ToArray(), Metadata = metadata });
                        }
                    }
                }
            }
            catch
            {
                // ignore parse errors
            }
        }
    }
}
