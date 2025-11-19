using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using RAGDemo.Embeddings;
using RAGDemo.VectorStore;

namespace RAGDemo
{
    public class Retriever
    {
        private readonly IEmbeddings _emb;
        private readonly IVectorStore _store;

        public Retriever(IEmbeddings emb, IVectorStore store)
        {
            _emb = emb;
            _store = store;
        }

        public async Task<IEnumerable<VectorResult>> RetrieveAsync(string query, int topK = 5)
        {
            var cfg = new Config();
            // Console.WriteLine($"[Retriever] Query: \"{query}\"");
            var vec = await _emb.EmbedAsync(query);

            // Hybrid strategy: get a larger candidate set by vector search then lexical-re-rank by metadata
            int candidateK = Math.Max(topK * 4, topK + 20);
            var candidates = _store.Query(vec, candidateK).ToList();

            // 使用 TokenizerBridge 进行分词（英文和中文都适用）
            string[] tokens;
            if (_emb is OnnxEmbeddings onnxEmb)
            {
                // Try to get tokenizer bridge for better tokenization
                try
                {
                    var tokenizer = new TokenizerBridge(cfg.ModelsDir);
                    if (tokenizer.HasNative)
                    {
                        tokens = tokenizer.TokenizeForSearch(query ?? string.Empty);
                    }
                    else
                    {
                        tokens = SimpleTokenize(query);
                    }
                }
                catch
                {
                    tokens = SimpleTokenize(query);
                }
            }
            else
            {
                tokens = SimpleTokenize(query);
            }

            // Console.WriteLine($"[Retriever] Tokens: {string.Join(", ", tokens)} (count: {tokens.Length})");

            var scored = new List<(VectorResult item, float combined)>();
            foreach (var c in candidates)
            {
                float boost = 0f;
                bool hasTokenMatch = false;
                bool hasFullQueryMatch = false;
                bool hasStrongSchemaMatch = false;
                
                try
                {
                    if (!string.IsNullOrEmpty(c.Metadata))
                    {
                        // Parse metadata JSON and look into 'text' and 'source'
                        using var doc = System.Text.Json.JsonDocument.Parse(c.Metadata);
                        var root = doc.RootElement;
                        string combinedText = string.Empty;
                        if (root.TryGetProperty("text", out var t)) combinedText += " " + (t.GetString() ?? string.Empty);
                        if (root.TryGetProperty("source", out var s)) combinedText += " " + (s.GetString() ?? string.Empty);
                        var lower = combinedText.ToLowerInvariant();
                        
                        // Console.WriteLine($"[Debug] Checking candidate text (first 100 chars): {combinedText.Substring(0, Math.Min(100, combinedText.Length))}");
                        
                        // Count token matches (case-insensitive, with partial matching support)
                        int matches = 0;
                        int exactMatches = 0;
                        foreach (var tok in tokens)
                        {
                            if (string.IsNullOrEmpty(tok)) continue;

                            bool isChineseToken = IsPureChineseToken(tok);
                            bool isSingleChar = tok.Length == 1;
                            string searchToken = isChineseToken ? tok : tok.ToLowerInvariant();
                            string searchText = isChineseToken ? combinedText : lower;

                            // Console.WriteLine($"[Debug] Checking token '{tok}' (isChineseToken={isChineseToken}, searchToken='{searchToken}')");

                            if (isChineseToken || isSingleChar)
                            {
                                if (searchText.Contains(searchToken, StringComparison.Ordinal))
                                {
                                    // Console.WriteLine($"[Debug] ✓ Matched single char '{tok}'");
                                    matches++;
                                    exactMatches++;
                                }
                                else
                                {
                                    // Console.WriteLine($"[Debug] ✗ No match for '{tok}'");
                                }
                            }
                            else
                            {
                                // Multi-character token (English word or number)
                                var tokLower = tok.ToLowerInvariant();

                                if (lower.Contains(" " + tokLower + " ") ||
                                    lower.Contains(" " + tokLower) ||
                                    lower.Contains(tokLower + " ") ||
                                    lower.StartsWith(tokLower + " ") ||
                                    lower.EndsWith(" " + tokLower) ||
                                    lower == tokLower)
                                {
                                    matches++;
                                    exactMatches++;
                                }
                                else if (lower.Contains(tokLower))
                                {
                                    matches++;
                                }
                            }
                        }
                        
                        if (matches > 0)
                        {
                            hasTokenMatch = true;
                            // Give more weight to exact matches
                            // Base boost: 0.15 per match, 0.20 per exact match
                            float baseBoost = 0.15f * matches + 0.05f * exactMatches;
                            boost = Math.Min(0.8f, baseBoost);
                        }
                        
                        // Extra boost for exact full-query substring match
                        if (!string.IsNullOrEmpty(query) && lower.Contains(query.ToLowerInvariant()))
                        {
                            hasFullQueryMatch = true;
                            boost = Math.Max(boost, 0.55f);
                        }

                        // // Enhanced schema recognition for table/field definitions
                        // bool looksLikeSchema = lower.Contains("create table")
                        //                      || lower.Contains("primary key")
                        //                      || lower.Contains(" comment '")
                        //                      || lower.Contains(" not null")
                        //                      || lower.Contains("varchar")
                        //                      || lower.Contains("int ")
                        //                      || lower.Contains("decimal")
                        //                      || lower.Contains("unique key")
                        //                      || lower.Contains("foreign key")
                        //                      || lower.Contains("alter table")
                        //                      || lower.Contains("add column");

                        // if ((hasTokenMatch || hasFullQueryMatch) && looksLikeSchema)
                        // {
                        //     boost += 0.25f;
                        //     hasStrongSchemaMatch = true;
                        // }
                        
                        // If schema and query matches field name pattern, extra boost
                        if (hasStrongSchemaMatch && tokens.Any(tok => lower.Contains(tok.ToLowerInvariant() + " ") || lower.Contains(" " + tok.ToLowerInvariant())))
                        {
                            boost += 0.15f;
                        }
                    }
                }
                catch
                {
                    // Ignore parse errors
                }

                var combinedScore = c.Score + boost;

                // Penalize if neither token match nor full query match (semantic only, may be irrelevant)
                if (!hasTokenMatch && !hasFullQueryMatch)
                {
                    combinedScore -= 0.3f;
                }

                // Penalize certain low-priority file extensions
                try
                {
                    if (!string.IsNullOrEmpty(c.Metadata))
                    {
                        using var doc2 = System.Text.Json.JsonDocument.Parse(c.Metadata);
                        var root2 = doc2.RootElement;
                        if (root2.TryGetProperty("source", out var src))
                        {
                            var path = src.GetString() ?? string.Empty;
                            var ext = System.IO.Path.GetExtension(path).ToLowerInvariant();
                            if (cfg.LowPriorityExtensions.Contains(ext))
                            {
                                combinedScore -= cfg.LowPriorityPenalty;
                            }
                        }
                    }
                }
                catch { }

                // Apply minimum score filter
                if (combinedScore >= cfg.MinScore)
                {
                    scored.Add((c, combinedScore));
                }
                
                // Console.WriteLine($"[Retriever] Candidate: {c.Id}, VectorScore: {c.Score:F3}, Boost: {boost:F3}, Combined: {combinedScore:F3}, TokenMatch: {hasTokenMatch}, FullMatch: {hasFullQueryMatch}");
            }

            var results = scored.OrderByDescending(x => x.combined).Take(topK).Select(x => x.item).ToList();
            // Console.WriteLine($"[Retriever] Returning {results.Count} results");
            return results;
        }

        private static string[] SimpleTokenize(string? text)
        {
            if (string.IsNullOrWhiteSpace(text)) return Array.Empty<string>();
            
            // Smart tokenization for mixed Chinese-English text
            var tokens = new List<string>();
            var currentToken = new System.Text.StringBuilder();
            var chineseBuffer = new System.Text.StringBuilder();
            bool lastWasChinese = false;

            void FlushChineseBuffer()
            {
                if (chineseBuffer.Length > 1)
                {
                    tokens.Add(chineseBuffer.ToString());
                }
                chineseBuffer.Clear();
            }
            
            foreach (var ch in text)
            {
                bool isChinese = ch >= 0x4e00 && ch <= 0x9fff;
                bool isLetterOrDigit = char.IsLetterOrDigit(ch);
                
                if (char.IsWhiteSpace(ch) || char.IsPunctuation(ch))
                {
                    FlushChineseBuffer();
                    if (currentToken.Length > 0)
                    {
                        tokens.Add(currentToken.ToString());
                        currentToken.Clear();
                    }
                    lastWasChinese = false;
                }
                else if (isChinese)
                {
                    if (currentToken.Length > 0 && !lastWasChinese)
                    {
                        tokens.Add(currentToken.ToString());
                        currentToken.Clear();
                    }
                    tokens.Add(ch.ToString());
                    chineseBuffer.Append(ch);
                    lastWasChinese = true;
                }
                else if (isLetterOrDigit)
                {
                    FlushChineseBuffer();
                    if (lastWasChinese && currentToken.Length > 0)
                    {
                        tokens.Add(currentToken.ToString());
                        currentToken.Clear();
                    }
                    currentToken.Append(ch);
                    lastWasChinese = false;
                }
                else
                {
                    FlushChineseBuffer();
                    if (currentToken.Length > 0)
                    {
                        tokens.Add(currentToken.ToString());
                        currentToken.Clear();
                    }
                    lastWasChinese = false;
                }
            }
            
            FlushChineseBuffer();
            if (currentToken.Length > 0)
            {
                tokens.Add(currentToken.ToString());
            }

            // Convert English tokens to lowercase, keep Chinese as-is
            var result = new List<string>();
            foreach (var token in tokens.Where(t => t.Length > 0).Distinct())
            {
                bool hasChinese = token.All(ch => ch >= 0x4e00 && ch <= 0x9fff);
                result.Add(hasChinese ? token : token.ToLowerInvariant());
            }
            
            return result.ToArray();
        }

        private static bool IsPureChineseToken(string token)
        {
            if (string.IsNullOrEmpty(token)) return false;
            foreach (var ch in token)
            {
                if (ch < 0x4e00 || ch > 0x9fff)
                {
                    return false;
                }
            }
            return true;
        }
    }
}
