using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace RAGDemo.Embeddings
{
    public interface IEmbeddings
    {
        Task<float[]> EmbedAsync(string text);
        int Dimension { get; }
    }

    public class OnnxEmbeddings : IEmbeddings, IDisposable
    {
        private InferenceSession? _session;
        private readonly TokenizerBridge _tokenizer;
        private readonly string _modelDir = ".";
        public int Dimension { get; }

        public OnnxEmbeddings(string modelPath, TokenizerBridge? tokenizer = null, int dimension = 384)
        {
            Dimension = dimension;
            _tokenizer = tokenizer ?? new TokenizerBridge(null);
            if (System.IO.File.Exists(modelPath))
            {
                try
                {
                    _session = new InferenceSession(modelPath);
                    _modelDir = Path.GetDirectoryName(modelPath) ?? ".";
                    // Print input metadata to help diagnose required tokenizer tensors
                    Console.WriteLine("ONNX model loaded. Input metadata:");
                    foreach (var kv in _session.InputMetadata)
                    {
                        Console.WriteLine($" - {kv.Key}: {kv.Value.ElementType} {string.Join('x', kv.Value.Dimensions ?? Array.Empty<int>())}");
                    }
                }
                catch (Exception ex)
                {
                    Console.WriteLine("Failed to load ONNX model: " + ex.Message);
                    _session = null;
                }
            }
        }

        public async Task<float[]> EmbedAsync(string text)
        {
            if (_session == null)
            {
                return await Task.FromResult(HashEmbedding(text, Dimension));
            }

            try
            {
                var inputMeta = _session.InputMetadata;
                List<NamedOnnxValue> inputs = new();

                // Use TokenizerBridge to get all required tensors
                var (inputIds, attentionMask, tokenTypeIds) = _tokenizer.TokenizeFull(text);

                if (inputIds.Length == 0)
                {
                    Console.WriteLine("[OnnxEmbeddings] Tokenization returned empty, using hash fallback");
                    return await Task.FromResult(HashEmbedding(text, Dimension));
                }

                var seqLen = Math.Max(1, inputIds.Length);
                // Prepare input tensors based on model's expected inputs
                foreach (var name in inputMeta.Keys)
                {
                    var lower = name.ToLowerInvariant();
                    if (lower.Contains("input"))
                    {
                        var t = new DenseTensor<long>(new[] { 1, seqLen });
                        for (int i = 0; i < seqLen; i++) t[0, i] = i < inputIds.Length ? inputIds[i] : 0;
                        inputs.Add(NamedOnnxValue.CreateFromTensor(name, t));
                    }
                    else if (lower.Contains("attention"))
                    {
                        var t = new DenseTensor<long>(new[] { 1, seqLen });
                        for (int i = 0; i < seqLen; i++) t[0, i] = i < attentionMask.Length ? attentionMask[i] : 1;
                        inputs.Add(NamedOnnxValue.CreateFromTensor(name, t));
                    }
                    else if (lower.Contains("token_type") || lower.Contains("segment"))
                    {
                        var t = new DenseTensor<long>(new[] { 1, seqLen });
                        for (int i = 0; i < seqLen; i++) t[0, i] = i < tokenTypeIds.Length ? tokenTypeIds[i] : 0;
                        inputs.Add(NamedOnnxValue.CreateFromTensor(name, t));
                    }
                    else
                    {
                        // Unknown input, attempt zeros
                        var t = new DenseTensor<long>(new[] { 1, seqLen });
                        for (int i = 0; i < seqLen; i++) t[0, i] = 0;
                        inputs.Add(NamedOnnxValue.CreateFromTensor(name, t));
                    }
                }

                using var results = _session.Run(inputs);
                foreach (var r in results)
                {
                    if (r.Value is Tensor<float> t)
                    {
                        var outDims = t.Dimensions.ToArray();
                        var outLen = outDims.Aggregate(1, (a, b) => a * b);
                        var flat = new float[outLen];
                        int idx = 0;
                        foreach (var v in t) flat[idx++] = v;
                        float[] res;
                        if (flat.Length >= Dimension)
                        {
                            res = new float[Dimension];
                            Array.Copy(flat, flat.Length - Dimension, res, 0, Dimension);
                        }
                        else
                        {
                            res = flat;
                        }
                        // 归一化处理
                        var norm = Math.Sqrt(res.Select(x => x * x).Sum());
                        if (norm > 0)
                        {
                            for (int i = 0; i < res.Length; i++) res[i] = (float)(res[i] / norm);
                        }
                        return res;
                    }
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine("[OnnxEmbeddings] ONNX run failed: " + ex.Message);
            }

            return await Task.FromResult(HashEmbedding(text, Dimension));
        }

        private static float[] HashEmbedding(string text, int dim)
        {
            // Deterministic embedding fallback using SHA256; map bytes to floats in [-1,1]
            var bytes = SHA256.HashData(Encoding.UTF8.GetBytes(text));
            var res = new float[dim];
            for (int i = 0; i < dim; i++)
            {
                var b = bytes[i % bytes.Length];
                res[i] = (b / 255f) * 2f - 1f;
            }
            // normalize
            var norm = Math.Sqrt(res.Select(x => x * x).Sum());
            if (norm > 0)
            {
                for (int i = 0; i < dim; i++) res[i] = (float)(res[i] / norm);
            }
            return res;
        }

        public void Dispose()
        {
            _session?.Dispose();
            _session = null;
        }

        /// <summary>
        /// Batch embed multiple texts in a single ONNX run when possible.
        /// Falls back to individual hash embeddings if ONNX cannot be executed.
        /// </summary>
        public async Task<float[][]> EmbedBatchAsync(string[] texts)
        {
            if (_session == null)
            {
                return texts.Select(t => HashEmbedding(t, Dimension)).ToArray();
            }

            try
            {
                var inputMeta = _session.InputMetadata;

                int batch = texts.Length;
                var allInputIds = new List<int[]>();
                var allAttentionMask = new List<int[]>();
                var allTokenTypeIds = new List<int[]>();
                int maxLen = 0;

                // Tokenize each text
                for (int i = 0; i < texts.Length; i++)
                {
                    var (inputIds, attentionMask, tokenTypeIds) = _tokenizer.TokenizeFull(texts[i]);
                    
                    if (inputIds.Length == 0)
                    {
                        // Use fallback for this specific text
                        inputIds = new int[] { 0 };
                        attentionMask = new int[] { 1 };
                        tokenTypeIds = new int[] { 0 };
                    }
                    
                    allInputIds.Add(inputIds);
                    allAttentionMask.Add(attentionMask);
                    allTokenTypeIds.Add(tokenTypeIds);
                    maxLen = Math.Max(maxLen, inputIds.Length);
                }

                if (maxLen == 0) maxLen = 1;

                // Build batch tensors
                List<NamedOnnxValue> inputs = new();
                foreach (var name in inputMeta.Keys)
                {
                    var lower = name.ToLowerInvariant();
                    if (lower.Contains("input"))
                    {
                        var t = new DenseTensor<long>(new[] { batch, maxLen });
                        for (int i = 0; i < batch; i++)
                        {
                            var ids = allInputIds[i];
                            for (int j = 0; j < maxLen; j++) t[i, j] = j < ids.Length ? ids[j] : 0;
                        }
                        inputs.Add(NamedOnnxValue.CreateFromTensor(name, t));
                    }
                    else if (lower.Contains("attention"))
                    {
                        var t = new DenseTensor<long>(new[] { batch, maxLen });
                        for (int i = 0; i < batch; i++)
                        {
                            var att = allAttentionMask[i];
                            for (int j = 0; j < maxLen; j++) t[i, j] = j < att.Length ? att[j] : 1;
                        }
                        inputs.Add(NamedOnnxValue.CreateFromTensor(name, t));
                    }
                    else if (lower.Contains("token_type") || lower.Contains("segment"))
                    {
                        var t = new DenseTensor<long>(new[] { batch, maxLen });
                        for (int i = 0; i < batch; i++)
                        {
                            var tt = allTokenTypeIds[i];
                            for (int j = 0; j < maxLen; j++) t[i, j] = j < tt.Length ? tt[j] : 0;
                        }
                        inputs.Add(NamedOnnxValue.CreateFromTensor(name, t));
                    }
                    else
                    {
                        var t = new DenseTensor<long>(new[] { batch, maxLen });
                        for (int i = 0; i < batch; i++) for (int j = 0; j < maxLen; j++) t[i, j] = 0;
                        inputs.Add(NamedOnnxValue.CreateFromTensor(name, t));
                    }
                }

                using var results = _session.Run(inputs);
                foreach (var r in results)
                {
                    if (r.Value is Tensor<float> t)
                    {
                        var dims = t.Dimensions.ToArray();
                        if (dims.Length >= 2 && dims[0] == batch)
                        {
                            int rowSize = dims.Skip(1).Aggregate(1, (a, b) => a * b);
                            var flat = new float[batch * rowSize];
                            int idx = 0;
                            foreach (var v in t) flat[idx++] = v;
                            var outArr = new float[batch][];
                            for (int i = 0; i < batch; i++)
                            {
                                var row = new float[rowSize];
                                Array.Copy(flat, i * rowSize, row, 0, rowSize);
                                float[] embedding;
                                if (row.Length >= Dimension)
                                {
                                    embedding = new float[Dimension];
                                    Array.Copy(row, row.Length - Dimension, embedding, 0, Dimension);
                                }
                                else
                                {
                                    embedding = row;
                                }
                                // Normalize
                                var norm = Math.Sqrt(embedding.Select(x => x * x).Sum());
                                if (norm > 0)
                                {
                                    for (int j = 0; j < embedding.Length; j++)
                                        embedding[j] = (float)(embedding[j] / norm);
                                }
                                outArr[i] = embedding;
                            }
                            return outArr;
                        }
                        else
                        {
                            // Fall back to per-sample hash if unexpected shape
                            break;
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine("[OnnxEmbeddings] Batch run failed: " + ex.Message);
            }

            // Fallback: embed individually
            var result = new List<float[]>();
            foreach (var text in texts)
            {
                result.Add(await EmbedAsync(text));
            }
            return result.ToArray();
        }
    }
}
