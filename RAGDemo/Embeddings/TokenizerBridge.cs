using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Tokenizers.DotNet;

namespace RAGDemo.Embeddings
{
    /// <summary>
    /// Tokenizer bridge using Tokenizers.DotNet for accurate tokenization.
    /// Supports BERT-like models with proper special tokens handling.
    /// </summary>
    public class TokenizerBridge
    {
        private Tokenizer? _nativeTokenizer;
        public bool HasNative => _nativeTokenizer != null;

        public TokenizerBridge(string? modelDir = null)
        {
            if (!string.IsNullOrEmpty(modelDir))
            {
                try
                {
                    var tokJson = Path.Combine(modelDir, "tokenizer.json");
                    if (File.Exists(tokJson))
                    {
                        _nativeTokenizer = new Tokenizer(vocabPath: tokJson);
                        Console.WriteLine($"[TokenizerBridge] Loaded tokenizer from: {tokJson}");
                        return;
                    }
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"[TokenizerBridge] Failed to load tokenizer: {ex.Message}");
                    _nativeTokenizer = null;
                }
            }
        }

        /// <summary>
        /// Tokenize text and return token IDs
        /// </summary>
        public int[] Tokenize(string text)
        {
            if (string.IsNullOrEmpty(text)) return Array.Empty<int>();
            
            if (_nativeTokenizer != null)
            {
                try
                {
                    var tokens = _nativeTokenizer.Encode(text);
                    return tokens.Select(t => (int)t).ToArray();
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"[TokenizerBridge] Tokenization failed: {ex.Message}");
                }
            }

            // Fallback: use simple whitespace tokenization
            return Array.Empty<int>();
        }

        /// <summary>
        /// Tokenize text and return all required tensors for BERT-like models
        /// </summary>
        public (int[] inputIds, int[] attentionMask, int[] tokenTypeIds) TokenizeFull(string text)
        {
            if (string.IsNullOrEmpty(text))
            {
                return (Array.Empty<int>(), Array.Empty<int>(), Array.Empty<int>());
            }

            if (_nativeTokenizer != null)
            {
                try
                {
                    var tokens = _nativeTokenizer.Encode(text);
                    var inputIds = tokens.Select(t => (int)t).ToArray();
                    var attentionMask = Enumerable.Repeat(1, inputIds.Length).ToArray();
                    var tokenTypeIds = Enumerable.Repeat(0, inputIds.Length).ToArray();
                    return (inputIds, attentionMask, tokenTypeIds);
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"[TokenizerBridge] Full tokenization failed: {ex.Message}");
                }
            }

            return (Array.Empty<int>(), Array.Empty<int>(), Array.Empty<int>());
        }

        /// <summary>
        /// Tokenize text into words/subwords for search matching (no IDs)
        /// Handles mixed Chinese-English text intelligently
        /// </summary>
        public string[] TokenizeForSearch(string text)
        {
            if (string.IsNullOrEmpty(text)) return Array.Empty<string>();

            if (_nativeTokenizer != null)
            {
                try
                {
                    // Use native tokenizer first - it handles mixed text well
                    var encoding = _nativeTokenizer.Encode(text);
                    // Since we can't easily get token strings from encoding,
                    // fall through to smart tokenization
                }
                catch { }
            }

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
                    // Separator: flush current token and Chinese buffer
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
                    // Chinese character: flush English token, append to Chinese buffer
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
                    // English letter/digit: flush Chinese buffer before building token
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
                    // Other characters: treat as separator
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

    }
}
