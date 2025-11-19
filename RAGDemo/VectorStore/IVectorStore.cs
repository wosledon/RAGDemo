using System.Collections.Generic;

namespace RAGDemo.VectorStore
{
    public record VectorResult(string Id, float Score, string? Metadata);

    public interface IVectorStore
    {
        void Upsert(string id, float[] vector, string? metadata = null);
        IEnumerable<VectorResult> Query(float[] vector, int topK = 5);
        void Save(string path);
        void Load(string path);
    }
}
