# Count parameters and print basic architecture facts for both models.
from sentence_transformers import SentenceTransformer

for name in ["google/embeddinggemma-300m", "Qwen/Qwen3-Embedding-0.6B"]:
    m = SentenceTransformer(name, device="cpu")
    n = sum(p.numel() for p in m.parameters())
    print(f"{name}: {n:,} parameters, dim {m.get_sentence_embedding_dimension()}")
    print(m)
