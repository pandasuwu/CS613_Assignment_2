from pathlib import Path

from src.config import EMBEDDINGS_CACHE_DIR, RESULTS_DIR

# 1. Models Taxonomy
# Contrastively trained dedicated sentence encoders:
EMBEDDING_MODELS: list[str] = [
    "google/embeddinggemma-300m",  # 308M parameters, native dimension d=768
    "Qwen/Qwen3-Embedding-0.6B",   # 600M parameters, native dimension d=1024
]

# Autoregressive next-token causal base language models (un-fine-tuned):
BASE_MODELS: list[str] = [
    "google/gemma-3-1b-pt",        # 1.0B parameters, native dimension d=1152
    "Qwen/Qwen3.5-0.8B-Base",      # 0.8B parameters, native dimension d=1024
]

# Base model token pooling modes:
BASE_POOLING_MODES: list[str] = [
    "mean_pooling",        # Average across active non-padding token positions
    "last_token_pooling",  # Hidden state of the final non-padding token
]

# 2. Benchmarks and Tasks (MTEB v2 English)
# Information retrieval benchmarks (asymmetric search):
RETRIEVAL_TASKS: list[str] = [
    "FiQA2018",   # Financial question answering (57k docs, 648 queries)
    "ArguAna",    # Counter-argument retrieval (8.6k docs, 1406 queries)
    "SCIDOCS",    # Scientific citation search (25.6k docs, 1000 queries)
    "TRECCOVID",  # Biomedical literature search (171k docs, 50 queries)
]

# Semantic textual similarity benchmarks (symmetric pair scoring):
SIMILARITY_TASKS: list[str] = [
    "STSBenchmark",  # Multi-domain sentence similarity synthesis (1379 pairs)
    "SICK-R",        # Relatedness and entailment (9927 pairs)
    "STS22.v2",      # Long-form cross-domain similarity (197 pairs)
]

# 3. Compression Ladder Target Dimensions (d -> k)
COMPRESSION_LADDER_K: list[int] = [512, 256, 128, 64]


def sanitize_model_id(model_id: str) -> str:
    """Convert Hugging Face model identifier to safe directory name.

    Example:
        'google/embeddinggemma-300m' -> 'google_embeddinggemma-300m'
    """
    return model_id.replace("/", "_")


def is_base_model(model_id: str) -> bool:
    """Check whether a model belongs to the un-fine-tuned base causal LLM tier."""
    return model_id in BASE_MODELS


def get_raw_cache_dir(
    dataset: str, model_id: str, pooling: str | None = None
) -> Path:
    """Resolve directory path for raw, un-transformed cached embeddings.

    Path Schema:
        Embedding models: cache/embeddings/{dataset}/embedding/{clean_id}/raw/
        Base models:      cache/embeddings/{dataset}/base/{clean_id}/{pooling}/raw/
    """
    clean_id = sanitize_model_id(model_id)
    if is_base_model(model_id):
        pool_name = pooling if pooling else "mean_pooling"
        return EMBEDDINGS_CACHE_DIR / dataset / "base" / clean_id / pool_name / "raw"
    return EMBEDDINGS_CACHE_DIR / dataset / "embedding" / clean_id / "raw"


def get_transformed_cache_dir(
    dataset: str,
    model_id: str,
    regime: str,
    method: str,
    pooling: str | None = None,
) -> Path:
    """Resolve directory path for post-processed cached embeddings.

    Path Schema:
        Embedding models: cache/embeddings/{dataset}/embedding/{clean_id}/{regime}/{method}/
        Base models:      cache/embeddings/{dataset}/base/{clean_id}/{pooling}/{regime}/{method}/
    """
    clean_id = sanitize_model_id(model_id)
    if is_base_model(model_id):
        pool_name = pooling if pooling else "mean_pooling"
        return (
            EMBEDDINGS_CACHE_DIR
            / dataset
            / "base"
            / clean_id
            / pool_name
            / regime
            / method
        )
    return EMBEDDINGS_CACHE_DIR / dataset / "embedding" / clean_id / regime / method


def get_result_dir(
    track: str,
    dataset: str,
    model_id: str,
    regime: str,
    pooling: str | None = None,
) -> Path:
    """Resolve directory path for atomic result CSV leaves.

    Path Schema:
        results/{track}/{dataset}/{tier}/{clean_id}/[{pooling}/]{regime}/
    """
    clean_id = sanitize_model_id(model_id)
    if is_base_model(model_id):
        pool_name = pooling if pooling else "mean_pooling"
        return RESULTS_DIR / track / dataset / "base" / clean_id / pool_name / regime
    return RESULTS_DIR / track / dataset / "embedding" / clean_id / regime
