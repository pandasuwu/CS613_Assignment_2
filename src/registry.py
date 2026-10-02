from pathlib import Path

from src.config import EMBEDDINGS_CACHE_DIR, RESULTS_DIR

# 1. Models Taxonomy
EMBEDDING_MODELS: list[str] = [
    "google/embeddinggemma-300m",
    "Qwen/Qwen3-Embedding-0.6B",
]

BASE_MODELS: list[str] = [
    "google/gemma-3-1b-pt",
    "Qwen/Qwen3.5-0.8B-Base",
]

BASE_POOLING_MODES: list[str] = [
    "mean_pooling",
    "last_token_pooling",
]

# 2. Benchmarks and Datasets (MTEB v2 English)
RETRIEVAL_CORE_TASKS: list[str] = [
    "FiQA2018",
    "ArguAna",
    "SCIDOCS",
    "TRECCOVID",
]

RETRIEVAL_SCALEUP_TASKS: list[str] = [
    "FEVERHardNegatives",
    "ClimateFEVERHardNegatives",
    "HotpotQAHardNegatives",
    "Touche2020Retrieval.v3",
    "CQADupstackGamingRetrieval",
    "CQADupstackUnixRetrieval",
]

SIMILARITY_CORE_TASKS: list[str] = [
    "STSBenchmark",
    "SICK-R",
    "STS22.v2",
]

# 3. Compression Ladder Target Dimensions
COMPRESSION_LADDER_K: list[int] = [512, 256, 128, 64]


def sanitize_model_id(model_id: str) -> str:
    """Convert model identifier to safe directory name (e.g. google/embeddinggemma-300m -> google_embeddinggemma-300m)."""
    return model_id.replace("/", "_")


def is_base_model(model_id: str) -> bool:
    """Check whether a model belongs to the base LLM tier."""
    return model_id in BASE_MODELS


def get_raw_cache_dir(
    dataset: str, model_id: str, pooling: str | None = None
) -> Path:
    """Resolve directory path for raw cached embeddings."""
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
    """Resolve directory path for transformed cached embeddings."""
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
    """Resolve directory path for atomic result CSV leaves."""
    clean_id = sanitize_model_id(model_id)
    if is_base_model(model_id):
        pool_name = pooling if pooling else "mean_pooling"
        return RESULTS_DIR / track / dataset / "base" / clean_id / pool_name / regime
    return RESULTS_DIR / track / dataset / "embedding" / clean_id / regime
