import argparse
import json
from typing import Any

import torch

from src.config import logger
from src.data import load_retrieval_dataset, load_similarity_dataset
from src.models import (
    encode_base_texts,
    encode_embedding_corpus,
    encode_embedding_queries,
    encode_embedding_sentences,
    load_base_model,
    load_embedding_model,
)
from src.registry import (
    BASE_MODELS,
    BASE_POOLING_MODES,
    EMBEDDING_MODELS,
    RETRIEVAL_CORE_TASKS,
    SIMILARITY_CORE_TASKS,
    get_raw_cache_dir,
    is_base_model,
)


def is_retrieval_task(task_name: str) -> bool:
    """Identify if a task is an information retrieval benchmark."""
    return task_name in (RETRIEVAL_CORE_TASKS + [
        "FEVERHardNegatives", "ClimateFEVERHardNegatives", "HotpotQAHardNegatives",
        "Touche2020Retrieval.v3", "CQADupstackGamingRetrieval", "CQADupstackUnixRetrieval", "SciFact"
    ])


def encode_single_combination(
    model_id: str,
    task_name: str,
    pooling: str | None = None,
    overwrite: bool = False,
    batch_size_docs: int = 16,
    batch_size_queries: int = 64,
    loaded_model: Any = None,
    loaded_tokenizer: Any = None,
) -> None:
    """Encode texts for a single model and task combination, saving FP32 tensors to raw cache."""
    cache_dir = get_raw_cache_dir(task_name, model_id, pooling=pooling)
    cache_dir.mkdir(parents=True, exist_ok=True)

    meta_file = cache_dir / "meta.json"
    retrieval = is_retrieval_task(task_name)

    if retrieval:
        corpus_file = cache_dir / "corpus.pt"
        queries_file = cache_dir / "queries.pt"
        if corpus_file.exists() and queries_file.exists() and meta_file.exists() and not overwrite:
            logger.info("Skipping %s on %s (already cached in %s)", model_id, task_name, cache_dir)
            return
    else:
        s1_file = cache_dir / "sentences1.pt"
        s2_file = cache_dir / "sentences2.pt"
        if s1_file.exists() and s2_file.exists() and meta_file.exists() and not overwrite:
            logger.info("Skipping %s on %s (already cached in %s)", model_id, task_name, cache_dir)
            return

    logger.info("Encoding %s on %s (pooling=%s)", model_id, task_name, pooling)

    # 1. Load Data
    if retrieval:
        ds_ret = load_retrieval_dataset(task_name)
    else:
        ds_sim = load_similarity_dataset(task_name)

    # 2. Encode
    if is_base_model(model_id):
        if loaded_model is None or loaded_tokenizer is None:
            model, tokenizer = load_base_model(model_id)
        else:
            model, tokenizer = loaded_model, loaded_tokenizer

        pool_mode = "mean" if (pooling == "mean_pooling" or pooling is None) else "last_token"

        if retrieval:
            corpus_tensor = encode_base_texts(ds_ret.doc_texts, model, tokenizer, pooling=pool_mode, batch_size=batch_size_docs)
            queries_tensor = encode_base_texts(ds_ret.query_texts, model, tokenizer, pooling=pool_mode, batch_size=batch_size_queries)
            torch.save(corpus_tensor, cache_dir / "corpus.pt")
            torch.save(queries_tensor, cache_dir / "queries.pt")
            with open(meta_file, "w", encoding="utf-8") as f:
                json.dump({"qids": ds_ret.query_ids, "dids": ds_ret.doc_ids, "qrels": ds_ret.qrels}, f)
        else:
            s1_tensor = encode_base_texts(ds_sim.sentences1, model, tokenizer, pooling=pool_mode, batch_size=batch_size_queries)
            s2_tensor = encode_base_texts(ds_sim.sentences2, model, tokenizer, pooling=pool_mode, batch_size=batch_size_queries)
            torch.save(s1_tensor, cache_dir / "sentences1.pt")
            torch.save(s2_tensor, cache_dir / "sentences2.pt")
            with open(meta_file, "w", encoding="utf-8") as f:
                json.dump({"scores": ds_sim.scores}, f)
    else:
        if loaded_model is None:
            model = load_embedding_model(model_id)
        else:
            model = loaded_model

        if retrieval:
            corpus_tensor = encode_embedding_corpus(ds_ret.doc_texts, model, model_id, batch_size=batch_size_docs)
            queries_tensor = encode_embedding_queries(ds_ret.query_texts, model, model_id, instruction=ds_ret.instruction, batch_size=batch_size_queries)
            torch.save(corpus_tensor, cache_dir / "corpus.pt")
            torch.save(queries_tensor, cache_dir / "queries.pt")
            with open(meta_file, "w", encoding="utf-8") as f:
                json.dump({"qids": ds_ret.query_ids, "dids": ds_ret.doc_ids, "qrels": ds_ret.qrels}, f)
        else:
            s1_tensor = encode_embedding_sentences(ds_sim.sentences1, model, batch_size=batch_size_queries)
            s2_tensor = encode_embedding_sentences(ds_sim.sentences2, model, batch_size=batch_size_queries)
            torch.save(s1_tensor, cache_dir / "sentences1.pt")
            torch.save(s2_tensor, cache_dir / "sentences2.pt")
            with open(meta_file, "w", encoding="utf-8") as f:
                json.dump({"scores": ds_sim.scores}, f)

    logger.info("Saved raw cached representations to %s", cache_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 1: Encode and cache raw representations.")
    parser.add_argument("--model", type=str, default=None, help="Model ID (e.g. google/embeddinggemma-300m). Default: all models.")
    parser.add_argument("--task", type=str, default=None, help="Task name (e.g. FiQA2018, STSBenchmark). Default: all core tasks.")
    parser.add_argument("--pooling", type=str, default=None, help="Pooling mode for base LLMs (mean_pooling or last_token_pooling).")
    parser.add_argument("--overwrite", action="store_true", help="Re-encode and overwrite existing cache.")
    parser.add_argument("--batch-size-docs", type=int, default=16, help="Batch size for encoding documents.")
    parser.add_argument("--batch-size-queries", type=int, default=64, help="Batch size for encoding queries/sentences.")

    args = parser.parse_args()

    models_to_run = [args.model] if args.model else (EMBEDDING_MODELS + BASE_MODELS)
    tasks_to_run = [args.task] if args.task else (RETRIEVAL_CORE_TASKS + SIMILARITY_CORE_TASKS)

    for model_id in models_to_run:
        # Load model once outside task loop to avoid reloading weights repeatedly
        if is_base_model(model_id):
            loaded_model, loaded_tokenizer = load_base_model(model_id)
            poolings = [args.pooling] if args.pooling else BASE_POOLING_MODES
        else:
            loaded_model = load_embedding_model(model_id)
            loaded_tokenizer = None
            poolings = [None]

        for task_name in tasks_to_run:
            for pool_mode in poolings:
                encode_single_combination(
                    model_id=model_id,
                    task_name=task_name,
                    pooling=pool_mode,
                    overwrite=args.overwrite,
                    batch_size_docs=args.batch_size_docs,
                    batch_size_queries=args.batch_size_queries,
                    loaded_model=loaded_model,
                    loaded_tokenizer=loaded_tokenizer,
                )


if __name__ == "__main__":
    main()
