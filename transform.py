import argparse
import json

import torch

from src.config import get_device, logger
from src.registry import (
    BASE_MODELS,
    BASE_POOLING_MODES,
    COMPRESSION_LADDER_K,
    EMBEDDING_MODELS,
    RETRIEVAL_CORE_TASKS,
    SIMILARITY_CORE_TASKS,
    get_raw_cache_dir,
    get_transformed_cache_dir,
    is_base_model,
)
from src.transforms import (
    transform_abtt,
    transform_baseline,
    transform_mc,
    transform_pca,
    transform_prefix,
    transform_r1,
    transform_r2,
    transform_rand,
    transform_random_truncation,
    transform_soft_zca,
    transform_spectemp,
    transform_standardization,
    transform_whitening,
)


def run_transforms_for_combination(
    model_id: str,
    task_name: str,
    pooling: str | None = None,
    overwrite: bool = False,
) -> None:
    """Apply all full and compression transformations to raw cached embeddings."""
    raw_dir = get_raw_cache_dir(task_name, model_id, pooling=pooling)
    is_retrieval = (raw_dir / "corpus.pt").exists()

    if is_retrieval:
        corpus = torch.load(raw_dir / "corpus.pt", weights_only=True)
        queries = torch.load(raw_dir / "queries.pt", weights_only=True)
    else:
        if not (raw_dir / "sentences1.pt").exists():
            logger.warning("Raw embeddings not found in %s. Run encode.py first.", raw_dir)
            return
        corpus = torch.load(raw_dir / "sentences1.pt", weights_only=True)
        queries = torch.load(raw_dir / "sentences2.pt", weights_only=True)

    device = get_device()
    corpus = corpus.to(device)
    queries = queries.to(device)

    # For STS, symmetrically pool all sentences across both splits; for retrieval, fit strictly on corpus documents
    fit_data = None if is_retrieval else torch.cat([corpus, queries], dim=0)

    native_d = corpus.shape[1]
    logger.info("Transforming %s on %s (native_d=%d, pooling=%s, device=%s)", model_id, task_name, native_d, pooling, device)

    # ==========================================
    # 1. Full-Dimension Transforms (d -> d)
    # ==========================================
    full_transforms = {
        "baseline": lambda c, q: transform_baseline(c, q),
        "standardization": lambda c, q: transform_standardization(c, q, fit_data=fit_data),
        "r1": lambda c, q: transform_r1(c, q, fit_data=fit_data),
        "r2": lambda c, q: transform_r2(c, q, fit_data=fit_data),
        "soft_zca": lambda c, q: transform_soft_zca(c, q, epsilon=1e-4, fit_data=fit_data),
        "abtt_1": lambda c, q: transform_abtt(c, q, d_components=1, fit_data=fit_data),
        "abtt_2": lambda c, q: transform_abtt(c, q, d_components=2, fit_data=fit_data),
        "abtt_3": lambda c, q: transform_abtt(c, q, d_components=3, fit_data=fit_data),
        "rand": lambda c, q: transform_rand(c, q),
        "mc": lambda c, q: transform_mc(c, q),
    }

    for method_name, func in full_transforms.items():
        out_dir = get_transformed_cache_dir(task_name, model_id, "full", method_name, pooling=pooling)
        out_dir.mkdir(parents=True, exist_ok=True)
        c_file = out_dir / ("corpus.pt" if is_retrieval else "sentences1.pt")
        q_file = out_dir / ("queries.pt" if is_retrieval else "sentences2.pt")

        if c_file.exists() and q_file.exists() and not overwrite:
            continue

        c_trans, q_trans = func(corpus, queries)
        torch.save(c_trans.cpu(), c_file)
        torch.save(q_trans.cpu(), q_file)

    # ==========================================
    # 2. Compression Transforms (d -> k)
    # ==========================================
    for k in COMPRESSION_LADDER_K:
        if k >= native_d:
            continue

        # Prefix
        out_prefix = get_transformed_cache_dir(task_name, model_id, "compression", f"prefix_k{k}", pooling=pooling)
        out_prefix.mkdir(parents=True, exist_ok=True)
        c_file = out_prefix / ("corpus.pt" if is_retrieval else "sentences1.pt")
        q_file = out_prefix / ("queries.pt" if is_retrieval else "sentences2.pt")
        if overwrite or not (c_file.exists() and q_file.exists()):
            cp, qp = transform_prefix(corpus, queries, target_dim=k)
            torch.save(cp.cpu(), c_file)
            torch.save(qp.cpu(), q_file)

        # Random Truncation (negative control)
        out_rand = get_transformed_cache_dir(task_name, model_id, "compression", f"random_truncation_k{k}", pooling=pooling)
        out_rand.mkdir(parents=True, exist_ok=True)
        c_file = out_rand / ("corpus.pt" if is_retrieval else "sentences1.pt")
        q_file = out_rand / ("queries.pt" if is_retrieval else "sentences2.pt")
        if overwrite or not (c_file.exists() and q_file.exists()):
            cr, qr = transform_random_truncation(corpus, queries, target_dim=k)
            torch.save(cr.cpu(), c_file)
            torch.save(qr.cpu(), q_file)

        # PCA
        out_pca = get_transformed_cache_dir(task_name, model_id, "compression", f"pca_k{k}", pooling=pooling)
        out_pca.mkdir(parents=True, exist_ok=True)
        c_file = out_pca / ("corpus.pt" if is_retrieval else "sentences1.pt")
        q_file = out_pca / ("queries.pt" if is_retrieval else "sentences2.pt")
        if overwrite or not (c_file.exists() and q_file.exists()):
            cpca, qpca = transform_pca(corpus, queries, target_dim=k, fit_data=fit_data)
            torch.save(cpca.cpu(), c_file)
            torch.save(qpca.cpu(), q_file)

        # Whitening
        out_white = get_transformed_cache_dir(task_name, model_id, "compression", f"whitening_k{k}", pooling=pooling)
        out_white.mkdir(parents=True, exist_ok=True)
        c_file = out_white / ("corpus.pt" if is_retrieval else "sentences1.pt")
        q_file = out_white / ("queries.pt" if is_retrieval else "sentences2.pt")
        if overwrite or not (c_file.exists() and q_file.exists()):
            cw, qw = transform_whitening(corpus, queries, target_dim=k, fit_data=fit_data)
            torch.save(cw.cpu(), c_file)
            torch.save(qw.cpu(), q_file)

        # SpecTemp
        out_spec = get_transformed_cache_dir(task_name, model_id, "compression", f"spectemp_k{k}", pooling=pooling)
        out_spec.mkdir(parents=True, exist_ok=True)
        c_file = out_spec / ("corpus.pt" if is_retrieval else "sentences1.pt")
        q_file = out_spec / ("queries.pt" if is_retrieval else "sentences2.pt")
        if overwrite or not (c_file.exists() and q_file.exists()):
            cs, qs, gamma = transform_spectemp(corpus, queries, target_dim=k, fit_data=fit_data)
            torch.save(cs.cpu(), c_file)
            torch.save(qs.cpu(), q_file)
            with open(out_spec / "gamma.json", "w", encoding="utf-8") as f:
                json.dump({"gamma": gamma}, f)

    logger.info("Completed all transforms for %s on %s", model_id, task_name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 2: Post-processing transformations and caching.")
    parser.add_argument("--model", type=str, default=None, help="Model ID. Default: all models.")
    parser.add_argument("--task", type=str, default=None, help="Task name. Default: all tasks.")
    parser.add_argument("--pooling", type=str, default=None, help="Pooling mode for base LLMs.")
    parser.add_argument("--overwrite", action="store_true", help="Re-transform and overwrite existing cache.")

    args = parser.parse_args()

    models_to_run = [args.model] if args.model else (EMBEDDING_MODELS + BASE_MODELS)
    tasks_to_run = [args.task] if args.task else (RETRIEVAL_CORE_TASKS + SIMILARITY_CORE_TASKS)

    for model_id in models_to_run:
        poolings = [args.pooling] if args.pooling else (BASE_POOLING_MODES if is_base_model(model_id) else [None])
        for task_name in tasks_to_run:
            for pool_mode in poolings:
                run_transforms_for_combination(
                    model_id=model_id,
                    task_name=task_name,
                    pooling=pool_mode,
                    overwrite=args.overwrite,
                )


if __name__ == "__main__":
    main()
