import argparse
import json

import torch

from src.config import get_device, logger
from src.registry import (
    BASE_MODELS,
    BASE_POOLING_MODES,
    COMPRESSION_LADDER_K,
    EMBEDDING_MODELS,
    RETRIEVAL_TASKS,
    SIMILARITY_TASKS,
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
    """Apply all full-dimension and compression transformations to raw cached embeddings."""
    raw_dir = get_raw_cache_dir(task_name, model_id, pooling=pooling)
    is_retrieval = (raw_dir / "corpus.pt").exists()

    if is_retrieval:
        x1 = torch.load(raw_dir / "corpus.pt", weights_only=True)
        x2 = torch.load(raw_dir / "queries.pt", weights_only=True)
        out_name1, out_name2 = "corpus.pt", "queries.pt"
        # In asymmetric IR, parameters (mean, covariance, basis) are estimated strictly on the
        # document corpus, mirroring offline document indexing in production search architectures.
        calib_data = x1
    else:
        if not (raw_dir / "sentences1.pt").exists():
            logger.warning("Raw embeddings not found in %s. Run encode.py first.", raw_dir)
            return
        x1 = torch.load(raw_dir / "sentences1.pt", weights_only=True)
        x2 = torch.load(raw_dir / "sentences2.pt", weights_only=True)
        out_name1, out_name2 = "sentences1.pt", "sentences2.pt"
        # In symmetric STS, sentence pairs originate from the identical distribution, so both
        # sentence sets are pooled to form the shared calibration manifold without artificial asymmetry.
        calib_data = torch.cat([x1, x2], dim=0)

    device = get_device()
    x1 = x1.to(device)
    x2 = x2.to(device)
    calib_data = calib_data.to(device)

    native_d = x1.shape[1]
    logger.info("Transforming %s on %s (native_d=%d, pooling=%s, device=%s)", model_id, task_name, native_d, pooling, device)

    # ==========================================
    # 1. Full-Dimension Transforms (d -> d)
    # ==========================================
    full_transforms = {
        "baseline": lambda a, b: transform_baseline(a, b),
        "standardization": lambda a, b: transform_standardization(a, b, calib_data),
        "r1": lambda a, b: transform_r1(a, b, calib_data),
        "r2": lambda a, b: transform_r2(a, b, calib_data),
        "soft_zca": lambda a, b: transform_soft_zca(a, b, calib_data, epsilon=1e-4),
        "abtt_1": lambda a, b: transform_abtt(a, b, calib_data, d_components=1),
        "abtt_2": lambda a, b: transform_abtt(a, b, calib_data, d_components=2),
        "abtt_3": lambda a, b: transform_abtt(a, b, calib_data, d_components=3),
        "rand": lambda a, b: transform_rand(a, b),
        "mc": lambda a, b: transform_mc(a, b, calib_data),
    }

    for method_name, func in full_transforms.items():
        out_dir = get_transformed_cache_dir(task_name, model_id, "full", method_name, pooling=pooling)
        out_dir.mkdir(parents=True, exist_ok=True)
        f1 = out_dir / out_name1
        f2 = out_dir / out_name2

        if f1.exists() and f2.exists() and not overwrite:
            continue

        x1_trans, x2_trans = func(x1, x2)
        torch.save(x1_trans.cpu(), f1)
        torch.save(x2_trans.cpu(), f2)

    # ==========================================
    # 2. Compression Transforms (d -> k)
    # ==========================================
    for k in COMPRESSION_LADDER_K:
        if k >= native_d:
            continue

        # Prefix coordinate slicing (Matryoshka representation baseline)
        out_prefix = get_transformed_cache_dir(task_name, model_id, "compression", f"prefix_k{k}", pooling=pooling)
        out_prefix.mkdir(parents=True, exist_ok=True)
        f1 = out_prefix / out_name1
        f2 = out_prefix / out_name2
        if overwrite or not (f1.exists() and f2.exists()):
            xp1, xp2 = transform_prefix(x1, x2, target_dim=k)
            torch.save(xp1.cpu(), f1)
            torch.save(xp2.cpu(), f2)

        # Uniform random coordinate sampling (Negative scientific control)
        out_rand = get_transformed_cache_dir(task_name, model_id, "compression", f"random_truncation_k{k}", pooling=pooling)
        out_rand.mkdir(parents=True, exist_ok=True)
        f1 = out_rand / out_name1
        f2 = out_rand / out_name2
        if overwrite or not (f1.exists() and f2.exists()):
            xr1, xr2 = transform_random_truncation(x1, x2, target_dim=k)
            torch.save(xr1.cpu(), f1)
            torch.save(xr2.cpu(), f2)

        # Principal Component Analysis
        out_pca = get_transformed_cache_dir(task_name, model_id, "compression", f"pca_k{k}", pooling=pooling)
        out_pca.mkdir(parents=True, exist_ok=True)
        f1 = out_pca / out_name1
        f2 = out_pca / out_name2
        if overwrite or not (f1.exists() and f2.exists()):
            x_pca1, x_pca2 = transform_pca(x1, x2, calib_data, target_dim=k)
            torch.save(x_pca1.cpu(), f1)
            torch.save(x_pca2.cpu(), f2)

        # Truncated Whitening-k
        out_white = get_transformed_cache_dir(task_name, model_id, "compression", f"whitening_k{k}", pooling=pooling)
        out_white.mkdir(parents=True, exist_ok=True)
        f1 = out_white / out_name1
        f2 = out_white / out_name2
        if overwrite or not (f1.exists() and f2.exists()):
            xw1, xw2 = transform_whitening(x1, x2, calib_data, target_dim=k)
            torch.save(xw1.cpu(), f1)
            torch.save(xw2.cpu(), f2)

        # SpecTemp (Adaptive SNR-tempered spectral projection)
        out_spec = get_transformed_cache_dir(task_name, model_id, "compression", f"spectemp_k{k}", pooling=pooling)
        out_spec.mkdir(parents=True, exist_ok=True)
        f1 = out_spec / out_name1
        f2 = out_spec / out_name2
        if overwrite or not (f1.exists() and f2.exists()):
            xs1, xs2, gamma = transform_spectemp(x1, x2, calib_data, target_dim=k)
            torch.save(xs1.cpu(), f1)
            torch.save(xs2.cpu(), f2)
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
    tasks_to_run = [args.task] if args.task else (RETRIEVAL_TASKS + SIMILARITY_TASKS)

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
