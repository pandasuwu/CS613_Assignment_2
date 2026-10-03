import argparse
import json
from pathlib import Path
from typing import Any

import torch

from src.config import logger
from src.metrics.similarity import compute_pearson_r, compute_spearman_rho
from src.registry import (
    BASE_MODELS,
    BASE_POOLING_MODES,
    COMPRESSION_LADDER_K,
    EMBEDDING_MODELS,
    SIMILARITY_CORE_TASKS,
    get_raw_cache_dir,
    get_result_dir,
    get_transformed_cache_dir,
    is_base_model,
)


def evaluate_similarity_for_method(
    s1: torch.Tensor,
    s2: torch.Tensor,
    gold_scores: list[float],
    out_csv: Path,
    extra_cols: dict[str, Any] | None = None,
) -> dict[str, float]:
    """Compute STS similarity metrics (Spearman rho, Pearson r) and save atomic CSV."""
    # Row-wise cosine similarity between paired sentences
    sims = torch.sum(s1 * s2, dim=1)

    spearman_rho = compute_spearman_rho(sims, gold_scores)
    pearson_r = compute_pearson_r(sims, gold_scores)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", encoding="utf-8") as f:
        if extra_cols:
            header_keys = list(extra_cols.keys()) + ["spearman_rho", "pearson_r"]
            header_vals = [str(extra_cols[k]) for k in extra_cols] + [f"{spearman_rho:.4f}", f"{pearson_r:.4f}"]
            f.write(",".join(header_keys) + "\n")
            f.write(",".join(header_vals) + "\n")
        else:
            f.write("metric,value\n")
            f.write(f"spearman_rho,{spearman_rho:.4f}\n")
            f.write(f"pearson_r,{pearson_r:.4f}\n")

    return {"spearman_rho": spearman_rho, "pearson_r": pearson_r}


def evaluate_similarity_combination(
    model_id: str,
    task_name: str,
    pooling: str | None = None,
    overwrite: bool = False,
) -> None:
    """Evaluate STS benchmark metrics across all full and compression transforms."""
    raw_dir = get_raw_cache_dir(task_name, model_id, pooling=pooling)
    meta_file = raw_dir / "meta.json"
    if not meta_file.exists():
        logger.warning("Metadata not found in %s. Run encode.py first.", raw_dir)
        return

    with open(meta_file, encoding="utf-8") as f:
        meta = json.load(f)
    gold_scores = meta["scores"]

    # 1. Full-Dimension Transforms
    full_methods = [
        "baseline", "standardization", "r1", "r2", "soft_zca",
        "abtt_1", "abtt_2", "abtt_3", "rand", "mc"
    ]
    for method in full_methods:
        csv_path = get_result_dir("similarity", task_name, model_id, "full", pooling=pooling) / f"{method}.csv"
        if csv_path.exists() and not overwrite:
            continue

        trans_dir = get_transformed_cache_dir(task_name, model_id, "full", method, pooling=pooling)
        s1_file = trans_dir / "sentences1.pt"
        s2_file = trans_dir / "sentences2.pt"
        if not (s1_file.exists() and s2_file.exists()):
            continue

        s1 = torch.load(s1_file, weights_only=True)
        s2 = torch.load(s2_file, weights_only=True)

        res = evaluate_similarity_for_method(s1, s2, gold_scores, csv_path)
        logger.info(
            "[%s | %s | full | %s] Spearman=%.2f  Pearson=%.2f",
            task_name, model_id, method, res["spearman_rho"], res["pearson_r"]
        )

    # 2. Compression Transforms
    compression_methods = ["prefix", "random_truncation", "pca", "whitening", "spectemp"]
    for method in compression_methods:
        for k in COMPRESSION_LADDER_K:
            sub_name = f"{method}_k{k}"
            csv_path = get_result_dir("similarity", task_name, model_id, "compression", pooling=pooling) / f"{sub_name}.csv"
            if csv_path.exists() and not overwrite:
                continue

            trans_dir = get_transformed_cache_dir(task_name, model_id, "compression", sub_name, pooling=pooling)
            s1_file = trans_dir / "sentences1.pt"
            s2_file = trans_dir / "sentences2.pt"
            if not (s1_file.exists() and s2_file.exists()):
                continue

            s1 = torch.load(s1_file, weights_only=True)
            s2 = torch.load(s2_file, weights_only=True)

            gamma_val: float | None = None
            if (trans_dir / "gamma.json").exists():
                with open(trans_dir / "gamma.json", encoding="utf-8") as gf:
                    gamma_val = json.load(gf).get("gamma")

            extra = {"k": k, "gamma": f"{gamma_val:.4f}" if gamma_val is not None else ""}
            res = evaluate_similarity_for_method(s1, s2, gold_scores, csv_path, extra_cols=extra)
            logger.info(
                "[%s | %s | compression | %s] Spearman=%.2f  Pearson=%.2f",
                task_name, model_id, sub_name, res["spearman_rho"], res["pearson_r"]
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 3: Semantic Textual Similarity (STS) evaluation (Spearman rho, Pearson r).")
    parser.add_argument("--model", type=str, default=None, help="Model ID. Default: all models.")
    parser.add_argument("--task", type=str, default=None, help="Task name. Default: all STS tasks.")
    parser.add_argument("--pooling", type=str, default=None, help="Pooling mode for base LLMs.")
    parser.add_argument("--overwrite", action="store_true", help="Re-evaluate and overwrite existing CSVs.")

    args = parser.parse_args()

    models_to_run = [args.model] if args.model else (EMBEDDING_MODELS + BASE_MODELS)
    tasks_to_run = [args.task] if args.task else SIMILARITY_CORE_TASKS

    for model_id in models_to_run:
        poolings = [args.pooling] if args.pooling else (BASE_POOLING_MODES if is_base_model(model_id) else [None])
        for task_name in tasks_to_run:
            for pool_mode in poolings:
                evaluate_similarity_combination(
                    model_id=model_id,
                    task_name=task_name,
                    pooling=pool_mode,
                    overwrite=args.overwrite,
                )


if __name__ == "__main__":
    main()
