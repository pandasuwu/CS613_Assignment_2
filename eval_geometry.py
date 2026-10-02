import argparse
from pathlib import Path
from typing import Any

import torch

from src.config import logger
from src.metrics.geometry import (
    compute_average_cosine,
    compute_centroid_norm,
    compute_isoscore,
    compute_mev,
    compute_nid,
)
from src.registry import (
    BASE_MODELS,
    BASE_POOLING_MODES,
    COMPRESSION_LADDER_K,
    EMBEDDING_MODELS,
    RETRIEVAL_CORE_TASKS,
    SIMILARITY_CORE_TASKS,
    get_raw_cache_dir,
    get_result_dir,
    get_transformed_cache_dir,
    is_base_model,
)


def evaluate_geometry_for_tensor(
    tensor: torch.Tensor,
    out_csv: Path,
    extra_cols: dict[str, Any] | None = None,
) -> dict[str, float]:
    """Compute the 5 intrinsic geometry metrics and write atomic CSV."""
    centroid_norm = compute_centroid_norm(tensor)
    average_cosine = compute_average_cosine(tensor)
    mev = compute_mev(tensor)
    nid = compute_nid(tensor)
    isoscore = compute_isoscore(tensor)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", encoding="utf-8") as f:
        if extra_cols:
            header_keys = list(extra_cols.keys()) + [
                "centroid_norm", "average_cosine", "mev_top1", "nid_spectral_entropy", "isoscore"
            ]
            header_vals = [str(extra_cols[k]) for k in extra_cols] + [
                f"{centroid_norm:.6f}", f"{average_cosine:.6f}", f"{mev:.6f}", f"{nid:.6f}", f"{isoscore:.6f}"
            ]
            f.write(",".join(header_keys) + "\n")
            f.write(",".join(header_vals) + "\n")
        else:
            f.write("metric,value\n")
            f.write(f"centroid_norm,{centroid_norm:.6f}\n")
            f.write(f"average_cosine,{average_cosine:.6f}\n")
            f.write(f"mev_top1,{mev:.6f}\n")
            f.write(f"nid_spectral_entropy,{nid:.6f}\n")
            f.write(f"isoscore,{isoscore:.6f}\n")

    return {
        "centroid_norm": centroid_norm,
        "average_cosine": average_cosine,
        "mev_top1": mev,
        "nid_spectral_entropy": nid,
        "isoscore": isoscore,
    }


def evaluate_geometry_combination(
    model_id: str,
    task_name: str,
    pooling: str | None = None,
    overwrite: bool = False,
) -> None:
    """Evaluate intrinsic geometry across baseline, full-dimension, and compression transforms."""
    raw_dir = get_raw_cache_dir(task_name, model_id, pooling=pooling)
    is_retrieval = (raw_dir / "corpus.pt").exists()
    pt_name = "corpus.pt" if is_retrieval else "sentences1.pt"

    # 1. Full-Dimension Transforms (including baseline)
    full_methods = [
        "baseline", "standardization", "r1", "r2", "soft_zca",
        "abtt_1", "abtt_2", "abtt_3", "rand", "mc"
    ]
    for method in full_methods:
        csv_path = get_result_dir("geometry", task_name, model_id, "full", pooling=pooling) / f"{method}.csv"
        if csv_path.exists() and not overwrite:
            continue

        trans_dir = get_transformed_cache_dir(task_name, model_id, "full", method, pooling=pooling)
        tensor_file = trans_dir / pt_name
        if not tensor_file.exists():
            continue

        tensor = torch.load(tensor_file, weights_only=True)
        res = evaluate_geometry_for_tensor(tensor, csv_path)
        logger.info(
            "[%s | %s | full | %s] Centroid=%.4f  AvgCos=%.4f  MEV=%.4f  NID=%.4f  IsoScore=%.4f",
            task_name, model_id, method,
            res["centroid_norm"], res["average_cosine"], res["mev_top1"], res["nid_spectral_entropy"], res["isoscore"]
        )

    # 2. Compression Transforms
    compression_methods = ["prefix", "random_truncation", "pca", "whitening", "spectemp", "r2_spectemp"]
    for method in compression_methods:
        for k in COMPRESSION_LADDER_K:
            sub_name = f"{method}_k{k}"
            csv_path = get_result_dir("geometry", task_name, model_id, "compression", pooling=pooling) / f"{sub_name}.csv"
            if csv_path.exists() and not overwrite:
                continue

            trans_dir = get_transformed_cache_dir(task_name, model_id, "compression", sub_name, pooling=pooling)
            tensor_file = trans_dir / pt_name
            if not tensor_file.exists():
                continue

            tensor = torch.load(tensor_file, weights_only=True)
            res = evaluate_geometry_for_tensor(tensor, csv_path, extra_cols={"k": k})
            logger.info(
                "[%s | %s | compression | %s] Centroid=%.4f  AvgCos=%.4f  MEV=%.4f  NID=%.4f  IsoScore=%.4f",
                task_name, model_id, sub_name,
                res["centroid_norm"], res["average_cosine"], res["mev_top1"], res["nid_spectral_entropy"], res["isoscore"]
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 3: Intrinsic space geometry diagnostics (Centroid, AvgCos, MEV, NID, IsoScore).")
    parser.add_argument("--model", type=str, default=None, help="Model ID. Default: all models.")
    parser.add_argument("--task", type=str, default=None, help="Task name. Default: all tasks.")
    parser.add_argument("--pooling", type=str, default=None, help="Pooling mode for base LLMs.")
    parser.add_argument("--overwrite", action="store_true", help="Re-evaluate and overwrite existing CSVs.")

    args = parser.parse_args()

    models_to_run = [args.model] if args.model else (EMBEDDING_MODELS + BASE_MODELS)
    tasks_to_run = [args.task] if args.task else (RETRIEVAL_CORE_TASKS + SIMILARITY_CORE_TASKS)

    for model_id in models_to_run:
        poolings = [args.pooling] if args.pooling else (BASE_POOLING_MODES if is_base_model(model_id) else [None])
        for task_name in tasks_to_run:
            for pool_mode in poolings:
                evaluate_geometry_combination(
                    model_id=model_id,
                    task_name=task_name,
                    pooling=pool_mode,
                    overwrite=args.overwrite,
                )


if __name__ == "__main__":
    main()
