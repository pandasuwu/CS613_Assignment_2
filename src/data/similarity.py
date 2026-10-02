from dataclasses import dataclass
from typing import Any

import mteb

from src.config import logger


@dataclass(frozen=True)
class SimilarityDataset:
    task_name: str
    sentences1: list[str]
    sentences2: list[str]
    scores: list[float]


def load_similarity_dataset(
    task_name: str, split_name: str = "test"
) -> SimilarityDataset:
    """Load and parse an MTEB v2 Semantic Textual Similarity (STS) benchmark dataset."""
    logger.info("Loading similarity task: %s (split=%s)", task_name, split_name)
    task: Any = mteb.get_task(task_name)
    task.load_data()

    dataset_dict = task.dataset
    if split_name in dataset_dict:
        split = dataset_dict[split_name]
    elif "default" in dataset_dict and split_name in dataset_dict["default"]:
        split = dataset_dict["default"][split_name]
    else:
        available_splits = list(dataset_dict.keys())
        raise ValueError(
            f"Split '{split_name}' not found for task '{task_name}'. Available: {available_splits}"
        )

    sentences1 = [str(s) for s in split["sentence1"]]
    sentences2 = [str(s) for s in split["sentence2"]]
    scores = [float(sc) for sc in split["score"]]

    logger.info(
        "Loaded %s: %d sentence pairs",
        task_name,
        len(sentences1),
    )

    return SimilarityDataset(
        task_name=task_name,
        sentences1=sentences1,
        sentences2=sentences2,
        scores=scores,
    )
