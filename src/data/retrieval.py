from dataclasses import dataclass
from typing import Any

import mteb

from src.config import logger


@dataclass(frozen=True)
class RetrievalDataset:
    task_name: str
    doc_texts: list[str]
    doc_ids: list[str]
    query_texts: list[str]
    query_ids: list[str]
    qrels: dict[str, dict[str, int]]
    instruction: str


def load_retrieval_dataset(
    task_name: str, split_name: str = "test"
) -> RetrievalDataset:
    """Load and parse an MTEB v2 retrieval benchmark dataset into clean Python types."""
    logger.info("Loading retrieval task: %s (split=%s)", task_name, split_name)
    task: Any = mteb.get_task(task_name)
    task.load_data()

    dataset_dict = task.dataset
    if split_name in dataset_dict:
        split = dataset_dict[split_name]
    elif "default" in dataset_dict and split_name in dataset_dict["default"]:
        split = dataset_dict["default"][split_name]
    elif "en" in dataset_dict and split_name in dataset_dict["en"]:
        split = dataset_dict["en"][split_name]
    else:
        available_splits = list(dataset_dict.keys())
        raise ValueError(
            f"Split '{split_name}' not found for task '{task_name}'. Available: {available_splits}"
        )

    corpus = split["corpus"]
    queries = split["queries"]
    raw_qrels = split["relevant_docs"]

    # Format document text joining title and body text: "title text"
    titles = corpus["title"] if "title" in corpus.column_names else [""] * len(corpus)
    doc_texts = [
        f"{t} {x}".strip() if t else x for t, x in zip(titles, corpus["text"])
    ]
    doc_ids = [str(did) for did in corpus["id"]]

    query_texts = [str(q) for q in queries["text"]]
    query_ids = [str(qid) for qid in queries["id"]]

    # Standardize qrels to dict[str, dict[str, int]]
    qrels: dict[str, dict[str, int]] = {}
    for qid, doc_dict in raw_qrels.items():
        qrels[str(qid)] = {str(did): int(rel) for did, rel in dict(doc_dict).items()}

    # Extract task query prompt instruction if present in metadata
    prompt = ""
    if hasattr(task, "metadata") and task.metadata is not None:
        raw_prompt = getattr(task.metadata, "prompt", "")
        if isinstance(raw_prompt, dict):
            prompt = raw_prompt.get("query", "")
        elif isinstance(raw_prompt, str):
            prompt = raw_prompt

    logger.info(
        "Loaded %s: %d documents, %d queries, %d qrel entries",
        task_name,
        len(doc_texts),
        len(query_texts),
        len(qrels),
    )

    return RetrievalDataset(
        task_name=task_name,
        doc_texts=doc_texts,
        doc_ids=doc_ids,
        query_texts=query_texts,
        query_ids=query_ids,
        qrels=qrels,
        instruction=prompt,
    )
