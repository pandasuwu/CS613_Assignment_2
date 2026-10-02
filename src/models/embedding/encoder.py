import numpy as np
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

from src.config import logger


def encode_embedding_queries(
    texts: list[str],
    model: SentenceTransformer,
    model_id: str,
    instruction: str = "",
    batch_size: int = 32,
    show_progress: bool = True,
) -> torch.Tensor:
    """Encode retrieval queries using model-mandated asymmetric instruction prompts."""
    logger.info("Encoding %d queries with model '%s'", len(texts), model_id)

    if "gemma" in model_id.lower() and hasattr(model, "encode_query"):
        embeddings = model.encode_query(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
        )
    elif "qwen" in model_id.lower():
        prompt = f"Instruct: {instruction}\nQuery: " if instruction else None
        embeddings = model.encode(
            texts,
            prompt=prompt,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
        )
    else:
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
        )

    if isinstance(embeddings, np.ndarray):
        tensor = torch.from_numpy(embeddings).to(torch.float32)
    elif isinstance(embeddings, torch.Tensor):
        tensor = embeddings.detach().cpu().to(torch.float32)
    else:
        tensor = torch.tensor(embeddings, dtype=torch.float32)

    return F.normalize(tensor, p=2, dim=1)


def encode_embedding_corpus(
    texts: list[str],
    model: SentenceTransformer,
    model_id: str,
    batch_size: int = 16,
    show_progress: bool = True,
) -> torch.Tensor:
    """Encode corpus documents as plain texts without instruction prompts."""
    logger.info("Encoding %d documents with model '%s'", len(texts), model_id)

    if "gemma" in model_id.lower() and hasattr(model, "encode_document"):
        embeddings = model.encode_document(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
        )
    else:
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
        )

    if isinstance(embeddings, np.ndarray):
        tensor = torch.from_numpy(embeddings).to(torch.float32)
    elif isinstance(embeddings, torch.Tensor):
        tensor = embeddings.detach().cpu().to(torch.float32)
    else:
        tensor = torch.tensor(embeddings, dtype=torch.float32)

    return F.normalize(tensor, p=2, dim=1)


def encode_embedding_sentences(
    texts: list[str],
    model: SentenceTransformer,
    batch_size: int = 32,
    show_progress: bool = True,
) -> torch.Tensor:
    """Encode symmetric STS sentences plain without retrieval prompts."""
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=show_progress,
    )
    if isinstance(embeddings, np.ndarray):
        tensor = torch.from_numpy(embeddings).to(torch.float32)
    elif isinstance(embeddings, torch.Tensor):
        tensor = embeddings.detach().cpu().to(torch.float32)
    else:
        tensor = torch.tensor(embeddings, dtype=torch.float32)

    return F.normalize(tensor, p=2, dim=1)
