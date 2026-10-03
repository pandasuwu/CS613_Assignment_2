import torch
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
            convert_to_tensor=True,
        )
    elif "qwen" in model_id.lower():
        prompt = f"Instruct: {instruction}\nQuery: " if instruction else None
        embeddings = model.encode(
            texts,
            prompt=prompt,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
            convert_to_tensor=True,
        )
    else:
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
            convert_to_tensor=True,
        )

    return torch.as_tensor(embeddings, dtype=torch.float32).detach().cpu()


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
            convert_to_tensor=True,
        )
    else:
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
            convert_to_tensor=True,
        )

    return torch.as_tensor(embeddings, dtype=torch.float32).detach().cpu()


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
        convert_to_tensor=True,
    )
    return torch.as_tensor(embeddings, dtype=torch.float32).detach().cpu()
