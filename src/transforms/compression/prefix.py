import torch

from src.transforms.common import l2_normalize


def transform_prefix(
    x1: torch.Tensor,
    x2: torch.Tensor,
    target_dim: int = 512,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prefix coordinate slicing (Matryoshka representation baseline).

    Mathematical Formulation:
        x' = x[:k] / ||x[:k]||_2

    Evaluates native Matryoshka Representation Learning (MRL) coordinate prefix slicing.
    Does not require parameter fitting or covariance calculation.

    Args:
        x1: First representation tensor of shape (N, d) (documents in IR, sentences1 in STS).
        x2: Second representation tensor of shape (M, d) (queries in IR, sentences2 in STS).
        target_dim: Target reduced dimension k.

    Returns:
        Tuple of (transformed_x1, transformed_x2), each in R^k and L2 unit-normalized.
    """
    x1_trunc = x1[:, :target_dim]
    x2_trunc = x2[:, :target_dim]
    return l2_normalize(x1_trunc), l2_normalize(x2_trunc)
