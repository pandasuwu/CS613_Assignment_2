import torch

from src.transforms.common import l2_normalize


def transform_baseline(
    x1: torch.Tensor,
    x2: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Raw un-transformed baseline representations with unit L2 normalization.

    Mathematical Formulation:
        x' = x / ||x||_2

    Preserves native representation geometry without centering, scaling, or deflation.

    Args:
        x1: First representation tensor of shape (N, d) (documents in IR, sentences1 in STS).
        x2: Second representation tensor of shape (M, d) (queries in IR, sentences2 in STS).

    Returns:
        Tuple of (transformed_x1, transformed_x2), each L2 unit-normalized.
    """
    return l2_normalize(x1), l2_normalize(x2)
