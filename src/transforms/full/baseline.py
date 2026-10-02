import torch

from src.transforms.common import l2_normalize


def transform_baseline(
    corpus: torch.Tensor, queries: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Raw un-transformed baseline with unit L2 normalization."""
    return l2_normalize(corpus), l2_normalize(queries)
