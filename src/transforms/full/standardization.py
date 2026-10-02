import torch

from src.transforms.common import l2_normalize


def transform_standardization(
    corpus: torch.Tensor, queries: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Coordinate-wise z-score standardization (Timkey & van Schijndel, 2021)."""
    mu = torch.mean(corpus, dim=0)
    sigma = torch.std(corpus, dim=0, unbiased=True)
    sigma = torch.clamp(sigma, min=1e-9)

    c_std = (corpus - mu) / sigma
    q_std = (queries - mu) / sigma

    return l2_normalize(c_std), l2_normalize(q_std)
