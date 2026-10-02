import torch

from src.transforms.compression.spectemp import transform_spectemp
from src.transforms.full.r2 import transform_r2


def transform_r2_spectemp(
    corpus: torch.Tensor,
    queries: torch.Tensor,
    target_dim: int = 512,
    gamma: float | None = None,
    kneedle_s: float = 0.5,
    epsilon: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Novel sequential composition: R2 mean deflation followed by adaptive SpecTemp compression."""
    c_r2, q_r2 = transform_r2(corpus, queries)
    return transform_spectemp(
        c_r2,
        q_r2,
        target_dim=target_dim,
        gamma=gamma,
        kneedle_s=kneedle_s,
        epsilon=epsilon,
    )
