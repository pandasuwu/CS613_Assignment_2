import torch

from src.transforms.common import l2_normalize


def transform_r1(
    corpus: torch.Tensor, queries: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Direct empirical mean subtraction and renormalization (Ren et al., 2025)."""
    mu = torch.mean(corpus, dim=0)

    c_r1 = corpus - mu
    q_r1 = queries - mu

    return l2_normalize(c_r1), l2_normalize(q_r1)
