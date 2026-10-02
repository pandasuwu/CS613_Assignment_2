import torch


def transform_mc(
    corpus: torch.Tensor, queries: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean centering without subsequent L2 renormalization negative control (Ren et al., 2025)."""
    mu = torch.mean(corpus, dim=0)
    return corpus - mu, queries - mu
