import torch

from src.transforms.common import l2_normalize


def transform_r1(
    corpus: torch.Tensor,
    queries: torch.Tensor,
    fit_data: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Direct empirical mean subtraction and renormalization (Ren et al., 2025)."""
    ref = fit_data if fit_data is not None else corpus
    mu = ref.mean(dim=0)
    return l2_normalize(corpus - mu), l2_normalize(queries - mu)
