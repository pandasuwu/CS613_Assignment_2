import torch

from src.transforms.common import l2_normalize


def transform_prefix(
    corpus: torch.Tensor, queries: torch.Tensor, target_dim: int = 512
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prefix coordinate slicing (Matryoshka representation baseline)."""
    c_trunc = corpus[:, :target_dim]
    q_trunc = queries[:, :target_dim]
    return l2_normalize(c_trunc), l2_normalize(q_trunc)
