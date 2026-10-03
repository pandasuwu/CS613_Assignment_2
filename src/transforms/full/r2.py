import torch
import torch.nn.functional as F

from src.transforms.common import l2_normalize


def transform_r2(
    corpus: torch.Tensor, queries: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean-subspace projection deflation and renormalization (Ren et al., 2025)."""
    mu = corpus.mean(dim=0)
    if torch.linalg.norm(mu) < 1e-12:
        return l2_normalize(corpus), l2_normalize(queries)

    u = F.normalize(mu, p=2, dim=0)

    c_r2 = corpus - (corpus @ u).unsqueeze(1) * u
    q_r2 = queries - (queries @ u).unsqueeze(1) * u

    return l2_normalize(c_r2), l2_normalize(q_r2)
