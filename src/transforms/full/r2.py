import torch

from src.transforms.common import l2_normalize


def transform_r2(
    corpus: torch.Tensor, queries: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean-subspace projection deflation and renormalization (Ren et al., 2025)."""
    mu = torch.mean(corpus, dim=0)
    mu_norm = torch.linalg.norm(mu)
    if mu_norm < 1e-12:
        return l2_normalize(corpus), l2_normalize(queries)

    u = mu / mu_norm  # Unit mean direction (d,)

    # Subtract parallel projection along mean direction: e - (e . u) u
    c_proj = torch.outer(torch.matmul(corpus, u), u)
    q_proj = torch.outer(torch.matmul(queries, u), u)

    c_r2 = corpus - c_proj
    q_r2 = queries - q_proj

    return l2_normalize(c_r2), l2_normalize(q_r2)
