import torch

from src.transforms.common import compute_mean_and_cov, l2_normalize


def transform_abtt(
    corpus: torch.Tensor,
    queries: torch.Tensor,
    d_components: int = 1,
    fit_data: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """All-but-the-top principal component deflation (Mu & Viswanath, 2018; Ren et al., 2025)."""
    ref = fit_data if fit_data is not None else corpus
    mu, _, eigenvectors = compute_mean_and_cov(ref)

    u_top = eigenvectors[:, :d_components]  # Top D principal components (d, D)

    c_cent = corpus - mu
    q_cent = queries - mu

    # Project centered vectors onto top D subspace and subtract: (I - U U^T)(x - mu)
    c_proj = torch.matmul(torch.matmul(c_cent, u_top), u_top.T)
    q_proj = torch.matmul(torch.matmul(q_cent, u_top), u_top.T)

    c_abtt = c_cent - c_proj
    q_abtt = q_cent - q_proj

    return l2_normalize(c_abtt), l2_normalize(q_abtt)
