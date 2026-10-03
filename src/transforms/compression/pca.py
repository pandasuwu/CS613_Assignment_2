import torch

from src.transforms.common import compute_mean_and_cov, l2_normalize


def transform_pca(
    corpus: torch.Tensor,
    queries: torch.Tensor,
    target_dim: int = 512,
    fit_data: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Unweighted projection onto leading k principal components (Li et al., 2026)."""
    ref = fit_data if fit_data is not None else corpus
    mu, _, eigenvectors = compute_mean_and_cov(ref)

    u_k = eigenvectors[:, :target_dim]  # (d, k)

    c_pca = torch.matmul(corpus - mu, u_k)
    q_pca = torch.matmul(queries - mu, u_k)

    return l2_normalize(c_pca), l2_normalize(q_pca)
