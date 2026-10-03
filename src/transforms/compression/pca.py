import torch

from src.transforms.common import compute_mean_and_cov, l2_normalize


def transform_pca(
    corpus: torch.Tensor, queries: torch.Tensor, target_dim: int = 512
) -> tuple[torch.Tensor, torch.Tensor]:
    """Unweighted projection onto leading k principal components (Li et al., 2026)."""
    mu, _, eigenvectors = compute_mean_and_cov(corpus)

    u_k = eigenvectors[:, :target_dim]  # (d, k)

    c_pca = torch.matmul(corpus - mu, u_k)
    q_pca = torch.matmul(queries - mu, u_k)

    return l2_normalize(c_pca), l2_normalize(q_pca)
