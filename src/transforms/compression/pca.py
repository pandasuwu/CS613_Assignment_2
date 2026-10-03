import torch

from src.transforms.common import compute_mean_and_cov, l2_normalize


def transform_pca(
    x1: torch.Tensor,
    x2: torch.Tensor,
    calib_data: torch.Tensor,
    target_dim: int = 512,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Unweighted projection onto leading k principal components (Li et al., 2026).

    Mathematical Formulation:
        mu, eigenvectors = cov(calib_data)
        U_k = eigenvectors[:, :target_dim]  # leading k eigenvectors of shape (d, k)
        x' = (x - mu) @ U_k
        x' = x' / ||x'||_2

    Maximizes retained variance in R^k without whitening or component scaling (gamma = 0).

    Args:
        x1: First representation tensor of shape (N, d) (documents in IR, sentences1 in STS).
        x2: Second representation tensor of shape (M, d) (queries in IR, sentences2 in STS).
        calib_data: Calibration tensor of shape (K, d) used to estimate mean and covariance.
        target_dim: Target reduced dimension k (e.g. 512, 256, 128, 64).

    Returns:
        Tuple of (transformed_x1, transformed_x2), each in R^k and L2 unit-normalized.
    """
    mu, _, eigenvectors = compute_mean_and_cov(calib_data)

    u_k = eigenvectors[:, :target_dim]  # Shape: (d, k)

    x1_pca = torch.matmul(x1 - mu, u_k)
    x2_pca = torch.matmul(x2 - mu, u_k)

    return l2_normalize(x1_pca), l2_normalize(x2_pca)
