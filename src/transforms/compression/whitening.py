import torch

from src.transforms.common import compute_mean_and_cov, l2_normalize


def transform_whitening(
    x1: torch.Tensor,
    x2: torch.Tensor,
    calib_data: torch.Tensor,
    target_dim: int = 512,
    epsilon: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Truncated Whitening-k transforming to identity covariance in R^k (Su et al., 2021).

    Mathematical Formulation:
        mu, Lambda, U = cov(calib_data)
        W_k = U_k @ diag(Lambda_k^(-1/2))  # shape (d, k)
        x' = (x - mu) @ W_k
        x' = x' / ||x'||_2

    Combines PCA dimensionality reduction with exact variance equalization along all
    k retained components (gamma = 1.0). In high target dimensions (k >= 512), whitening
    amplifies high-frequency tail noise, degrading retrieval performance.

    Args:
        x1: First representation tensor of shape (N, d) (documents in IR, sentences1 in STS).
        x2: Second representation tensor of shape (M, d) (queries in IR, sentences2 in STS).
        calib_data: Calibration tensor of shape (K, d) used to estimate mean and covariance.
        target_dim: Target reduced dimension k (e.g. 512, 256, 128, 64).
        epsilon: Numerical stability parameter (default: 1e-6).

    Returns:
        Tuple of (transformed_x1, transformed_x2), each in R^k and L2 unit-normalized.
    """
    mu, eigenvalues, eigenvectors = compute_mean_and_cov(calib_data)

    u_k = eigenvectors[:, :target_dim]
    lam_k = eigenvalues[:target_dim]

    # Full whitening scales each principal component by 1 / sqrt(lambda_i)
    scales = 1.0 / torch.sqrt(torch.clamp(lam_k + epsilon, min=1e-12))
    w_k = u_k * scales.unsqueeze(0)  # Shape: (d, k)

    x1_white = torch.matmul(x1 - mu, w_k)
    x2_white = torch.matmul(x2 - mu, w_k)

    return l2_normalize(x1_white), l2_normalize(x2_white)
