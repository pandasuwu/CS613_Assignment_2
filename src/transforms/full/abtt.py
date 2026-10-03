import torch

from src.transforms.common import compute_mean_and_cov, l2_normalize


def transform_abtt(
    x1: torch.Tensor,
    x2: torch.Tensor,
    calib_data: torch.Tensor,
    d_components: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """All-but-the-top principal component deflation (Mu & Viswanath, 2018; Ren et al., 2025).

    Mathematical Formulation:
        mu, eigenvectors = cov(calib_data)
        U_D = eigenvectors[:, :d_components]  # leading D eigenvectors
        x_centered = x - mu
        x' = x_centered - x_centered @ U_D @ U_D^T
        x' = x' / ||x'||_2

    Projects centered representations onto the orthogonal complement of the top D
    principal components, removing dominant shared background/frequency directions.

    Args:
        x1: First representation tensor of shape (N, d) (documents in IR, sentences1 in STS).
        x2: Second representation tensor of shape (M, d) (queries in IR, sentences2 in STS).
        calib_data: Calibration tensor of shape (K, d) used to estimate mean and principal components.
        d_components: Number of leading principal components to deflate (D in {1, 2, 3}).

    Returns:
        Tuple of (transformed_x1, transformed_x2), each L2 unit-normalized.
    """
    mu, _, eigenvectors = compute_mean_and_cov(calib_data)

    u_top = eigenvectors[:, :d_components]  # Top D principal components of shape (d, D)

    x1_cent = x1 - mu
    x2_cent = x2 - mu

    # Subtract projection onto the leading D subspace: (I - U_D U_D^T)(x - mu)
    x1_proj = torch.matmul(torch.matmul(x1_cent, u_top), u_top.T)
    x2_proj = torch.matmul(torch.matmul(x2_cent, u_top), u_top.T)

    x1_abtt = x1_cent - x1_proj
    x2_abtt = x2_cent - x2_proj

    return l2_normalize(x1_abtt), l2_normalize(x2_abtt)
