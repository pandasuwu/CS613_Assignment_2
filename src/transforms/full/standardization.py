import torch

from src.transforms.common import l2_normalize


def transform_standardization(
    x1: torch.Tensor,
    x2: torch.Tensor,
    calib_data: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Coordinate-wise z-score standardization followed by L2 renormalization (Timkey & van Schijndel, 2021).

    Mathematical Formulation:
        mu_j = mean(calib_data[:, j])
        sigma_j = std(calib_data[:, j])
        x'_j = (x_j - mu_j) / sigma_j
        x' = x' / ||x'||_2

    Transformers often develop 1 to 5 rogue dimensions with disproportionately large variance
    and coordinate offset that dominate inner products. Standardizing each coordinate to zero
    mean and unit variance eliminates outlier coordinate dominance.

    Args:
        x1: First representation tensor of shape (N, d) (documents in IR, sentences1 in STS).
        x2: Second representation tensor of shape (M, d) (queries in IR, sentences2 in STS).
        calib_data: Calibration tensor of shape (K, d) used to compute per-dimension mean and std.

    Returns:
        Tuple of (transformed_x1, transformed_x2), each L2 unit-normalized.
    """
    mu = torch.mean(calib_data, dim=0)
    sigma = torch.std(calib_data, dim=0, unbiased=True)
    sigma = torch.clamp(sigma, min=1e-9)

    x1_std = (x1 - mu) / sigma
    x2_std = (x2 - mu) / sigma

    return l2_normalize(x1_std), l2_normalize(x2_std)
