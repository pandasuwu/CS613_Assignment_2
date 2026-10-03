import torch


def transform_mc(
    x1: torch.Tensor,
    x2: torch.Tensor,
    calib_data: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean centering without subsequent L2 renormalization negative control (Ren et al., 2025).

    Mathematical Formulation:
        mu = mean(calib_data, dim=0)
        x' = x - mu

    Omits hyperspherical L2 normalization. Isolates the mathematical necessity of
    unit renormalization on inner product metrics after origin translation.

    Args:
        x1: First representation tensor of shape (N, d) (documents in IR, sentences1 in STS).
        x2: Second representation tensor of shape (M, d) (queries in IR, sentences2 in STS).
        calib_data: Calibration tensor of shape (K, d) used to estimate the empirical mean vector.

    Returns:
        Tuple of (transformed_x1, transformed_x2) unnormalized.
    """
    mu = torch.mean(calib_data, dim=0)
    return x1 - mu, x2 - mu
