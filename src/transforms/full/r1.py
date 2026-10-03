import torch

from src.transforms.common import l2_normalize


def transform_r1(
    x1: torch.Tensor,
    x2: torch.Tensor,
    calib_data: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Direct empirical mean subtraction followed by L2 renormalization (Ren et al., 2025).

    Mathematical Formulation:
        mu = mean(calib_data, dim=0)
        x' = (x - mu) / ||x - mu||_2

    Args:
        x1: First representation tensor of shape (N, d) (documents in IR, sentences1 in STS).
        x2: Second representation tensor of shape (M, d) (queries in IR, sentences2 in STS).
        calib_data: Calibration tensor of shape (K, d) used to estimate the empirical mean vector.
            In retrieval, this is the document corpus; in STS, it is pooled [sentences1; sentences2].

    Returns:
        Tuple of (transformed_x1, transformed_x2), each L2 unit-normalized.
    """
    mu = calib_data.mean(dim=0)
    return l2_normalize(x1 - mu), l2_normalize(x2 - mu)
