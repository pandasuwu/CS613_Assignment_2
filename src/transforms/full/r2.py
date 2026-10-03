import torch
import torch.nn.functional as F

from src.transforms.common import l2_normalize


def transform_r2(
    x1: torch.Tensor,
    x2: torch.Tensor,
    calib_data: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean-subspace projection deflation followed by L2 renormalization (Ren et al., 2025).

    Mathematical Formulation:
        mu = mean(calib_data, dim=0)
        u = mu / ||mu||_2
        x' = (x - (x . u) * u) / ||x - (x . u) * u||_2

    Unlike R1 (direct subtraction), R2 projects representations onto the orthogonal
    complement of the unit mean direction vector u. As proven by Ren et al. (2025),
    this first-order orthogonal projection cancels out parallel sample-mean estimation
    noise (alpha * u) without degrading the remaining d-1 dimensional subspace.

    Args:
        x1: First representation tensor of shape (N, d) (documents in IR, sentences1 in STS).
        x2: Second representation tensor of shape (M, d) (queries in IR, sentences2 in STS).
        calib_data: Calibration tensor of shape (K, d) used to estimate the empirical mean direction.

    Returns:
        Tuple of (transformed_x1, transformed_x2), each L2 unit-normalized.
    """
    mu = calib_data.mean(dim=0)
    if torch.linalg.norm(mu) < 1e-12:
        return l2_normalize(x1), l2_normalize(x2)

    u = F.normalize(mu, p=2, dim=0)

    # Orthogonal projection: subtract the scalar projection along unit direction u
    x1_r2 = x1 - (x1 @ u).unsqueeze(1) * u
    x2_r2 = x2 - (x2 @ u).unsqueeze(1) * u

    return l2_normalize(x1_r2), l2_normalize(x2_r2)
