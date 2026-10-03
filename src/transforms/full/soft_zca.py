import torch

from src.transforms.common import compute_mean_and_cov, l2_normalize


def transform_soft_zca(
    x1: torch.Tensor,
    x2: torch.Tensor,
    calib_data: torch.Tensor,
    epsilon: float = 1e-4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Eigenvalue-regularized Zero-phase Component Analysis (Soft-ZCA) whitening (Diera et al., 2024).

    Mathematical Formulation:
        Sigma = cov(calib_data) = U * diag(lambda) * U^T
        W_zca = U * diag((lambda + epsilon)^(-1/2)) * U^T
        x' = (x - mu) @ W_zca^T
        x' = x' / ||x'||_2

    Standard ZCA whitening inverts unregularized eigenvalues lambda^(-1/2), which causes
    catastrophic noise amplification along trailing dimensions where lambda -> 0.
    Soft-ZCA adds an eigenvalue regularizer epsilon >= 0 to bound component amplification
    by epsilon^(-1/2), preserving primary semantic structure while decorrelating axes.

    Args:
        x1: First representation tensor of shape (N, d) (documents in IR, sentences1 in STS).
        x2: Second representation tensor of shape (M, d) (queries in IR, sentences2 in STS).
        calib_data: Calibration tensor of shape (K, d) used to estimate mean and covariance.
        epsilon: Additive spectral regularization parameter (default: 1e-4).

    Returns:
        Tuple of (transformed_x1, transformed_x2), each L2 unit-normalized.
    """
    mu, eigenvalues, eigenvectors = compute_mean_and_cov(calib_data)

    # Regularized inverse square root eigenvalues: bound maximum amplification
    scales = 1.0 / torch.sqrt(torch.clamp(eigenvalues + epsilon, min=1e-12))
    w_zca = torch.matmul(eigenvectors, torch.matmul(torch.diag(scales), eigenvectors.T))

    x1_trans = torch.matmul(x1 - mu, w_zca.T)
    x2_trans = torch.matmul(x2 - mu, w_zca.T)

    return l2_normalize(x1_trans), l2_normalize(x2_trans)
