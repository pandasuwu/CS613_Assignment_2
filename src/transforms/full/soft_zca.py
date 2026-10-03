import torch

from src.transforms.common import compute_mean_and_cov, l2_normalize


def transform_soft_zca(
    corpus: torch.Tensor,
    queries: torch.Tensor,
    epsilon: float = 1e-4,
    fit_data: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Eigenvalue-regularized Zero-phase Component Analysis (Soft-ZCA) whitening (Diera et al., 2024)."""
    ref = fit_data if fit_data is not None else corpus
    mu, eigenvalues, eigenvectors = compute_mean_and_cov(ref)

    # Regularized inverse square root eigenvalues: bound noise amplification
    scales = 1.0 / torch.sqrt(torch.clamp(eigenvalues + epsilon, min=1e-12))
    w_zca = torch.matmul(eigenvectors, torch.matmul(torch.diag(scales), eigenvectors.T))

    c_trans = torch.matmul(corpus - mu, w_zca.T)
    q_trans = torch.matmul(queries - mu, w_zca.T)

    return l2_normalize(c_trans), l2_normalize(q_trans)
