import torch
import torch.nn.functional as F


def l2_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Normalize tensor along dimension 1 to unit L2 norm."""
    return F.normalize(x, p=2, dim=1, eps=eps)


def compute_mean_and_cov(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute empirical mean, covariance, and ordered eigendecomposition in FP32.

    Returns:
        mu: Mean vector of shape (d,)
        eigenvalues: Descending ordered eigenvalues of shape (d,)
        eigenvectors: Orthogonal eigenvectors matrix U of shape (d, d)
        cov: Centered covariance matrix of shape (d, d)
    """
    mu = torch.mean(x, dim=0)
    centered = x - mu
    n = x.shape[0]

    cov = torch.matmul(centered.T, centered) / max(n - 1, 1)
    # Enforce exact matrix symmetry to eliminate floating point asymmetry
    cov = 0.5 * (cov + cov.T)

    eigenvalues, eigenvectors = torch.linalg.eigh(cov)

    # Sort in descending order (highest variance first)
    descending_idx = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[descending_idx]
    eigenvectors = eigenvectors[:, descending_idx]

    return mu, eigenvalues, eigenvectors, cov
