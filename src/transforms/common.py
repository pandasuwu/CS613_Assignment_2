import torch
import torch.nn.functional as F


def l2_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Normalize tensor along dimension 1 to unit L2 norm."""
    return F.normalize(x, p=2, dim=1, eps=eps)


def compute_mean_and_cov(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute empirical mean, descending eigenvalues, and orthogonal eigenvectors in FP32.

    Returns:
        mu: Mean vector of shape (d,)
        eigenvalues: Descending ordered eigenvalues of shape (d,)
        eigenvectors: Orthogonal eigenvectors matrix U of shape (d, d)
    """
    mu = torch.mean(x, dim=0)
    cov = torch.cov(x.T)
    eigenvalues, eigenvectors = torch.linalg.eigh(cov)
    eigenvalues = torch.clamp(eigenvalues, min=0.0)

    descending_idx = torch.argsort(eigenvalues, descending=True)
    return mu, eigenvalues[descending_idx], eigenvectors[:, descending_idx]
