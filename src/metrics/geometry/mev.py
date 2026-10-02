import torch

from src.transforms.common import compute_mean_and_cov


def compute_mev(x: torch.Tensor) -> float:
    """Compute Maximum Explainable Variance (MEV): ratio of variance explained by the top principal component."""
    _, eigenvalues, _, _ = compute_mean_and_cov(x)
    total_var = torch.sum(eigenvalues)
    if total_var <= 1e-12:
        return 0.0
    return float((eigenvalues[0] / total_var).item())
