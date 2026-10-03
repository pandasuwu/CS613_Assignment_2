import torch


def compute_mev(x: torch.Tensor) -> float:
    """Compute Maximum Explainable Variance (MEV): ratio of variance explained by the top principal component."""
    vals = torch.clamp(torch.linalg.eigvalsh(torch.cov(x.T)), min=0.0)
    total_var = vals.sum()
    if total_var <= 1e-12:
        return 0.0
    return float((vals[-1] / total_var).item())
