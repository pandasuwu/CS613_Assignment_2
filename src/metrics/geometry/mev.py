import torch


def compute_mev(x: torch.Tensor) -> float:
    """Compute Maximum Explainable Variance (MEV) of the leading principal component.

    Mathematical Formulation:
        Sigma = cov(x)
        MEV = lambda_max / sum_{j=1}^d lambda_j

    Quantifies the fraction of total representational variance concentrated in the single
    dominant principal axis. High MEV (e.g. > 0.5) indicates that a single Rogue direction
    dominates Euclidean distances and inner products across the space.

    Args:
        x: Input tensor of representations of shape (N, d).

    Returns:
        Ratio of maximum eigenvalue to total trace variance as a float in [0.0, 1.0].
    """
    vals = torch.clamp(torch.linalg.eigvalsh(torch.cov(x.T)), min=0.0)
    total_var = vals.sum()
    if total_var <= 1e-12:
        return 0.0
    return float((vals[-1] / total_var).item())
