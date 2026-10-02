import torch


def compute_centroid_norm(x: torch.Tensor) -> float:
    """Compute the L2 norm of the empirical mean vector ||mu||_2 (Ren et al., 2025)."""
    mu = torch.mean(x, dim=0)
    return float(torch.linalg.norm(mu).item())
