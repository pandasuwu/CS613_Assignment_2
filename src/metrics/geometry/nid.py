import math

import torch

from src.transforms.common import compute_mean_and_cov


def compute_nid(x: torch.Tensor) -> float:
    """Compute Normalized Intrinsic Dimensionality (NID) via normalized spectral entropy (Yokoi et al., 2024)."""
    _, eigenvalues, _, _ = compute_mean_and_cov(x)
    d = eigenvalues.shape[0]
    if d <= 1:
        return 1.0

    total_var = torch.sum(eigenvalues)
    if total_var <= 1e-12:
        return 0.0

    probs = torch.clamp(eigenvalues / total_var, min=1e-12)
    entropy = -torch.sum(probs * torch.log(probs))
    max_entropy = math.log(d)

    nid = entropy / max_entropy
    return float(torch.clamp(nid, min=0.0, max=1.0).item())
