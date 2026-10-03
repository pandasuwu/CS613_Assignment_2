import math

import torch


def compute_nid(x: torch.Tensor) -> float:
    """Compute Normalized Intrinsic Dimensionality (NID) via normalized spectral entropy (Yokoi et al., 2024)."""
    vals = torch.clamp(torch.linalg.eigvalsh(torch.cov(x.T)), min=0.0)
    d = vals.shape[0]
    if d <= 1:
        return 1.0

    total_var = vals.sum()
    if total_var <= 1e-12:
        return 0.0

    probs = torch.clamp(vals / total_var, min=1e-12)
    entropy = -torch.sum(probs * torch.log(probs))
    max_entropy = math.log(d)

    nid = entropy / max_entropy
    return float(torch.clamp(nid, min=0.0, max=1.0).item())
