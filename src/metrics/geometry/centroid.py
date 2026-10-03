import torch


def compute_centroid_norm(x: torch.Tensor) -> float:
    """Compute the L2 norm of the empirical mean vector ||mu||_2 (Ren et al., 2025).

    Mathematical Formulation:
        mu = (1 / N) * sum_{i=1}^N x_i
        centroid_norm = ||mu||_2

    For unit-normalized vectors on S^{d-1}, ||mu||_2 in [0, 1] measures origin displacement:
    - ||mu||_2 ~ 1.0 indicates severe collapse into a narrow, localized cone.
    - ||mu||_2 ~ 0.0 indicates origin-centered, balanced distribution across the hypersphere.

    Args:
        x: Input tensor of representations of shape (N, d).

    Returns:
        Euclidean norm of the empirical centroid as a float.
    """
    return float(torch.linalg.norm(x.mean(dim=0)).item())
