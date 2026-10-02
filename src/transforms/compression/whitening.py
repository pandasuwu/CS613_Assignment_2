import torch

from src.transforms.common import compute_mean_and_cov, l2_normalize


def transform_whitening(
    corpus: torch.Tensor,
    queries: torch.Tensor,
    target_dim: int = 512,
    epsilon: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Truncated Whitening-k transforming to identity covariance in R^k (Su et al., 2021)."""
    mu, eigenvalues, eigenvectors, _ = compute_mean_and_cov(corpus)

    u_k = eigenvectors[:, :target_dim]
    lam_k = eigenvalues[:target_dim]

    scales = 1.0 / torch.sqrt(torch.clamp(lam_k + epsilon, min=1e-12))
    w_k = u_k * scales.unsqueeze(0)  # (d, k)

    c_white = torch.matmul(corpus - mu, w_k)
    q_white = torch.matmul(queries - mu, w_k)

    return l2_normalize(c_white), l2_normalize(q_white)
