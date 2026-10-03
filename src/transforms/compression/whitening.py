import torch

from src.transforms.common import compute_mean_and_cov, l2_normalize


def transform_whitening(
    corpus: torch.Tensor,
    queries: torch.Tensor,
    target_dim: int = 512,
    epsilon: float = 1e-6,
    fit_data: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Truncated Whitening-k transforming to identity covariance in R^k (Su et al., 2021)."""
    ref = fit_data if fit_data is not None else corpus
    mu, eigenvalues, eigenvectors = compute_mean_and_cov(ref)

    u_k = eigenvectors[:, :target_dim]
    lam_k = eigenvalues[:target_dim]

    scales = 1.0 / torch.sqrt(torch.clamp(lam_k + epsilon, min=1e-12))
    w_k = u_k * scales.unsqueeze(0)  # (d, k)

    c_white = torch.matmul(corpus - mu, w_k)
    q_white = torch.matmul(queries - mu, w_k)

    return l2_normalize(c_white), l2_normalize(q_white)
