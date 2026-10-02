import numpy as np
import torch
from kneed import KneeLocator

from src.transforms.common import compute_mean_and_cov, l2_normalize


def calculate_snr_curve(
    eigenvalues: torch.Tensor, tail_percentile: float = 0.9
) -> tuple[torch.Tensor, torch.Tensor]:
    """Calculate local signal-to-noise ratio curve from eigenvalue spectrum."""
    d = len(eigenvalues)
    tail_start = int(d * tail_percentile)
    noise_variance = torch.mean(eigenvalues[tail_start:])

    snr_curve = torch.clamp((eigenvalues[:tail_start] - noise_variance) / torch.clamp(noise_variance, min=1e-12), min=0.0)
    return snr_curve, noise_variance


def find_optimal_gamma(
    eigenvalues: torch.Tensor,
    target_dim: int,
    kneedle_s: float = 0.5,
    tail_percentile: float = 0.9,
) -> float:
    """Derive adaptive whitening exponent gamma(k) via Kneedle knee detection on SNR curve (Li et al., 2026)."""
    snr_curve, _ = calculate_snr_curve(eigenvalues, tail_percentile=tail_percentile)
    snr_np = snr_curve.cpu().numpy()
    n_points = len(snr_np)

    kneedle = KneeLocator(
        np.arange(1, n_points + 1),
        snr_np,
        curve="convex",
        direction="decreasing",
        S=kneedle_s,
    )

    knee_point = kneedle.knee
    if knee_point is None or knee_point < 1:
        knee_point = max(1, min(target_dim // 2, n_points))

    snr_knee = snr_np[knee_point - 1]
    if snr_knee <= 1e-12:
        return 1.0

    target_idx = min(target_dim, n_points) - 1
    snr_target = snr_np[target_idx]

    gamma = float(snr_target / snr_knee)
    return max(0.0, min(1.0, gamma))


def transform_spectemp(
    corpus: torch.Tensor,
    queries: torch.Tensor,
    target_dim: int = 512,
    gamma: float | None = None,
    kneedle_s: float = 0.5,
    epsilon: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Adaptive SNR-tempered spectral projection (SpecTemp, Li et al., 2026)."""
    mu, eigenvalues, eigenvectors, _ = compute_mean_and_cov(corpus)

    if gamma is None:
        gamma_val = find_optimal_gamma(eigenvalues, target_dim=target_dim, kneedle_s=kneedle_s)
    else:
        gamma_val = float(gamma)

    u_k = eigenvectors[:, :target_dim]
    lam_k = eigenvalues[:target_dim]

    # Fractional eigenvalue scaling: (lambda_i + eps)^(-gamma / 2)
    scales = torch.clamp(lam_k + epsilon, min=1e-12) ** (-gamma_val / 2.0)
    w_k = u_k * scales.unsqueeze(0)

    c_trans = torch.matmul(corpus - mu, w_k)
    q_trans = torch.matmul(queries - mu, w_k)

    return l2_normalize(c_trans), l2_normalize(q_trans), gamma_val
