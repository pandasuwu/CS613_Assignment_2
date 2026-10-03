import torch

from src.transforms.common import l2_normalize


def compute_average_cosine(
    x: torch.Tensor, max_samples: int = 2000, seed: int = 2026
) -> float:
    """Compute average pairwise cosine similarity between random representations (Ethayarajh, 2019).

    Mathematical Formulation:
        avg_cos = (1 / (S * (S - 1))) * sum_{i != j} (x_i . x_j) / (||x_i||_2 * ||x_j||_2)

    Measures representation cone sharpness (anisotropy). When all representations cluster
    within an acute cone, average cosine approaches 1.0. For uniformly distributed isotropic
    vectors on S^{d-1}, average cosine approaches 0.0.

    Args:
        x: Input tensor of representations of shape (N, d).
        max_samples: Maximum number of vectors to subsample for pairwise evaluation (default: 2000).
        seed: Random seed for deterministic subsampling.

    Returns:
        Average pairwise cosine similarity as a float in [-1.0, 1.0].
    """
    n = x.shape[0]
    if n < 2:
        return 1.0

    normalized = l2_normalize(x)
    sample_size = min(n, max_samples)

    if sample_size < n:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        indices = torch.randperm(n, generator=generator)[:sample_size].to(x.device)
        sample = normalized[indices]
    else:
        sample = normalized

    # Gram matrix: G_ij = cos(x_i, x_j)
    gram = torch.matmul(sample, sample.T)
    # Exclude diagonal self-similarities (cos(x_i, x_i) = 1.0)
    off_diag_sum = torch.sum(gram) - torch.trace(gram)
    num_pairs = sample_size * (sample_size - 1)

    return float((off_diag_sum / num_pairs).item())
