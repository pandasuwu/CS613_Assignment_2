import torch

from src.transforms.common import l2_normalize


def compute_average_cosine(
    x: torch.Tensor, max_samples: int = 2000, seed: int = 2026
) -> float:
    """Compute average pairwise cosine similarity between random representations (Ethayarajh, 2019)."""
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
    # Sum of off-diagonal elements
    off_diag_sum = torch.sum(gram) - torch.trace(gram)
    num_pairs = sample_size * (sample_size - 1)

    return float((off_diag_sum / num_pairs).item())
