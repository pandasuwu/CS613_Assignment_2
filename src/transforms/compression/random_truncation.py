import torch

from src.transforms.common import l2_normalize


def transform_random_truncation(
    x1: torch.Tensor,
    x2: torch.Tensor,
    target_dim: int = 512,
    seed: int = 2026,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Uniform random subsampling of k coordinates without replacement (negative control).

    Mathematical Formulation:
        idx ~ UniformRandomSubset({1, ..., d}, size=k)
        x' = x[idx] / ||x[idx]||_2

    Functions as an empirical negative scientific control for prefix slicing and PCA.
    Tests whether compression gains arise from structured coordinate ordering or merely
    from random subspace dimensionality reduction (Johnson-Lindenstrauss baseline).

    Args:
        x1: First representation tensor of shape (N, d) (documents in IR, sentences1 in STS).
        x2: Second representation tensor of shape (M, d) (queries in IR, sentences2 in STS).
        target_dim: Target reduced dimension k.
        seed: Deterministic random seed for index generation.

    Returns:
        Tuple of (transformed_x1, transformed_x2), each in R^k and L2 unit-normalized.
    """
    generator = torch.Generator(device="cpu").manual_seed(seed)
    d = x1.shape[1]
    indices = torch.randperm(d, generator=generator)[:target_dim].to(x1.device)

    x1_rand = x1[:, indices]
    x2_rand = x2[:, indices]
    return l2_normalize(x1_rand), l2_normalize(x2_rand)
