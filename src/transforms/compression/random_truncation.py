import torch

from src.transforms.common import l2_normalize


def transform_random_truncation(
    corpus: torch.Tensor,
    queries: torch.Tensor,
    target_dim: int = 512,
    seed: int = 2026,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Uniform random subsampling of k coordinates without replacement (negative control)."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    d = corpus.shape[1]
    indices = torch.randperm(d, generator=generator)[:target_dim].to(corpus.device)

    c_rand = corpus[:, indices]
    q_rand = queries[:, indices]
    return l2_normalize(c_rand), l2_normalize(q_rand)
