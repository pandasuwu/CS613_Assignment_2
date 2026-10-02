import torch

from src.transforms.common import l2_normalize


def transform_rand(
    corpus: torch.Tensor, queries: torch.Tensor, seed: int = 2026
) -> tuple[torch.Tensor, torch.Tensor]:
    """Random direction deflation negative control (Ren et al., 2025)."""
    generator = torch.Generator(device=corpus.device).manual_seed(seed)
    d = corpus.shape[1]
    v = torch.randn(d, generator=generator, device=corpus.device, dtype=corpus.dtype)
    u = v / torch.linalg.norm(v)

    c_proj = torch.outer(torch.matmul(corpus, u), u)
    q_proj = torch.outer(torch.matmul(queries, u), u)

    c_rand = corpus - c_proj
    q_rand = queries - q_proj

    return l2_normalize(c_rand), l2_normalize(q_rand)
