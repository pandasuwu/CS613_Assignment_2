import numpy as np
import scipy.stats
import torch


def compute_spearman_rho(
    predictions: torch.Tensor | list[float] | np.ndarray,
    targets: torch.Tensor | list[float] | np.ndarray,
) -> float:
    """Compute Spearman rank correlation coefficient (rho * 100) between predictions and targets.

    Mathematical Formulation:
        rho = 1 - (6 * sum(d_i^2)) / (n * (n^2 - 1))
        where d_i = rank(pred_i) - rank(target_i)

    Args:
        predictions: Predicted sentence pair cosine similarities.
        targets: Human ground-truth semantic similarity ratings.

    Returns:
        Spearman rank correlation scaled as rho * 100 in [-100.0, 100.0].
    """
    if isinstance(predictions, torch.Tensor):
        p = predictions.detach().cpu().numpy()
    else:
        p = np.asarray(predictions)

    if isinstance(targets, torch.Tensor):
        t = targets.detach().cpu().numpy()
    else:
        t = np.asarray(targets)

    corr, _ = scipy.stats.spearmanr(p, t)
    if np.isnan(corr):
        return 0.0
    return float(corr * 100.0)
