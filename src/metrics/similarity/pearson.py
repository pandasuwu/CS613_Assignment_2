import numpy as np
import scipy.stats
import torch


def compute_pearson_r(
    predictions: torch.Tensor | list[float] | np.ndarray,
    targets: torch.Tensor | list[float] | np.ndarray,
) -> float:
    """Compute Pearson correlation coefficient (r * 100) between predictions and targets.

    Mathematical Formulation:
        r = sum((p - mean(p)) * (t - mean(t))) / (sqrt(sum((p - mean(p))^2)) * sqrt(sum((t - mean(t))^2)))

    Args:
        predictions: Predicted sentence pair cosine similarities.
        targets: Human ground-truth semantic similarity ratings.

    Returns:
        Pearson linear correlation scaled as r * 100 in [-100.0, 100.0].
    """
    if isinstance(predictions, torch.Tensor):
        p = predictions.detach().cpu().numpy()
    else:
        p = np.asarray(predictions)

    if isinstance(targets, torch.Tensor):
        t = targets.detach().cpu().numpy()
    else:
        t = np.asarray(targets)

    corr, _ = scipy.stats.pearsonr(p, t)
    if np.isnan(corr):
        return 0.0
    return float(corr * 100.0)
