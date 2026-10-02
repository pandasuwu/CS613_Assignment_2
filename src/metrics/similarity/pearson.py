import numpy as np
import scipy.stats
import torch


def compute_pearson_r(
    predictions: torch.Tensor | list[float] | np.ndarray,
    targets: torch.Tensor | list[float] | np.ndarray,
) -> float:
    """Compute Pearson correlation coefficient (r * 100) between predictions and targets."""
    if isinstance(predictions, torch.Tensor):
        p = predictions.detach().cpu().numpy()
    else:
        p = np.asarray(predictions)

    if isinstance(targets, torch.Tensor):
        t = targets.detach().cpu().numpy()
    else:
        t = np.asarray(targets)

    corr, _ = scipy.stats.pearsonr(p, t)
    return float(corr * 100.0)
