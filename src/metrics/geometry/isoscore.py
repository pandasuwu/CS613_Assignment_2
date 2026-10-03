import math

import torch


def compute_isoscore(points: torch.Tensor) -> float:
    """Compute IsoScore measuring covariance diagonal defect from identity (Rudman et al., 2022).

    IsoScore is 1.0 for perfect spherical isotropy (all eigenvalues equal)
    and 0.0 for total dimensional collapse (single non-zero eigenvalue).
    """
    cov_diag = torch.linalg.eigvalsh(torch.cov(points.T))
    d = cov_diag.shape[0]
    if d <= 1:
        return 1.0

    cov_diag_norm = torch.linalg.norm(cov_diag)
    if cov_diag_norm <= 1e-12:
        return 0.0

    # Step 4: Normalize diagonal to unit variance scaling
    cov_diag_normalized = (cov_diag * math.sqrt(d)) / cov_diag_norm

    # Step 5: Distance from uniform diagonal (vector of all ones)
    iso_diag = torch.ones(d, device=points.device, dtype=points.dtype)
    l2_distance = torch.linalg.norm(cov_diag_normalized - iso_diag)
    normalization_constant = math.sqrt(2.0 * (d - math.sqrt(d)))

    if normalization_constant <= 1e-12:
        return 1.0

    isotropy_defect = l2_distance / normalization_constant

    # Steps 6 and 7: Scale-invariant score mapping to [0, 1]
    defect_term = (isotropy_defect**2) * (d - math.sqrt(d))
    score = ((d - defect_term) ** 2 - d) / (d * (d - 1))

    return float(torch.clamp(score, min=0.0, max=1.0).item())
