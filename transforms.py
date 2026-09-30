# Step 3: the post-processing methods. Each takes (Q, D) and returns new (Q, D).
import numpy as np
from kneed import KneeLocator


def l2(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def r1(Q, D, mu):
    # Subtract the whole mean vector, renormalise.
    return l2(Q - mu), l2(D - mu)


def r2(Q, D, mu):
    # Remove only the part of each vector that points along the mean, renormalise.
    u = mu / np.linalg.norm(mu)
    return l2(Q - np.outer(Q @ u, u)), l2(D - np.outer(D @ u, u))


def auto_gamma(lam, k):
    # SpecTemp's rule: noise = mean of last 10% eigenvalues,
    # SNR curve, knee via Kneedle, gamma = SNR(k) / SNR(knee), capped at 1.
    t = int(len(lam) * 0.9)
    noise = lam[t:].mean()
    snr = np.maximum(0, (lam[:t] - noise) / noise)
    knee = KneeLocator(np.arange(1, t + 1), snr, curve="convex", direction="decreasing", S=0.5).knee
    return min(1.0, snr[min(k, t) - 1] / snr[knee - 1])


def spectemp(Q, D, k, gamma=None):
    # Fit on documents only. gamma=None -> automatic; 0 -> PCA; 1 -> full whitening.
    X = D.astype(np.float64)
    mu = X.mean(0)
    lam, U = np.linalg.eigh(np.cov(X.T))
    order = np.argsort(lam)[::-1]
    lam, U = lam[order], U[:, order]
    if gamma is None:
        gamma = auto_gamma(lam, k)
    P = U[:, :k] * (lam[:k] + 1e-6) ** (-gamma / 2)
    return l2((Q - mu) @ P).astype(np.float32), l2((D - mu) @ P).astype(np.float32)
