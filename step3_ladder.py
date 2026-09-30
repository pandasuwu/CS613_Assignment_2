# Step 3: run every method on the same saved embeddings and compare to "none".
import json

import numpy as np
from scipy.stats import ttest_rel

from scoring import ndcg10
from transforms import r1, r2, spectemp

Q, D = np.load("emb/Q.npy"), np.load("emb/D.npy")
meta = json.load(open("emb/meta.json"))
mu = D.mean(0)  # corpus mean, same data SpecTemp fits on

runs = {
    "none": (Q, D),
    "r1": r1(Q, D, mu),
    "r2": r2(Q, D, mu),
    "spectemp_768": spectemp(Q, D, 768),
    "spectemp_256": spectemp(Q, D, 256),
    "spectemp_128": spectemp(Q, D, 128),
    "pca_128": spectemp(Q, D, 128, gamma=0),
    "whiten_128": spectemp(Q, D, 128, gamma=1),
}

base_mean, base = ndcg10(Q, D, meta)
for name, (q, d) in runs.items():
    mean, per_q = ndcg10(q, d, meta)
    p = ttest_rel(base, per_q).pvalue if name != "none" else float("nan")
    print(f"{name:14s} {mean:.4f}   diff {mean - base_mean:+.4f}   p {p:.3g}")
