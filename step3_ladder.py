# Step 3: run every method on every saved embedding folder, write results/results.csv.
# Usage: python step3_ladder.py   (uses every folder in emb/)
import csv
import glob
import json
import os

import numpy as np
from scipy.stats import ttest_rel

from scoring import avg_cos, evaluate
from transforms import r1, r2, r2_spectemp, spectemp


def methods(Q, D):
    mu = D.mean(0)
    dim = D.shape[1]
    yield "none", dim, None, (Q, D)
    yield "r1", dim, None, r1(Q, D, mu)
    yield "r2", dim, None, r2(Q, D, mu)
    for k in [64, 128, 256, dim]:
        q, d, g = spectemp(Q, D, k)
        yield "spectemp", k, g, (q, d)
        q, d, g = spectemp(Q, D, k, gamma=0)
        yield "pca", k, g, (q, d)
        q, d, g = spectemp(Q, D, k, gamma=1)
        yield "whitening", k, g, (q, d)
        q, d, g = r2_spectemp(Q, D, k)
        yield "r2_spectemp", k, g, (q, d)
    for g in [0.25, 0.5, 0.75]:  # gamma grid at k=128 (ablation)
        q, d, _ = spectemp(Q, D, 128, gamma=g)
        yield "gamma_grid", 128, g, (q, d)
    for S in [1.0, 2.0]:  # Kneedle sensitivity ablation at k=128 (default S=0.5 is the spectemp row)
        q, d, g = spectemp(Q, D, 128, S=S)
        yield f"spectemp_S{S}", 128, g, (q, d)


os.makedirs("results", exist_ok=True)
rows = []
for path in sorted(glob.glob("emb/*/")):
    name = os.path.basename(path.rstrip("/\\"))
    model, task = name.split("_", 1)
    Q, D = np.load(f"{path}/Q.npy"), np.load(f"{path}/D.npy")
    meta = json.load(open(f"{path}/meta.json"))
    base = None
    for method, k, g, (q, d) in methods(Q, D):
        ndcg, recall = evaluate(q, d, meta)
        if base is None:
            base = ndcg
        p = ttest_rel(base, ndcg).pvalue if method != "none" else float("nan")
        row = {
            "model": model, "task": task, "method": method, "k": k,
            "gamma": "" if g is None else round(g, 4),
            "ndcg10": round(ndcg.mean(), 4), "diff": round(ndcg.mean() - base.mean(), 4),
            "recall100": round(recall.mean(), 4), "avgcos": round(float(avg_cos(d)), 4),
            "p": round(float(p), 4),
        }
        rows.append(row)
        print(row)

with open("results/results.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print("wrote results/results.csv")
