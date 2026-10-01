# Step 2: score saved embeddings. nDCG@10 is the main metric (same as MTEB and SpecTemp); Recall@100 is extra.
# Usage: python scoring.py emb/gemma_FiQA2018
import numpy as np
import pytrec_eval


def evaluate(Q, D, meta):
    S = Q @ D.T  # cosine similarity, because rows have length 1
    top = np.argpartition(-S, 100, axis=1)[:, :100]  # 100 best docs per query
    run = {
        qid: {meta["dids"][j]: float(S[i, j]) for j in top[i]}
        for i, qid in enumerate(meta["qids"])
    }
    res = pytrec_eval.RelevanceEvaluator(meta["qrels"], {"ndcg_cut.10", "recall.100"}).evaluate(run)
    ndcg = np.array([res[q]["ndcg_cut_10"] for q in meta["qids"]])
    recall = np.array([res[q]["recall_100"] for q in meta["qids"]])
    return ndcg, recall


def avg_cos(D, n=2000, seed=0):
    # Anisotropy check: average cosine between random document pairs (high = narrow cone).
    X = D[np.random.default_rng(seed).choice(len(D), min(n, len(D)), replace=False)]
    G = X @ X.T
    return (G.sum() - np.trace(G)) / (len(X) * (len(X) - 1))


if __name__ == "__main__":
    import json
    import sys

    path = sys.argv[1]
    Q, D = np.load(f"{path}/Q.npy"), np.load(f"{path}/D.npy")
    meta = json.load(open(f"{path}/meta.json"))
    ndcg, recall = evaluate(Q, D, meta)
    print(f"{path}  nDCG@10 {ndcg.mean():.4f}  Recall@100 {recall.mean():.4f}  AvgCos {avg_cos(D):.4f}")
