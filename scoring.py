# Step 2: score saved embeddings with nDCG@10 (same metric as MTEB and SpecTemp).
import numpy as np
import pytrec_eval


def ndcg10(Q, D, meta):
    S = Q @ D.T  # cosine similarity, because rows have length 1
    top = np.argpartition(-S, 100, axis=1)[:, :100]  # 100 best docs per query
    run = {
        qid: {meta["dids"][j]: float(S[i, j]) for j in top[i]}
        for i, qid in enumerate(meta["qids"])
    }
    per_query = pytrec_eval.RelevanceEvaluator(meta["qrels"], {"ndcg_cut.10"}).evaluate(run)
    scores = np.array([per_query[q]["ndcg_cut_10"] for q in meta["qids"]])
    return scores.mean(), scores


if __name__ == "__main__":
    import json

    Q, D = np.load("emb/Q.npy"), np.load("emb/D.npy")
    meta = json.load(open("emb/meta.json"))
    mean, _ = ndcg10(Q, D, meta)
    print(f"none  nDCG@10 = {mean:.4f}   (MTEB leaderboard: 0.4774)")
