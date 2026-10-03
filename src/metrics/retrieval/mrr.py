import pytrec_eval


def compute_mrr_at_k(
    qrels: dict[str, dict[str, int]],
    run: dict[str, dict[str, float]],
    k: int = 10,
) -> float:
    """Compute Mean Reciprocal Rank (MRR@K) using pytrec_eval."""
    # Truncate run to top k documents per query for MRR@K
    truncated_run: dict[str, dict[str, float]] = {}
    for qid, doc_dict in run.items():
        sorted_docs = sorted(doc_dict.items(), key=lambda item: item[1], reverse=True)[:k]
        truncated_run[qid] = dict(sorted_docs)

    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {"recip_rank"})
    results = evaluator.evaluate(truncated_run)

    query_scores = [
        results[qid]["recip_rank"] for qid in qrels if qid in results
    ]
    if not query_scores:
        return 0.0
    return float(sum(query_scores) / len(query_scores))
