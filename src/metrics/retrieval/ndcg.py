import pytrec_eval


def compute_ndcg_at_k(
    qrels: dict[str, dict[str, int]],
    run: dict[str, dict[str, float]],
    k: int = 10,
) -> float:
    """Compute Normalized Discounted Cumulative Gain at rank k (nDCG@K) using pytrec_eval."""
    metric_str = f"ndcg_cut.{k}"
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {metric_str})
    results = evaluator.evaluate(run)

    query_scores = [
        results[qid][f"ndcg_cut_{k}"] for qid in qrels if qid in results
    ]
    if not query_scores:
        return 0.0
    return float(sum(query_scores) / len(query_scores))
