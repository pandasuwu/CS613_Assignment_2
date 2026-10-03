import pytrec_eval


def compute_recall_at_k(
    qrels: dict[str, dict[str, int]],
    run: dict[str, dict[str, float]],
    k: int = 100,
) -> float:
    """Compute Recall coverage at rank k (Recall@K) using pytrec_eval."""
    if not qrels:
        return 0.0

    metric_str = f"recall.{k}"
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {metric_str})
    results = evaluator.evaluate(run)

    query_scores = [
        float(results[qid][f"recall_{k}"]) if qid in results else 0.0
        for qid in qrels
    ]
    return float(sum(query_scores) / len(qrels))
