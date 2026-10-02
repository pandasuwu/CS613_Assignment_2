import numpy as np
import pytrec_eval


def compute_recall_at_k(
    qrels: dict[str, dict[str, int]],
    run: dict[str, dict[str, float]],
    k: int = 100,
) -> float:
    """Compute Recall coverage at rank k (Recall@K) using pytrec_eval."""
    metric_str = f"recall.{k}"
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {metric_str})
    results = evaluator.evaluate(run)

    query_scores = [
        results[qid][f"recall_{k}"] for qid in qrels if qid in results
    ]
    if not query_scores:
        return 0.0
    return float(np.mean(query_scores))
