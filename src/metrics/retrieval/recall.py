import pytrec_eval


def compute_recall_at_k(
    qrels: dict[str, dict[str, int]],
    run: dict[str, dict[str, float]],
    k: int = 100,
) -> float:
    """Compute Recall coverage at rank k (Recall@K) using pytrec_eval.

    Mathematical Formulation:
        Recall@K = |{retrieved docs in top K} intersection {relevant docs}| / |{relevant docs}|

    In accordance with TREC and BEIR evaluation protocols, the benchmark average is
    computed across all target queries defined in qrels. Queries with no retrieved
    documents receive a score of 0.0.

    Args:
        qrels: Ground-truth relevance mapping {query_id: {doc_id: relevance_score}}.
        run: Model retrieval run predictions {query_id: {doc_id: similarity_score}}.
        k: Truncation cutoff rank (default: 100).

    Returns:
        Mean Recall@K score in [0.0, 1.0] across all benchmark queries in qrels.
    """
    if not qrels:
        return 0.0

    metric_str = f"recall.{k}"
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {metric_str})
    results = evaluator.evaluate(run)

    # Standard TREC evaluation: average over all benchmark queries in qrels
    query_scores = [
        float(results[qid][f"recall_{k}"]) if qid in results else 0.0
        for qid in qrels
    ]
    return float(sum(query_scores) / len(qrels))
