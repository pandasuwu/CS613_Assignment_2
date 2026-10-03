import pytrec_eval


def compute_ndcg_at_k(
    qrels: dict[str, dict[str, int]],
    run: dict[str, dict[str, float]],
    k: int = 10,
) -> float:
    """Compute Normalized Discounted Cumulative Gain at rank k (nDCG@K) using pytrec_eval.

    Mathematical Formulation:
        DCG@K = sum_{i=1}^K (2^{rel_i} - 1) / log_2(i + 1)
        IDCG@K = Ideal DCG@K achieved by perfectly ordering all relevant documents
        nDCG@K = DCG@K / IDCG@K

    In accordance with TREC and BEIR evaluation protocols, the benchmark average is
    computed across all target queries defined in qrels. Queries with no retrieved
    documents receive a score of 0.0.

    Args:
        qrels: Ground-truth relevance mapping {query_id: {doc_id: relevance_score}}.
        run: Model retrieval run predictions {query_id: {doc_id: similarity_score}}.
        k: Truncation cutoff rank (default: 10).

    Returns:
        Mean nDCG@K score in [0.0, 1.0] across all benchmark queries in qrels.
    """
    if not qrels:
        return 0.0

    metric_str = f"ndcg_cut.{k}"
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {metric_str})
    results = evaluator.evaluate(run)

    # Standard TREC evaluation: average over all benchmark queries in qrels
    query_scores = [
        float(results[qid][f"ndcg_cut_{k}"]) if qid in results else 0.0
        for qid in qrels
    ]
    return float(sum(query_scores) / len(qrels))
