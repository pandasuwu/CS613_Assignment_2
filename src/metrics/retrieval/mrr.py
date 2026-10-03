import pytrec_eval


def compute_mrr_at_k(
    qrels: dict[str, dict[str, int]],
    run: dict[str, dict[str, float]],
    k: int = 10,
) -> float:
    """Compute Mean Reciprocal Rank truncated at rank k (MRR@K) using pytrec_eval.

    Mathematical Formulation:
        RR@K = 1 / rank_first_relevant  if rank <= K else 0.0
        MRR@K = (1 / |Q|) * sum_{q in Q} RR@K(q)

    In accordance with TREC and BEIR evaluation protocols, the benchmark average is
    computed across all target queries defined in qrels. Queries with no retrieved
    documents receive a score of 0.0.

    Args:
        qrels: Ground-truth relevance mapping {query_id: {doc_id: relevance_score}}.
        run: Model retrieval run predictions {query_id: {doc_id: similarity_score}}.
        k: Cutoff rank beyond which reciprocal rank is set to 0.0 (default: 10).

    Returns:
        Mean Reciprocal Rank score in [0.0, 1.0] across all benchmark queries in qrels.
    """
    if not qrels:
        return 0.0

    # Truncate run candidates to top-K per query prior to evaluating reciprocal rank
    truncated_run: dict[str, dict[str, float]] = {}
    for qid, doc_dict in run.items():
        sorted_docs = sorted(doc_dict.items(), key=lambda item: item[1], reverse=True)[:k]
        truncated_run[qid] = dict(sorted_docs)

    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {"recip_rank"})
    results = evaluator.evaluate(truncated_run)

    # Standard TREC evaluation: average over all benchmark queries in qrels
    query_scores = [
        float(results[qid]["recip_rank"]) if qid in results else 0.0
        for qid in qrels
    ]
    return float(sum(query_scores) / len(qrels))
