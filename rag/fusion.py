"""Merge ranked lists from different retrievers."""


def reciprocal_rank_fusion(
    ranked_lists: list[list[str]], k: int = 60, weights: list[float] | None = None
) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion: score(d) = sum_i w_i / (k + rank_i(d)), rank starting at 1.

    Uses ranks only, so BM25 scores and cosine similarities never need to be on
    the same scale. Returns (id, fused_score) best first; ties keep first-seen order.
    """
    weights = weights or [1.0] * len(ranked_lists)
    if len(weights) != len(ranked_lists):
        raise ValueError("weights must match ranked_lists")
    scores: dict[str, float] = {}
    for w, ids in zip(weights, ranked_lists):
        for rank, doc_id in enumerate(ids, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + w / (k + rank)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
