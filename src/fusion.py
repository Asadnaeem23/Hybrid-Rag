from typing import List, Dict, Any


def reciprocal_rank_fusion(
    result_lists: List[List[Dict[str, Any]]],
    k: int = 60,
):
    fused_scores = {}
    result_lookup = {}

    for results in result_lists:

        for rank, result in enumerate(results, start=1):

            chunk_id = result["id"] if isinstance(result, dict) else getattr(result, "id")

            # RRF score
            score = 1 / (k + rank)

            fused_scores[chunk_id] = (
                fused_scores.get(chunk_id, 0) + score
            )

            # Keep the actual result so we can return the text
            result_lookup[chunk_id] = result

    # Sort highest RRF score first
    ranked_ids = sorted(
        fused_scores,
        key=fused_scores.get,
        reverse=True,
    )

    fused_results = []

    for chunk_id in ranked_ids:

        raw_result = result_lookup[chunk_id]
        if isinstance(raw_result, dict):
            result = raw_result.copy()
        else:
            meta = getattr(raw_result, "metadata", {}) or {}
            result = {
                "id": getattr(raw_result, "id"),
                "score": float(getattr(raw_result, "score")),
                "metadata": meta,
                "text": meta.get("text", "") if isinstance(meta, dict) else "",
            }

        result["rrf_score"] = fused_scores[chunk_id]

        fused_results.append(result)

    return fused_results