from typing import List, Dict, Any
import voyageai

from src.config import VOYAGE_API_KEY


class VoyageReranker:

    def __init__(
        self,
        model: str = "rerank-2.5-lite",
        api_key: str | None = None,
    ):
        self.model = model

        key = api_key or VOYAGE_API_KEY

        self.client = voyageai.Client(
            api_key=key
        )

    def rerank(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_k: int = 5,
    ):
        if not results:
            return []

        documents = [
            result["text"]
            for result in results
        ]

        response = self.client.rerank(
            query=query,
            documents=documents,
            model=self.model,
            top_k=top_k,
        )

        reranked_results = []

        for item in response.results:

            original_result = results[item.index].copy()

            original_result["rerank_score"] = (
                item.relevance_score
            )

            reranked_results.append(
                original_result
            )

        return reranked_results