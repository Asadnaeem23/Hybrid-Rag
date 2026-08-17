from typing import Optional

from src.embeddings import VoyageEmbedder
from src.vector_store import PineconeStore



class DenseRetriever:
    def __init__(
        self,
        index_name: str,
        top_k: int = 5,
        voyage_api_key: str | None = None,
        pinecone_api_key: str | None = None,
        embedder: VoyageEmbedder | None = None,
        vector_store: PineconeStore | None = None,
    ):
        self.embedder = embedder or VoyageEmbedder(api_key=voyage_api_key)
        self.vector_store = vector_store or PineconeStore(
            index_name,
            api_key=pinecone_api_key,
        )
        self.top_k = top_k

    def retrieve(self, query: str, top_k: Optional[int] = None):
        k = top_k if top_k is not None else self.top_k
        query_vector = self.embedder.embed_query(query)

        results = self.vector_store.search(
            vector=query_vector,
            top_k=k,
        )


        formatted_results = []
        for match in results["matches"]:
            if isinstance(match, dict):
                m_id = match["id"]
                score = match["score"]
                metadata = match.get("metadata", {})
            else:
                m_id = getattr(match, "id")
                score = getattr(match, "score")
                metadata = getattr(match, "metadata", {}) or {}

            formatted_results.append({
                "id": m_id,
                "score": float(score),
                "metadata": metadata,
                "text": metadata.get("text", "") if isinstance(metadata, dict) else "",
            })

        return formatted_results
