from typing import List, Dict, Any

from src.retrieval import DenseRetriever
from src.sparse_retrieval import BM25Retriever
from src.fusion import reciprocal_rank_fusion
from src.reranking import VoyageReranker


class HybridRetriever:

    def __init__(
        self,
        dense_retriever: DenseRetriever,
        sparse_retriever: BM25Retriever,
        reranker: VoyageReranker,
        dense_top_k: int = 10,
        sparse_top_k: int = 10,
        rerank_top_k: int = 5,
    ):
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.reranker = reranker

        self.dense_top_k = dense_top_k
        self.sparse_top_k = sparse_top_k
        self.rerank_top_k = rerank_top_k

    def retrieve(
        self,
        query: str,
    ) -> List[Dict[str, Any]]:

        # --------------------------------------------------
        # 1. Dense retrieval
        # --------------------------------------------------

        dense_results = self.dense_retriever.retrieve(
            query,
            top_k=self.dense_top_k,
        )

        # --------------------------------------------------
        # 2. Sparse retrieval (BM25)
        # --------------------------------------------------

        sparse_results = self.sparse_retriever.retrieve(
            query,
            top_k=self.sparse_top_k,
        )

        # --------------------------------------------------
        # 3. Reciprocal Rank Fusion
        # --------------------------------------------------

        fused_results = reciprocal_rank_fusion(
            [
                dense_results,
                sparse_results,
            ]
        )

        # --------------------------------------------------
        # 4. Reranking
        # --------------------------------------------------

        reranked_results = self.reranker.rerank(
            query=query,
            results=fused_results,
            top_k=self.rerank_top_k,
        )

        return reranked_results