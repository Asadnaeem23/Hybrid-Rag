from dataclasses import dataclass

from langchain_core.documents import Document

from src.config import RuntimeConfig
from src.graph import build_rag_graph
from src.hybrid_retrieval import HybridRetriever
from src.llm import get_generation_llm
from src.reranking import VoyageReranker
from src.retrieval import DenseRetriever
from src.sparse_retrieval import BM25Retriever


@dataclass
class RAGPipeline:
    graph: object
    hybrid_retriever: HybridRetriever


def build_hybrid_retriever(
    runtime_config: RuntimeConfig,
    sparse_chunks: list[Document] | None = None,
) -> HybridRetriever:
    dense_retriever = DenseRetriever(
        runtime_config.pinecone_index,
        voyage_api_key=runtime_config.voyage_api_key,
        pinecone_api_key=runtime_config.pinecone_api_key,
    )

    sparse_retriever = BM25Retriever(sparse_chunks or [])
    reranker = VoyageReranker(api_key=runtime_config.voyage_api_key)

    return HybridRetriever(
        dense_retriever=dense_retriever,
        sparse_retriever=sparse_retriever,
        reranker=reranker,
        dense_top_k=runtime_config.dense_top_k,
        sparse_top_k=runtime_config.sparse_top_k,
        rerank_top_k=runtime_config.rerank_top_k,
    )


def build_rag_pipeline(
    runtime_config: RuntimeConfig,
    sparse_chunks: list[Document] | None = None,
    checkpointer=None,
) -> RAGPipeline:
    runtime_config.require_chat_settings()

    hybrid_retriever = build_hybrid_retriever(
        runtime_config=runtime_config,
        sparse_chunks=sparse_chunks,
    )
    llm = get_generation_llm(runtime_config)
    
    try:
        from src.query_rewriter import get_query_rewriter
        rewriter = get_query_rewriter(api_key=runtime_config.groq_api_key)
        enable_rewrite = runtime_config.enable_query_rewrite
    except Exception:
        rewriter = None
        enable_rewrite = False
        
    graph = build_rag_graph(
        hybrid_retriever=hybrid_retriever,
        llm=llm,
        rewriter=rewriter,
        enable_rewrite=enable_rewrite,
        min_score=runtime_config.retrieval_min_score,
        checkpointer=checkpointer,
    )

    return RAGPipeline(
        graph=graph,
        hybrid_retriever=hybrid_retriever,
    )
