from langgraph.graph import END, START, StateGraph

from src.graph_nodes import (
    check_retrieval_node,
    generate_node,
    retrieve_node,
    rewrite_query_node,
    route_after_quality_check,
    trim_messages_node,
)
from src.graph_state import RAGState
from src.hybrid_retrieval import HybridRetriever


def build_rag_graph(
    hybrid_retriever: HybridRetriever,
    llm,
    rewriter=None,
    knowledge_context: str | None = None,
    min_score: float = 0.50,
    enable_rewrite: bool = True,
    checkpointer=None,
):
    builder = StateGraph(RAGState)

    builder.add_node(
        "retrieve",
        lambda state: retrieve_node(
            state,
            hybrid_retriever,
        ),
    )

    builder.add_node(
        "check_retrieval",
        lambda state: check_retrieval_node(
            state,
            min_score,
        ),
    )

    builder.add_node(
        "rewrite_query",
        lambda state: rewrite_query_node(
            state,
            rewriter,
            knowledge_context,
        ),
    )

    builder.add_node(
        "generate",
        lambda state: generate_node(
            state,
            llm,
        ),
    )

    builder.add_node(
        "trim_messages",
        trim_messages_node,
    )

    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "check_retrieval")

    if enable_rewrite and rewriter is not None:
        builder.add_conditional_edges(
            "check_retrieval",
            route_after_quality_check,
            {
                "generate": "generate",
                "rewrite": "rewrite_query",
            },
        )
        builder.add_edge("rewrite_query", "retrieve")
    else:
        builder.add_edge("check_retrieval", "generate")

    builder.add_edge("generate", "trim_messages")
    builder.add_edge("trim_messages", END)

    return builder.compile(checkpointer=checkpointer)
