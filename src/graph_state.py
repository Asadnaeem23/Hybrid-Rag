from typing import Annotated, Any, Dict, List, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class RAGState(TypedDict):

    # Original user question
    query: str

    # Query produced by the rewriter
    rewritten_query: str

    # Retrieved and reranked documents
    retrieved_documents: List[Dict[str, Any]]

    # Whether we have already attempted rewriting
    rewrite_attempted: bool

    # Result of retrieval quality check
    retrieval_quality: str

    # Final answer
    answer: str

    # Recent conversational history
    messages: Annotated[List[BaseMessage], add_messages]
