from typing import Any, Optional
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, RemoveMessage

from src.graph_state import RAGState
from src.hybrid_retrieval import HybridRetriever
from src.query_rewriter import rewrite_query


MAX_CONVERSATION_MESSAGES = 10


def _format_messages(messages: list[BaseMessage]) -> str:
    lines = []

    for message in messages[-MAX_CONVERSATION_MESSAGES:]:
        role = "Assistant" if isinstance(message, AIMessage) else "User"
        content = str(message.content).strip()
        if content:
            lines.append(f"{role}: {content}")

    return "\n".join(lines)

def retrieve_node(
    state: RAGState,
    hybrid_retriever: HybridRetriever,
) -> dict:

    query = state.get("rewritten_query") or state["query"]
    conversation = _format_messages(state.get("messages", []))

    retrieval_query = query
    if conversation:
        retrieval_query = f"""
Recent conversation:
{conversation}

Current question:
{query}
"""

    results = hybrid_retriever.retrieve(retrieval_query)

    return {
        "retrieved_documents": results
    }


def check_retrieval_node(
    state: RAGState,
    min_score: float,
) -> dict:

    documents = state.get("retrieved_documents", [])

    if not documents:
        quality = "BAD"
    else:
        max_score = max((doc.get("rerank_score", 0.0) for doc in documents), default=0.0)
        if max_score >= min_score:
            quality = "GOOD"
        else:
            quality = "BAD"

    return {
        "retrieval_quality": quality
    }


def rewrite_query_node(
    state: RAGState,
    rewriter: Any,
    knowledge_context: Optional[str] = None,
) -> dict:

    query = state["query"]

    rewritten = rewrite_query(
        query=query,
        llm=rewriter,
        knowledge_context=knowledge_context,
    )

    return {
        "rewritten_query": rewritten,
        "rewrite_attempted": True
    }


def route_after_quality_check(state: RAGState) -> str:
    quality = state.get("retrieval_quality")
    attempted = state.get("rewrite_attempted", False)

    if quality == "GOOD" or attempted:
        return "generate"
    return "rewrite"


def generate_node(
    state: RAGState,
    llm,
) -> dict:

    query = state["query"]
    conversation = _format_messages(state.get("messages", []))

    documents = state.get(
        "retrieved_documents",
        []
    )

    context_parts = []

    for i, document in enumerate(
        documents,
        start=1,
    ):

        context_parts.append(
            f"""
SOURCE {i}
ID: {document['id']}


{document['text']}
"""
        )

    context = "\n".join(context_parts)

    prompt = f"""
You are a helpful question-answering assistant.


Answer the user's current question using ONLY the retrieved document context
and the recent conversation history.


Rules:


- Do not invent facts.
- Do not use outside knowledge.
- If the context does not contain enough information,
  clearly say that the information is not available
  in the provided documents.
- Treat retrieved document context as the primary factual source.
- Give a direct and useful answer.
- Preserve important technical details.


RECENT CONVERSATION:
{conversation or "No previous conversation."}


USER QUESTION:
{query}


RETRIEVED DOCUMENT CONTEXT:
{context}


ANSWER:
"""

    response = llm.invoke(prompt)

    answer = response.content

    return {
        "answer": answer,
        "messages": [
            HumanMessage(content=query),
            AIMessage(content=answer),
        ],
    }


def trim_messages_node(state: RAGState) -> dict:
    messages = state.get("messages", [])
    excess_count = max(0, len(messages) - MAX_CONVERSATION_MESSAGES)

    if excess_count == 0:
        return {}

    return {
        "messages": [
            RemoveMessage(id=message.id)
            for message in messages[:excess_count]
            if message.id
        ]
    }
